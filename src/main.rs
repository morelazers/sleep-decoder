use crate::phase_analysis::SleepPhase;
use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use chrono::{DateTime, Utc};
use clap::Parser;
use env_logger;
use serde::{Deserialize, Serialize};
use serde_bytes;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use arrow::array::{Array, StringArray, ListArray, Int32Array};
use arrow::ipc::reader::FileReaderBuilder;

mod phase_analysis;

/// Process sleep data from RAW files
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory containing .RAW files or path to input CSV
    #[arg(help = "Directory containing .RAW files or path to input CSV")]
    input_path: PathBuf,

    /// Whether the input is a CSV file
    #[arg(long)]
    csv_input: bool,

    /// Whether the input is a Feather file
    #[arg(long)]
    feather_input: bool,

    /// Start time (format: YYYY-MM-DD HH:MM), defaults to 24 hours ago
    #[arg(long, env = "ANALYSIS_START_TIME")]
    start_time: Option<String>,

    /// End time (format: YYYY-MM-DD HH:MM), defaults to now
    #[arg(long, env = "ANALYSIS_END_TIME")]
    end_time: Option<String>,

    /// CSV output file prefix (e.g. /path/to/output/prefix)
    #[arg(long, env = "CSV_OUTPUT")]
    csv_output: Option<String>,

    /// Window size for smoothing HR results
    #[arg(long, default_value = "60")]
    hr_smoothing_window: usize,

    /// Heart rate window size in seconds
    #[arg(long, default_value = "10.0")]
    hr_window_seconds: f32,

    /// Heart rate window overlap percentage (0.0 to 1.0)
    #[arg(long, default_value = "0.1")]
    hr_window_overlap: f32,

    /// Breathing rate window size in seconds
    #[arg(long, default_value = "120.0")]
    br_window_seconds: f32,

    /// Breathing rate window overlap percentage (0.0 to 1.0)
    #[arg(long, default_value = "0.0")]
    br_window_overlap: f32,

    /// Merge left and right signals for analysis
    #[arg(long)]
    merge_sides: bool,

    /// Percentile threshold for HR outlier detection (0.0 to 0.5, default 0.05 means using 5th and 95th percentiles)
    #[arg(long, default_value = "0.05")]
    hr_outlier_percentile: f32,

    /// Number of recent heart rate measurements to consider for outlier detection
    #[arg(long, default_value = "60")]
    hr_history_window: usize,

    /// Base penalty for breathing harmonics when close to previous HR (0.0 to 1.0)
    #[arg(long, default_value = "0.8")]
    harmonic_penalty_close: f32,

    /// Base penalty for breathing harmonics when far from previous HR (0.0 to 1.0)
    #[arg(long, default_value = "0.5")]
    harmonic_penalty_far: f32,

    /// BPM difference threshold for "close" harmonic penalty (in BPM)
    #[arg(long, default_value = "5.0")]
    harmonic_close_threshold: f32,

    /// BPM difference threshold for "far" harmonic penalty (in BPM)
    #[arg(long, default_value = "10.0")]
    harmonic_far_threshold: f32,

    /// Smoothing strength for heart rate data (0.1 to 2.0, higher = more smoothing, default 0.25)
    #[arg(long, default_value = "0.25")]
    hr_smoothing_strength: f32,
}

#[derive(Debug, Deserialize)]
struct BatchItem {
    seq: u32,
    data: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BedTempSide {
    side: f32,
    out: f32,
    cen: f32,
    #[serde(rename = "in")]
    _in: f32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum SensorData {
    #[serde(rename = "piezo-dual")]
    PiezoDual {
        ts: i64,
        adc: u8,
        freq: u16,
        gain: u16,
        #[serde(with = "serde_bytes")]
        left1: Vec<u8>,
        #[serde(default)]
        #[serde(with = "serde_bytes")]
        left2: Option<Vec<u8>>,
        #[serde(with = "serde_bytes")]
        right1: Vec<u8>,
        #[serde(default)]
        #[serde(with = "serde_bytes")]
        right2: Option<Vec<u8>>,
    }
}

#[derive(Debug, Serialize, Copy, Clone)]
struct ProcessedData {
    timestamp: DateTime<Utc>,
    left1_mean: f32,
    left1_std: f32,
    left2_mean: f32,
    left2_std: f32,
    right1_mean: f32,
    right1_std: f32,
    right2_mean: f32,
    right2_std: f32,
}

#[derive(Debug, Clone)]
struct BedPresence {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    confidence: f32,
}

#[derive(Debug)]
struct PeriodAnalysis {
    fft_heart_rates: Vec<(DateTime<Utc>, f32)>, // Results from FFT analysis
    breathing_rates: Vec<(DateTime<Utc>, f32)>, // Results from breathing analysis
}

#[derive(Debug)]
struct SideAnalysis {
    combined: PeriodAnalysis,
    period_num: usize,
    sleep_phases: Vec<(DateTime<Utc>, SleepPhase)>,
}

#[derive(Debug)]
struct BedAnalysis {
    left_side: Vec<SideAnalysis>,
    right_side: Vec<SideAnalysis>,
}

pub struct RawDataView<'a> {
    raw_data: &'a [(u32, CombinedSensorData)],
    start_idx: usize,
    end_idx: usize,
}

impl<'a> RawDataView<'a> {
    pub fn get_data_at(&self, idx: usize) -> Option<RawPeriodData<'a>> {
        if idx >= self.end_idx - self.start_idx {
            return None;
        }

        let data = &self.raw_data[self.start_idx + idx].1;
        Some(RawPeriodData {
            timestamp: DateTime::from_timestamp(data.ts, 0).unwrap(),
            left: &data.left,
            right: &data.right,
        })
    }

    pub fn len(&self) -> usize {
        self.end_idx - self.start_idx
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug)]
pub struct RawPeriodData<'a> {
    pub timestamp: DateTime<Utc>,
    pub left: &'a [i32],
    pub right: &'a [i32],
}

fn create_raw_data_view<'a>(
    raw_sensor_data: &'a [(u32, CombinedSensorData)],
    period: &'_ BedPresence,
) -> RawDataView<'a> {
    let start_idx = raw_sensor_data
        .partition_point(|(_, data)| DateTime::from_timestamp(data.ts, 0).unwrap() < period.start);

    let end_idx = start_idx
        + raw_sensor_data[start_idx..].partition_point(|(_, data)| {
            DateTime::from_timestamp(data.ts, 0).unwrap() <= period.end
        });

    RawDataView {
        raw_data: raw_sensor_data,
        start_idx,
        end_idx,
    }
}

fn calculate_stats(data: &[i32]) -> (f32, f32) {
    // First remove outliers from raw data
    let cleaned_data = heart_analysis::interpolate_outliers(data, 2);

    let n = cleaned_data.len() as f32;
    if n == 0.0 {
        return (0.0, 0.0);
    }

    let mean = cleaned_data.iter().sum::<f32>() / n;
    let variance = cleaned_data
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / n;

    let std_dev = variance.sqrt();

    // Round to 2 decimal places like Python
    (
        (mean * 100.0).round() / 100.0,
        (std_dev * 100.0).round() / 100.0,
    )
}

fn process_piezo_data(data: &CombinedSensorData) -> Option<ProcessedData> {
    let timestamp = DateTime::from_timestamp(data.ts, 0).unwrap();

    // Calculate stats for combined signals
    let (left_mean, left_std) = calculate_stats(&data.left);
    let (right_mean, right_std) = calculate_stats(&data.right);

    Some(ProcessedData {
        timestamp,
        left1_mean: left_mean,
        left1_std: left_std,
        left2_mean: left_mean, // Use same values since signals are combined
        left2_std: left_std,
        right1_mean: right_mean,
        right1_std: right_std,
        right2_mean: right_mean, // Use same values since signals are combined
        right2_std: right_std,
    })
}

fn remove_stat_outliers_for_sensor(data: &mut Vec<ProcessedData>, sensor: &str) {
    // First collect the values we need
    let (mean_data, std_data): (Vec<f32>, Vec<f32>) = match sensor {
        "left1" => data.iter().map(|d| (d.left1_mean, d.left1_std)).unzip(),
        "left2" => data.iter().map(|d| (d.left2_mean, d.left2_std)).unzip(),
        "right1" => data.iter().map(|d| (d.right1_mean, d.right1_std)).unzip(),
        "right2" => data.iter().map(|d| (d.right2_mean, d.right2_std)).unzip(),
        _ => unreachable!(),
    };

    let len_before = data.len();

    // Calculate percentiles for mean
    let mean_percentiles = {
        let mut indices: Vec<usize> = (0..mean_data.len()).collect();
        indices.sort_by(|&a, &b| mean_data[a].partial_cmp(&mean_data[b]).unwrap());
        let mean_lower = mean_data[indices[(data.len() as f32 * 0.02) as usize]];
        let mean_upper = mean_data[indices[(data.len() as f32 * 0.98) as usize]];
        (mean_lower, mean_upper)
    };

    // Calculate percentiles for std
    let std_percentiles = {
        let mut indices: Vec<usize> = (0..std_data.len()).collect();
        indices.sort_by(|&a, &b| std_data[a].partial_cmp(&std_data[b]).unwrap());
        let std_lower = std_data[indices[(data.len() as f32 * 0.02) as usize]];
        let std_upper = std_data[indices[(data.len() as f32 * 0.98) as usize]];
        (std_lower, std_upper)
    };

    // Filter data
    data.retain(|d| {
        let (mean, std) = match sensor {
            "left1" => (d.left1_mean, d.left1_std),
            "left2" => (d.left2_mean, d.left2_std),
            "right1" => (d.right1_mean, d.right1_std),
            "right2" => (d.right2_mean, d.right2_std),
            _ => unreachable!(),
        };
        mean >= mean_percentiles.0
            && mean <= mean_percentiles.1
            && std >= std_percentiles.0
            && std <= std_percentiles.1
    });

    println!(
        "Removed {} rows as mean/std outliers for {}",
        len_before - data.len(),
        sensor
    );
    println!("Remaining rows: {}", data.len());
}

fn remove_stat_outliers(data: &mut Vec<ProcessedData>) {
    // Process each sensor independently
    for sensor in ["left1", "left2", "right1", "right2"] {
        remove_stat_outliers_for_sensor(data, sensor);
    }
}

fn remove_time_outliers(data: &mut Vec<ProcessedData>) {
    if data.is_empty() {
        return;
    }

    // Convert timestamps to i64 for calculations
    let timestamps: Vec<i64> = data.iter().map(|d| d.timestamp.timestamp()).collect();

    // Sort for percentile calculation
    let mut sorted = timestamps.clone();
    sorted.sort();

    // Calculate 1st and 99th percentiles like Python
    let lower_idx = (timestamps.len() as f32 * 0.01) as usize;
    let upper_idx = (timestamps.len() as f32 * 0.99) as usize;
    let lower_bound = sorted[lower_idx];
    let upper_bound = sorted[upper_idx];

    let len_before = data.len();
    data.retain(|d| {
        let ts = d.timestamp.timestamp();
        ts >= lower_bound && ts <= upper_bound
    });

    println!("Removed {} rows as date outliers", len_before - data.len());
    println!("Remaining rows: {}", data.len());
}

fn decode_batch_item(file_path: &PathBuf) -> Result<Vec<(u32, SensorData)>> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut items = Vec::new();

    loop {
        let batch_item: BatchItem = match ciborium::from_reader(&mut reader) {
            Ok(item) => item,
            Err(ciborium::de::Error::Io(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                eprintln!("Warning: Skipping malformed CBOR data: {}", e);
                continue;
            }
        };

        match ciborium::from_reader(batch_item.data.as_slice()) {
            Ok(sensor_data) => {
                items.push((batch_item.seq, sensor_data));
            }
            Err(_) => {
                continue;
            }
        }
    }

    Ok(items)
}

fn scale_data(data: &mut Vec<ProcessedData>) {
    // Helper functions remain the same
    fn median(values: &[f32]) -> f32 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    fn quartiles(values: &[f32]) -> (f32, f32) {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1_pos = (sorted.len() as f32 * 0.25) as usize;
        let q3_pos = (sorted.len() as f32 * 0.75) as usize;
        (sorted[q1_pos], sorted[q3_pos])
    }

    // Scale each field individually
    let fields = [
        "left1_mean",
        "left2_mean",
        "right1_mean",
        "right2_mean",
        "left1_std",
        "left2_std",
        "right1_std",
        "right2_std",
    ];

    for field in fields.iter() {
        // Get values for this field
        let values: Vec<f32> = match *field {
            "left1_mean" => data.iter().map(|d| d.left1_mean).collect(),
            "left2_mean" => data.iter().map(|d| d.left2_mean).collect(),
            "right1_mean" => data.iter().map(|d| d.right1_mean).collect(),
            "right2_mean" => data.iter().map(|d| d.right2_mean).collect(),
            "left1_std" => data.iter().map(|d| d.left1_std).collect(),
            "left2_std" => data.iter().map(|d| d.left2_std).collect(),
            "right1_std" => data.iter().map(|d| d.right1_std).collect(),
            "right2_std" => data.iter().map(|d| d.right2_std).collect(),
            _ => unreachable!(),
        };

        let center = median(&values);
        let (q1, q3) = quartiles(&values);
        let iqr = q3 - q1;

        // Apply scaling - just like RobustScaler
        for d in data.iter_mut() {
            let value = match *field {
                "left1_mean" => &mut d.left1_mean,
                "left2_mean" => &mut d.left2_mean,
                "right1_mean" => &mut d.right1_mean,
                "right2_mean" => &mut d.right2_mean,
                "left1_std" => &mut d.left1_std,
                "left2_std" => &mut d.left2_std,
                "right1_std" => &mut d.right1_std,
                "right2_std" => &mut d.right2_std,
                _ => unreachable!(),
            };

            *value = if iqr != 0.0 {
                (*value - center) / iqr // This is the key change - matches RobustScaler
            } else {
                *value - center
            };
        }
    }
}

fn detect_bed_presence(data: &[ProcessedData], side: &str) -> Vec<BedPresence> {
    // First, calculate the overall std distribution for this side
    let stds: Vec<f32> = match side {
        "left" => data
            .iter()
            .map(|d| (d.left1_std + d.left2_std) / 2.0)
            .collect(),
        "right" => data
            .iter()
            .map(|d| (d.right1_std + d.right2_std) / 2.0)
            .collect(),
        _ => unreachable!(),
    };

    // Calculate adaptive threshold using percentiles
    let mut sorted_stds = stds.clone();
    sorted_stds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = sorted_stds[(sorted_stds.len() as f32 * 0.60) as usize]; // Keep 60th percentile

    // Calculate rolling average
    let window_size = 10;
    let rolling_stds: Vec<f32> = stds
        .windows(window_size)
        .map(|window| window.iter().sum::<f32>() / window_size as f32)
        .collect();

    // Find continuous periods above threshold
    let mut periods = Vec::new();
    let mut current_start: Option<(DateTime<Utc>, usize)> = None;
    let min_duration = chrono::Duration::minutes(10); // Keep 10 minutes minimum
    let max_gap = chrono::Duration::minutes(60); // Increased to 60 minutes for deep sleep periods

    let mut i = 0;
    while i < rolling_stds.len() {
        let is_active = rolling_stds[i] > threshold;

        match (is_active, &current_start) {
            (true, None) => {
                current_start = Some((data[i].timestamp, i));
            }
            (false, Some((start_time, start_idx))) => {
                // Look ahead for activity within max_gap
                let mut gap_length = 1;
                let mut found_activity = false;
                while i + gap_length < rolling_stds.len()
                    && data[i + gap_length].timestamp - data[i].timestamp <= max_gap
                {
                    if rolling_stds[i + gap_length] > threshold {
                        found_activity = true;
                        i += gap_length;
                        break;
                    }
                    gap_length += 1;
                }

                if !found_activity {
                    // End the current period
                    let duration = data[i].timestamp - *start_time;
                    if duration >= min_duration {
                        let period_stds = &rolling_stds[*start_idx..i];
                        let avg_elevation = period_stds
                            .iter()
                            .map(|s| (s - threshold).max(0.0))
                            .sum::<f32>()
                            / period_stds.len() as f32;

                        periods.push(BedPresence {
                            start: *start_time,
                            end: data[i].timestamp,
                            confidence: avg_elevation,
                        });
                    }
                    current_start = None;
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Handle case where presence extends to end of data
    if let Some((start_time, start_idx)) = current_start {
        let end_time = data.last().unwrap().timestamp;
        let duration = end_time - start_time;
        if duration >= min_duration {
            let period_stds = &rolling_stds[start_idx..];
            let avg_elevation = period_stds
                .iter()
                .map(|s| (s - threshold).max(0.0))
                .sum::<f32>()
                / period_stds.len() as f32;

            periods.push(BedPresence {
                start: start_time,
                end: end_time,
                confidence: avg_elevation,
            });
        }
    }

    // Merge periods that are close together (also using 30-minute max gap)
    let mut merged_periods = Vec::new();
    let mut current_period: Option<BedPresence> = None;

    for period in periods {
        match &mut current_period {
            Some(curr) => {
                if period.start - curr.end <= max_gap {
                    // Merge periods
                    curr.end = period.end;
                    curr.confidence = (curr.confidence + period.confidence) / 2.0;
                } else {
                    // Start new period
                    merged_periods.push(curr.clone());
                    current_period = Some(period);
                }
            }
            None => current_period = Some(period),
        }
    }

    if let Some(period) = current_period {
        merged_periods.push(period);
    }

    merged_periods
}

fn extract_raw_data_for_period<'a>(
    raw_sensor_data: &'a [(u32, CombinedSensorData)],
    period: &BedPresence,
) -> RawDataView<'a> {
    create_raw_data_view(raw_sensor_data, period)
}

mod heart_analysis;

fn analyse_sensor_data(
    signal: &[i32],
    raw_data: &RawDataView,
    samples_per_segment_hr: usize,
    step_size_hr: usize,
    samples_per_segment_br: usize,
    step_size_br: usize,
    hr_outlier_percentile: f32,
    hr_history_window: usize,
    harmonic_penalty_close: f32,
    harmonic_penalty_far: f32,
    harmonic_close_threshold: f32,
    harmonic_far_threshold: f32,
) -> PeriodAnalysis {
    let mut fft_heart_rates = Vec::new();
    let mut breathing_rates = Vec::new();
    let mut prev_fft_hr = None;

    // Create FFT contexts for both heart rate and breathing rate
    let mut hr_fft_context = heart_analysis::FftContext::new(samples_per_segment_hr);
    let mut br_fft_context = heart_analysis::FftContext::new(samples_per_segment_br);
    let mut hr_history = heart_analysis::HeartRateHistory::new(15.0 * 60.0);

    // Process breathing rate windows first...
    let br_windows = heart_analysis::SignalWindowIterator::new(
        signal,
        raw_data,
        samples_per_segment_br,
        step_size_br,
    );

    // Collect breathing rates
    for window in br_windows {
        if let Some(breathing_rate) = heart_analysis::analyze_breathing_rate_fft(
            &window.processed_signal,
            500.0,
            &mut br_fft_context,
        ) {
            breathing_rates.push((window.timestamp, breathing_rate));
        }
    }

    // Calculate breathing rate stability using a rolling window
    let window_size = 5; // Use 5 measurements for stability calculation
    let breathing_rates_with_stability: Vec<(DateTime<Utc>, f32, f32)> = breathing_rates
        .iter()
        .enumerate()
        .map(|(i, &(timestamp, rate))| {
            // Get window of measurements centered on current point
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(breathing_rates.len());
            let window = &breathing_rates[start..end];

            // Calculate local variance
            let mean = window.iter().map(|(_, r)| r).sum::<f32>() / window.len() as f32;
            let variance =
                window.iter().map(|(_, r)| (r - mean).powi(2)).sum::<f32>() / window.len() as f32;

            // Convert variance to stability score (0 to 1)
            let stability = 1.0 / (1.0 + variance);

            (timestamp, rate, stability)
        })
        .collect();

    // Process heart rate windows...
    let hr_windows = heart_analysis::SignalWindowIterator::new(
        signal,
        raw_data,
        samples_per_segment_hr,
        step_size_hr,
    );

    let time_step = samples_per_segment_hr as f32 / 500.0;

    for window in hr_windows {
        // Find the closest breathing rate measurement with stability
        let breathing_data = breathing_rates_with_stability
            .iter()
            .min_by_key(|(br_time, _, _)| {
                (br_time.timestamp() - window.timestamp.timestamp()).abs() as u64
            })
            .map(|(_, rate, stability)| (*rate, *stability));

        if let Some(fft_hr) = heart_analysis::analyze_heart_rate_fft(
            &window.processed_signal,
            500.0,
            prev_fft_hr,
            window.timestamp,
            &mut hr_history,
            &mut hr_fft_context,
            time_step,
            breathing_data,
            hr_outlier_percentile,
            hr_history_window,
            harmonic_penalty_close,
            harmonic_penalty_far,
            harmonic_close_threshold,
            harmonic_far_threshold,
        ) {
            fft_heart_rates.push((window.timestamp, fft_hr));
            prev_fft_hr = Some(fft_hr);
        }
    }

    PeriodAnalysis {
        fft_heart_rates,
        breathing_rates,
    }
}

fn analyze_bed_presence_periods(
    raw_sensor_data: &[(u32, CombinedSensorData)],
    left_periods: &[BedPresence],
    right_periods: &[BedPresence],
    args: &Args,
) -> Result<BedAnalysis> {
    let mut bed_analysis = BedAnalysis {
        left_side: Vec::new(),
        right_side: Vec::new(),
    };

    println!("\nAnalyzing bed presence periods:");

    let samples_per_segment_hr = (args.hr_window_seconds * 500.0) as usize;
    let overlap_samples_hr = (samples_per_segment_hr as f32 * args.hr_window_overlap) as usize;
    let step_size_hr = samples_per_segment_hr - overlap_samples_hr;

    let samples_per_segment_br = (args.br_window_seconds * 500.0) as usize;
    let overlap_samples_br = (samples_per_segment_br as f32 * args.br_window_overlap) as usize;
    let step_size_br = samples_per_segment_br - overlap_samples_br;

    println!("\n  Processing with:");
    println!(
        "    (hr) Segment width : {} seconds",
        args.hr_window_seconds
    );
    println!("    (hr) Overlap: {}%", args.hr_window_overlap * 100.0);
    println!("    (hr) Samples per segment: {}", samples_per_segment_hr);
    println!("    (hr) Overlap samples: {}", overlap_samples_hr);
    println!("    (hr) Step size: {} samples", step_size_hr);

    println!("\n  --:");
    println!(
        "    (br) Segment width : {} seconds",
        args.br_window_seconds
    );
    println!("    (br) Overlap: {}%", args.br_window_overlap * 100.0);
    println!("    (br) Samples per segment: {}", samples_per_segment_br);
    println!("    (br) Overlap samples: {}", overlap_samples_br);
    println!("    (br) Step size: {} samples", step_size_br);

    // Analyze left side periods
    for (i, period) in left_periods.iter().enumerate() {
        println!("\nLeft side period {}", i + 1);
        println!(
            "  {} to {}",
            period.start.format("%Y-%m-%d %H:%M"),
            period.end.format("%Y-%m-%d %H:%M")
        );
        println!(
            "  Duration: {} minutes",
            (period.end - period.start).num_minutes()
        );

        // Extract and analyze raw data for the entire period
        let raw_data_view = extract_raw_data_for_period(raw_sensor_data, period);

        // Extract signals
        let mut combined_signal: Vec<i32> = (0..raw_data_view.len())
            .filter_map(|idx| raw_data_view.get_data_at(idx))
            .map(|data| data.left.to_vec())
            .flatten()
            .collect();

        // If merging sides, add right side data
        if args.merge_sides {
            let right_signal: Vec<i32> = (0..raw_data_view.len())
                .filter_map(|idx| raw_data_view.get_data_at(idx))
                .map(|data| data.right.to_vec())
                .flatten()
                .collect();

            // Average the signals
            for (left, right) in combined_signal.iter_mut().zip(right_signal.iter()) {
                *left = ((*left as i64 + *right as i64) / 2) as i32;
            }
        }

        let analysis_combined = analyse_sensor_data(
            &combined_signal,
            &raw_data_view,
            samples_per_segment_hr,
            step_size_hr,
            samples_per_segment_br,
            step_size_br,
            args.hr_outlier_percentile,
            args.hr_history_window,
            args.harmonic_penalty_close,
            args.harmonic_penalty_far,
            args.harmonic_close_threshold,
            args.harmonic_far_threshold,
        );

        // Analyze sleep phases starting 30 minutes after period start
        let sleep_onset = period.start;
        let sleep_phases = phase_analysis::analyze_sleep_phases(&analysis_combined, sleep_onset);

        // Print sleep phase summary
        phase_analysis::summarize_sleep_phases(&sleep_phases);

        bed_analysis.left_side.push(SideAnalysis {
            combined: analysis_combined,
            period_num: i,
            sleep_phases,
        });
    }

    // Only analyze right side if not merging
    if !args.merge_sides {
        // Analyze right side periods
        for (i, period) in right_periods.iter().enumerate() {
            println!("\nRight side period {}", i + 1);
            println!(
                "  {} to {}",
                period.start.format("%Y-%m-%d %H:%M"),
                period.end.format("%Y-%m-%d %H:%M")
            );
            println!(
                "  Duration: {} minutes",
                (period.end - period.start).num_minutes()
            );

            // Extract and analyze raw data for the entire period
            let raw_data_view = extract_raw_data_for_period(raw_sensor_data, period);

            // Extract signals
            let combined_signal: Vec<i32> = (0..raw_data_view.len())
                .filter_map(|idx| raw_data_view.get_data_at(idx))
                .map(|data| data.right.to_vec())
                .flatten()
                .collect();

            let analysis_combined = analyse_sensor_data(
                &combined_signal,
                &raw_data_view,
                samples_per_segment_hr,
                step_size_hr,
                samples_per_segment_br,
                step_size_br,
                args.hr_outlier_percentile,
                args.hr_history_window,
                args.harmonic_penalty_close,
                args.harmonic_penalty_far,
                args.harmonic_close_threshold,
                args.harmonic_far_threshold,
            );

            // Analyze sleep phases starting 30 minutes after period start
            let sleep_onset = period.start;
            let sleep_phases =
                phase_analysis::analyze_sleep_phases(&analysis_combined, sleep_onset);

            // Print sleep phase summary
            phase_analysis::summarize_sleep_phases(&sleep_phases);

            println!("  Detected {} sleep phase transitions", sleep_phases.len());

            bed_analysis.right_side.push(SideAnalysis {
                combined: analysis_combined,
                period_num: i,
                sleep_phases,
            });
        }
    }

    Ok(bed_analysis)
}

fn write_analysis_to_csv(
    base_path: &str,
    sensor_id: &str,
    period_num: usize,
    analysis: &PeriodAnalysis,
    hr_smoothing_window: usize,
    hr_smoothing_strength: f32,
    sleep_phases: &[(DateTime<Utc>, SleepPhase)],
) -> Result<()> {
    let path = std::path::Path::new(base_path);
    let dir = path.parent().unwrap_or(std::path::Path::new("."));

    // Create directory if it doesn't exist
    std::fs::create_dir_all(dir)?;

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("results");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("csv");

    let filename = format!("{}_{}_period_{}.{}", stem, sensor_id, period_num, ext);
    let full_path = dir.join(filename);

    println!("Writing results to {}", full_path.display());
    let file = std::fs::File::create(full_path)?;
    let mut writer = csv::Writer::from_writer(file);

    // Collect all timestamps
    let mut timestamps: Vec<DateTime<Utc>> = Vec::new();
    timestamps.extend(analysis.fft_heart_rates.iter().map(|(t, _)| *t));
    timestamps.extend(analysis.breathing_rates.iter().map(|(t, _)| *t));
    timestamps.extend(sleep_phases.iter().map(|(t, _)| *t));
    timestamps.sort_unstable();
    timestamps.dedup();

    // Interpolate and smooth heart rates
    let smoothed_fft_hr = heart_analysis::interpolate_and_smooth(
        &timestamps,
        &analysis.fft_heart_rates,
        hr_smoothing_window,
        hr_smoothing_strength,
    );

    // Write header
    writer.write_record(&[
        "timestamp",
        "fft_hr",
        "fft_hr_smoothed",
        "breathing_rate",
        "sleep_phase",
    ])?;

    // Write data for each timestamp
    for &timestamp in &timestamps {
        let fft_hr = analysis
            .fft_heart_rates
            .iter()
            .find(|(t, _)| *t == timestamp)
            .map(|(_, hr)| hr.to_string())
            .unwrap_or_default();

        let fft_hr_smoothed = smoothed_fft_hr
            .iter()
            .find(|(t, _)| *t == timestamp)
            .map(|(_, hr)| hr.to_string())
            .unwrap_or_default();

        let br = analysis
            .breathing_rates
            .iter()
            .find(|(t, _)| *t == timestamp)
            .map(|(_, br)| br.to_string())
            .unwrap_or_default();

        // Find the current sleep phase
        let phase = sleep_phases
            .iter()
            .rev()
            .find(|(t, _)| *t <= timestamp)
            .map(|(_, phase)| format!("{:?}", phase))
            .unwrap_or_default();

        writer.write_record(&[
            timestamp.format("%Y-%m-%d %H:%M").to_string(),
            fft_hr,
            fft_hr_smoothed,
            br,
            phase,
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn parse_datetime(datetime_str: &str) -> Result<DateTime<Utc>> {
    let naive = NaiveDateTime::parse_from_str(datetime_str, "%Y-%m-%d %H:%M")?;
    Ok(DateTime::from_naive_utc_and_offset(naive, Utc))
}

#[derive(Debug)]
struct RawFileInfo {
    path: PathBuf,
    first_seq: u32,
    last_seq: u32,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
}

fn build_raw_file_index(raw_dir: &PathBuf) -> Result<Vec<RawFileInfo>> {
    let mut file_index = Vec::new();

    for entry in std::fs::read_dir(raw_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("RAW") {
            println!("Indexing file: {}", path.display());

            // Load file to extract metadata
            let items = decode_batch_item(&path)?;

            if items.is_empty() {
                println!("Skipping empty file: {}", path.display());
                continue;
            }

            // Find sequence range and time range
            let mut first_seq = u32::MAX;
            let mut last_seq = 0;
            let mut start_time = DateTime::<Utc>::MAX_UTC;
            let mut end_time = DateTime::<Utc>::MIN_UTC;

            for (seq, data) in &items {
                if let SensorData::PiezoDual { ts, .. } = data {
                    first_seq = first_seq.min(*seq);
                    last_seq = last_seq.max(*seq);

                    let timestamp = DateTime::from_timestamp(*ts, 0).unwrap();
                    start_time = start_time.min(timestamp);
                    end_time = end_time.max(timestamp);
                }
            }

            file_index.push(RawFileInfo {
                path,
                first_seq,
                last_seq,
                start_time,
                end_time,
            });
        }
    }

    // Sort by first sequence number
    file_index.sort_by_key(|info| info.first_seq);

    println!("\nFound {} RAW files:", file_index.len());
    for info in &file_index {
        println!(
            "  {} (seq: {} to {}, time: {} to {})",
            info.path.file_name().unwrap().to_string_lossy(),
            info.first_seq,
            info.last_seq,
            info.start_time.format("%Y-%m-%d %H:%M:%S"),
            info.end_time.format("%Y-%m-%d %H:%M:%S")
        );
    }

    Ok(file_index)
}

#[derive(Debug)]
struct CombinedSensorData {
    ts: i64,
    left: Vec<i32>,
    right: Vec<i32>,
}

impl<'a> From<&'a SensorData> for Option<CombinedSensorData> {
    fn from(data: &'a SensorData) -> Option<CombinedSensorData> {
        if let SensorData::PiezoDual {
            ts,
            left1,
            left2,
            right1,
            right2,
            ..
        } = data
        {
            // Convert left side data
            let left: Vec<i32> = match left2 {
                // If we have left2, average left1 and left2
                Some(left2_data) => left1
                    .chunks_exact(4)
                    .zip(left2_data.chunks_exact(4))
                    .map(|(a, b)| {
                        let val1 = i32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                        let val2 = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        ((val1 as i64 + val2 as i64) / 2) as i32
                    })
                    .collect(),
                // If no left2, just convert left1
                None => left1
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            };

            // Convert right side data
            let right: Vec<i32> = match right2 {
                // If we have right2, average right1 and right2
                Some(right2_data) => right1
                    .chunks_exact(4)
                    .zip(right2_data.chunks_exact(4))
                    .map(|(a, b)| {
                        let val1 = i32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                        let val2 = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        ((val1 as i64 + val2 as i64) / 2) as i32
                    })
                    .collect(),
                // If no right2, just convert right1
                None => right1
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            };

            Some(CombinedSensorData {
                ts: *ts,
                left,
                right,
            })
        } else {
            None
        }
    }
}

// Add new function to read CSV data
fn read_csv_file(path: &PathBuf) -> Result<Vec<(u32, CombinedSensorData)>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)  // Handle variable number of fields
        .from_reader(file);
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result?;

        // Parse the timestamp from datetime string
        let dt = NaiveDateTime::parse_from_str(record.get(1).unwrap(), "%Y-%m-%d %H:%M:%S")?;
        let ts = dt.timestamp();

        let freq: u16 = record.get(2).unwrap().parse()?;
        let adc: u8 = record.get(3).unwrap().parse()?;
        let gain: u16 = record.get(4).unwrap().parse()?;

        // Parse left1 and right1 as space-separated integers
        let parse_signal = |s: &str| -> Result<Vec<i32>> {
            // Remove square brackets and split on whitespace
            let cleaned = s.trim_matches(|c| c == '[' || c == ']' || c == ' ')
                .split_whitespace()  // This handles multiple spaces and newlines
                .filter(|s| !s.is_empty());

            cleaned
                .map(|s| s.parse::<i32>())
                .collect::<Result<Vec<i32>, _>>()
                .map_err(|e| anyhow::anyhow!("Failed to parse signal data: {}", e))
        };

        let left1 = parse_signal(record.get(5).unwrap())?;
        let right1 = parse_signal(record.get(6).unwrap())?;
        let seq: u32 = record.get(7).unwrap().parse()?;

        // Create CombinedSensorData
        let combined = CombinedSensorData {
            ts,
            left: left1,
            right: right1,
        };

        data.push((seq, combined));
    }

    Ok(data)
}

// Add new function to read Feather data
fn read_feather_file(path: &PathBuf) -> Result<Vec<(u32, CombinedSensorData)>> {
    let file = File::open(path)?;
    let reader = FileReaderBuilder::new().build(file)?;
    let mut data = Vec::new();
    let mut seq = 0;

    for batch in reader {
        let batch = batch?;

        // Get column arrays
        let type_col = batch.column_by_name("type").expect("type column missing")
            .as_any().downcast_ref::<StringArray>().expect("type column should be strings");
        let ts_col = batch.column_by_name("ts").expect("ts column missing")
            .as_any().downcast_ref::<StringArray>().expect("ts column should be strings");
        let left1_col = batch.column_by_name("left1").expect("left1 column missing")
            .as_any().downcast_ref::<ListArray>().expect("left1 column should be a list");
        let right1_col = batch.column_by_name("right1").expect("right1 column missing")
            .as_any().downcast_ref::<ListArray>().expect("right1 column should be a list");

        // Process each row
        for row in 0..batch.num_rows() {
            // Only process piezo-dual records
            if type_col.value(row) == "piezo-dual" {
                // Parse timestamp from string
                let ts = NaiveDateTime::parse_from_str(ts_col.value(row), "%Y-%m-%d %H:%M:%S")?
                    .timestamp();

                // Get left and right signals as i32 arrays
                let left_values = left1_col.value(row);
                let right_values = right1_col.value(row);

                let left = left_values.as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("left values should be i32")
                    .values()
                    .to_vec();

                let right = right_values.as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("right values should be i32")
                    .values()
                    .to_vec();

                let combined = CombinedSensorData {
                    ts,
                    left,
                    right,
                };

                data.push((seq, combined));
                seq += 1;
            }
        }
    }

    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logger
    env_logger::init();

    let args = Args::parse();

    // Only parse and apply time range if provided
    let (start_time, end_time) = if args.start_time.is_some() || args.end_time.is_some() {
        let end_time = match args.end_time {
            Some(ref t) => parse_datetime(t)?,
            None => Utc::now(),
        };

        let start_time = match args.start_time {
            Some(ref t) => parse_datetime(t)?,
            None => end_time - chrono::Duration::days(1),
        };

        println!("\nAnalyzing data from {} to {}", start_time, end_time);
        (Some(start_time), Some(end_time))
    } else {
        (None, None)
    };

    // Load raw sensor data based on input type
    let raw_sensor_data = if args.csv_input {
        println!("Reading from CSV file: {}", args.input_path.display());
        let mut data = read_csv_file(&args.input_path)?;

        println!("Loaded {} rows from CSV", data.len());

        // Only filter by time range if provided
        if let (Some(start), Some(end)) = (start_time, end_time) {
            data.retain(|(_, combined)| {
                let timestamp = DateTime::from_timestamp(combined.ts, 0).unwrap();
                timestamp >= start && timestamp <= end
            });
        }
        data
    } else if args.feather_input {
        println!("Reading from Feather file: {}", args.input_path.display());
        let mut data = read_feather_file(&args.input_path)?;

        println!("Loaded {} rows from Feather file", data.len());

        // Only filter by time range if provided
        if let (Some(start), Some(end)) = (start_time, end_time) {
            data.retain(|(_, combined)| {
                let timestamp = DateTime::from_timestamp(combined.ts, 0).unwrap();
                timestamp >= start && timestamp <= end
            });
        }
        data
    } else {
        println!("Building RAW file index...");
        let file_index = build_raw_file_index(&args.input_path)?;

        if file_index.is_empty() {
            println!("No RAW files found");
            return Ok(());
        }

        // Only filter files by time range if provided
        let relevant_files = if let (Some(start), Some(end)) = (start_time, end_time) {
            file_index
                .into_iter()
                .filter(|info| info.end_time >= start && info.start_time <= end)
                .collect()
        } else {
            file_index
        };

        println!(
            "\nProcessing {} RAW files",
            relevant_files.len()
        );

        // Load and process files in sequence order
        let mut data = Vec::new();
        for file_info in relevant_files {
            println!("Loading file: {}", file_info.path.display());
            let items = decode_batch_item(&file_info.path)?;

            // Filter data by timestamp and convert to combined format
            for (seq, sensor_data) in items {
                if let Some(combined) = Option::<CombinedSensorData>::from(&sensor_data) {
                    if let (Some(start), Some(end)) = (start_time, end_time) {
                        let timestamp = DateTime::from_timestamp(combined.ts, 0).unwrap();
                        if timestamp >= start && timestamp <= end {
                            data.push((seq, combined));
                        }
                    } else {
                        data.push((seq, combined));
                    }
                }
            }
        }
        data
    };

    // Sort by sequence number and timestamp
    let mut raw_sensor_data = raw_sensor_data;
    raw_sensor_data.sort_by_key(|(seq, data)| (*seq, data.ts));

    println!("Loaded {} raw sensor data points", raw_sensor_data.len());

    // Process all sensor data at once
    let mut all_processed_data: Vec<ProcessedData> = raw_sensor_data
        .iter()
        .map(|(_, data)| process_piezo_data(data))
        .flatten()
        .collect();

    // First remove time outliers
    remove_time_outliers(&mut all_processed_data);

    // Then remove stat outliers
    let len_before = all_processed_data.len();
    remove_stat_outliers(&mut all_processed_data);

    println!(
        "Removed {} rows as mean/std outliers",
        len_before - all_processed_data.len()
    );

    println!("Total processed entries: {}", all_processed_data.len());

    // Scale the data
    scale_data(&mut all_processed_data);

    // Determine bed presence periods based on whether time window is specified
    let (left_periods, right_periods) = if let (Some(start), Some(end)) = (start_time, end_time) {
        // Create a single period for both sides
        let single_period = BedPresence {
            start,
            end,
            confidence: 1.0,
        };

        // Use the same period for both sides
        (vec![single_period.clone()], vec![single_period])
    } else {
        // Use detection method when no time window is specified
        (
            detect_bed_presence(&all_processed_data, "left"),
            detect_bed_presence(&all_processed_data, "right"),
        )
    };

    // Print period information
    println!("\nDetected bed presence periods:");
    println!("Left side periods: {}", left_periods.len());
    println!("Right side periods: {}", right_periods.len());

    let bed_analysis =
        analyze_bed_presence_periods(&raw_sensor_data, &left_periods, &right_periods, &args)?;

    // Write CSV files if output prefix was provided
    if let Some(prefix) = &args.csv_output {
        for analysis in &bed_analysis.left_side {
            write_analysis_to_csv(
                prefix,
                "left_combined",
                analysis.period_num,
                &analysis.combined,
                args.hr_smoothing_window,
                args.hr_smoothing_strength,
                &analysis.sleep_phases,
            )?;
        }

        for analysis in &bed_analysis.right_side {
            write_analysis_to_csv(
                prefix,
                "right_combined",
                analysis.period_num,
                &analysis.combined,
                args.hr_smoothing_window,
                args.hr_smoothing_strength,
                &analysis.sleep_phases,
            )?;
        }
    }

    Ok(())
}
