use anyhow::Context;
use chrono::NaiveDateTime;
use chrono::{DateTime, Utc};
use clap::Parser;
use env_logger;
use log::trace;
use serde::{Deserialize, Serialize};
use serde_bytes;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

/// Process sleep data from RAW files
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory containing .RAW files
    #[arg(help = "Directory containing .RAW files")]
    raw_dir: PathBuf,

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
    #[arg(long, env = "HR_SMOOTHING_WINDOW", default_value = "60")]
    hr_smoothing_window: Option<usize>,

    /// Split sensors into separate CSV files
    #[arg(long, env = "SPLIT_SENSORS")]
    split_sensors: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct BatchItem {
    seq: u32,
    data: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct BedTempSide {
    _side: f32,
    _out: f32,
    _cen: f32,
    #[serde(rename = "in")]
    _in: f32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum SensorData {
    #[serde(rename = "piezo-dual")]
    PiezoDual {
        ts: i64,
        adc: u8,
        freq: u16,
        gain: u16,
        #[serde(with = "serde_bytes")]
        left1: Vec<u8>,
        #[serde(with = "serde_bytes")]
        left2: Vec<u8>,
        #[serde(with = "serde_bytes")]
        right1: Vec<u8>,
        #[serde(with = "serde_bytes")]
        right2: Vec<u8>,
    },
    #[serde(rename = "bedTemp")]
    BedTemp {
        _ts: i64,
        _mcu: f32,
        _amb: f32,
        _left: BedTempSide,
        _right: BedTempSide,
    },
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
    sensor1: PeriodAnalysis,
    sensor2: PeriodAnalysis,
    combined: PeriodAnalysis,
    period_num: usize,
}

#[derive(Debug)]
struct BedAnalysis {
    left_side: Vec<SideAnalysis>,
    right_side: Vec<SideAnalysis>,
}

pub struct RawDataView<'a> {
    raw_data: &'a [(u32, SensorData)],
    start_idx: usize,
    end_idx: usize,
}

impl<'a> RawDataView<'a> {
    pub fn get_data_at(&self, idx: usize) -> Option<RawPeriodData<'a>> {
        if idx >= self.end_idx - self.start_idx {
            return None;
        }

        if let SensorData::PiezoDual {
            ts,
            left1,
            left2,
            right1,
            right2,
            ..
        } = &self.raw_data[self.start_idx + idx].1
        {
            Some(RawPeriodData {
                timestamp: DateTime::from_timestamp(*ts, 0).unwrap(),
                left1,
                left2,
                right1,
                right2,
            })
        } else {
            None
        }
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
    pub left1: &'a [u8],
    pub left2: &'a [u8],
    pub right1: &'a [u8],
    pub right2: &'a [u8],
}

fn create_raw_data_view<'a>(
    raw_sensor_data: &'a [(u32, SensorData)],
    period: &'_ BedPresence,
) -> RawDataView<'a> {
    let start_idx = raw_sensor_data.partition_point(|(_, data)| {
        if let SensorData::PiezoDual { ts, .. } = data {
            DateTime::from_timestamp(*ts, 0).unwrap() < period.start
        } else {
            true
        }
    });

    let end_idx = start_idx
        + raw_sensor_data[start_idx..].partition_point(|(_, data)| {
            if let SensorData::PiezoDual { ts, .. } = data {
                DateTime::from_timestamp(*ts, 0).unwrap() <= period.end
            } else {
                false
            }
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

fn process_piezo_data(data: &SensorData) -> Option<ProcessedData> {
    if let SensorData::PiezoDual {
        ts,
        left1,
        left2,
        right1,
        right2,
        ..
    } = data
    {
        // Convert bytes to i32 values using from_le_bytes instead of from_ne_bytes
        let left1_data: Vec<i32> = left1
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let left2_data: Vec<i32> = left2
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let right1_data: Vec<i32> = right1
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let right2_data: Vec<i32> = right2
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let (left1_mean, left1_std) = calculate_stats(&left1_data);
        let (left2_mean, left2_std) = calculate_stats(&left2_data);
        let (right1_mean, right1_std) = calculate_stats(&right1_data);
        let (right2_mean, right2_std) = calculate_stats(&right2_data);

        Some(ProcessedData {
            timestamp: DateTime::from_timestamp(*ts, 0).unwrap(),
            left1_mean,
            left1_std,
            left2_mean,
            left2_std,
            right1_mean,
            right1_std,
            right2_mean,
            right2_std,
        })
    } else {
        None
    }
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

fn decode_batch_item(file_path: &PathBuf) -> anyhow::Result<Vec<(u32, SensorData)>> {
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
    raw_sensor_data: &'a [(u32, SensorData)],
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
        500.0,
    );

    // Collect breathing rates with timestamps
    for window in br_windows {
        if let Some(breathing_rate) = heart_analysis::analyze_breathing_rate_fft(
            &window.processed_signal,
            500.0,
            &mut br_fft_context,
        ) {
            breathing_rates.push((window.timestamp, breathing_rate));
        }
    }

    // Process heart rate windows...
    let hr_windows = heart_analysis::SignalWindowIterator::new(
        signal,
        raw_data,
        samples_per_segment_hr,
        step_size_hr,
        500.0,
    );

    let time_step = samples_per_segment_hr as f32 / 500.0;

    for window in hr_windows {
        // Find the closest breathing rate measurement
        let breathing_rate = breathing_rates
            .iter()
            .min_by_key(|(br_time, _)| {
                (br_time.timestamp() - window.timestamp.timestamp()).abs() as u64
            })
            .map(|(_, rate)| *rate);

        if let Some(fft_hr) = heart_analysis::analyze_heart_rate_fft(
            &window.processed_signal,
            500.0,
            prev_fft_hr,
            window.timestamp,
            &mut hr_history,
            &mut hr_fft_context,
            time_step,
            breathing_rate,
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
    raw_sensor_data: &[(u32, SensorData)],
    left_periods: &[BedPresence],
    right_periods: &[BedPresence],
) -> anyhow::Result<BedAnalysis> {
    let mut bed_analysis = BedAnalysis {
        left_side: Vec::new(),
        right_side: Vec::new(),
    };

    println!("\nAnalyzing bed presence periods:");

    // Get segment width and overlap from environment variables or use defaults
    let segment_width_hr: f32 = env::var("HR_WINDOW_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10.0);

    let overlap_percent_hr: f32 = env::var("HR_WINDOW_OVERLAP_PERCENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);

    let segment_width_br: f32 = env::var("BR_WINDOW_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120.0);

    let overlap_percent_br: f32 = env::var("BR_WINDOW_OVERLAP_PERCENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let samples_per_segment_hr = (segment_width_hr * 500.0) as usize;
    let overlap_samples_hr = (samples_per_segment_hr as f32 * overlap_percent_hr) as usize;
    let step_size_hr = samples_per_segment_hr - overlap_samples_hr;

    let samples_per_segment_br = (segment_width_br * 500.0) as usize;
    let overlap_samples_br = (samples_per_segment_br as f32 * overlap_percent_br) as usize;
    let step_size_br = samples_per_segment_br - overlap_samples_br;

    println!("\n  Processing with:");
    println!("    (hr) Segment width : {} seconds", segment_width_hr);
    println!("    (hr) Overlap: {}%", overlap_percent_hr * 100.0);
    println!("    (hr) Samples per segment: {}", samples_per_segment_hr);
    println!("    (hr) Overlap samples: {}", overlap_samples_hr);
    println!("    (hr) Step size: {} samples", step_size_hr);

    println!("\n  --:");
    println!("    (br) Segment width : {} seconds", segment_width_br);
    println!("    (br) Overlap: {}%", overlap_percent_br * 100.0);
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
        let signal1: Vec<i32> = (0..raw_data_view.len())
            .filter_map(|idx| raw_data_view.get_data_at(idx))
            .flat_map(|data| {
                data.left1
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect();

        let signal2: Vec<i32> = (0..raw_data_view.len())
            .filter_map(|idx| raw_data_view.get_data_at(idx))
            .flat_map(|data| {
                data.left2
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect();

        println!("\n  Analyzing left side period...");
        let analysis1 = analyse_sensor_data(
            &signal1,
            &raw_data_view,
            samples_per_segment_hr,
            step_size_hr,
            samples_per_segment_br,
            step_size_br,
        );

        let analysis2 = analyse_sensor_data(
            &signal2,
            &raw_data_view,
            samples_per_segment_hr,
            step_size_hr,
            samples_per_segment_br,
            step_size_br,
        );

        // Combine signals by averaging
        let combined_signal: Vec<i32> = signal1
            .iter()
            .zip(signal2.iter())
            .map(|(a, b)| ((*a as i64 + *b as i64) / 2) as i32)
            .collect();

        let analysis_combined = analyse_sensor_data(
            &combined_signal,
            &raw_data_view,
            samples_per_segment_hr,
            step_size_hr,
            samples_per_segment_br,
            step_size_br,
        );

        bed_analysis.left_side.push(SideAnalysis {
            sensor1: analysis1,
            sensor2: analysis2,
            combined: analysis_combined,
            period_num: i,
        });
    }

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
        let signal1: Vec<i32> = (0..raw_data_view.len())
            .filter_map(|idx| raw_data_view.get_data_at(idx))
            .flat_map(|data| {
                data.right1
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect();

        let signal2: Vec<i32> = (0..raw_data_view.len())
            .filter_map(|idx| raw_data_view.get_data_at(idx))
            .flat_map(|data| {
                data.right2
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect();

        println!("\n  Analyzing right side period...");
        let analysis1 = analyse_sensor_data(
            &signal1,
            &raw_data_view,
            samples_per_segment_hr,
            step_size_hr,
            samples_per_segment_br,
            step_size_br,
        );

        let analysis2 = analyse_sensor_data(
            &signal2,
            &raw_data_view,
            samples_per_segment_hr,
            step_size_hr,
            samples_per_segment_br,
            step_size_br,
        );

        // Combine signals by averaging
        let combined_signal: Vec<i32> = signal1
            .iter()
            .zip(signal2.iter())
            .map(|(a, b)| ((*a as i64 + *b as i64) / 2) as i32)
            .collect();

        let analysis_combined = analyse_sensor_data(
            &combined_signal,
            &raw_data_view,
            samples_per_segment_hr,
            step_size_hr,
            samples_per_segment_br,
            step_size_br,
        );

        bed_analysis.right_side.push(SideAnalysis {
            sensor1: analysis1,
            sensor2: analysis2,
            combined: analysis_combined,
            period_num: i,
        });
    }

    Ok(bed_analysis)
}

fn write_analysis_to_csv(
    base_path: &str,
    sensor_id: &str,
    period_num: usize,
    analysis: &PeriodAnalysis,
    hr_smoothing_window: usize,
) -> anyhow::Result<()> {
    let path = std::path::Path::new(base_path);
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
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
    timestamps.sort_unstable();
    timestamps.dedup();

    // Interpolate and smooth heart rates
    let smoothed_fft_hr = heart_analysis::interpolate_and_smooth(
        &timestamps,
        &analysis.fft_heart_rates,
        hr_smoothing_window,
    );

    // Write header
    writer.write_record(&["timestamp", "fft_hr", "fft_hr_smoothed", "breathing_rate"])?;

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

        writer.write_record(&[
            timestamp.format("%Y-%m-%d %H:%M").to_string(),
            fft_hr,
            fft_hr_smoothed,
            br,
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn parse_datetime(datetime_str: &str) -> anyhow::Result<DateTime<Utc>> {
    let naive = NaiveDateTime::parse_from_str(datetime_str, "%Y-%m-%d %H:%M")?;
    Ok(DateTime::from_naive_utc_and_offset(naive, Utc))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    let args = Args::parse();
    let mut raw_sensor_data = Vec::new();

    // First, load all RAW files and collect sensor data
    for entry in std::fs::read_dir(args.raw_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("RAW") {
            println!("Loading file: {}", path.display());
            let items = decode_batch_item(&path)?;
            raw_sensor_data.extend(items);
        }
    }

    // Sort by sequence number and timestamp
    raw_sensor_data.sort_by_key(|(seq, data)| {
        if let SensorData::PiezoDual { ts, .. } = data {
            (*seq, *ts)
        } else {
            (*seq, 0)
        }
    });

    println!("Loaded {} raw sensor data points", raw_sensor_data.len());

    if raw_sensor_data.is_empty() {
        println!("No raw sensor data found");
        return Ok(());
    }

    // Only filter by timeframe if start/end times were provided
    if args.start_time.is_some() || args.end_time.is_some() {
        let end_time = match args.end_time {
            Some(ref t) => parse_datetime(t)?,
            None => Utc::now(),
        };

        let start_time = match args.start_time {
            Some(ref t) => parse_datetime(t)?,
            None => end_time - chrono::Duration::days(1),
        };

        println!("Analyzing data from {} to {}", start_time, end_time);

        // Filter data by timestamp after sorting
        let initial_count = raw_sensor_data.len();
        raw_sensor_data.retain(|(_seq, data)| {
            if let SensorData::PiezoDual { ts, .. } = data {
                let timestamp = DateTime::from_timestamp(*ts, 0).unwrap();
                timestamp >= start_time && timestamp <= end_time
            } else {
                false
            }
        });

        println!(
            "Filtered {} data points to {} within specified timeframe",
            initial_count,
            raw_sensor_data.len()
        );
    }

    // Process all sensor data at once
    let mut all_processed_data: Vec<ProcessedData> = raw_sensor_data
        .iter()
        .filter_map(|(seq, data)| {
            let processed = process_piezo_data(data);
            if processed.is_some() {
                trace!("Processing seq {}", seq);
            }
            processed
        })
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
    let (left_periods, right_periods) = if args.start_time.is_some() || args.end_time.is_some() {
        let end_time = match args.end_time {
            Some(ref t) => parse_datetime(t)?,
            None => Utc::now(),
        };

        let start_time = match args.start_time {
            Some(ref t) => parse_datetime(t)?,
            None => end_time - chrono::Duration::days(1),
        };

        // Create a single period for both sides
        let single_period = BedPresence {
            start: start_time,
            end: end_time,
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
        analyze_bed_presence_periods(&raw_sensor_data, &left_periods, &right_periods)?;

    let hr_smoothing_window = args.hr_smoothing_window.unwrap_or(60);

    // Write CSV files if output prefix was provided
    if let Some(prefix) = &args.csv_output {
        for analysis in &bed_analysis.left_side {
            if let Some(split_sensors) = &args.split_sensors {
                if *split_sensors == true {
                    write_analysis_to_csv(
                        prefix,
                        "left",
                        analysis.period_num,
                        &analysis.sensor1,
                        hr_smoothing_window,
                    )?;
                    write_analysis_to_csv(
                        prefix,
                        "left2",
                        analysis.period_num,
                        &analysis.sensor2,
                        hr_smoothing_window,
                    )?;
                }
            }
            write_analysis_to_csv(
                prefix,
                "left_combined",
                analysis.period_num,
                &analysis.combined,
                hr_smoothing_window,
            )?;
        }

        for analysis in &bed_analysis.right_side {
            if let Some(split_sensors) = &args.split_sensors {
                if *split_sensors == true {
                    write_analysis_to_csv(
                        prefix,
                        "right",
                        analysis.period_num,
                        &analysis.sensor1,
                        hr_smoothing_window,
                    )?;
                    write_analysis_to_csv(
                        prefix,
                        "right2",
                        analysis.period_num,
                        &analysis.sensor2,
                        hr_smoothing_window,
                    )?;
                }
            }
            write_analysis_to_csv(
                prefix,
                "right_combined",
                analysis.period_num,
                &analysis.combined,
                hr_smoothing_window,
            )?;
        }
    }

    Ok(())
}
