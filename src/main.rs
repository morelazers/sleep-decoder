use anyhow::Context;
use chrono::Timelike;
use chrono::{DateTime, Utc};
use clap::Parser;
use env_logger;
use log::{debug, trace};
use plotters::prelude::*;
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
}

#[derive(Debug, Deserialize)]
struct BatchItem {
    seq: u32,
    data: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct BedTempSide {
    side: f32,
    out: f32,
    cen: f32,
    #[serde(rename = "in")]
    in_: f32,
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
        ts: i64,
        mcu: f32,
        amb: f32,
        left: BedTempSide,
        right: BedTempSide,
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
struct RawPeriodData {
    timestamp: DateTime<Utc>,
    left1: Vec<u8>,
    left2: Vec<u8>,
    right1: Vec<u8>,
    right2: Vec<u8>,
}

#[derive(Debug)]
struct PeriodAnalysis {
    peak_heart_rates: Vec<(DateTime<Utc>, f32)>, // Results from peak detection
    fft_heart_rates: Vec<(DateTime<Utc>, f32)>,  // Results from FFT analysis
    breathing_rates: Vec<(DateTime<Utc>, f32)>,  // Results from breathing analysis
    side: String,
    period_num: usize,
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
    let mut sorted_means = mean_data.clone();
    sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_lower = sorted_means[(data.len() as f32 * 0.02) as usize];
    let mean_upper = sorted_means[(data.len() as f32 * 0.98) as usize];

    // Calculate percentiles for std
    let mut sorted_stds = std_data.clone();
    sorted_stds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let std_lower = sorted_stds[(data.len() as f32 * 0.02) as usize];
    let std_upper = sorted_stds[(data.len() as f32 * 0.98) as usize];

    // Filter data
    data.retain(|d| {
        let (mean, std) = match sensor {
            "left1" => (d.left1_mean, d.left1_std),
            "left2" => (d.left2_mean, d.left2_std),
            "right1" => (d.right1_mean, d.right1_std),
            "right2" => (d.right2_mean, d.right2_std),
            _ => unreachable!(),
        };
        mean >= mean_lower && mean <= mean_upper && std >= std_lower && std <= std_upper
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
    let max_gap = chrono::Duration::minutes(45); // Increased to 45 minutes for deep sleep periods

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

fn extract_raw_data_for_period(
    raw_sensor_data: &[(u32, SensorData)],
    period: &BedPresence,
) -> Vec<RawPeriodData> {
    raw_sensor_data
        .iter()
        .filter_map(|(_, data)| {
            if let SensorData::PiezoDual {
                ts,
                left1,
                left2,
                right1,
                right2,
                ..
            } = data
            {
                let timestamp = DateTime::from_timestamp(*ts, 0).unwrap();
                if timestamp >= period.start && timestamp <= period.end {
                    Some(RawPeriodData {
                        timestamp,
                        left1: left1.clone(),
                        left2: left2.clone(),
                        right1: right1.clone(),
                        right2: right2.clone(),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

mod heart_analysis;

fn analyze_period_signals(
    raw_data: &[RawPeriodData],
    side: &str,
    period_num: usize,
) -> PeriodAnalysis {
    const SAMPLING_RATE: f32 = 500.0;

    // Get segment width and overlap from environment variables or use defaults
    let segment_width: f32 = env::var("SEGMENT_WIDTH_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120.0); // Default: 120 second segments

    let segment_overlap: f32 = env::var("SEGMENT_OVERLAP_PERCENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0); // Default: 0% overlap

    let samples_per_segment: usize = (segment_width * SAMPLING_RATE) as usize;
    let overlap_samples: usize = (samples_per_segment as f32 * segment_overlap) as usize;
    let step_size: usize = samples_per_segment - overlap_samples;

    println!(
        "\n  Processing segments with:
    Width: {} seconds
    Overlap: {}%
    Samples per segment: {}
    Overlap samples: {}
    Step size: {} samples",
        segment_width,
        segment_overlap * 100.0,
        samples_per_segment,
        overlap_samples,
        step_size
    );

    // Extract signals based on side
    let (signal1, signal2) = match side {
        "left" => {
            let s1: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.left1.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();
            let s2: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.left2.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();
            (s1, s2)
        },
        "right" => {
            let s1: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.right1.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();
            let s2: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.right2.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();
            (s1, s2)
        },
        _ => unreachable!(),
    };

    // Analyze sensor1
    println!("\nAnalyzing {} sensor 1:", side);
    let sensor1_results = analyze_single_sensor(
        &signal1,
        raw_data,
        samples_per_segment,
        step_size,
        &format!("{}_sensor1", side),
    );
    print_analysis_results(&sensor1_results, &format!("{} Sensor 1", side));

    // Analyze sensor2
    println!("\nAnalyzing {} sensor 2:", side);
    let sensor2_results = analyze_single_sensor(
        &signal2,
        raw_data,
        samples_per_segment,
        step_size,
        &format!("{}_sensor2", side),
    );
    print_analysis_results(&sensor2_results, &format!("{} Sensor 2", side));

    // Analyze combined signals
    println!("\nAnalyzing {} combined signals:", side);
    let combined_signal: Vec<i32> = signal1.iter()
        .zip(signal2.iter())
        .map(|(a, b)| ((*a as i64 + *b as i64) / 2) as i32)
        .collect();
    let combined_results = analyze_single_sensor(
        &combined_signal,
        raw_data,
        samples_per_segment,
        step_size,
        &format!("{}_combined", side),
    );
    print_analysis_results(&combined_results, &format!("{} Combined", side));

    // Return the combined results as the main analysis
    combined_results
}

fn analyze_single_sensor(
    signal: &[i32],
    raw_data: &[RawPeriodData],
    samples_per_segment: usize,
    step_size: usize,
    sensor_id: &str,
) -> PeriodAnalysis {
    let mut peak_heart_rates = Vec::new();
    let mut fft_heart_rates = Vec::new();
    let mut breathing_rates = Vec::new();

    // Initialize CSV writer if environment variable is set
    let mut csv_writer = if let Ok(base_path) = env::var("CSV_OUTPUT") {
        // Split the path into directory and filename parts
        let path = std::path::Path::new(&base_path);
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("results");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("csv");

        // Create new filename with sensor id
        let filename = format!("{}_{}.{}", stem, sensor_id, ext);
        let full_path = dir.join(filename);

        println!("Writing results to {}", full_path.display());
        let file = std::fs::File::create(full_path).expect("Failed to create CSV file");
        let mut writer = csv::Writer::from_writer(file);
        writer.write_record(&["timestamp", "peak_hr", "fft_hr", "breathing_rate"])
            .expect("Failed to write CSV header");
        Some(writer)
    } else {
        None
    };

    // Process segments
    let total_samples = signal.len();
    let num_segments = (total_samples - samples_per_segment) / step_size + 1;

    for segment_idx in 0..num_segments {
        let start_sample = segment_idx * step_size;
        let end_sample = (start_sample + samples_per_segment).min(total_samples);

        // Skip if we don't have enough samples for a full segment
        if end_sample - start_sample < samples_per_segment / 2 {
            continue;
        }

        // Extract segment
        let segment = &signal[start_sample..end_sample];

        // Remove outliers from the segment
        let cleaned_segment = heart_analysis::interpolate_outliers(segment, 2);

        // Convert to f32 for processing
        let segment_f32: Vec<f32> = cleaned_segment.iter().map(|&x| x as f32).collect();

        // Scale the segment for breathing rate analysis
        let scaled_segment = heart_analysis::scale_data(&segment_f32, 0.0, 1024.0);
        if let Some(breathing_rate) = heart_analysis::analyze_breathing_rate_fft(&scaled_segment, 500.0) {
            let segment_time = raw_data[start_sample / 500].timestamp;
            breathing_rates.push((segment_time, breathing_rate));
        }

        // Process for heart rate
        let processed_segment = heart_analysis::remove_baseline_wander(&scaled_segment, 500.0, 0.05);
        let (working_data, measures) = heart_analysis::process(&processed_segment, 500.0, 0.75);

        let segment_time = raw_data[start_sample / 500].timestamp;

        // Store peak detection results if confident
        if measures.confidence > 0.5 {
            peak_heart_rates.push((segment_time, measures.bpm));
        }

        // Store FFT results
        if let Some(fft_hr) = heart_analysis::analyze_heart_rate_fft(&processed_segment, 500.0) {
            fft_heart_rates.push((segment_time, fft_hr));
        }

        // Write to CSV if enabled
        if let Some(writer) = csv_writer.as_mut() {
            let peak_hr = peak_heart_rates.last().map(|(_, hr)| hr.to_string()).unwrap_or_default();
            let fft_hr = fft_heart_rates.last().map(|(_, hr)| hr.to_string()).unwrap_or_default();
            let br = breathing_rates.last().map(|(_, br)| br.to_string()).unwrap_or_default();
            let record = [
                segment_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                peak_hr,
                fft_hr,
                br,
            ];
            writer.write_record(&record).expect("Failed to write CSV record");
        }
    }

    // Ensure CSV is flushed
    if let Some(mut writer) = csv_writer {
        writer.flush().expect("Failed to flush CSV writer");
    }

    PeriodAnalysis {
        peak_heart_rates,
        fft_heart_rates,
        breathing_rates,
        side: String::from(sensor_id),
        period_num: 0,
    }
}

fn analyze_bed_presence_periods(
    raw_sensor_data: &[(u32, SensorData)],
    all_processed_data: &[ProcessedData],
) -> anyhow::Result<()> {
    // Detect presence for both sides
    let left_periods = detect_bed_presence(all_processed_data, "left");
    let right_periods = detect_bed_presence(all_processed_data, "right");

    println!("\nAnalyzing bed presence periods:");

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
        let raw_data = extract_raw_data_for_period(raw_sensor_data, period);
        println!("  Found {} raw data points", raw_data.len());

        if !raw_data.is_empty() {
            // Get segment width and overlap from environment variables or use defaults
            let segment_width: f32 = env::var("SEGMENT_WIDTH_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(120.0); // Default: 120 second segments

            let overlap_percent: f32 = env::var("SEGMENT_OVERLAP_PERCENT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0); // Default: 0% overlap

            let samples_per_segment = (segment_width * 500.0) as usize;  // 500 Hz sampling rate
            let overlap_samples = (samples_per_segment as f32 * overlap_percent) as usize;
            let step_size = samples_per_segment - overlap_samples;

            println!("\n  Processing with:");
            println!("    Segment width: {} seconds", segment_width);
            println!("    Overlap: {}%", overlap_percent * 100.0);
            println!("    Samples per segment: {}", samples_per_segment);
            println!("    Overlap samples: {}", overlap_samples);
            println!("    Step size: {} samples", step_size);

            // Extract signals
            let signal1: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.left1.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();

            let signal2: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.left2.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();

            println!("\n  Analyzing left side period...");
            let sensor1_results = analyze_single_sensor(
                &signal1,
                &raw_data,
                samples_per_segment,
                step_size,
                &format!("left_sensor1_{}", i)
            );

            let sensor2_results = analyze_single_sensor(
                &signal2,
                &raw_data,
                samples_per_segment,
                step_size,
                &format!("left_sensor2_{}", i)
            );

            // Combine signals by averaging
            let combined_signal: Vec<i32> = signal1.iter()
                .zip(signal2.iter())
                .map(|(a, b)| ((*a as i64 + *b as i64) / 2) as i32)
                .collect();

            let combined_results = analyze_single_sensor(
                &combined_signal,
                &raw_data,
                samples_per_segment,
                step_size,
                &format!("left_combined_{}", i)
            );

            println!("\nLeft Sensor 1 Results:");
            print_detailed_comparison(&sensor1_results, "Left Sensor 1");

            println!("\nLeft Sensor 2 Results:");
            print_detailed_comparison(&sensor2_results, "Left Sensor 2");

            println!("\nLeft Combined Results:");
            print_detailed_comparison(&combined_results, "Left Combined");
        } else {
            println!("  No raw data found for this period!");
        }
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
        let raw_data = extract_raw_data_for_period(raw_sensor_data, period);
        println!("  Found {} raw data points", raw_data.len());

        if !raw_data.is_empty() {
            // Get segment width and overlap from environment variables or use defaults
            let segment_width: f32 = env::var("SEGMENT_WIDTH_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(120.0); // Default: 120 second segments

            let overlap_percent: f32 = env::var("SEGMENT_OVERLAP_PERCENT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0); // Default: 0% overlap

            let samples_per_segment = (segment_width * 500.0) as usize;  // 500 Hz sampling rate
            let overlap_samples = (samples_per_segment as f32 * overlap_percent) as usize;
            let step_size = samples_per_segment - overlap_samples;

            println!("\n  Processing with:");
            println!("    Segment width: {} seconds", segment_width);
            println!("    Overlap: {}%", overlap_percent * 100.0);
            println!("    Samples per segment: {}", samples_per_segment);
            println!("    Overlap samples: {}", overlap_samples);
            println!("    Step size: {} samples", step_size);

            // Extract signals
            let signal1: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.right1.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();

            let signal2: Vec<i32> = raw_data.iter().flat_map(|d| {
                d.right2.chunks_exact(4).map(|chunk| {
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                })
            }).collect();

            println!("\n  Analyzing right side period...");
            let sensor1_results = analyze_single_sensor(
                &signal1,
                &raw_data,
                samples_per_segment,
                step_size,
                &format!("right_sensor1_{}", i)
            );

            let sensor2_results = analyze_single_sensor(
                &signal2,
                &raw_data,
                samples_per_segment,
                step_size,
                &format!("right_sensor2_{}", i)
            );

            // Combine signals by averaging
            let combined_signal: Vec<i32> = signal1.iter()
                .zip(signal2.iter())
                .map(|(a, b)| ((*a as i64 + *b as i64) / 2) as i32)
                .collect();

            let combined_results = analyze_single_sensor(
                &combined_signal,
                &raw_data,
                samples_per_segment,
                step_size,
                &format!("right_combined_{}", i)
            );

            println!("\nRight Sensor 1 Results:");
            print_detailed_comparison(&sensor1_results, "Right Sensor 1");

            println!("\nRight Sensor 2 Results:");
            print_detailed_comparison(&sensor2_results, "Right Sensor 2");

            println!("\nRight Combined Results:");
            print_detailed_comparison(&combined_results, "Right Combined");
        } else {
            println!("  No raw data found for this period!");
        }
    }

    Ok(())
}

fn detect_sleep_onset(
    processed_data: &[ProcessedData],
    side: &str,
    start_time: DateTime<Utc>,
) -> Option<DateTime<Utc>> {
    const ROLLING_WINDOW: usize = 20; // Increased from 5 to 20 for better stability
    const MIN_QUIET_PERIOD: usize = 12; // Increased from 8 to 12 (3 minutes of low activity)
    const TRANSITION_WINDOW: usize = 40; // 10 minute window to detect gradual transition

    // Get the relevant standard deviations based on side
    let std_values: Vec<f32> = processed_data
        .iter()
        .filter(|d| d.timestamp >= start_time)
        .map(|d| match side {
            "left" => (d.left1_std + d.left2_std) / 2.0,
            "right" => (d.right1_std + d.right2_std) / 2.0,
            _ => unreachable!(),
        })
        .collect();

    let timestamps: Vec<DateTime<Utc>> = processed_data
        .iter()
        .filter(|d| d.timestamp >= start_time)
        .map(|d| d.timestamp)
        .collect();

    // Check if we have enough data for all windows
    let min_required_length = TRANSITION_WINDOW + MIN_QUIET_PERIOD + ROLLING_WINDOW;
    if std_values.len() < min_required_length {
        println!("\n  Debug: Not enough data for sleep onset detection");
        println!("    Required length: {}", min_required_length);
        println!("    Available length: {}", std_values.len());
        return None;
    }

    // Calculate distribution statistics for the entire period
    let mut sorted_stds = std_values.clone();
    sorted_stds.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q25_idx = (sorted_stds.len() as f32 * 0.25) as usize;
    let q75_idx = (sorted_stds.len() as f32 * 0.75) as usize;
    let median_idx = sorted_stds.len() / 2;

    let q25 = sorted_stds[q25_idx];
    let q75 = sorted_stds[q75_idx];
    let median = sorted_stds[median_idx];
    let iqr = q75 - q25;

    // Define multiple thresholds for different activity states
    let high_activity_threshold = q75 + iqr * 0.5;
    let medium_activity_threshold = median + iqr * 0.25;
    let low_activity_threshold = median - iqr * 0.25;
    let sleep_threshold = q25 - iqr * 0.25;

    println!("\n  Debug: Activity Thresholds ({} side):", side);
    println!("    High activity: > {:.2}", high_activity_threshold);
    println!("    Medium activity: > {:.2}", medium_activity_threshold);
    println!("    Low activity: > {:.2}", low_activity_threshold);
    println!("    Sleep level: <= {:.2}", sleep_threshold);

    // Calculate rolling statistics
    let rolling_stds: Vec<f32> = std_values
        .windows(ROLLING_WINDOW)
        .map(|window| window.iter().sum::<f32>() / window.len() as f32)
        .collect();

    // Print debug information for the first hour
    println!("\n  Debug: Activity levels for first hour:");
    let hour_duration = chrono::Duration::hours(1);
    let five_minutes = chrono::Duration::minutes(5);
    let mut current_time = start_time;
    let end_time = start_time + hour_duration;

    while current_time < end_time {
        let next_time = current_time + five_minutes;
        let window_stds: Vec<f32> = std_values
            .iter()
            .zip(&timestamps)
            .filter(|(_, ts)| **ts >= current_time && **ts < next_time)
            .map(|(std, _)| *std)
            .collect();

        if !window_stds.is_empty() {
            let avg_std = window_stds.iter().sum::<f32>() / window_stds.len() as f32;
            let activity_level = if avg_std > high_activity_threshold {
                "High"
            } else if avg_std > medium_activity_threshold {
                "Medium"
            } else if avg_std > low_activity_threshold {
                "Low"
            } else {
                "Sleep"
            };

            println!(
                "    {}: {:.2} ({} activity, {} samples)",
                current_time.format("%H:%M"),
                avg_std,
                activity_level,
                window_stds.len()
            );
        }
        current_time = next_time;
    }

    // Look for sleep onset pattern using state transitions
    for i in ROLLING_WINDOW
        ..rolling_stds
            .len()
            .saturating_sub(MIN_QUIET_PERIOD + ROLLING_WINDOW)
    {
        // Ensure we have enough data for all windows
        if i + MIN_QUIET_PERIOD + ROLLING_WINDOW > rolling_stds.len() {
            break;
        }

        // Analyze transition period (before potential sleep onset)
        let transition_window = &rolling_stds[i - ROLLING_WINDOW..i];
        let transition_trend = analyze_trend(transition_window);

        // Analyze quiet period (potential sleep onset)
        let quiet_window = &rolling_stds[i..i + MIN_QUIET_PERIOD];
        let quiet_avg = quiet_window.iter().sum::<f32>() / quiet_window.len() as f32;
        let quiet_stability = calculate_stability(quiet_window);

        // Analyze post-onset period
        let post_window =
            &rolling_stds[i + MIN_QUIET_PERIOD..i + MIN_QUIET_PERIOD + ROLLING_WINDOW];
        let post_avg = post_window.iter().sum::<f32>() / post_window.len() as f32;
        let post_stability = calculate_stability(post_window);

        // Check for sleep onset pattern:
        // 1. Transition period shows declining trend
        // 2. Quiet period is stable and below sleep threshold
        // 3. Post-onset period remains stable and low
        if transition_trend < -0.2  // Declining trend
            && quiet_avg < sleep_threshold
            && quiet_stability < iqr * 0.2
            && post_avg < low_activity_threshold
            && post_stability < iqr * 0.25
        {
            println!("\n  Debug: Found sleep onset pattern:");
            println!("    Transition trend: {:.3}", transition_trend);
            println!("    Quiet period average: {:.2}", quiet_avg);
            println!("    Quiet period stability: {:.2}", quiet_stability);
            println!("    Post-onset average: {:.2}", post_avg);
            println!("    Post-onset stability: {:.2}", post_stability);

            // Return timestamp from middle of quiet period
            let timestamp_idx = i + MIN_QUIET_PERIOD / 2;
            if timestamp_idx < timestamps.len() {
                return Some(timestamps[timestamp_idx]);
            }
        }
    }

    None
}

/// Calculate the trend in a window of values
/// Returns a value between -1 and 1, where:
/// - Negative values indicate declining trend
/// - Positive values indicate increasing trend
/// - Magnitude indicates strength of trend
fn analyze_trend(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f32;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f32;
        let x_diff = x - mean_x;
        let y_diff = y - mean_y;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    if denominator == 0.0 {
        0.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

/// Calculate stability of a window of values
/// Returns the coefficient of variation (standard deviation / mean)
fn calculate_stability(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if mean == 0.0 {
        return f32::INFINITY;
    }

    let variance = values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / values.len() as f32;

    variance.sqrt() / mean
}

/// Remove outliers from raw sensor data
fn clean_raw_sensor_data(data: &mut Vec<(u32, SensorData)>) {
    data.retain(|(_, sensor_data)| {
        if let SensorData::PiezoDual {
            left1,
            left2,
            right1,
            right2,
            ..
        } = sensor_data
        {
            // Convert bytes to i32 and remove outliers
            let l1: Vec<i32> = left1.iter().map(|&x| x as i32).collect();
            let l2: Vec<i32> = left2.iter().map(|&x| x as i32).collect();
            let r1: Vec<i32> = right1.iter().map(|&x| x as i32).collect();
            let r2: Vec<i32> = right2.iter().map(|&x| x as i32).collect();

            // If any sensor has all values as outliers, remove this data point
            !heart_analysis::interpolate_outliers(&l1, 2).is_empty()
                && !heart_analysis::interpolate_outliers(&l2, 2).is_empty()
                && !heart_analysis::interpolate_outliers(&r1, 2).is_empty()
                && !heart_analysis::interpolate_outliers(&r2, 2).is_empty()
        } else {
            true // Keep non-piezo data
        }
    });
}

fn plot_signal_stages(
    raw: &[f32],
    scaled: &[f32],
    processed: &[f32],
    cleaned: &[f32],
    segment_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create plots directory if it doesn't exist
    let plot_dir = env::var("PLOT_DIR")?;

    // Error if PLOT_DIR is not set
    if plot_dir.is_empty() {
        return Err("PLOT_DIR is not set".into());
    }

    std::fs::create_dir_all(plot_dir.clone())?;

    let path = format!("{}/segment_{}_signals.png", plot_dir, segment_idx);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((4, 1));

    // Plot raw signal
    let raw_min = raw.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let raw_max = raw.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut chart = ChartBuilder::on(&areas[0])
        .caption("Raw Signal", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0..raw.len(), raw_min..raw_max)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(
        raw.iter().enumerate().map(|(i, &v)| (i, v)),
        &BLUE,
    ))?;

    // Plot scaled signal
    let mut chart = ChartBuilder::on(&areas[1])
        .caption("Scaled Signal", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0..scaled.len(), 0f32..1024f32)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(
        scaled.iter().enumerate().map(|(i, &v)| (i, v)),
        &RED,
    ))?;

    // Plot processed signal
    let proc_min = processed.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let proc_max = processed.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut chart = ChartBuilder::on(&areas[2])
        .caption("After Baseline Removal", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0..processed.len(), proc_min..proc_max)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(
        processed.iter().enumerate().map(|(i, &v)| (i, v)),
        &GREEN,
    ))?;

    // Plot cleaned signal
    let clean_min = cleaned.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let clean_max = cleaned.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut chart = ChartBuilder::on(&areas[3])
        .caption("After Cleaning", ("sans-serif", 20))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0..cleaned.len(), clean_min..clean_max)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(
        cleaned.iter().enumerate().map(|(i, &v)| (i, v)),
        &BLUE,
    ))?;

    debug!("  Generated plot: {}", path);
    Ok(())
}

fn print_heart_rate_comparison(analysis: &PeriodAnalysis) {
    if analysis.peak_heart_rates.is_empty()
        && analysis.fft_heart_rates.is_empty()
        && analysis.breathing_rates.is_empty()
    {
        println!("No data available from any method");
        return;
    }

    // Get time range
    let start_time = [
        analysis.peak_heart_rates.first().map(|(t, _)| *t),
        analysis.fft_heart_rates.first().map(|(t, _)| *t),
        analysis.breathing_rates.first().map(|(t, _)| *t),
    ]
    .iter()
    .filter_map(|&x| x)
    .min()
    .unwrap_or_default();

    let end_time = [
        analysis.peak_heart_rates.last().map(|(t, _)| *t),
        analysis.fft_heart_rates.last().map(|(t, _)| *t),
        analysis.breathing_rates.last().map(|(t, _)| *t),
    ]
    .iter()
    .filter_map(|&x| x)
    .max()
    .unwrap_or_default();

    // Initialize CSV writer if environment variable is set
    let mut csv_writer = if let Ok(base_path) = env::var("CSV_OUTPUT") {
        // Split the path into directory and filename parts
        let path = std::path::Path::new(&base_path);
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("results");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("csv");

        // Create new filename with side and period number
        let filename = format!(
            "{}_{}_period_{}.{}",
            stem, analysis.side, analysis.period_num, ext
        );
        let full_path = dir.join(filename);

        println!("Writing results to {}", full_path.display());
        let file = std::fs::File::create(full_path).expect("Failed to create CSV file");
        let mut writer = csv::Writer::from_writer(file);
        writer
            .write_record(&["timestamp", "peak_hr", "fft_hr", "breathing_rate"])
            .expect("Failed to write CSV header");
        Some(writer)
    } else {
        None
    };

    println!("\n  5-minute interval comparison:");
    println!("  Time      Peak HR    FFT HR    Breathing");
    println!("  ------------------------------------------");

    let interval = chrono::Duration::minutes(5);
    let mut current_time = start_time;

    while current_time <= end_time {
        let next_time = current_time + interval;

        // Get peak detection results for this interval
        let peak_rates: Vec<f32> = analysis
            .peak_heart_rates
            .iter()
            .filter(|(ts, _)| *ts >= current_time && *ts < next_time)
            .map(|(_, hr)| *hr)
            .collect();

        // Get FFT results for this interval
        let fft_rates: Vec<f32> = analysis
            .fft_heart_rates
            .iter()
            .filter(|(ts, _)| *ts >= current_time && *ts < next_time)
            .map(|(_, hr)| *hr)
            .collect();

        // Get breathing rates for this interval
        let breathing_rates: Vec<f32> = analysis
            .breathing_rates
            .iter()
            .filter(|(ts, _)| *ts >= current_time && *ts < next_time)
            .map(|(_, br)| *br)
            .collect();

        let peak_avg = if !peak_rates.is_empty() {
            Some(peak_rates.iter().sum::<f32>() / peak_rates.len() as f32)
        } else {
            None
        };

        let fft_avg = if !fft_rates.is_empty() {
            Some(fft_rates.iter().sum::<f32>() / fft_rates.len() as f32)
        } else {
            None
        };

        let breathing_avg = if !breathing_rates.is_empty() {
            Some(breathing_rates.iter().sum::<f32>() / breathing_rates.len() as f32)
        } else {
            None
        };

        // Print to terminal
        print!(
            "  {:02}:{:02}     ",
            current_time.hour(),
            current_time.minute()
        );

        match peak_avg {
            Some(avg) => print!("{:6.1}    ", avg),
            None => print!("  --      "),
        };

        match fft_avg {
            Some(avg) => print!("{:6.1}    ", avg),
            None => print!("  --      "),
        };

        match breathing_avg {
            Some(avg) => println!("{:6.1}", avg),
            None => println!("  --"),
        };

        // Write to CSV if enabled
        if let Some(writer) = csv_writer.as_mut() {
            writer
                .write_record(&[
                    current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                    peak_avg.map(|v| v.to_string()).unwrap_or_default(),
                    fft_avg.map(|v| v.to_string()).unwrap_or_default(),
                    breathing_avg.map(|v| v.to_string()).unwrap_or_default(),
                ])
                .expect("Failed to write CSV record");
        }

        current_time = next_time;
    }

    // Flush CSV writer if it exists
    if let Some(mut writer) = csv_writer {
        writer.flush().expect("Failed to flush CSV writer");
    }
}

fn print_analysis_results(analysis: &PeriodAnalysis, sensor_name: &str) {
    println!("\nResults for {}:", sensor_name);
    println!("Peak Detection Method:");
    println!("  Valid segments: {}", analysis.peak_heart_rates.len());
    if !analysis.peak_heart_rates.is_empty() {
        let avg_hr: f32 = analysis.peak_heart_rates.iter().map(|(_, hr)| hr).sum::<f32>()
            / analysis.peak_heart_rates.len() as f32;
        println!("  Average heart rate: {:.1} BPM", avg_hr);
    }

    println!("\nFFT Method:");
    println!("  Valid segments: {}", analysis.fft_heart_rates.len());
    if !analysis.fft_heart_rates.is_empty() {
        let avg_hr: f32 = analysis.fft_heart_rates.iter().map(|(_, hr)| hr).sum::<f32>()
            / analysis.fft_heart_rates.len() as f32;
        println!("  Average heart rate: {:.1} BPM", avg_hr);
    }

    println!("\nBreathing Rate:");
    println!("  Valid segments: {}", analysis.breathing_rates.len());
    if !analysis.breathing_rates.is_empty() {
        let avg_br: f32 = analysis.breathing_rates.iter().map(|(_, br)| br).sum::<f32>()
            / analysis.breathing_rates.len() as f32;
        println!("  Average breathing rate: {:.1} breaths/min", avg_br);
    }
}

fn print_detailed_comparison(analysis: &PeriodAnalysis, sensor_name: &str) {
    println!("\nDetailed comparison for {} (5-minute intervals):", sensor_name);
    println!("Time      Peak HR    FFT HR    Breathing");
    println!("------------------------------------------");

    let mut current_time = analysis.peak_heart_rates.first()
        .or(analysis.fft_heart_rates.first())
        .or(analysis.breathing_rates.first())
        .map(|(t, _)| *t)
        .unwrap_or_default();

    let end_time = analysis.peak_heart_rates.last()
        .or(analysis.fft_heart_rates.last())
        .or(analysis.breathing_rates.last())
        .map(|(t, _)| *t)
        .unwrap_or_default();

    while current_time <= end_time {
        let interval_end = current_time + chrono::Duration::minutes(5);

        // Calculate averages for this interval
        let peak_hr = calculate_interval_average(&analysis.peak_heart_rates, current_time, interval_end);
        let fft_hr = calculate_interval_average(&analysis.fft_heart_rates, current_time, interval_end);
        let br = calculate_interval_average(&analysis.breathing_rates, current_time, interval_end);

        print!("  {:02}:{:02}     ", current_time.hour(), current_time.minute());

        if let Some(hr) = peak_hr {
            print!("{:6.1}    ", hr);
        } else {
            print!("   --     ");
        }

        if let Some(hr) = fft_hr {
            print!("{:6.1}    ", hr);
        } else {
            print!("   --     ");
        }

        if let Some(br) = br {
            println!("{:6.1}", br);
        } else {
            println!("   --");
        }

        current_time = interval_end;
    }
}

fn calculate_interval_average(data: &[(DateTime<Utc>, f32)], start: DateTime<Utc>, end: DateTime<Utc>) -> Option<f32> {
    let values: Vec<f32> = data.iter()
        .filter(|(t, _)| *t >= start && *t < end)
        .map(|(_, v)| *v)
        .collect();

    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f32>() / values.len() as f32)
    }
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

    // Analyze the raw data during bed presence periods
    analyze_bed_presence_periods(&raw_sensor_data, &all_processed_data)?;

    Ok(())
}
