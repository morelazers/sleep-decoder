use anyhow::Result;
use chrono::NaiveDateTime;
use chrono::{DateTime, Utc};
use clap::Parser;
use env_logger;
use serde::Deserialize;
use sleep_decoder::{
    config::{Args, SensorSelection},
    data_loading::{
        build_raw_file_index, decode_batch_item, read_csv_file, read_feather_file,
        CombinedSensorData,
    },
    heart_analysis, output,
    phase_analysis::{self},
    preprocessing::{self, ProcessedData},
    BedAnalysis, BedPresence, PeriodAnalysis, RawDataView, SideAnalysis,
};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BedTempSide {
    side: f32,
    out: f32,
    cen: f32,
    #[serde(rename = "in")]
    _in: f32,
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
    enable_median_reprocessing: bool,
) -> PeriodAnalysis {
    let mut breathing_rates = Vec::new();
    let mut br_fft_context = heart_analysis::FftContext::new(samples_per_segment_br);
    let mut hr_fft_context = heart_analysis::FftContext::new(samples_per_segment_hr);
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
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(breathing_rates.len());
            let window = &breathing_rates[start..end];

            let mean = window.iter().map(|(_, r)| r).sum::<f32>() / window.len() as f32;
            let variance =
                window.iter().map(|(_, r)| (r - mean).powi(2)).sum::<f32>() / window.len() as f32;
            let stability = 1.0 / (1.0 + variance);

            (timestamp, rate, stability)
        })
        .collect();

    // First pass: Process all windows to get initial estimates
    let hr_windows: Vec<_> = heart_analysis::SignalWindowIterator::new(
        signal,
        raw_data,
        samples_per_segment_hr,
        step_size_hr,
    )
    .collect();

    let time_step = samples_per_segment_hr as f32 / 500.0;
    let mut fft_heart_rates = Vec::new();
    let mut prev_hr = None;

    // First pass - process all windows
    for window in &hr_windows {
        let breathing_data = breathing_rates_with_stability
            .iter()
            .min_by_key(|(br_time, _, _)| {
                (br_time.timestamp() - window.timestamp.timestamp()).abs() as u64
            })
            .map(|(_, rate, stability)| (*rate, *stability));

        if let Some(fft_hr) = heart_analysis::analyze_heart_rate_fft(
            &window.processed_signal,
            500.0,
            prev_hr,
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
            None,
            None,
        ) {
            fft_heart_rates.push((window.timestamp, fft_hr));
            prev_hr = Some(fft_hr);
        }
    }

    // Only perform median-based reprocessing if enabled
    if enable_median_reprocessing {
        // Calculate median HR from all valid estimates
        let median_hr = if !fft_heart_rates.is_empty() {
            let mut hrs: Vec<f32> = fft_heart_rates.iter().map(|(_, hr)| *hr).collect();
            hrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Some(hrs[hrs.len() / 2])
        } else {
            None
        };

        // Second pass - conditionally reprocess early windows
        if let Some(median) = median_hr {
            const REPROCESS_THRESHOLD: f32 = 7.5; // BPM threshold for reprocessing
            const EARLY_WINDOW_HOURS: i64 = 4;

            // Clear history for second pass
            hr_history.clear();
            prev_hr = None;

            // Get the cutoff time for early windows
            let start_time = hr_windows.first().map(|w| w.timestamp);
            if let Some(start) = start_time {
                let early_cutoff = start + chrono::Duration::hours(EARLY_WINDOW_HOURS);

                // Create a new vector for the final results
                let mut final_heart_rates = Vec::new();

                for (i, window) in hr_windows.iter().enumerate() {
                    if window.timestamp <= early_cutoff {
                        // For early windows, check if we need to reprocess
                        let current_hr = fft_heart_rates.get(i).map(|(_, hr)| *hr);

                        let needs_reprocessing =
                            current_hr.map_or(true, |hr| (hr - median).abs() > REPROCESS_THRESHOLD);

                        if needs_reprocessing {
                            // Reprocess with median-informed range
                            let breathing_data = breathing_rates_with_stability
                                .iter()
                                .min_by_key(|(br_time, _, _)| {
                                    (br_time.timestamp() - window.timestamp.timestamp()).abs()
                                        as u64
                                })
                                .map(|(_, rate, stability)| (*rate, *stability));

                            if let Some(new_hr) = heart_analysis::analyze_heart_rate_fft(
                                &window.processed_signal,
                                500.0,
                                prev_hr,
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
                                Some(median - 20.0),
                                Some(median + 15.0),
                            ) {
                                final_heart_rates.push((window.timestamp, new_hr));
                                prev_hr = Some(new_hr);
                            }
                        } else {
                            // Keep original estimate
                            if let Some((ts, hr)) = fft_heart_rates.get(i) {
                                final_heart_rates.push((*ts, *hr));
                                prev_hr = Some(*hr);
                            }
                        }
                    } else {
                        // For later windows, keep original estimates
                        if let Some((ts, hr)) = fft_heart_rates.get(i) {
                            final_heart_rates.push((*ts, *hr));
                            prev_hr = Some(*hr);
                        }
                    }
                }

                fft_heart_rates = final_heart_rates;
            }
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

        let analysis_combined = if let Some(SensorSelection::Choose) = args.sensor {
            // Run analysis on both sensors
            let signal1: Vec<i32> = (0..raw_data_view.len())
                .filter_map(|idx| raw_data_view.get_data_at(idx))
                .map(|data| data.left1.to_vec())
                .flatten()
                .collect();

            let signal2: Vec<i32> = (0..raw_data_view.len())
                .filter_map(|idx| raw_data_view.get_data_at(idx))
                .map(|data| data.left2.unwrap_or_default().to_vec())
                .flatten()
                .collect();

            // Analyze both sensors
            let analysis1 = analyse_sensor_data(
                &signal1,
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
                args.enable_median_reprocessing,
            );

            let analysis2 = analyse_sensor_data(
                &signal2,
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
                args.enable_median_reprocessing,
            );

            // Compare and choose the better analysis
            heart_analysis::compare_sensor_analysis(&analysis1, &analysis2)
                .map(|analysis| analysis.clone())
                .unwrap_or(analysis1)
        } else {
            // Extract signal based on sensor selection
            let mut signal: Vec<i32> = (0..raw_data_view.len())
                .filter_map(|idx| raw_data_view.get_data_at(idx))
                .map(|data| match args.sensor {
                    Some(SensorSelection::First) => data.left1.to_vec(),
                    Some(SensorSelection::Second) => data.left2.unwrap_or_default().to_vec(),
                    Some(SensorSelection::Combined) | None => data.left.to_vec(),
                    Some(SensorSelection::Choose) => unreachable!(),
                })
                .flatten()
                .collect();

            // Rest of the existing analysis code...
            if args.merge_sides {
                let right_signal: Vec<i32> = (0..raw_data_view.len())
                    .filter_map(|idx| raw_data_view.get_data_at(idx))
                    .map(|data| data.right.to_vec())
                    .flatten()
                    .collect();

                // Average the signals
                for (left, right) in signal.iter_mut().zip(right_signal.iter()) {
                    *left = ((*left as i64 + *right as i64) / 2) as i32;
                }
            }

            analyse_sensor_data(
                &signal,
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
                args.enable_median_reprocessing,
            )
        };

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

            let analysis_combined = if let Some(SensorSelection::Choose) = args.sensor {
                // Run analysis on both sensors
                let signal1: Vec<i32> = (0..raw_data_view.len())
                    .filter_map(|idx| raw_data_view.get_data_at(idx))
                    .map(|data| data.right1.to_vec())
                    .flatten()
                    .collect();

                let signal2: Vec<i32> = (0..raw_data_view.len())
                    .filter_map(|idx| raw_data_view.get_data_at(idx))
                    .map(|data| data.right2.unwrap_or_default().to_vec())
                    .flatten()
                    .collect();

                // Analyze both sensors
                let analysis1 = analyse_sensor_data(
                    &signal1,
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
                    args.enable_median_reprocessing,
                );

                let analysis2 = analyse_sensor_data(
                    &signal2,
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
                    args.enable_median_reprocessing,
                );

                // Compare and choose the better analysis
                heart_analysis::compare_sensor_analysis(&analysis1, &analysis2)
                    .map(|analysis| analysis.clone())
                    .unwrap_or(analysis1)
            } else {
                // Extract signal based on sensor selection
                let signal: Vec<i32> = (0..raw_data_view.len())
                    .filter_map(|idx| raw_data_view.get_data_at(idx))
                    .map(|data| match args.sensor {
                        Some(SensorSelection::First) => data.right1.to_vec(),
                        Some(SensorSelection::Second) => data.right2.unwrap_or_default().to_vec(),
                        Some(SensorSelection::Combined) | None => data.right.to_vec(),
                        Some(SensorSelection::Choose) => unreachable!(),
                    })
                    .flatten()
                    .collect();

                analyse_sensor_data(
                    &signal,
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
                    args.enable_median_reprocessing,
                )
            };

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

fn parse_datetime(datetime_str: &str) -> Result<DateTime<Utc>> {
    let naive = NaiveDateTime::parse_from_str(datetime_str, "%Y-%m-%d %H:%M")?;
    Ok(DateTime::from_naive_utc_and_offset(naive, Utc))
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

fn main() -> Result<()> {
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
            return Err(anyhow::anyhow!(
                "No RAW files found in directory: {}",
                args.input_path.display()
            ));
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

        println!("\nProcessing {} RAW files", relevant_files.len());

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
        .map(|(_, data)| preprocessing::process_piezo_data(data))
        .flatten()
        .collect();

    // First remove time outliers
    preprocessing::remove_time_outliers(&mut all_processed_data);

    // Then remove stat outliers
    let len_before = all_processed_data.len();
    preprocessing::remove_stat_outliers(&mut all_processed_data);

    println!(
        "Removed {} rows as mean/std outliers",
        len_before - all_processed_data.len()
    );

    println!("Total processed entries: {}", all_processed_data.len());

    // Scale the data
    preprocessing::scale_data(&mut all_processed_data);

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
            output::write_analysis_to_csv(
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
            output::write_analysis_to_csv(
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
