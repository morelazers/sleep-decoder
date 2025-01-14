use chrono::{DateTime, Utc};
use log::debug;
use rustfft::{num_complex::Complex, FftPlanner};
use sci_rs::signal::filter::{design::Sos, sosfiltfilt_dyn};
use std::f32::consts::PI;

/// Create a Hann window of the specified size
fn create_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Design notch or peak digital filter.
fn design_notch_peak_filter(mut w0: f32, q: f32, ftype: &str, fs: f32) -> (Vec<f32>, Vec<f32>) {
    // Validate fs
    assert!(fs > 0.0, "fs must be positive");

    // Guarantee inputs are floats (already handled by Rust's type system)
    w0 = 2.0 * w0 / fs;

    // Check if w0 is within range
    assert!(w0 > 0.0 && w0 < 1.0, "w0 should be such that 0 < w0 < 1");

    // Get bandwidth
    let mut bw = w0 / q;

    // Normalize inputs
    bw = bw * PI;
    let w0 = w0 * PI;

    assert!(ftype == "notch" || ftype == "peak", "Unknown ftype");

    // Compute beta (see Python comments for reference)
    let beta = (bw / 2.0).tan();

    // Compute gain
    let gain = 1.0 / (1.0 + beta);

    // Compute numerator b and denominator a
    let b = if ftype == "notch" {
        vec![1.0, -2.0 * w0.cos(), 1.0]
            .iter()
            .map(|x| x * gain)
            .collect()
    } else {
        vec![1.0, 0.0, -1.0]
            .iter()
            .map(|x| x * (1.0 - gain))
            .collect()
    };

    let a = vec![1.0, -2.0 * gain * w0.cos(), 2.0 * gain - 1.0];

    (b, a)
}

/// Creates a notch filter using the design_notch_peak_filter function
fn create_notch_filter(freq: f32, sample_rate: f32) -> (Vec<f32>, Vec<f32>) {
    let q = 0.005; // Fixed Q factor
    let (b, a) = design_notch_peak_filter(freq, q, "notch", sample_rate);
    (b, a)
}

/// Removes baseline wander using a notch filter
pub fn remove_baseline_wander(data: &[f32], sample_rate: f32, cutoff: f32) -> Vec<f32> {
    let (b, a) = create_notch_filter(cutoff, sample_rate);

    // Convert to SOS format and use sosfiltfilt
    let sos = tf2sos(&b, &a);

    // Convert to sci-rs's Sos format
    let sos_array = vec![Sos::new(
        [sos[0][0], sos[0][1], sos[0][2]],
        [sos[0][3], sos[0][4], sos[0][5]],
    )];

    sosfiltfilt_dyn(data.iter(), &sos_array)
}

/// Replace outliers with linear interpolation from neighboring points
pub fn interpolate_outliers(data: &[i32], i: i32) -> Vec<f32> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    // Sort for percentile calculation
    let mut sorted = data_f32.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (data_f32.len() as f32 * (i as f32 / 100.0)) as usize;
    let upper_idx = (data_f32.len() as f32 * (1.0 - (i as f32 / 100.0))) as usize;

    let lower_bound = sorted[lower_idx];
    let upper_bound = sorted[upper_idx];

    let mut result = data_f32.clone();

    // Replace outliers with interpolated values
    for i in 0..data_f32.len() {
        if data_f32[i] < lower_bound || data_f32[i] > upper_bound {
            // Find previous valid value
            let prev_val = if i > 0 {
                let mut j = i - 1;
                while j > 0 && (data_f32[j] < lower_bound || data_f32[j] > upper_bound) {
                    j -= 1;
                }
                data_f32[j]
            } else {
                data_f32[i] // If no previous value, use current
            };

            // Find next valid value
            let next_val = if i < data_f32.len() - 1 {
                let mut j = i + 1;
                while j < data_f32.len() - 1
                    && (data_f32[j] < lower_bound || data_f32[j] > upper_bound)
                {
                    j += 1;
                }
                data_f32[j]
            } else {
                data_f32[i] // If no next value, use current
            };

            // Replace with average of previous and next valid values
            result[i] = (prev_val + next_val) / 2.0;
        }
    }

    result
}

/// Scale data to specified range
pub fn scale_data(data: &[f32], lower: f32, upper: f32) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    data.iter()
        .map(|&x| (upper - lower) * ((x - min_val) / range) + lower)
        .collect()
}

/// Convert transfer function coefficients to second-order sections
fn tf2sos(b: &[f32], a: &[f32]) -> Vec<[f32; 6]> {
    // For a second-order filter (like our notch filter), we just need one section
    // Each section is [b0, b1, b2, 1, a1, a2]
    vec![[b[0], b[1], b[2], 1.0, a[1] / a[0], a[2] / a[0]]]
}

/// Analyze heart rate using FFT, considering previous heart rate for continuity
pub fn analyze_heart_rate_fft(
    signal: &[f32],
    sample_rate: f32,
    prev_hr: Option<f32>,
) -> Option<f32> {
    // Apply Hann window
    let window = create_hann_window(signal.len());
    let windowed_signal: Vec<f32> = signal
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| s * w)
        .collect();

    // Prepare FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());

    // Convert to complex numbers
    let mut buffer: Vec<Complex<f32>> = windowed_signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    // Perform FFT
    fft.process(&mut buffer);

    // Calculate frequency resolution
    let freq_resolution = sample_rate / signal.len() as f32;

    // Look at magnitude spectrum in the heart rate range (40-100 BPM = 0.67-1.67 Hz)
    let min_bin = (0.67 / freq_resolution) as usize;
    let max_bin = (1.67 / freq_resolution) as usize;

    // Find all peaks in the heart rate range
    let mut peaks = Vec::new();
    for bin in min_bin + 1..=max_bin - 1 {
        let magnitude = (buffer[bin].norm_sqr() as f32).sqrt();
        let prev_magnitude = (buffer[bin - 1].norm_sqr() as f32).sqrt();
        let next_magnitude = (buffer[bin + 1].norm_sqr() as f32).sqrt();

        // Check if this is a local maximum
        if magnitude > prev_magnitude && magnitude > next_magnitude {
            let freq = bin as f32 * freq_resolution;
            let bpm = freq * 60.0;
            peaks.push((bpm, magnitude));
        }
    }

    // Sort peaks by magnitude
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // If we have a previous heart rate, try to find a peak close to it
    if let Some(prev_hr) = prev_hr {
        // Look at top 3 peaks (if available) and choose the one closest to previous HR
        let top_peaks: Vec<(f32, f32)> = peaks
            .iter()
            .take(3)
            .filter(|(_bpm, magnitude)| {
                // Must be at least 20% of strongest peak's magnitude
                *magnitude >= peaks[0].1 * 0.2
            })
            .cloned()
            .collect();

        if !top_peaks.is_empty() {
            // Find the peak closest to previous heart rate
            let closest_peak = top_peaks.iter().min_by(|(bpm1, _), (bpm2, _)| {
                let diff1 = (bpm1 - prev_hr).abs();
                let diff2 = (bpm2 - prev_hr).abs();
                diff1.partial_cmp(&diff2).unwrap()
            });

            if let Some(&(bpm, _)) = closest_peak {
                // Only accept if it's not too far from previous HR (max 15 BPM difference)
                if (bpm - prev_hr).abs() <= 15.0 {
                    return Some(bpm);
                }
            }
        }
    }

    // If no previous HR or no suitable peaks found near previous HR,
    // return the strongest peak if it's in valid range
    peaks
        .first()
        .map(|&(bpm, _)| bpm)
        .filter(|&bpm| bpm >= 40.0 && bpm <= 180.0)
}

/// Analyze breathing rate using FFT on the scaled signal
pub fn analyze_breathing_rate_fft(signal: &[f32], sample_rate: f32) -> Option<f32> {
    // Check if we have enough samples for reliable breathing rate detection
    // Need at least 30 seconds of data for good frequency resolution
    if signal.len() < (30.0 * sample_rate) as usize {
        debug!("Window too small for reliable breathing rate detection. Need at least 30 seconds of data.");
        return None;
    }

    // Apply Hann window
    let window = create_hann_window(signal.len());
    let windowed_signal: Vec<f32> = signal
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| s * w)
        .collect();

    // Prepare FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());

    // Convert to complex numbers
    let mut buffer: Vec<Complex<f32>> = windowed_signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    // Perform FFT
    fft.process(&mut buffer);

    // Calculate frequency resolution
    let freq_resolution = sample_rate / signal.len() as f32;
    debug!("Frequency resolution: {} Hz", freq_resolution);

    // Look at magnitude spectrum in the breathing rate range (8-20 BPM = 0.133-0.333 Hz)
    let min_bin = (0.133 / freq_resolution) as usize;
    let max_bin = (0.333 / freq_resolution) as usize;

    debug!(
        "Analyzing bins {} to {} for breathing rate",
        min_bin, max_bin
    );

    // Find peak in breathing rate range
    let mut max_magnitude = 0.0;
    let mut peak_freq = 0.0;

    for bin in min_bin..=max_bin {
        let magnitude = (buffer[bin].norm_sqr() as f32).sqrt();
        if magnitude > max_magnitude {
            max_magnitude = magnitude;
            peak_freq = bin as f32 * freq_resolution;
        }
    }

    // Convert peak frequency to breaths per minute
    let breaths_per_minute = peak_freq * 60.0;

    // Basic validation
    if breaths_per_minute >= 8.0 && breaths_per_minute <= 20.0 {
        debug!("Found breathing rate: {:.1} BPM", breaths_per_minute);
        Some(breaths_per_minute)
    } else {
        debug!(
            "Breathing rate {:.1} BPM outside valid range",
            breaths_per_minute
        );
        None
    }
}

/// Interpolate missing values and smooth the time series
pub fn interpolate_and_smooth(
    timestamps: &[DateTime<Utc>],
    values: &[(DateTime<Utc>, f32)],
    window_size: usize,
) -> Vec<(DateTime<Utc>, f32)> {
    if values.is_empty() || timestamps.is_empty() {
        return Vec::new();
    }

    // First, interpolate missing values
    let mut interpolated: Vec<(DateTime<Utc>, f32)> = Vec::with_capacity(timestamps.len());

    for &timestamp in timestamps {
        // Find the value at this timestamp or interpolate
        let value = match values.binary_search_by_key(&timestamp, |&(t, _)| t) {
            Ok(idx) => values[idx].1,
            Err(idx) => {
                if idx == 0 {
                    // Before first value, use first value
                    values[0].1
                } else if idx >= values.len() {
                    // After last value, use last value
                    values[values.len() - 1].1
                } else {
                    // Interpolate between two points
                    let (t1, v1) = values[idx - 1];
                    let (t2, v2) = values[idx];

                    // Calculate weights based on time differences
                    let total_duration = (t2 - t1).num_seconds() as f32;
                    let first_duration = (timestamp - t1).num_seconds() as f32;

                    // Linear interpolation
                    let weight = first_duration / total_duration;
                    v1 + (v2 - v1) * weight
                }
            }
        };
        interpolated.push((timestamp, value));
    }

    // Then apply moving average smoothing
    let mut smoothed = Vec::with_capacity(interpolated.len());
    let half_window = window_size / 2;

    for i in 0..interpolated.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(interpolated.len());
        let window = &interpolated[start..end];

        // Calculate Gaussian-weighted moving average
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        let sigma = (window.len() as f32) / 4.0; // Controls how quickly weights fall off
        let center = window.len() as f32 / 2.0;

        for (j, &(_, value)) in window.iter().enumerate() {
            let x = (j as f32 - center) / sigma;
            let weight = (-0.5 * x * x).exp(); // Gaussian weight
            weighted_sum += value * weight;
            weight_sum += weight;
        }

        let smoothed_value = weighted_sum / weight_sum;
        smoothed.push((interpolated[i].0, smoothed_value));
    }

    smoothed
}
