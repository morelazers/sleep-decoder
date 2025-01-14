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

/// Calculate physiologically possible heart rate range based on previous HR
fn get_adaptive_hr_range(prev_hr: f32, time_step: f32) -> (f32, f32) {
    // Reduce maximum allowed changes
    const MAX_INSTANT_CHANGE: f32 = 6.0;
    const MAX_GRADUAL_CHANGE: f32 = 1.5;

    // Calculate maximum possible change for this time step
    let max_change = if time_step <= 5.0 {
        MAX_INSTANT_CHANGE * time_step
    } else {
        (MAX_INSTANT_CHANGE * 5.0) + (MAX_GRADUAL_CHANGE * (time_step - 5.0))
    };

    // Make the range slightly asymmetric - less aggressive downward bias
    let upward_change = max_change * 0.85; // Changed from 0.7 - allow more upward movement
    let downward_change = max_change * 0.9; // Add small restriction to downward movement too

    // Calculate range with smaller buffer
    let min_hr = (prev_hr - downward_change - 3.0).max(35.0);
    let max_hr = (prev_hr + upward_change + 3.0).min(95.0);

    // Less aggressive HR zone restrictions
    let max_hr = if prev_hr < 60.0 {
        // Low HR zone - allow more upward movement
        (prev_hr + upward_change * 0.8).min(80.0) // Changed from 0.6 and 75.0
    } else if prev_hr < 75.0 {
        // Medium HR zone
        (prev_hr + upward_change * 0.9).min(90.0) // Changed from 0.8 and 85.0
    } else {
        // High HR zone
        max_hr
    };

    (min_hr, max_hr)
}

/// Calculate signal quality metric
fn calculate_signal_quality(signal: &[f32]) -> f32 {
    // Calculate signal variance
    let mean = signal.iter().sum::<f32>() / signal.len() as f32;
    let variance = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / signal.len() as f32;

    // Calculate zero crossings
    let zero_crossings = signal
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();

    // Combine metrics (lower score = noisier signal)
    let quality_score = 1.0 / (1.0 + variance * (zero_crossings as f32 / signal.len() as f32));

    quality_score
}

/// Detect if a heart rate change is suspicious based on multiple criteria
fn is_suspicious_change(
    bpm: f32,
    magnitude: f32,
    max_magnitude: f32,
    prev_hr: f32,
    quality: f32,
) -> bool {
    let hr_change = (bpm - prev_hr).abs();
    let normalized_magnitude = magnitude / max_magnitude;

    // Criteria for suspicious changes:
    // 1. Large change with weak peak
    // 2. Change size relative to signal quality
    // 3. Magnitude threshold increases with larger changes

    let required_magnitude = 0.3 + (hr_change / 20.0).min(0.3); // Higher threshold for bigger changes

    (hr_change > 8.0 && normalized_magnitude < required_magnitude)
        || (hr_change > 12.0 && quality < 0.4)
        || (hr_change > 15.0 && normalized_magnitude < 0.6)
}

/// Store historical heart rate data
pub struct HeartRateHistory {
    rates: Vec<(DateTime<Utc>, f32)>, // (timestamp, heart_rate)
    window_duration: f32,             // Duration to look back in seconds
}

impl HeartRateHistory {
    pub fn new(window_duration: f32) -> Self {
        Self {
            rates: Vec::new(),
            window_duration,
        }
    }

    pub fn add_measurement(&mut self, timestamp: DateTime<Utc>, rate: f32) {
        self.rates.push((timestamp, rate));

        // Remove old measurements
        let cutoff = timestamp - chrono::Duration::seconds(self.window_duration as i64);
        self.rates.retain(|(ts, _)| *ts >= cutoff);
    }

    pub fn get_trend(&self) -> Option<f32> {
        if self.rates.len() < 2 {
            return None;
        }

        // Calculate rate of change in BPM per minute
        let (first_ts, first_rate) = self.rates.first()?;
        let (last_ts, last_rate) = self.rates.last()?;

        let duration_mins = (*last_ts - first_ts).num_seconds() as f32 / 60.0;
        if duration_mins < 1.0 {
            return None;
        }

        Some((last_rate - first_rate) / duration_mins)
    }
}

/// Modify the heart rate analysis function to use history
pub fn analyze_heart_rate_fft(
    signal: &[f32],
    sample_rate: f32,
    prev_hr: Option<f32>,
    timestamp: DateTime<Utc>,
    history: &mut HeartRateHistory,
    time_step: f32,
) -> Option<f32> {
    let quality = calculate_signal_quality(signal);

    // Check both trend and physiological plausibility
    if let Some(prev_hr) = prev_hr {
        // Check overall trend
        if let Some(trend) = history.get_trend() {
            if !validate_rate_of_change(trend, trend < 0.0) {
                debug!("Trend too steep ({:.2} BPM/min), being conservative", trend);
                let smoothed_bpm = prev_hr;
                history.add_measurement(timestamp, smoothed_bpm);
                return Some(smoothed_bpm);
            }
        }
    }

    // If signal is too noisy and we have a previous value, return previous
    if quality < 0.35 && prev_hr.is_some() {
        return prev_hr;
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

    // Calculate frequency range
    let (min_hr, max_hr) = if let Some(prev_hr) = prev_hr {
        get_adaptive_hr_range(prev_hr, time_step)
    } else {
        (40.0, 100.0) // Default range if no previous HR
    };

    // Convert BPM to Hz
    let min_freq = min_hr / 60.0;
    let max_freq = max_hr / 60.0;

    // Calculate bin range for FFT
    let freq_resolution = sample_rate / signal.len() as f32;
    let min_bin = (min_freq / freq_resolution) as usize;
    let max_bin = (max_freq / freq_resolution) as usize;

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
        let top_peaks: Vec<(f32, f32)> = peaks
            .iter()
            .take(7)
            .filter(|(_bpm, magnitude)| *magnitude >= peaks[0].1 * 0.2)
            .cloned()
            .collect();

        if !top_peaks.is_empty() {
            let best_peak = top_peaks
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            let (bpm, magnitude) = *best_peak;

            // After finding best peak but before applying it:
            // Check if the change is physiologically plausible
            let window_duration = if let Some((first_ts, _)) = history.rates.first() {
                (timestamp - *first_ts).num_seconds() as f32
            } else {
                time_step
            };

            if !is_physiologically_plausible(bpm, prev_hr, window_duration) {
                debug!(
                    "Change not physiologically plausible: {:.1} -> {:.1} BPM",
                    prev_hr, bpm
                );
                let smoothed_bpm = prev_hr;
                history.add_measurement(timestamp, smoothed_bpm);
                return Some(smoothed_bpm);
            }

            // Check if this change is suspicious
            if is_suspicious_change(bpm, magnitude, peaks[0].1, prev_hr, quality) {
                let maintained_hr = prev_hr;
                history.add_measurement(timestamp, maintained_hr);
                return Some(maintained_hr);
            }

            // Calculate and apply confidence-based weighting
            let confidence = {
                let normalized_magnitude = magnitude / peaks[0].1;
                let hr_change = (bpm - prev_hr).abs();
                let change_penalty = (hr_change / 20.0).min(0.5);
                let base_confidence = normalized_magnitude * quality;
                (base_confidence - change_penalty).max(0.0)
            };

            let smoothed_bpm = if confidence >= 0.8 {
                let weight = 0.8;
                bpm * weight + prev_hr * (1.0 - weight)
            } else if confidence >= 0.6 {
                let weight = confidence / 2.5;
                bpm * weight + prev_hr * (1.0 - weight)
            } else {
                let weight = 0.05;
                bpm * weight + prev_hr * (1.0 - weight)
            };

            history.add_measurement(timestamp, smoothed_bpm);
            return Some(smoothed_bpm);
        }
    }

    // If no previous HR or no suitable peaks found near previous HR,
    // return the strongest peak if it's in valid range
    let result = peaks
        .first()
        .map(|&(bpm, _)| bpm)
        .filter(|&bpm| bpm >= 40.0 && bpm <= 100.0);

    if let Some(bpm) = result {
        history.add_measurement(timestamp, bpm);
    }

    result
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

/// At module level
const MAX_DECREASE_RATE: f32 = 1.2; // BPM per minute
const MAX_INCREASE_RATE: f32 = 1.5; // BPM per minute

/// Validate rate of change against physiological limits
fn validate_rate_of_change(rate_of_change: f32, is_decrease: bool) -> bool {
    if is_decrease {
        rate_of_change >= -MAX_DECREASE_RATE
    } else {
        rate_of_change <= MAX_INCREASE_RATE
    }
}

/// Check if heart rate change is physiologically plausible
fn is_physiologically_plausible(current_bpm: f32, prev_hr: f32, window_duration: f32) -> bool {
    let hr_change = current_bpm - prev_hr;
    let rate_of_change = hr_change / (window_duration / 60.0); // Convert to per minute
    validate_rate_of_change(rate_of_change, hr_change < 0.0)
}
