use crate::PeriodAnalysis;
use crate::RawDataView;
use chrono::{DateTime, Duration, Utc};
use log::debug;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use sci_rs::signal::filter::{design::Sos, sosfiltfilt_dyn};
use std::f32::consts::PI;
use std::sync::Arc;

/// A window of signal data with its metadata
pub struct SignalWindow {
    pub timestamp: DateTime<Utc>,
    pub processed_signal: Vec<f32>,
}

/// Iterator that yields processed windows of signal data
pub struct SignalWindowIterator<'a> {
    signal: &'a [i32],
    raw_data: &'a RawDataView<'a>,
    window_size: usize,
    step_size: usize,
    current_idx: usize,
    buffers: SignalProcessingBuffers,
}

impl<'a> SignalWindowIterator<'a> {
    pub fn new(
        signal: &'a [i32],
        raw_data: &'a RawDataView<'a>,
        window_size: usize,
        step_size: usize,
    ) -> Self {
        Self {
            signal,
            raw_data,
            window_size,
            step_size,
            current_idx: 0,
            buffers: SignalProcessingBuffers::new(window_size),
        }
    }
}

impl<'a> Iterator for SignalWindowIterator<'a> {
    type Item = SignalWindow;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.signal.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.window_size).min(self.signal.len());

        // Skip if window is too small
        if end_idx - self.current_idx < self.window_size / 2 {
            return None;
        }

        let segment = &self.signal[self.current_idx..end_idx];
        let timestamp = self.raw_data.get_data_at(self.current_idx / 500)?.timestamp;

        // Use the buffers to process the window
        let processed_signal = self.buffers.process_window(segment).to_vec();

        self.current_idx += self.step_size;

        Some(SignalWindow {
            timestamp,
            processed_signal,
        })
    }
}

pub struct SignalProcessingBuffers {
    cleaned: Vec<f32>,
    float_buffer: Vec<f32>,
    scaled: Vec<f32>,
    processed: Vec<f32>,
}

impl SignalProcessingBuffers {
    pub fn new(capacity: usize) -> Self {
        Self {
            cleaned: Vec::with_capacity(capacity),
            float_buffer: Vec::with_capacity(capacity),
            scaled: Vec::with_capacity(capacity),
            processed: Vec::with_capacity(capacity),
        }
    }

    pub fn process_window(&mut self, segment: &[i32]) -> &[f32] {
        // Reuse buffers instead of creating new ones
        self.cleaned.clear();
        self.float_buffer.clear();
        self.scaled.clear();
        self.processed.clear();

        interpolate_outliers_into(segment, 2, &mut self.cleaned);
        convert_to_f32(&self.cleaned, &mut self.float_buffer);
        scale_data_into(&self.float_buffer, 0.0, 1024.0, &mut self.scaled);
        remove_baseline_wander_into(&self.scaled, 500.0, 0.05, &mut self.processed);

        &self.processed
    }
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
        let cutoff = timestamp - Duration::seconds(self.window_duration as i64);
        self.rates.retain(|(ts, _)| *ts >= cutoff);
    }

    /// Clear all stored heart rate measurements
    pub fn clear(&mut self) {
        self.rates.clear();
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

    pub fn get_recent_rates(&self, count: usize) -> Vec<f32> {
        // Take the last 'count' measurements
        self.rates
            .iter()
            .rev() // Reverse to get most recent first
            .take(count) // Take only the number we want
            .map(|(_, rate)| *rate)
            .collect()
    }
}

/// FFT context holding reusable buffers and planner
pub struct FftContext {
    window: Vec<f32>,
    windowed_signal: Vec<f32>,
    complex_buffer: Vec<Complex<f32>>,
    fft: Arc<dyn Fft<f32>>,
    buffer_size: usize,
}

impl FftContext {
    pub fn new(size: usize) -> Self {
        // Create Hann window once
        let window = create_hann_window(size);

        // Pre-allocate buffers
        let windowed_signal = Vec::with_capacity(size);
        let complex_buffer = Vec::with_capacity(size);

        // Create FFT planner once
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);

        Self {
            window,
            windowed_signal,
            complex_buffer,
            fft,
            buffer_size: size,
        }
    }

    fn process(&mut self, signal: &[f32]) -> &[Complex<f32>] {
        // Pad or truncate signal to match buffer size
        self.windowed_signal.clear();
        if signal.len() <= self.buffer_size {
            // Pad with zeros
            self.windowed_signal
                .extend(signal.iter().zip(self.window.iter()).map(|(&s, &w)| s * w));
            self.windowed_signal.resize(self.buffer_size, 0.0);
        } else {
            // Truncate
            self.windowed_signal.extend(
                signal[..self.buffer_size]
                    .iter()
                    .zip(self.window.iter())
                    .map(|(&s, &w)| s * w),
            );
        }

        // Rest of processing remains the same...
        self.complex_buffer.clear();
        self.complex_buffer
            .extend(self.windowed_signal.iter().map(|&x| Complex::new(x, 0.0)));

        self.fft.process(&mut self.complex_buffer);

        &self.complex_buffer
    }
}

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

    // Calculate bounds using indices to avoid cloning
    let mut indices: Vec<usize> = (0..data_f32.len()).collect();
    indices.sort_by(|&a, &b| data_f32[a].partial_cmp(&data_f32[b]).unwrap());

    let lower_idx = (data_f32.len() as f32 * (i as f32 / 100.0)) as usize;
    let upper_idx = (data_f32.len() as f32 * (1.0 - (i as f32 / 100.0))) as usize;

    let lower_bound = data_f32[indices[lower_idx]];
    let upper_bound = data_f32[indices[upper_idx]];

    let mut result = data_f32; // Take ownership instead of cloning

    // Replace outliers with interpolated values
    for i in 0..result.len() {
        if result[i] < lower_bound || result[i] > upper_bound {
            // Find previous valid value
            let prev_val = if i > 0 {
                let mut j = i - 1;
                while j > 0 && (result[j] < lower_bound || result[j] > upper_bound) {
                    j -= 1;
                }
                result[j]
            } else {
                result[i] // If no previous value, use current
            };

            // Find next valid value
            let next_val = if i < result.len() - 1 {
                let mut j = i + 1;
                while j < result.len() - 1 && (result[j] < lower_bound || result[j] > upper_bound) {
                    j += 1;
                }
                result[j]
            } else {
                result[i] // If no next value, use current
            };

            // Replace with average of previous and next valid values
            result[i] = (prev_val + next_val) / 2.0;
        }
    }

    result
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
    let upward_change = max_change * 0.85;
    let downward_change = max_change * 0.9;

    // Calculate range with smaller buffer
    let min_hr = (prev_hr - downward_change - 3.0).max(35.0);
    let max_hr = (prev_hr + upward_change + 3.0).min(90.0);

    // Less aggressive HR zone restrictions
    let max_hr = if prev_hr < 65.0 {
        // Low HR zone - allow more upward movement
        (prev_hr + upward_change * 0.8).min(85.0)
    } else if prev_hr < 75.0 {
        // Medium HR zone
        (prev_hr + upward_change * 0.9).min(90.0)
    } else {
        // High HR zone
        max_hr
    };

    (min_hr, max_hr)
}

/// Detect if a heart rate change is suspicious based on multiple criteria
fn is_suspicious_change(bpm: f32, magnitude: f32, max_magnitude: f32, prev_hr: f32) -> bool {
    let hr_change = (bpm - prev_hr).abs();
    let normalized_magnitude = magnitude / max_magnitude;

    // Require stronger evidence for higher heart rates
    let base_required_magnitude = 0.3 + (hr_change / 20.0).min(0.3);
    let high_hr_penalty = if bpm > 80.0 {
        // Additional magnitude requirement for rates above 80 BPM
        ((bpm - 80.0) / 20.0).min(0.3)
    } else {
        0.0
    };

    let required_magnitude = base_required_magnitude + high_hr_penalty;

    // More stringent checks for higher rates
    (hr_change > 8.0 && normalized_magnitude < required_magnitude)
        || (hr_change > 15.0 && normalized_magnitude < 0.6)
        || (bpm > 80.0 && normalized_magnitude < (0.5 + high_hr_penalty))
}

/// Calculate the harmonic penalty factor for a given frequency
fn calculate_harmonic_penalty(
    freq: f32,
    bpm: f32,
    breathing_data: Option<(f32, f32)>,
    prev_hr: Option<f32>,
    harmonic_penalty_close: f32,
    harmonic_penalty_far: f32,
    harmonic_close_threshold: f32,
    harmonic_far_threshold: f32,
) -> f32 {
    // Base penalty from existing logic
    let mut penalty = if let Some((br, stability)) = breathing_data {
        // Calculate breathing harmonics
        let fundamental = br / 60.0; // Convert BPM to Hz
        let harmonics = vec![
            fundamental * 2.0, // Second harmonic
            fundamental * 3.0, // Third harmonic
            fundamental * 4.0, // Fourth harmonic
            fundamental * 5.0, // Fifth harmonic - added for higher coverage
        ];

        // Check if frequency is near a harmonic with tighter bounds for higher rates
        let harmonic_threshold = if bpm > 80.0 {
            0.03 // Tighter threshold for higher rates
        } else {
            0.05 // Original threshold
        };

        let is_harmonic = harmonics
            .iter()
            .any(|&h| (freq - h).abs() < harmonic_threshold);

        if is_harmonic {
            // If we have a previous heart rate, check if this peak is close to it
            if let Some(prev_hr) = prev_hr {
                let freq_diff = (bpm - prev_hr).abs();
                if freq_diff < harmonic_close_threshold {
                    1.0 // No penalty if very close to previous HR
                } else {
                    // Scale penalty by breathing stability and heart rate
                    let base_penalty = if freq_diff < harmonic_far_threshold {
                        harmonic_penalty_close
                    } else {
                        harmonic_penalty_far
                    };

                    // Increase penalty for higher heart rates
                    let hr_factor = if bpm > 80.0 {
                        1.0 + ((bpm - 80.0) / 40.0).min(0.5)
                    } else {
                        1.0
                    };

                    1.0 - ((1.0 - base_penalty) * stability * hr_factor)
                }
            } else {
                // No previous HR, scale penalty by stability
                1.0 - (0.5 * stability)
            }
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Additional penalty for higher heart rates without strong historical support
    if bpm > 80.0 {
        if let Some(prev_hr) = prev_hr {
            if prev_hr < 75.0 {
                // Stronger penalty for jumping to high HR from low HR
                penalty *= 0.8;
            }
        }
    }

    penalty
}

/// Modify the heart rate analysis function to use breathing rate data
pub fn analyze_heart_rate_fft(
    signal: &[f32],
    sample_rate: f32,
    prev_hr: Option<f32>,
    timestamp: DateTime<Utc>,
    history: &mut HeartRateHistory,
    fft_context: &mut FftContext,
    time_step: f32,
    breathing_data: Option<(f32, f32)>,
    hr_outlier_percentile: f32,
    hr_history_window: usize,
    harmonic_penalty_close: f32,
    harmonic_penalty_far: f32,
    harmonic_close_threshold: f32,
    harmonic_far_threshold: f32,
    min_hr: Option<f32>,
    max_hr: Option<f32>,
) -> Option<f32> {
    // Calculate initial range - broader when no history exists
    let (min_hr, max_hr) = if let Some(prev_hr) = prev_hr {
        // If min/max HR are provided, use them to constrain the adaptive range
        let (adaptive_min, adaptive_max) = get_adaptive_hr_range(prev_hr, time_step);
        (
            min_hr.map_or(adaptive_min, |min| min.max(adaptive_min)),
            max_hr.map_or(adaptive_max, |max| max.min(adaptive_max)),
        )
    } else {
        // No previous HR, use provided range or default
        (min_hr.unwrap_or(40.0), max_hr.unwrap_or(100.0))
    };

    // Check both trend and physiological plausibility
    if let Some(prev_hr) = prev_hr {
        // Check overall trend with more stringent validation for higher rates
        if let Some(trend) = history.get_trend() {
            let max_allowed_trend = if prev_hr > 80.0 || trend > 0.0 {
                0.8 // More conservative for high rates or increasing trends
            } else {
                1.2 // Original value for decreasing trends at normal rates
            };

            if !validate_rate_of_change(trend, trend < 0.0) {
                debug!("Trend too steep ({:.2} BPM/min), being conservative", trend);
                let smoothed_bpm = prev_hr;
                history.add_measurement(timestamp, smoothed_bpm);
                return Some(smoothed_bpm);
            }
        }
    }

    // Use FFT context to process signal
    let buffer = fft_context.process(signal);

    // Calculate frequency resolution and bin ranges
    let freq_resolution = sample_rate / signal.len() as f32;
    let min_bin = (min_hr / 60.0 / freq_resolution) as usize;
    let max_bin = (max_hr / 60.0 / freq_resolution) as usize;

    // Find peaks with enhanced validation
    let mut peaks = Vec::new();
    for bin in min_bin + 1..=max_bin - 1 {
        let magnitude = (buffer[bin].norm_sqr() as f32).sqrt();
        let prev_magnitude = (buffer[bin - 1].norm_sqr() as f32).sqrt();
        let next_magnitude = (buffer[bin + 1].norm_sqr() as f32).sqrt();

        if magnitude > prev_magnitude && magnitude > next_magnitude {
            let freq = bin as f32 * freq_resolution;
            let bpm = freq * 60.0;

            // Calculate harmonic penalty with enhanced validation
            let harmonic_factor = calculate_harmonic_penalty(
                freq,
                bpm,
                breathing_data,
                prev_hr,
                harmonic_penalty_close,
                harmonic_penalty_far,
                harmonic_close_threshold,
                harmonic_far_threshold,
            );

            // Additional validation for higher heart rates
            let high_hr_factor = if bpm > 80.0 {
                // Require increasingly strong evidence for higher rates
                let excess = bpm - 80.0;
                0.8 - (excess / 40.0).min(0.3)
            } else {
                1.0
            };

            peaks.push((bpm, magnitude * harmonic_factor * high_hr_factor));
        }
    }

    // Sort peaks by adjusted magnitude
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find peaks and validate them...
    let result = if let Some(prev_hr) = prev_hr {
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

            // Check if the change is physiologically plausible
            let window_duration = if let Some((first_ts, _)) = history.rates.first() {
                (timestamp - *first_ts).num_seconds() as f32
            } else {
                time_step
            };

            if !is_physiologically_plausible(bpm, prev_hr, window_duration) {
                debug!(
                    "Time: {:?} Change not physiologically plausible: {:.1} -> {:.1} BPM",
                    timestamp, prev_hr, bpm
                );
                return None;
            }

            // Check if this change is suspicious
            if is_suspicious_change(bpm, magnitude, peaks[0].1, prev_hr) {
                debug!(
                    "Time: {:?} Rejecting suspicious HR change: {:.1} -> {:.1} BPM",
                    timestamp, prev_hr, bpm
                );
                return None;
            }

            // Check if this is an outlier compared to recent history
            let recent_rates = history.get_recent_rates(hr_history_window);
            let is_valid = if !recent_rates.is_empty() {
                let mut sorted_rates = recent_rates;
                sorted_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let lower_idx = (sorted_rates.len() as f32 * hr_outlier_percentile) as usize;
                let upper_idx =
                    (sorted_rates.len() as f32 * (1.0 - hr_outlier_percentile)) as usize;
                let mut lower_bound = sorted_rates[lower_idx];
                let mut upper_bound = sorted_rates[upper_idx];

                if (upper_bound - lower_bound) < 25.0 {
                    let mean: f32 = sorted_rates.iter().sum::<f32>() / sorted_rates.len() as f32;
                    lower_bound = mean - 10.0;
                    upper_bound = mean + 10.0;
                }

                debug!(
                    "Time: {:?} Lower bound: {:?} Upper bound: {:?}",
                    timestamp, lower_bound, upper_bound
                );

                if bpm <= lower_bound || bpm >= upper_bound {
                    debug!(
                        "Time: {:?} Rejecting outlier HR {:.1} BPM (bounds: {:.1}-{:.1})",
                        timestamp, bpm, lower_bound, upper_bound
                    );
                    false
                } else {
                    true
                }
            } else {
                true
            };

            if is_valid {
                history.add_measurement(timestamp, bpm);
                Some(bpm)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        // No previous HR, use strongest peak in valid range
        peaks
            .first()
            .map(|&(bpm, _)| bpm)
            .filter(|&bpm| bpm >= 40.0 && bpm <= 100.0)
    };

    // Add to history if we found a valid rate
    if let Some(bpm) = result {
        if prev_hr.is_none() {
            history.add_measurement(timestamp, bpm);
        }
    }

    result
}

/// Analyze breathing rate using FFT on the scaled signal
pub fn analyze_breathing_rate_fft(
    signal: &[f32],
    sample_rate: f32,
    fft_context: &mut FftContext,
) -> Option<f32> {
    // Check if we have enough samples for reliable breathing rate detection
    if signal.len() < (30.0 * sample_rate) as usize {
        debug!("Window too small for reliable breathing rate detection");
        return None;
    }

    // Use FFT context instead of creating new buffers
    let buffer = fft_context.process(signal);

    // Calculate frequency resolution
    let freq_resolution = sample_rate / signal.len() as f32;

    // Look at magnitude spectrum in the breathing rate range (8-20 BPM = 0.133-0.333 Hz)
    let min_bin = (0.133 / freq_resolution) as usize;
    let max_bin = (0.333 / freq_resolution) as usize;

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

    if breaths_per_minute >= 8.0 && breaths_per_minute <= 20.0 {
        Some(breaths_per_minute)
    } else {
        None
    }
}

/// Interpolate missing values and smooth the time series
pub fn interpolate_and_smooth(
    timestamps: &[DateTime<Utc>],
    values: &[(DateTime<Utc>, f32)],
    window_size: usize,
    smoothing_strength: f32,
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

        // Invert the smoothing strength (higher value = more smoothing)
        let sigma = (window.len() as f32) * smoothing_strength;
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

fn interpolate_outliers_into(data: &[i32], i: i32, output: &mut Vec<f32>) {
    // First create a temporary vector for the f32 values
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    // Create indices vector for sorting
    let mut indices: Vec<usize> = (0..data_f32.len()).collect();
    indices.sort_by(|&a, &b| data_f32[a].partial_cmp(&data_f32[b]).unwrap());

    let lower_idx = (data_f32.len() as f32 * (i as f32 / 100.0)) as usize;
    let upper_idx = (data_f32.len() as f32 * (1.0 - (i as f32 / 100.0))) as usize;

    let lower_bound = data_f32[indices[lower_idx]];
    let upper_bound = data_f32[indices[upper_idx]];

    // Now extend output with processed values
    output.extend(data_f32.iter().map(|&x| {
        if x < lower_bound || x > upper_bound {
            // Find neighbors and interpolate...
            // (For now just use the bound)
            if x < lower_bound {
                lower_bound
            } else {
                upper_bound
            }
        } else {
            x
        }
    }));
}

fn convert_to_f32(input: &[f32], output: &mut Vec<f32>) {
    output.extend_from_slice(input);
}

fn scale_data_into(data: &[f32], lower: f32, upper: f32, output: &mut Vec<f32>) {
    if data.is_empty() {
        return;
    }

    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    output.extend(
        data.iter()
            .map(|&x| (upper - lower) * ((x - min_val) / range) + lower),
    );
}

fn remove_baseline_wander_into(data: &[f32], sample_rate: f32, cutoff: f32, output: &mut Vec<f32>) {
    let filtered = remove_baseline_wander(data, sample_rate, cutoff);
    output.extend_from_slice(&filtered);
}

/// Compare two sensor analyses and return the one that appears more physiologically plausible
pub fn compare_sensor_analysis<'a>(
    sensor1: &'a PeriodAnalysis,
    sensor2: &'a PeriodAnalysis,
) -> Option<&'a PeriodAnalysis> {
    // Helper function to compute statistics for a sensor's heart rate data
    fn compute_stats(hrs: &[(DateTime<Utc>, f32)]) -> Option<(f32, f32)> {
        // returns (median, variance)
        if hrs.is_empty() {
            return None;
        }

        // Calculate median
        let mut values: Vec<f32> = hrs.iter().map(|(_, hr)| *hr).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = values[values.len() / 2];

        // Calculate variance
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        Some((median, variance))
    }

    // Compute statistics for both sensors
    let stats1 = compute_stats(&sensor1.fft_heart_rates);
    let stats2 = compute_stats(&sensor2.fft_heart_rates);

    match (stats1, stats2) {
        (Some((median1, variance1)), Some((median2, variance2))) => {
            // Score each sensor based on multiple factors
            let score1 = score_sensor_output(median1, variance1);
            let score2 = score_sensor_output(median2, variance2);

            // Return the sensor with the better score
            if score1 >= score2 {
                Some(sensor1)
            } else {
                Some(sensor2)
            }
        }
        (Some(_), None) => Some(sensor1),
        (None, Some(_)) => Some(sensor2),
        (None, None) => None,
    }
}

/// Score a sensor's output based on physiological plausibility and consistency
fn score_sensor_output(median: f32, variance: f32) -> f32 {
    let mut score = 0.0;

    // Factor 1: Median HR should be in a reasonable range for sleep
    // Ideal sleeping HR is typically between 50-70 BPM
    let median_score = if median >= 45.0 && median <= 85.0 {
        // Higher score for values closer to the ideal range
        let distance_from_ideal = if median < 50.0 {
            (median - 50.0).abs()
        } else if median > 70.0 {
            (median - 70.0).abs()
        } else {
            0.0
        };
        1.0 - (distance_from_ideal / 35.0).min(1.0) // Normalize to [0,1]
    } else {
        0.0 // Highly unlikely sleeping HR
    };

    // Factor 2: Lower variance is generally better (indicates more stable measurements)
    // Typical variance during sleep might be around 25-100
    let variance_score = 1.0 - (variance / 200.0).min(1.0);

    // Combine scores (currently weighted equally)
    score = median_score * 0.6 + variance_score * 0.4;

    score
}
