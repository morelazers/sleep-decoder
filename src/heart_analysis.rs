use log::{debug, trace};
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
///
/// Parameters:
/// - w0: Normalized frequency to remove from signal. If fs is specified, this is in the same units as fs.
///       By default, it is a normalized scalar that must satisfy 0 < w0 < 1, with w0 = 1 corresponding
///       to half of the sampling frequency.
/// - Q: Quality factor. Dimensionless parameter that characterizes notch filter -3 dB bandwidth bw
///      relative to its center frequency, Q = w0/bw.
/// - ftype: The type of IIR filter to design ("notch" or "peak")
/// - fs: The sampling frequency of the digital system (default: 2.0)
fn design_notch_peak_filter(mut w0: f32, Q: f32, ftype: &str, fs: f32) -> (Vec<f32>, Vec<f32>) {
    // Validate fs
    assert!(fs > 0.0, "fs must be positive");

    // Guarantee inputs are floats (already handled by Rust's type system)
    w0 = 2.0 * w0 / fs;

    // Check if w0 is within range
    assert!(w0 > 0.0 && w0 < 1.0, "w0 should be such that 0 < w0 < 1");

    // Get bandwidth
    let mut bw = w0 / Q;

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
    let q = 0.005; // Fixed Q factor to match HeartPy
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
        [1.0, sos[0][4], sos[0][5]],
    )];

    sosfiltfilt_dyn(data.iter(), &sos_array)
}

/// Represents the working data during heart rate analysis
#[derive(Debug)]
pub struct WorkingData {
    pub hr: Vec<f32>,
    pub sample_rate: f32,
    pub rolling_mean: Vec<f32>,
    pub peaklist: Vec<usize>,
    pub ybeat: Vec<f32>,
    pub rr_list: Vec<f32>,
    pub rr_indices: Vec<usize>,
    pub rr_diff: Vec<f32>,
    pub rr_sqdiff: Vec<f32>,
    pub binary_peaklist: Vec<usize>,
    pub removed_beats: Vec<usize>,
    pub removed_beats_y: Vec<f32>,
    pub rejected_segments: Vec<usize>,
    pub rr_masklist: Vec<usize>,
    pub rr_list_cor: Vec<f32>,
}

/// Represents the computed heart rate measures
#[derive(Debug)]
pub struct HeartMeasures {
    pub bpm: f32,
    pub ibi: f32,
    pub sdnn: f32,
    pub sdsd: f32,
    pub rmssd: f32,
    pub pnn20: f32,
    pub pnn50: f32,
    pub breathing_rate: Option<f32>,
    pub confidence: f32,
}

/// Calculate rolling mean of the signal, matching HeartPy's implementation
fn rolling_mean(data: &[f32], window_size: f32, sample_rate: f32) -> Vec<f32> {
    let size = (window_size * sample_rate) as usize;
    debug!("Rolling mean window size: {} samples", size);
    let mut result = vec![0.0; data.len()];

    // Implement uniform_filter1d like scipy
    for i in 0..data.len() {
        let start = if i < size / 2 { 0 } else { i - size / 2 };
        let end = ((i + size / 2 + 1).min(data.len()));
        let window_size = end - start;

        // Simple mean over the window (uniform filter)
        result[i] = data[start..end].iter().sum::<f32>() / window_size as f32;
    }

    // Debug first few rolling mean values
    if result.len() >= 10 {
        debug!("First 10 rolling mean values: {:?}", &result[..10]);
    }

    result
}

/// Calculate adaptive threshold for peak detection
fn calculate_adaptive_threshold(
    data: &[f32],
    rolling_mean: &[f32],
    window_size: usize,
) -> Vec<f32> {
    let mut thresholds = vec![0.0; data.len()];
    let half_window = window_size / 2;

    for i in 0..data.len() {
        // Get local window indices
        let start = if i > half_window { i - half_window } else { 0 };
        let end = (i + half_window + 1).min(data.len());

        // Calculate local statistics
        let window_data = &data[start..end];
        let local_mean = window_data.iter().sum::<f32>() / window_data.len() as f32;
        let local_std = (window_data
            .iter()
            .map(|&x| (x - local_mean).powi(2))
            .sum::<f32>()
            / window_data.len() as f32)
            .sqrt();

        // Adaptive threshold based on local signal characteristics
        let base_threshold = rolling_mean[i];
        let dynamic_component = local_std * 1.5; // Adjust based on local variance

        // Use higher threshold if signal is noisy
        let noise_factor = if local_std > local_mean * 0.1 {
            1.2
        } else {
            1.0
        };

        thresholds[i] = (base_threshold + dynamic_component) * noise_factor;
    }

    thresholds
}

/// Detect peaks using moving average, matching HeartPy's implementation exactly
fn detect_peaks(data: &[f32], rol_mean: &[f32], ma_perc: f32) -> (Vec<usize>, Vec<f32>) {
    // Calculate threshold exactly like HeartPy:
    // mn = np.mean(rmean / 100) * ma_perc
    // rol_mean = rmean + mn
    let mn = rol_mean.iter().map(|&x| x / 100.0).sum::<f32>() / rol_mean.len() as f32 * ma_perc;
    let rol_mean: Vec<f32> = rol_mean.iter().map(|&r| r + mn).collect();

    // Log input data ranges
    let data_min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let data_max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    // Log adjusted rolling mean range
    let rol_mean_min = rol_mean.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let rol_mean_max = rol_mean.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    debug!(
        "data range - data range: [{:.1}, {:.1}] - rolling mean range: [{:.1}, {:.1}] (mn adjustment: {:.1})",
        data_min, data_max, rol_mean_min, rol_mean_max, mn
    );

    // Find all points above rol_mean
    let mut peaksx = Vec::new();
    let mut peaksy = Vec::new();

    // First collect all points above rol_mean
    for (i, (&d, &t)) in data.iter().zip(rol_mean.iter()).enumerate() {
        if d > t {
            peaksx.push(i);
            peaksy.push(d);
        }
    }

    // If no points above rol_mean, return empty results
    if peaksx.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Find edges of consecutive sequences (where diff > 1)
    let mut peakedges = vec![0];
    for i in 1..peaksx.len() {
        if peaksx[i] - peaksx[i - 1] > 1 {
            peakedges.push(i);
        }
    }
    peakedges.push(peaksx.len());

    // Find maximum in each group of consecutive points
    let mut final_peaksx = Vec::new();
    let mut final_peaksy = Vec::new();

    for i in 0..peakedges.len() - 1 {
        let start = peakedges[i];
        let end = peakedges[i + 1];

        // Find maximum in this group
        if let Some(max_idx) =
            (start..end).max_by(|&a, &b| peaksy[a].partial_cmp(&peaksy[b]).unwrap())
        {
            final_peaksx.push(peaksx[max_idx]);
            final_peaksy.push(peaksy[max_idx]);
        }
    }

    (final_peaksx, final_peaksy)
}

/// Fit peaks to the signal using moving average, following HeartPy's implementation
fn fit_peaks(
    data: &[f32],
    rolling_mean: &[f32],
    sample_rate: f32,
    bpm_min: f32,
    bpm_max: f32,
) -> (Vec<usize>, Vec<f32>) {
    // Use the same moving average percentages as HeartPy
    let ma_percs = vec![
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        150.0, 200.0, 300.0,
    ];

    let mut rrsd_results = Vec::new();

    for ma_perc in ma_percs {
        // Detect peaks with current moving average percentage
        let (peaklist, ybeat) = detect_peaks(data, rolling_mean, ma_perc);

        // Calculate BPM exactly like HeartPy
        let bpm = (peaklist.len() as f32 / (data.len() as f32 / sample_rate)) * 60.0;

        debug!(
            "ma_perc: {}, peaks: {}, bpm: {:.1}",
            ma_perc,
            peaklist.len(),
            bpm
        );

        // Calculate RR intervals and RRSD if we have peaks
        let rrsd = if !peaklist.is_empty() {
            let rr_intervals: Vec<f32> = peaklist
                .windows(2)
                .map(|w| (w[1] - w[0]) as f32 / sample_rate * 1000.0)
                .collect();

            if !rr_intervals.is_empty() {
                let mean_rr = rr_intervals.iter().sum::<f32>() / rr_intervals.len() as f32;
                let rrsd = (rr_intervals
                    .iter()
                    .map(|&rr| (rr - mean_rr).powi(2))
                    .sum::<f32>()
                    / (rr_intervals.len() - 1) as f32)
                    .sqrt();
                debug!("  mean_rr: {:.1}ms, rrsd: {:.1}", mean_rr, rrsd);
                rrsd
            } else {
                f32::INFINITY
            }
        } else {
            f32::INFINITY
        };

        rrsd_results.push((rrsd, bpm, ma_perc, peaklist, ybeat));
    }

    // Filter valid results using HeartPy's criteria
    let valid_results: Vec<_> = rrsd_results
        .into_iter()
        .filter(|(rrsd, bpm, _, _, _)| *rrsd > 0.1 && *bpm >= bpm_min && *bpm <= bpm_max)
        .collect();

    debug!("Found {} valid results", valid_results.len());

    // Find the result with minimum RRSD, just like HeartPy
    if let Some((rrsd, bpm, ma_perc, best_peaklist, best_ybeat)) = valid_results
        .into_iter()
        .min_by(|(rrsd1, _, _, _, _), (rrsd2, _, _, _, _)| rrsd1.partial_cmp(rrsd2).unwrap())
    {
        debug!(
            "Selected result - rrsd: {:.1}, bpm: {:.1}, ma_perc: {}",
            rrsd, bpm, ma_perc
        );
        (best_peaklist, best_ybeat)
    } else {
        debug!("No valid peaks found with any ma_perc");
        (Vec::new(), Vec::new())
    }
}

/// Calculate RR intervals and indices, matching HeartPy's implementation exactly
fn calc_rr(peaklist: &[usize], sample_rate: f32) -> (Vec<f32>, Vec<usize>) {
    if peaklist.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let rr_list: Vec<f32> = peaklist
        .windows(2)
        .map(|window| (window[1] - window[0]) as f32 * 1000.0 / sample_rate)
        .collect();

    let rr_indices: Vec<usize> = peaklist[1..].to_vec();

    (rr_list, rr_indices)
}

/// Calculate time-series measures
fn calc_ts_measures(rr_list: &[f32], rr_diff: &[f32], rr_sqdiff: &[f32]) -> HeartMeasures {
    // Return zeros if we don't have enough data
    if rr_list.is_empty() {
        return HeartMeasures {
            bpm: 0.0,
            ibi: 0.0,
            sdnn: 0.0,
            sdsd: 0.0,
            rmssd: 0.0,
            pnn20: 0.0,
            pnn50: 0.0,
            breathing_rate: None,
            confidence: 0.0,
        };
    }

    let mean_rr = rr_list.iter().sum::<f32>() / rr_list.len() as f32;
    let sdnn = if rr_list.len() > 1 {
        (rr_list
            .iter()
            .map(|&x| {
                let diff = x - mean_rr;
                diff * diff
            })
            .sum::<f32>()
            / (rr_list.len() - 1) as f32)
            .sqrt()
    } else {
        0.0
    };

    let rmssd = if !rr_sqdiff.is_empty() {
        (rr_sqdiff.iter().sum::<f32>() / rr_sqdiff.len() as f32).sqrt()
    } else {
        0.0
    };

    let sdsd = if rr_diff.len() > 1 {
        let mean_diff = rr_diff.iter().sum::<f32>() / rr_diff.len() as f32;
        (rr_diff
            .iter()
            .map(|&x| {
                let diff = x - mean_diff;
                diff * diff
            })
            .sum::<f32>()
            / (rr_diff.len() - 1) as f32)
            .sqrt()
    } else {
        0.0
    };

    let nn20 = rr_diff.iter().filter(|&&x| x > 20.0).count();
    let nn50 = rr_diff.iter().filter(|&&x| x > 50.0).count();

    let (pnn20, pnn50) = if !rr_diff.is_empty() {
        (
            (nn20 as f32 / rr_diff.len() as f32) * 100.0,
            (nn50 as f32 / rr_diff.len() as f32) * 100.0,
        )
    } else {
        (0.0, 0.0)
    };

    HeartMeasures {
        bpm: if mean_rr > 0.0 {
            60_000.0 / mean_rr
        } else {
            0.0
        },
        ibi: mean_rr,
        sdnn,
        sdsd,
        rmssd,
        pnn20,
        pnn50,
        breathing_rate: None,
        confidence: 0.0,
    }
}

/// Check peaks for anomalies and update working data with filtered peaks.
/// Returns a confidence score (0.0 to 1.0) based on the quality of the peaks.
fn check_peaks(working_data: &mut WorkingData) -> f32 {
    // If we don't have enough peaks, return 0 confidence
    if working_data.peaklist.len() < 2 {
        debug!("check_peaks: Not enough peaks found (<2)");
        return 0.0;
    }

    // Calculate mean RR interval
    let mean_rr = working_data.rr_list.iter().sum::<f32>() / working_data.rr_list.len() as f32;

    // Define thresholds: mean ± 30%, with a minimum of 300ms
    let thirty_percent = 0.3 * mean_rr;
    let (lower_threshold, upper_threshold) = if thirty_percent <= 300.0 {
        (mean_rr - 300.0, mean_rr + 300.0)
    } else {
        (mean_rr - thirty_percent, mean_rr + thirty_percent)
    };

    debug!(
        "check_peaks: RR interval thresholds - lower: {:.0}ms, upper: {:.0}ms",
        lower_threshold, upper_threshold
    );

    // Create binary mask for peaks (1 for valid, 0 for invalid)
    let mut binary_peaklist = vec![1usize; working_data.peaklist.len()];
    let mut removed_beats = Vec::new();
    let mut removed_beats_y = Vec::new();

    // Mark invalid peaks based on RR intervals
    for (i, rr) in working_data.rr_list.iter().enumerate() {
        if *rr <= lower_threshold || *rr >= upper_threshold {
            binary_peaklist[i + 1] = 0; // Mark the second peak of the interval as invalid
            removed_beats.push(working_data.peaklist[i + 1]);
            removed_beats_y.push(working_data.ybeat[i + 1]);
        }
    }

    // Update working data with removed peaks information
    working_data.binary_peaklist = binary_peaklist;
    working_data.removed_beats = removed_beats;
    working_data.removed_beats_y = removed_beats_y;

    // Update RR intervals and differences based on binary peaklist
    update_rr(working_data);

    // Calculate percentage of valid intervals using the corrected RR list
    let valid_intervals = working_data.rr_list_cor.len();
    let total_intervals = working_data.rr_list.len();
    let valid_interval_ratio = valid_intervals as f32 / total_intervals as f32;

    debug!(
        "check_peaks: {}/{} intervals within physiological range (300-2000ms)",
        valid_intervals, total_intervals
    );

    // Sort intervals for further analysis
    let mut sorted_intervals = working_data.rr_list_cor.clone();
    sorted_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Get max and min intervals
    // Handle the case where the interval list is empty
    let max_ibi = *sorted_intervals.last().unwrap_or(&0.0);
    let min_ibi = sorted_intervals.first().unwrap_or(&0.0);
    debug!(
        "check_peaks: RR interval range: {:.0}ms - {:.0}ms",
        min_ibi, max_ibi
    );

    // Calculate measures of spread exactly like HeartPy
    let mean_ibi = sorted_intervals.iter().sum::<f32>() / sorted_intervals.len() as f32;
    let max_deviation = (max_ibi - mean_ibi).max(mean_ibi - min_ibi);
    let spread = max_deviation / mean_ibi;

    debug!(
        "check_peaks: Mean IBI: {:.0}ms, Max deviation: {:.0}ms, Spread: {:.2}",
        mean_ibi, max_deviation, spread
    );

    // Calculate spread confidence (1.0 if spread <= 0.5, decreasing linearly to 0.0 at spread = 1.0)
    let spread_confidence = if spread <= 0.5 {
        1.0
    } else if spread >= 1.0 {
        0.0
    } else {
        (1.0 - spread) / 0.5
    };

    // Combine confidences with weights
    // We weight valid intervals more heavily (0.7) than spread (0.3)
    let confidence = 0.7 * valid_interval_ratio + 0.3 * spread_confidence;

    debug!(
        "check_peaks: Confidence score: {:.2} (valid_ratio: {:.2}, spread_confidence: {:.2})",
        confidence, valid_interval_ratio, spread_confidence
    );

    confidence
}

/// Main processing function, similar to HeartPy's process()
pub fn process(signal: &[f32], sample_rate: f32, window_size: f32) -> (WorkingData, HeartMeasures) {
    let mut working_data = WorkingData {
        hr: signal.to_vec(),
        sample_rate,
        rolling_mean: rolling_mean(signal, window_size, sample_rate),
        peaklist: Vec::new(),
        ybeat: Vec::new(),
        rr_list: Vec::new(),
        rr_indices: Vec::new(),
        rr_diff: Vec::new(),
        rr_sqdiff: Vec::new(),
        binary_peaklist: Vec::new(),
        removed_beats: Vec::new(),
        removed_beats_y: Vec::new(),
        rejected_segments: Vec::new(),
        rr_masklist: Vec::new(),
        rr_list_cor: Vec::new(),
    };

    // Find peaks in the filtered signal
    let (peaklist, ybeat) = fit_peaks(signal, &working_data.rolling_mean, sample_rate, 40.0, 180.0);

    // Update working data with initial peak detections
    working_data.peaklist = peaklist;
    working_data.ybeat = ybeat;

    // Calculate RR intervals and indices first
    let (rr_list, rr_indices) = calc_rr(&working_data.peaklist, sample_rate);
    working_data.rr_list = rr_list;
    working_data.rr_indices = rr_indices;

    // Check peaks and get confidence score
    let confidence = check_peaks(&mut working_data);

    // Calculate time-series measures using the corrected RR list
    let mut measures = calc_ts_measures(
        &working_data.rr_list_cor,
        &working_data.rr_diff,
        &working_data.rr_sqdiff,
    );

    // Set confidence
    measures.confidence = confidence;

    // Calculate breathing rate using Welch's method
    measures.breathing_rate = calc_breathing(&working_data.rr_list_cor, "welch");

    (working_data, measures)
}

/// Replace outliers with linear interpolation from neighboring points
pub fn clamp_outliers(data: &[i32], i: i32) -> Vec<f32> {
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

/// Represents the results of segmentwise processing
#[derive(Debug)]
pub struct SegmentResults {
    pub timestamp: f32,      // Timestamp in seconds from start of recording
    pub segment_size: usize, // Size of the segment
    pub working_data: WorkingData,
    pub measures: HeartMeasures,
    pub quality: f32, // Signal quality metric (0-1)
    pub valid: bool,  // Whether the segment is considered valid
}

/// Combine two sensor signals by simple averaging
pub fn combine_signals(signal1: &[i32], signal2: &[i32]) -> Vec<i32> {
    signal1
        .iter()
        .zip(signal2.iter())
        .map(|(&s1, &s2)| ((s1 as i64 + s2 as i64) / 2) as i32)
        .collect()
}

/// Determine if a segment is valid based on quality metrics and measures
pub fn is_valid_segment(
    working_data: &WorkingData,
    measures: &HeartMeasures,
    quality: f32,
) -> bool {
    // Check if we have enough peaks
    if working_data.peaklist.len() < 4 {
        return false;
    }

    // Check if heart rate is in reasonable range (30-200 BPM)
    if measures.bpm < 30.0 || measures.bpm > 200.0 {
        return false;
    }

    // Check if quality is above threshold
    if quality < 0.5 {
        return false;
    }

    // Check if RR intervals are reasonable
    if measures.sdnn > 300.0 || measures.rmssd > 300.0 {
        return false;
    }

    true
}

/// Calculate breathing rate from RR intervals using frequency analysis
pub fn calc_breathing(rr_list: &[f32], method: &str) -> Option<f32> {
    if rr_list.len() < 4 {
        return None;
    }

    // Interpolate RR intervals to get evenly spaced samples
    let sample_rate = 4.0; // 4 Hz interpolation rate
    let duration = rr_list.len() as f32 / 1000.0 * rr_list.iter().sum::<f32>(); // Total duration in seconds
    let num_samples = (duration * sample_rate) as usize;

    // Create time points for original data (cumulative sum of RR intervals)
    let mut time_points = Vec::with_capacity(rr_list.len());
    let mut t = 0.0;
    for &rr in rr_list {
        time_points.push(t);
        t += rr / 1000.0; // Convert ms to seconds
    }

    // Create evenly spaced time points for interpolation
    let interp_times: Vec<f32> = (0..num_samples).map(|i| i as f32 / sample_rate).collect();

    // Linear interpolation of RR intervals
    let mut interpolated_rr = Vec::with_capacity(num_samples);
    let mut j = 0;
    for &t in &interp_times {
        while j < time_points.len() - 1 && time_points[j + 1] < t {
            j += 1;
        }
        if j >= time_points.len() - 1 {
            break;
        }
        let t1 = time_points[j];
        let t2 = time_points[j + 1];
        let rr1 = rr_list[j];
        let rr2 = rr_list[j + 1];
        let alpha = (t - t1) / (t2 - t1);
        interpolated_rr.push(rr1 + alpha * (rr2 - rr1));
    }

    // Remove mean
    let mean = interpolated_rr.iter().sum::<f32>() / interpolated_rr.len() as f32;
    for x in &mut interpolated_rr {
        *x -= mean;
    }

    match method {
        "welch" => calc_breathing_welch(&interpolated_rr, sample_rate),
        "fft" => calc_breathing_fft(&interpolated_rr, sample_rate),
        _ => None,
    }
}

/// Calculate breathing rate using Welch's method
fn calc_breathing_welch(signal: &[f32], sample_rate: f32) -> Option<f32> {
    let segment_length = (signal.len() as f32 / 4.0).floor() as usize;
    if segment_length < 4 {
        return None;
    }

    let overlap = segment_length / 2;
    let num_segments = (signal.len() - segment_length) / (segment_length - overlap) + 1;

    let mut psd = vec![0.0; segment_length / 2 + 1];
    let window = create_hann_window(segment_length);

    for i in 0..num_segments {
        let start = i * (segment_length - overlap);
        let segment = &signal[start..start + segment_length];

        // Apply window and calculate FFT
        let mut windowed = vec![Complex::new(0.0, 0.0); segment_length];
        for (j, &x) in segment.iter().enumerate() {
            windowed[j] = Complex::new(x * window[j], 0.0);
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(segment_length);
        fft.process(&mut windowed);

        // Accumulate power
        for j in 0..=segment_length / 2 {
            psd[j] += windowed[j].norm_sqr();
        }
    }

    // Average and normalize
    for x in &mut psd {
        *x /= num_segments as f32;
    }

    // Find peak in breathing frequency range (0.1-0.4 Hz = 6-24 breaths/min)
    let freq_resolution = sample_rate / segment_length as f32;
    let min_idx = (0.1 / freq_resolution).ceil() as usize;
    let max_idx = (0.4 / freq_resolution).floor() as usize;

    let mut max_power = 0.0;
    let mut peak_freq = 0.0;
    for i in min_idx..=max_idx {
        if psd[i] > max_power {
            max_power = psd[i];
            peak_freq = i as f32 * freq_resolution;
        }
    }

    if peak_freq > 0.0 {
        Some(peak_freq * 60.0) // Convert Hz to breaths/min
    } else {
        None
    }
}

/// Calculate breathing rate using FFT
fn calc_breathing_fft(signal: &[f32], sample_rate: f32) -> Option<f32> {
    if signal.len() < 4 {
        return None;
    }

    // Pad signal to power of 2
    let n = signal.len().next_power_of_two();
    let mut padded = vec![Complex::new(0.0, 0.0); n];
    for (i, &x) in signal.iter().enumerate() {
        padded[i] = Complex::new(x, 0.0);
    }

    // Calculate FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut padded);

    // Calculate power spectrum
    let mut psd = vec![0.0; n / 2 + 1];
    for i in 0..=n / 2 {
        psd[i] = padded[i].norm_sqr();
    }

    // Find peak in breathing frequency range (0.1-0.4 Hz = 6-24 breaths/min)
    let freq_resolution = sample_rate / n as f32;
    let min_idx = (0.1 / freq_resolution).ceil() as usize;
    let max_idx = (0.4 / freq_resolution).floor() as usize;

    let mut max_power = 0.0;
    let mut peak_freq = 0.0;
    for i in min_idx..=max_idx {
        if psd[i] > max_power {
            max_power = psd[i];
            peak_freq = i as f32 * freq_resolution;
        }
    }

    if peak_freq > 0.0 {
        Some(peak_freq * 60.0) // Convert Hz to breaths/min
    } else {
        None
    }
}

/// Scale data to specified range, matching HeartPy's scale_data function exactly
pub fn scale_data(data: &[f32], lower: f32, upper: f32) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    // Note: Python's numpy will handle division by zero by returning inf/nan
    // We'll do the same by not checking for zero range
    data.iter()
        .map(|&x| (upper - lower) * ((x - min_val) / range) + lower)
        .collect()
}

// Add a default version that uses HeartPy's default parameters
pub fn scale_data_default(data: &[f32]) -> Vec<f32> {
    scale_data(data, 0.0, 1024.0)
}

/// Convert transfer function coefficients to second-order sections
fn tf2sos(b: &[f32], a: &[f32]) -> Vec<[f32; 6]> {
    // For a second-order filter (like our notch filter), we just need one section
    // Each section is [b0, b1, b2, 1, a1, a2]
    vec![[b[0], b[1], b[2], 1.0, a[1] / a[0], a[2] / a[0]]]
}

/// Updates RR intervals and differences based on binary peaklist
/// Similar to HeartPy's update_rr function
fn update_rr(working_data: &mut WorkingData) {
    let rr_source = &working_data.rr_list;
    let b_peaklist = &working_data.binary_peaklist;

    // Create corrected RR list only from valid peaks (where both peaks are marked as 1)
    let rr_list_cor: Vec<f32> = rr_source
        .iter()
        .enumerate()
        .filter(|&(i, _)| b_peaklist[i] + b_peaklist[i + 1] == 2)
        .map(|(_, &rr)| rr)
        .collect();

    // Create mask where 0 indicates valid RR intervals and 1 indicates masked intervals
    let rr_masklist: Vec<usize> = rr_source
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if b_peaklist[i] + b_peaklist[i + 1] == 2 {
                0
            } else {
                1
            }
        })
        .collect();

    // Calculate differences between adjacent RR intervals (only for valid intervals)
    let mut rr_diff = Vec::new();
    let mut prev_valid_rr: Option<f32> = None;

    // Simulate NumPy's masked array behavior for calculating differences
    for (i, &rr) in rr_source.iter().enumerate() {
        if rr_masklist[i] == 0 {
            // If current interval is valid
            if let Some(prev_rr) = prev_valid_rr {
                rr_diff.push((rr - prev_rr).abs());
            }
            prev_valid_rr = Some(rr);
        }
    }

    // Calculate squared differences
    let rr_sqdiff: Vec<f32> = rr_diff.iter().map(|&x| x * x).collect();

    // Update working data
    working_data.rr_masklist = rr_masklist;
    working_data.rr_list_cor = rr_list_cor;
    working_data.rr_diff = rr_diff;
    working_data.rr_sqdiff = rr_sqdiff;
}

/// Analyze heart rate using FFT
pub fn analyze_heart_rate_fft(signal: &[f32], sample_rate: f32) -> Option<f32> {
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

    // Look at magnitude spectrum in the heart rate range (40-80 BPM = 0.67-1.33 Hz)
    let min_bin = (0.67 / freq_resolution) as usize;
    let max_bin = (1.33 / freq_resolution) as usize;

    // Find peak in heart rate range
    let mut max_magnitude = 0.0;
    let mut peak_freq = 0.0;

    for bin in min_bin..=max_bin {
        let magnitude = (buffer[bin].norm_sqr() as f32).sqrt();
        if magnitude > max_magnitude {
            max_magnitude = magnitude;
            peak_freq = bin as f32 * freq_resolution;
        }
    }

    // Convert peak frequency to BPM
    let bpm = peak_freq * 60.0;

    // Basic validation
    if bpm >= 40.0 && bpm <= 180.0 {
        Some(bpm)
    } else {
        None
    }
}

/// Analyze breathing rate using FFT on the scaled signal
pub fn analyze_breathing_rate_fft(signal: &[f32], sample_rate: f32) -> Option<f32> {
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

    // Look at magnitude spectrum in the breathing rate range (6-24 BPM = 0.1-0.4 Hz)
    let min_bin = (0.1 / freq_resolution) as usize;
    let max_bin = (0.4 / freq_resolution) as usize;

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
    if breaths_per_minute >= 6.0 && breaths_per_minute <= 24.0 {
        Some(breaths_per_minute)
    } else {
        None
    }
}
