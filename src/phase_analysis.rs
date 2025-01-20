use crate::PeriodAnalysis;
use chrono::{DateTime, Utc};

#[derive(Debug, PartialEq, Clone)]
pub enum SleepPhase {
    Deep,
    Light,
    Awake,
    REM,
}

// Configuration for sleep phase detection
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    // Timing constraints
    pub min_phase_duration_minutes: i64,
    pub typical_cycle_duration_minutes: i64,

    // Heart rate thresholds
    pub baseline_hr_threshold: f32,    // Base threshold for HR variability
    pub deep_sleep_hr_max: f32,       // Maximum HR for deep sleep
    pub rem_hr_variability_mult: f32, // Multiplier for REM HR variability

    // Breathing rate thresholds
    pub baseline_br_threshold: f32,    // Base threshold for BR variability
    pub deep_sleep_br_max: f32,       // Maximum BR for deep sleep
    pub rem_br_variability_mult: f32,  // Multiplier for REM BR variability

    // Time-based adjustments
    pub early_night_factor: f32,      // Adjustment factor for early night
    pub late_night_factor: f32,       // Adjustment factor for late night
    pub early_night_hours: f32,       // Duration of "early night" in hours
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            min_phase_duration_minutes: 15,
            typical_cycle_duration_minutes: 90,

            baseline_hr_threshold: 3.0,
            deep_sleep_hr_max: 80.0,
            rem_hr_variability_mult: 1.5,

            baseline_br_threshold: 1.0,
            deep_sleep_br_max: 14.0,
            rem_br_variability_mult: 1.5,

            early_night_factor: 0.8,   // More lenient thresholds early
            late_night_factor: 1.2,    // Stricter thresholds late
            early_night_hours: 3.0,    // First 3 hours considered early
        }
    }
}

pub fn calculate_std_dev<I>(values: I) -> f32
where
    I: Iterator<Item = f32>,
{
    let mut count = 0;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for value in values {
        count += 1;
        sum += value;
        sum_sq += value * value;
    }

    if count < 2 {
        return 0.0;
    }

    let mean = sum / count as f32;
    let variance = (sum_sq / count as f32) - (mean * mean);
    variance.max(0.0).sqrt()
}

#[derive(Debug)]
struct PhaseMetrics {
    hr_mean: f32,
    hr_std_dev: f32,
    br_mean: f32,
    br_std_dev: f32,
    time_of_night: f32,     // Hours since sleep onset
    cycle_position: f32,    // Position in sleep cycle (0.0 to 1.0)
}

fn calculate_adaptive_thresholds(metrics: &PhaseMetrics, config: &PhaseConfig) -> (f32, f32) {
    // Base thresholds
    let mut hr_threshold = config.baseline_hr_threshold;
    let mut br_threshold = config.baseline_br_threshold;

    // Adjust for time of night
    let time_factor = if metrics.time_of_night < config.early_night_hours {
        config.early_night_factor
    } else {
        config.late_night_factor
    };

    // Adjust thresholds based on time factor
    hr_threshold *= time_factor;
    br_threshold *= time_factor;

    // Further adjust based on cycle position
    let cycle_factor = 1.0 + (metrics.cycle_position * 0.2); // Up to 20% variation based on cycle
    hr_threshold *= cycle_factor;
    br_threshold *= cycle_factor;

    (hr_threshold, br_threshold)
}

fn determine_sleep_phase(
    metrics: &PhaseMetrics,
    prev_phase: Option<&SleepPhase>,
    config: &PhaseConfig
) -> SleepPhase {
    let (hr_threshold, br_threshold) = calculate_adaptive_thresholds(metrics, config);

    // Calculate base phase based on current metrics
    let base_phase = if metrics.hr_std_dev < hr_threshold && metrics.br_std_dev < br_threshold {
        if metrics.hr_mean < config.deep_sleep_hr_max && metrics.br_mean < config.deep_sleep_br_max {
            SleepPhase::Deep
        } else {
            SleepPhase::Light
        }
    } else if metrics.hr_std_dev > hr_threshold * config.rem_hr_variability_mult
        || metrics.br_std_dev > br_threshold * config.rem_br_variability_mult {
        if metrics.time_of_night > config.early_night_hours {
            SleepPhase::REM
        } else {
            SleepPhase::Light
        }
    } else {
        SleepPhase::Light
    };

    // Apply hysteresis - resist phase changes
    if let Some(prev) = prev_phase {
        match (prev, &base_phase) {
            // Require stronger evidence to transition out of deep sleep
            (SleepPhase::Deep, _) if metrics.hr_std_dev < hr_threshold * 1.3 => SleepPhase::Deep,
            // Maintain REM if conditions are close
            (SleepPhase::REM, _) if metrics.hr_std_dev > hr_threshold * 0.8 => SleepPhase::REM,
            _ => base_phase,
        }
    } else {
        base_phase
    }
}

pub fn analyze_sleep_phases(
    analysis: &PeriodAnalysis,
    sleep_onset: DateTime<Utc>,
    config: Option<PhaseConfig>,
) -> Vec<(DateTime<Utc>, SleepPhase)> {
    let config = config.unwrap_or_default();
    let window_size = 15; // Increased window size (15 measurements)
    let mut phases = Vec::new();
    let mut current_phase: Option<SleepPhase> = None;
    let mut current_phase_start: Option<DateTime<Utc>> = None;
    let mut cycle_start = sleep_onset;

    if analysis.breathing_rates.is_empty() || analysis.fft_heart_rates.is_empty() {
        return phases;
    }

    let post_onset_breathing: Vec<_> = analysis
        .breathing_rates
        .iter()
        .filter(|(t, _)| *t >= sleep_onset)
        .collect();

    for window in post_onset_breathing.windows(window_size) {
        let time = window[window_size / 2].0; // Use middle of window as current time
        let hours_since_onset = (time - sleep_onset).num_seconds() as f32 / 3600.0;

        // Calculate cycle position (0.0 to 1.0)
        let minutes_in_cycle = (time - cycle_start).num_minutes() as f32;
        let cycle_position = (minutes_in_cycle / config.typical_cycle_duration_minutes as f32) % 1.0;

        // Start new cycle if needed
        if minutes_in_cycle >= config.typical_cycle_duration_minutes as f32 {
            cycle_start = time;
        }

        // Calculate breathing metrics
        let br_values: Vec<f32> = window.iter().map(|(_, rate)| *rate).collect();
        let br_mean = br_values.iter().sum::<f32>() / br_values.len() as f32;
        let br_std_dev = calculate_std_dev(br_values.iter().cloned());

        // Get corresponding heart rate window
        let hr_window: Vec<_> = analysis
            .fft_heart_rates
            .iter()
            .filter(|(t, _)| {
                *t >= window[0].0 && *t <= window[window.len() - 1].0
            })
            .collect();

        if hr_window.is_empty() {
            continue;
        }

        // Calculate heart rate metrics
        let hr_values: Vec<f32> = hr_window.iter().map(|(_, rate)| *rate).collect();
        let hr_mean = hr_values.iter().sum::<f32>() / hr_values.len() as f32;
        let hr_std_dev = calculate_std_dev(hr_values.iter().cloned());

        let metrics = PhaseMetrics {
            hr_mean,
            hr_std_dev,
            br_mean,
            br_std_dev,
            time_of_night: hours_since_onset,
            cycle_position,
        };

        let candidate_phase = determine_sleep_phase(&metrics, current_phase.as_ref(), &config);

        match (&current_phase, current_phase_start) {
            (Some(phase), Some(start)) if phase == &candidate_phase => {
                // Continue current phase
                continue;
            }
            (Some(phase), Some(start)) => {
                // Check if current phase has met minimum duration
                let phase_duration = time - start;
                if phase_duration.num_minutes() >= config.min_phase_duration_minutes {
                    // Record the completed phase
                    phases.push((start, phase.clone()));
                    current_phase = Some(candidate_phase);
                    current_phase_start = Some(time);
                }
                // Otherwise, maintain current phase
            }
            (None, None) => {
                // Start first phase
                current_phase = Some(candidate_phase);
                current_phase_start = Some(time);
            }
            _ => unreachable!(),
        }
    }

    // Add final phase if it exists
    if let (Some(phase), Some(start)) = (&current_phase, current_phase_start) {
        phases.push((start, phase.clone()));
    }

    phases
}
