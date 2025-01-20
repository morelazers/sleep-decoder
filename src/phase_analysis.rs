use super::PeriodAnalysis;
use chrono::{DateTime, Utc};

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum SleepPhase {
    Light,
    Deep,
    REM,
}

pub fn analyze_sleep_phases(
    analysis: &PeriodAnalysis,
    sleep_onset: DateTime<Utc>,
) -> Vec<(DateTime<Utc>, SleepPhase)> {
    let window_size = 5; // 5-minute windows
    let mut phases = Vec::new();

    if analysis.breathing_rates.is_empty() || analysis.fft_heart_rates.is_empty() {
        return phases;
    }

    // Constants for sleep cycle rules
    const MIN_LIGHT_DURATION_MINS: i64 = 15;
    const BASE_MIN_DEEP_DURATION_MINS: i64 = 20; // Base duration for early night
    const MIN_REM_DURATION_MINS: i64 = 10;
    const FIRST_REM_AFTER_MINS: i64 = 0; // First REM typically after 0 mins

    let post_onset_breathing: Vec<_> = analysis
        .breathing_rates
        .iter()
        .filter(|(t, _)| *t >= sleep_onset)
        .collect();

    let mut current_phase = SleepPhase::Light; // Always start with Light sleep
    let mut phase_start_time = sleep_onset;
    let mut last_phase_change = sleep_onset;

    // Analyze sliding windows
    for window in post_onset_breathing.windows(window_size) {
        let time = window[0].0;
        let mins_since_onset = (time - sleep_onset).num_minutes();
        let mins_in_current_phase = (time - phase_start_time).num_minutes();

        // Calculate variances (same as before)
        let br_variance = calculate_variance(window.iter().map(|(_, rate)| *rate));
        let hr_window: Vec<_> = analysis
            .fft_heart_rates
            .iter()
            .filter(|(t, _)| *t >= window[0].0 && *t <= window[window.len() - 1].0)
            .collect();
        let hr_variance = calculate_variance(hr_window.iter().map(|(_, rate)| *rate));

        // Get time-based weight for deep sleep
        let time_weight = calculate_time_weight(sleep_onset, time);
        // Determine candidate next phase based on measurements
        // Adjust deep sleep threshold based on time weight
        let deep_sleep_threshold = 1.5 * (time_weight); // Increased base threshold from 1.0 to 1.5
        let candidate_phase = match (br_variance, hr_variance) {
            (br_var, hr_var) if br_var > 3.0 || hr_var > 20.0 => SleepPhase::REM,
            (br_var, hr_var) if br_var < deep_sleep_threshold &&
                hr_var < deep_sleep_threshold * 8.0 && // Scale for HR variance
                hr_var < hr_window.iter().map(|(_, rate)| rate).sum::<f32>() / hr_window.len() as f32 * 0.25 // Increased from 0.15 to 0.25
                => SleepPhase::Deep,
            _ => SleepPhase::Light,
        };

        // Apply sleep cycle rules
        let next_phase = match (current_phase, candidate_phase) {
            // Can't enter REM until after 0 minutes
            (_, SleepPhase::REM) if mins_since_onset < FIRST_REM_AFTER_MINS => current_phase,

            // Enforce minimum durations, with adaptive deep sleep minimum
            (SleepPhase::Light, _) if mins_in_current_phase < MIN_LIGHT_DURATION_MINS => {
                SleepPhase::Light
            }
            (SleepPhase::Deep, candidate) => {
                let time_weight = calculate_time_weight(sleep_onset, time);
                let adaptive_min_deep = (BASE_MIN_DEEP_DURATION_MINS as f32 * time_weight) as i64;
                if mins_in_current_phase < adaptive_min_deep {
                    SleepPhase::Deep
                } else if candidate == SleepPhase::REM {
                    // Can only enter REM from Light sleep
                    SleepPhase::Light
                } else {
                    candidate
                }
            }
            (SleepPhase::REM, _) if mins_in_current_phase < MIN_REM_DURATION_MINS => {
                SleepPhase::REM
            }

            // Allow the transition if all rules pass
            _ => candidate_phase,
        };

        if next_phase != current_phase {
            phase_start_time = time;
            last_phase_change = time;
        }
        current_phase = next_phase;
        phases.push((time, current_phase));
    }

    phases
}

/// Add a time-based weight to sleep phase classification
fn calculate_time_weight(start_time: DateTime<Utc>, current_time: DateTime<Utc>) -> f32 {
    let hours_elapsed = (current_time - start_time).num_minutes() as f32 / 60.0;

    // Weight starts at 1.0 and decreases to 0.2 over 8 hours
    // Using sigmoid function with center at 3 hours instead of 2
    let base_weight = 1.0 / (1.0 + (hours_elapsed - 3.0).exp());

    // Scale to range 0.2 - 1.0 so deep sleep is still possible later
    0.2 + (0.8 * base_weight)
}

fn calculate_variance<I>(iter: I) -> f32
where
    I: Iterator<Item = f32> + Clone,
{
    let count = iter.clone().count();
    if count == 0 {
        return 0.0;
    }
    let mean = iter.clone().sum::<f32>() / count as f32;
    iter.map(|x| {
        let diff = x - mean;
        diff * diff
    })
    .sum::<f32>()
        / count as f32
}

pub fn summarize_sleep_phases(phases: &[(DateTime<Utc>, SleepPhase)]) {
    if phases.is_empty() {
        println!("No sleep phases to summarize");
        return;
    }

    let mut current_phase = phases[0].1;
    let mut phase_start = phases[0].0;
    let mut total_minutes = std::collections::HashMap::new();

    for &(time, phase) in &phases[1..] {
        if phase != current_phase {
            let duration = time - phase_start;
            let minutes = duration.num_minutes();

            *total_minutes.entry(current_phase).or_insert(0) += minutes;

            current_phase = phase;
            phase_start = time;
        }
    }

    // Print last phase
    let duration = phases.last().unwrap().0 - phase_start;
    let minutes = duration.num_minutes();
    *total_minutes.entry(current_phase).or_insert(0) += minutes;

    // Print totals
    println!("\nTotal Time in Each Phase:");
    println!("------------------------");
    for (phase, &minutes) in total_minutes.iter() {
        let hours = minutes / 60;
        let remaining_minutes = minutes % 60;
        println!("{:?}: {}h {}min", phase, hours, remaining_minutes);
    }
}