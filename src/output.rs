use crate::{heart_analysis, phase_analysis::SleepPhase, PeriodAnalysis};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::path::Path;

pub fn write_analysis_to_csv(
    base_path: &str,
    sensor_id: &str,
    period_num: usize,
    analysis: &PeriodAnalysis,
    hr_smoothing_window: usize,
    hr_smoothing_strength: f32,
    sleep_phases: &[(DateTime<Utc>, SleepPhase)],
) -> Result<()> {
    let path = Path::new(base_path);
    let dir = path.parent().unwrap_or(Path::new("."));

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

    // Write header
    writer.write_record(&[
        "timestamp",
        "fft_hr_smoothed",
        "breathing_rate",
        "sleep_phase",
    ])?;

    // First smooth the heart rate data using its own timeline
    let hr_timestamps: Vec<DateTime<Utc>> =
        analysis.fft_heart_rates.iter().map(|(t, _)| *t).collect();
    let smoothed_hr = heart_analysis::interpolate_and_smooth(
        &hr_timestamps,
        &analysis.fft_heart_rates,
        hr_smoothing_window,
        hr_smoothing_strength,
    );

    // Use breathing rate timestamps as the primary timeline
    let br_timestamps: Vec<_> = analysis.breathing_rates.iter().map(|(t, _)| t).collect();

    // For each breathing rate timestamp, find or interpolate the corresponding heart rate
    for (br_timestamp, br) in &analysis.breathing_rates {
        // Find the closest heart rate values before and after this timestamp
        let hr = if let Some((_t, hr)) = smoothed_hr.iter().find(|(t, _)| t == br_timestamp) {
            // Exact match found
            Some(*hr)
        } else {
            // Find surrounding points for linear interpolation
            let before = smoothed_hr.iter().rev().find(|(t, _)| t < br_timestamp);
            let after = smoothed_hr.iter().find(|(t, _)| t > br_timestamp);

            match (before, after) {
                (Some((t1, v1)), Some((t2, v2))) => {
                    // Linear interpolation
                    let total_duration = (*t2 - *t1).num_seconds() as f32;
                    let first_duration = (*br_timestamp - *t1).num_seconds() as f32;
                    let weight = first_duration / total_duration;
                    Some(v1 + (v2 - v1) * weight)
                }
                (Some((_t, v)), None) => Some(*v), // Use last value
                (None, Some((_t, v))) => Some(*v), // Use first value
                (None, None) => None,              // No heart rate data available
            }
        };

        let phase = sleep_phases
            .iter()
            .find(|(t, _)| t == br_timestamp)
            .map(|(_, phase)| phase);

        writer.write_record(&[
            br_timestamp.format("%Y-%m-%d %H:%M").to_string(),
            hr.map(|v| v.to_string()).unwrap_or_default(),
            br.to_string(),
            phase.map(|p| format!("{:?}", p)).unwrap_or_default(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}
