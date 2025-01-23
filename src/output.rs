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

    // Collect all timestamps
    let mut timestamps: Vec<DateTime<Utc>> = Vec::new();
    timestamps.extend(analysis.fft_heart_rates.iter().map(|(t, _)| *t));
    timestamps.extend(analysis.breathing_rates.iter().map(|(t, _)| *t));
    timestamps.extend(sleep_phases.iter().map(|(t, _)| *t));
    timestamps.sort_unstable();
    timestamps.dedup();

    // Interpolate and smooth heart rates
    let smoothed_fft_hr = heart_analysis::interpolate_and_smooth(
        &timestamps,
        &analysis.fft_heart_rates,
        hr_smoothing_window,
        hr_smoothing_strength,
    );

    // Write header
    writer.write_record(&[
        "timestamp",
        "fft_hr",
        "fft_hr_smoothed",
        "breathing_rate",
        "sleep_phase",
    ])?;

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

        // Find the current sleep phase
        let phase = sleep_phases
            .iter()
            .rev()
            .find(|(t, _)| *t <= timestamp)
            .map(|(_, phase)| format!("{:?}", phase))
            .unwrap_or_default();

        writer.write_record(&[
            timestamp.format("%Y-%m-%d %H:%M").to_string(),
            fft_hr,
            fft_hr_smoothed,
            br,
            phase,
        ])?;
    }

    writer.flush()?;
    Ok(())
}
