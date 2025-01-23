use chrono::{DateTime, Utc};
use serde::Serialize;

#[derive(Debug, Serialize, Copy, Clone)]
pub struct ProcessedData {
    pub timestamp: DateTime<Utc>,
    pub left1_mean: f32,
    pub left1_std: f32,
    pub left2_mean: f32,
    pub left2_std: f32,
    pub right1_mean: f32,
    pub right1_std: f32,
    pub right2_mean: f32,
    pub right2_std: f32,
}

pub fn calculate_stats(data: &[i32]) -> (f32, f32) {
    // First remove outliers from raw data
    let cleaned_data = crate::heart_analysis::interpolate_outliers(data, 2);

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

pub fn process_piezo_data(data: &crate::data_loading::CombinedSensorData) -> Option<ProcessedData> {
    let timestamp = DateTime::from_timestamp(data.ts, 0).unwrap();

    // Calculate stats for combined signals
    let (left_mean, left_std) = calculate_stats(&data.left);
    let (right_mean, right_std) = calculate_stats(&data.right);

    Some(ProcessedData {
        timestamp,
        left1_mean: left_mean,
        left1_std: left_std,
        left2_mean: left_mean, // Use same values since signals are combined
        left2_std: left_std,
        right1_mean: right_mean,
        right1_std: right_std,
        right2_mean: right_mean, // Use same values since signals are combined
        right2_std: right_std,
    })
}

pub fn remove_stat_outliers_for_sensor(data: &mut Vec<ProcessedData>, sensor: &str) {
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
    let mean_percentiles = {
        let mut indices: Vec<usize> = (0..mean_data.len()).collect();
        indices.sort_by(|&a, &b| mean_data[a].partial_cmp(&mean_data[b]).unwrap());
        let mean_lower = mean_data[indices[(data.len() as f32 * 0.02) as usize]];
        let mean_upper = mean_data[indices[(data.len() as f32 * 0.98) as usize]];
        (mean_lower, mean_upper)
    };

    // Calculate percentiles for std
    let std_percentiles = {
        let mut indices: Vec<usize> = (0..std_data.len()).collect();
        indices.sort_by(|&a, &b| std_data[a].partial_cmp(&std_data[b]).unwrap());
        let std_lower = std_data[indices[(data.len() as f32 * 0.02) as usize]];
        let std_upper = std_data[indices[(data.len() as f32 * 0.98) as usize]];
        (std_lower, std_upper)
    };

    // Filter data
    data.retain(|d| {
        let (mean, std) = match sensor {
            "left1" => (d.left1_mean, d.left1_std),
            "left2" => (d.left2_mean, d.left2_std),
            "right1" => (d.right1_mean, d.right1_std),
            "right2" => (d.right2_mean, d.right2_std),
            _ => unreachable!(),
        };
        mean >= mean_percentiles.0
            && mean <= mean_percentiles.1
            && std >= std_percentiles.0
            && std <= std_percentiles.1
    });

    println!(
        "Removed {} rows as mean/std outliers for {}",
        len_before - data.len(),
        sensor
    );
    println!("Remaining rows: {}", data.len());
}

pub fn remove_stat_outliers(data: &mut Vec<ProcessedData>) {
    // Process each sensor independently
    for sensor in ["left1", "left2", "right1", "right2"] {
        remove_stat_outliers_for_sensor(data, sensor);
    }
}

pub fn remove_time_outliers(data: &mut Vec<ProcessedData>) {
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

pub fn scale_data(data: &mut Vec<ProcessedData>) {
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
