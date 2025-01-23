use clap::Parser;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub enum SensorSelection {
    Combined, // 0: Use combined sensors (default)
    First,    // 1: Use left1/right1
    Second,   // 2: Use left2/right2
}

impl FromStr for SensorSelection {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(SensorSelection::Combined),
            "1" => Ok(SensorSelection::First),
            "2" => Ok(SensorSelection::Second),
            _ => Err(format!("Invalid sensor selection: {}. Use 0 for combined sensors (default), 1 for first sensors (left1/right1), or 2 for second sensors (left2/right2)", s)),
        }
    }
}

/// Process sleep data from RAW files
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Directory containing .RAW files or path to input CSV
    #[arg(help = "Directory containing .RAW files or path to input CSV")]
    pub input_path: PathBuf,

    /// Whether the input is a CSV file
    #[arg(long)]
    pub csv_input: bool,

    /// Whether the input is a Feather file
    #[arg(long)]
    pub feather_input: bool,

    /// Select sensor for analysis (0=combined [default], 1=first sensors, 2=second sensors)
    #[arg(long)]
    pub sensor: Option<SensorSelection>,

    /// Start time (format: YYYY-MM-DD HH:MM), defaults to 24 hours ago
    #[arg(long)]
    pub start_time: Option<String>,

    /// End time (format: YYYY-MM-DD HH:MM), defaults to now
    #[arg(long)]
    pub end_time: Option<String>,

    /// CSV output file prefix (e.g. /path/to/output/prefix)
    #[arg(long)]
    pub csv_output: Option<String>,

    /// Window size for smoothing HR results
    #[arg(long, default_value = "60")]
    pub hr_smoothing_window: usize,

    /// Heart rate window size in seconds
    #[arg(long, default_value = "10.0")]
    pub hr_window_seconds: f32,

    /// Heart rate window overlap percentage (0.0 to 1.0)
    #[arg(long, default_value = "0.1")]
    pub hr_window_overlap: f32,

    /// Breathing rate window size in seconds
    #[arg(long, default_value = "120.0")]
    pub br_window_seconds: f32,

    /// Breathing rate window overlap percentage (0.0 to 1.0)
    #[arg(long, default_value = "0.0")]
    pub br_window_overlap: f32,

    /// Merge left and right signals for analysis
    #[arg(long)]
    pub merge_sides: bool,

    /// Percentile threshold for HR outlier detection (0.0 to 0.5, default 0.05 means using 5th and 95th percentiles)
    #[arg(long, default_value = "0.05")]
    pub hr_outlier_percentile: f32,

    /// Number of recent heart rate measurements to consider for outlier detection
    #[arg(long, default_value = "60")]
    pub hr_history_window: usize,

    /// Base penalty for breathing harmonics when close to previous HR (0.0 to 1.0)
    #[arg(long, default_value = "0.8")]
    pub harmonic_penalty_close: f32,

    /// Base penalty for breathing harmonics when far from previous HR (0.0 to 1.0)
    #[arg(long, default_value = "0.5")]
    pub harmonic_penalty_far: f32,

    /// BPM difference threshold for "close" harmonic penalty (in BPM)
    #[arg(long, default_value = "5.0")]
    pub harmonic_close_threshold: f32,

    /// BPM difference threshold for "far" harmonic penalty (in BPM)
    #[arg(long, default_value = "10.0")]
    pub harmonic_far_threshold: f32,

    /// Smoothing strength for heart rate data (0.1 to 2.0, higher = more smoothing, default 0.25)
    #[arg(long, default_value = "0.25")]
    pub hr_smoothing_strength: f32,
}
