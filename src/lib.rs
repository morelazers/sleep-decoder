pub mod config;
pub mod data_loading;
pub mod heart_analysis;
pub mod output;
pub mod phase_analysis;
pub mod preprocessing;

use chrono::{DateTime, Utc};
use phase_analysis::SleepPhase;

#[derive(Debug, Clone)]
pub struct BedPresence {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct PeriodAnalysis {
    pub fft_heart_rates: Vec<(DateTime<Utc>, f32)>, // Results from FFT analysis
    pub breathing_rates: Vec<(DateTime<Utc>, f32)>, // Results from breathing analysis
}

#[derive(Debug, Clone)]
pub struct SideAnalysis {
    pub combined: PeriodAnalysis,
    pub period_num: usize,
    pub sleep_phases: Vec<(DateTime<Utc>, SleepPhase)>,
}

#[derive(Debug, Clone)]
pub struct BedAnalysis {
    pub left_side: Vec<SideAnalysis>,
    pub right_side: Vec<SideAnalysis>,
}

pub struct RawDataView<'a> {
    pub raw_data: &'a [(u32, data_loading::CombinedSensorData)],
    pub start_idx: usize,
    pub end_idx: usize,
}

pub struct RawPeriodData<'a> {
    pub timestamp: DateTime<Utc>,
    pub left1: &'a [i32],
    pub left2: Option<&'a [i32]>,
    pub right1: &'a [i32],
    pub right2: Option<&'a [i32]>,
    pub left: &'a [i32],
    pub right: &'a [i32],
}

impl<'a> RawDataView<'a> {
    pub fn get_data_at(&self, idx: usize) -> Option<RawPeriodData<'a>> {
        if idx >= self.end_idx - self.start_idx {
            return None;
        }

        let data = &self.raw_data[self.start_idx + idx].1;
        Some(RawPeriodData {
            timestamp: DateTime::from_timestamp(data.ts, 0).unwrap(),
            left1: &data.left1,
            left2: data.left2.as_deref(),
            right1: &data.right1,
            right2: data.right2.as_deref(),
            left: &data.left,
            right: &data.right,
        })
    }

    pub fn len(&self) -> usize {
        self.end_idx - self.start_idx
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
