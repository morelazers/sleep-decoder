use anyhow::{Context, Result};
use arrow::array::{Array, Int32Array, ListArray, StringArray};
use arrow::ipc::reader::FileReaderBuilder;
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use serde_bytes;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct BatchItem {
    seq: u32,
    data: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum SensorData {
    #[serde(rename = "piezo-dual")]
    PiezoDual {
        ts: i64,
        adc: u8,
        freq: u16,
        gain: u16,
        #[serde(with = "serde_bytes")]
        left1: Vec<u8>,
        #[serde(default)]
        #[serde(with = "serde_bytes")]
        left2: Option<Vec<u8>>,
        #[serde(with = "serde_bytes")]
        right1: Vec<u8>,
        #[serde(default)]
        #[serde(with = "serde_bytes")]
        right2: Option<Vec<u8>>,
    },
}

#[derive(Debug)]
pub struct CombinedSensorData {
    pub ts: i64,
    pub left1: Vec<i32>,
    pub left2: Option<Vec<i32>>,
    pub right1: Vec<i32>,
    pub right2: Option<Vec<i32>>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
}

impl<'a> From<&'a SensorData> for Option<CombinedSensorData> {
    fn from(data: &'a SensorData) -> Option<CombinedSensorData> {
        let SensorData::PiezoDual {
            ts,
            left1,
            left2,
            right1,
            right2,
            ..
        } = data;
        // Convert individual sensors
        let left1_vec: Vec<i32> = left1
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let left2_vec: Option<Vec<i32>> = left2.as_ref().map(|data| {
            data.chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        });

        let right1_vec: Vec<i32> = right1
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let right2_vec: Option<Vec<i32>> = right2.as_ref().map(|data| {
            data.chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        });

        // Compute combined signals
        let left = match &left2_vec {
            Some(left2_data) => left1_vec
                .iter()
                .zip(left2_data.iter())
                .map(|(a, b)| (((*a as i64 + *b as i64) / 2) as i32))
                .collect(),
            None => left1_vec.clone(),
        };

        let right = match &right2_vec {
            Some(right2_data) => right1_vec
                .iter()
                .zip(right2_data.iter())
                .map(|(a, b)| (((*a as i64 + *b as i64) / 2) as i32))
                .collect(),
            None => right1_vec.clone(),
        };

        Some(CombinedSensorData {
            ts: *ts,
            left1: left1_vec,
            left2: left2_vec,
            right1: right1_vec,
            right2: right2_vec,
            left,
            right,
        })
    }
}

#[derive(Debug)]
pub struct RawFileInfo {
    pub path: PathBuf,
    pub first_seq: u32,
    pub last_seq: u32,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

pub fn decode_batch_item(file_path: &PathBuf) -> Result<Vec<(u32, SensorData)>> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut items = Vec::new();

    loop {
        let batch_item: BatchItem = match ciborium::from_reader(&mut reader) {
            Ok(item) => item,
            Err(ciborium::de::Error::Io(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                eprintln!("Warning: Skipping malformed CBOR data: {}", e);
                continue;
            }
        };

        match ciborium::from_reader(batch_item.data.as_slice()) {
            Ok(sensor_data) => {
                items.push((batch_item.seq, sensor_data));
            }
            Err(_) => {
                continue;
            }
        }
    }

    Ok(items)
}

pub fn build_raw_file_index(raw_dir: &PathBuf) -> Result<Vec<RawFileInfo>> {
    let mut file_index = Vec::new();

    for entry in std::fs::read_dir(raw_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("RAW") {
            println!("Indexing file: {}", path.display());

            // Load file to extract metadata
            let items = decode_batch_item(&path)?;

            if items.is_empty() {
                println!("Skipping empty file: {}", path.display());
                continue;
            }

            // Find sequence range and time range
            let mut first_seq = u32::MAX;
            let mut last_seq = 0;
            let mut start_time = DateTime::<Utc>::MAX_UTC;
            let mut end_time = DateTime::<Utc>::MIN_UTC;

            for (seq, data) in &items {
                let SensorData::PiezoDual { ts, .. } = data;
                first_seq = first_seq.min(*seq);
                last_seq = last_seq.max(*seq);

                let timestamp = DateTime::from_timestamp(*ts, 0).unwrap();
                start_time = start_time.min(timestamp);
                end_time = end_time.max(timestamp);
            }

            file_index.push(RawFileInfo {
                path,
                first_seq,
                last_seq,
                start_time,
                end_time,
            });
        }
    }

    // Sort by first sequence number
    file_index.sort_by_key(|info| info.first_seq);

    println!("\nFound {} RAW files:", file_index.len());
    for info in &file_index {
        println!(
            "  {} (seq: {} to {}, time: {} to {})",
            info.path.file_name().unwrap().to_string_lossy(),
            info.first_seq,
            info.last_seq,
            info.start_time.format("%Y-%m-%d %H:%M:%S"),
            info.end_time.format("%Y-%m-%d %H:%M:%S")
        );
    }

    Ok(file_index)
}

pub fn read_csv_file(path: &PathBuf) -> Result<Vec<(u32, CombinedSensorData)>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true) // Handle variable number of fields
        .from_reader(file);
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result?;

        // Parse the timestamp from datetime string
        let dt = NaiveDateTime::parse_from_str(record.get(1).unwrap(), "%Y-%m-%d %H:%M:%S")?;
        let ts = dt.and_utc().timestamp();

        // Parse left1 and right1 as space-separated integers
        let parse_signal = |s: &str| -> Result<Vec<i32>> {
            // Remove square brackets and split on whitespace
            let cleaned = s
                .trim_matches(|c| c == '[' || c == ']' || c == ' ')
                .split_whitespace() // This handles multiple spaces and newlines
                .filter(|s| !s.is_empty());

            cleaned
                .map(|s| s.parse::<i32>())
                .collect::<Result<Vec<i32>, _>>()
                .map_err(|e| anyhow::anyhow!("Failed to parse signal data: {}", e))
        };

        let left1 = parse_signal(record.get(5).unwrap())?;
        let right1 = parse_signal(record.get(6).unwrap())?;
        let seq: u32 = record.get(7).unwrap().parse()?;

        // Create CombinedSensorData
        let combined = CombinedSensorData {
            ts,
            left1: left1.clone(),
            left2: None,
            right1: right1.clone(),
            right2: None,
            left: left1,
            right: right1,
        };

        data.push((seq, combined));
    }

    Ok(data)
}

pub fn read_feather_file(path: &PathBuf) -> Result<Vec<(u32, CombinedSensorData)>> {
    let file = File::open(path)?;
    let reader = FileReaderBuilder::new().build(file)?;
    let mut data = Vec::new();
    let mut seq = 0;

    for batch in reader {
        let batch = batch?;

        // Get column arrays
        let type_col = batch
            .column_by_name("type")
            .expect("type column missing")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("type column should be strings");
        let ts_col = batch
            .column_by_name("ts")
            .expect("ts column missing")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("ts column should be strings");
        let left1_col = batch
            .column_by_name("left1")
            .expect("left1 column missing")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("left1 column should be a list");
        let left2_col = batch
            .column_by_name("left2")
            .expect("left2 column missing")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("left2 column should be a list");
        let right1_col = batch
            .column_by_name("right1")
            .expect("right1 column missing")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("right1 column should be a list");
        let right2_col = batch
            .column_by_name("right2")
            .expect("right2 column missing")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("right2 column should be a list");

        // Process each row
        for row in 0..batch.num_rows() {
            // Only process piezo-dual records
            if type_col.value(row) == "piezo-dual" {
                // Parse timestamp from string
                let ts = NaiveDateTime::parse_from_str(ts_col.value(row), "%Y-%m-%d %H:%M:%S")?
                    .and_utc()
                    .timestamp();

                // Get left and right signals as i32 arrays
                let left1_values = left1_col.value(row);
                let right1_values = right1_col.value(row);
                let left2_values = left2_col.value(row);
                let right2_values = right2_col.value(row);

                let left1 = left1_values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("left values should be i32")
                    .values()
                    .to_vec();

                let left2 = left2_values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("left2 values should be i32")
                    .values()
                    .to_vec();

                let right1 = right1_values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("right values should be i32")
                    .values()
                    .to_vec();

                let right2 = right2_values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("right2 values should be i32")
                    .values()
                    .to_vec();

                let combined = CombinedSensorData {
                    ts,
                    left1: left1.clone(),
                    left2: Some(left2.clone()),
                    right1: right1.clone(),
                    right2: Some(right2.clone()),
                    left: left1
                        .iter()
                        .zip(left2.iter())
                        .map(|(a, b)| ((*a as i64 + *b as i64) / 2) as i32)
                        .collect(),
                    right: right1
                        .iter()
                        .zip(right2.iter())
                        .map(|(a, b)| ((*a as i64 + *b as i64) / 2) as i32)
                        .collect(),
                };

                data.push((seq, combined));
                seq += 1;
            }
        }
    }

    Ok(data)
}
