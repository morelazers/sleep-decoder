# Sleep Decoder

## Setup

Make sure you have Rust installed.

## Usage

```bash
cargo run --release /path/to/raws [OPTIONS]
```

### Required Arguments

- `/path/to/raws`: Directory containing .RAW files or path to input CSV
- `--csv-input`: Whether the input is a CSV file

### Time Range Options

- `--start-time=YYYY-MM-DD HH:MM`: Start time in UTC (defaults to 24 hours ago)
- `--end-time=YYYY-MM-DD HH:MM`: End time in UTC (defaults to now)

### Output Options

- `--csv-output=/path/to/output/prefix`: CSV output file prefix
- `--merge-sides`: Whether to merge left and right sensor data (single occupant)

### Heart Rate Analysis Options

- `--hr-window-seconds=10.0`: Heart rate window size in seconds
- `--hr-window-overlap=0.1`: Heart rate window overlap (0.0 to 1.0)
- `--hr-smoothing-window=60`: Window size for smoothing HR results
- `--hr-smoothing-strength=0.25`: Smoothing strength (0.1 to 2.0, higher = more smoothing)
- `--hr-outlier-percentile=0.05`: Percentile threshold for HR outlier detection (0.0 to 0.5)
- `--hr-history-window=60`: Number of recent heart rate measurements for outlier detection

### Breathing Rate Analysis Options

- `--br-window-seconds=120.0`: Breathing rate window size in seconds
- `--br-window-overlap=0.0`: Breathing rate window overlap (0.0 to 1.0)

### Harmonic Analysis Options

- `--harmonic-penalty-close=0.8`: Base penalty for breathing harmonics when close to previous HR (0.0 to 1.0)
- `--harmonic-penalty-far=0.5`: Base penalty for breathing harmonics when far from previous HR (0.0 to 1.0)
- `--harmonic-close-threshold=5.0`: BPM difference threshold for "close" harmonic penalty
- `--harmonic-far-threshold=10.0`: BPM difference threshold for "far" harmonic penalty

## Example

```bash
cargo run --release /path/to/raws \                # Input directory with RAW files
  --csv-input \                                    # Use if input is a CSV file
  --feather-input \                                # Use if input is a Feather file
  --start-time="2025-01-01 22:00" \                # UTC!
  --end-time="2025-01-02 09:00" \                  # UTC!
  --csv-output=/path/to/output/dir/somename \      # Output file prefix
  --merge-sides \                                  # Merge left/right sensor data
  --hr-window-seconds=10.0 \                       # Window size for HR analysis
  --hr-window-overlap=0.1 \                        # HR window overlap (0.0-1.0)
  --hr-smoothing-window=60 \                       # Window size for smoothing
  --hr-smoothing-strength=0.25 \                   # Smoothing strength (0.1-2.0)
  --hr-outlier-percentile=0.05 \                   # HR outlier detection threshold
  --hr-history-window=60 \                         # Recent HR measurements to consider
  --br-window-seconds=120.0 \                      # Window size for BR analysis
  --br-window-overlap=0.0 \                        # BR window overlap (0.0-1.0)
  --harmonic-penalty-close=0.8 \                   # Penalty for close harmonics (default 80% penalty)
  --harmonic-penalty-far=0.5 \                     # Penalty for far harmonics (default 50% penalty)
  --harmonic-close-threshold=5.0 \                 # "Close" threshold in BPM
  --harmonic-far-threshold=10.0                    # "Far" threshold in BPM
```

## Build

```bash
cargo build --release --target aarch64-unknown-linux-musl

# or
./build.sh
```
