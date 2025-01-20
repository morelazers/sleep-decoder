# Sleep Decoder

## Setup

Make sure you have Rust installed.

```bash
cargo run --release /path/ro/raws \
  --start-time=2025-01-01 22:00 \ # UTC!
  --end-time=2025-01-02 09:00 \ # UTC!
  --csv-output=/path/to/output/dir/somename \
  --merge-sides \ # Whether to merge left and right sensor data (single occupant, omit for false)
  --hr-window-seconds=10 \ # Window size for HR analysis
  --hr-window-overlap-percent=0.1 \ # Overlap percentage for HR analysis
  --br-window-seconds=120 \ # Window size for BR analysis
  --br-window-overlap-percent=0 # Overlap percentage for BR analysis
```

## Build

```bash
cargo build --release --target aarch64-unknown-linux-musl

# or
./build.sh
```
