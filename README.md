# Sleep Decoder

## Setup

Make sure you have Rust installed.

```bash
cargo run --release /path/ro/raws \
  --start-time=2025-01-01 22:00 \ # UTC!
  --end-time=2025-01-02 09:00 \ # UTC!
  --csv-output=/path/to/output/dir/somename \
  --split-sensors=false # Will split csv outputs into left and right (and combined) if true
```

### Available env variables

```
HR_WINDOW_SECONDS=10 # Seems to work best at 10 seconds, but 120 is the default
HR_WINDOW_OVERLAP_PERCENT=0.1
BR_WINDOW_SECONDS=60 # Seems to work best at 60 seconds, but 120 is the default
BR_WINDOW_OVERLAP_PERCENT=0.5
```

## Build

```bash
cargo build --release --target aarch64-unknown-linux-musl

# or
./build.sh
```
