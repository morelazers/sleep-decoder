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
HR_WINDOW_SECONDS=10
HR_WINDOW_OVERLAP_PERCENT=0.1 # 10% overlap
BR_WINDOW_SECONDS=120
BR_WINDOW_OVERLAP_PERCENT=0
```

## Build

```bash
cargo build --release --target aarch64-unknown-linux-musl

# or
./build.sh
```
