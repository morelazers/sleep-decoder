# Sleep Decoder

## Setup

Make sure you have Rust installed.

```bash
cargo run --release /path/ro/raws
```

### Available variables

```
HR_WINDOW_SECONDS=10 # Seems to work best at 10 seconds, but 120 is the default
HR_WINDOW_OVERLAP_PERCENT=0.1
BR_WINDOW_SECONDS=60 # Seems to work best at 60 seconds, but 120 is the default
BR_WINDOW_OVERLAP_PERCENT=0.5
CSV_OUTPUT=/path/to/output/dir
```

## Build

```bash
cargo build --release --target aarch64-unknown-linux-musl
```
