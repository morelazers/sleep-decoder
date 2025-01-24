use anyhow::Result;
use arrow::ipc::reader::{FileReader, FileReaderBuilder};
use std::fs::File;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: {} <feather_file>", args[0]);
        std::process::exit(1);
    }

    let file = File::open(&args[1])?;
    let reader = FileReaderBuilder::new().build(file)?;

    println!("\nSchema:");
    println!("{:#?}", reader.schema());

    // println!("\nFirst few records:");
    // for batch in reader.into_iter().take(1) {
    //     let batch = batch?;
    //     println!("{:#?}", batch);
    // }

    Ok(())
}