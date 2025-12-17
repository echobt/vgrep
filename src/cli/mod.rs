//! Command-line interface.

mod commands;
mod interactive;

pub use commands::Cli;
pub use interactive::run_interactive_config;

use anyhow::Result;
use clap::Parser;

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    cli.run()
}
