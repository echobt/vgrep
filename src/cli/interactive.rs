use anyhow::Result;
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};

use crate::config::{Config, Mode};
use crate::ui;

pub fn run_interactive_config(config: &mut Config) -> Result<()> {
    ui::print_banner();
    println!("  {}Interactive Configuration", style("⚙️  ").cyan());
    println!("  {}", style("─".repeat(40)).dim());
    println!();

    let items = vec![
        "Mode (server/local)",
        "Server Settings",
        "Models",
        "Search Settings",
        "Indexing Settings",
        "Performance Settings",
        "Reset to Defaults",
        "Save & Exit",
        "Exit without Saving",
    ];

    loop {
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select category to configure")
            .items(&items)
            .default(0)
            .interact()?;

        match selection {
            0 => configure_mode(config)?,
            1 => configure_server(config)?,
            2 => configure_models(config)?,
            3 => configure_search(config)?,
            4 => configure_indexing(config)?,
            5 => configure_performance(config)?,
            6 => {
                if Confirm::with_theme(&ColorfulTheme::default())
                    .with_prompt("Reset all settings to defaults?")
                    .default(false)
                    .interact()?
                {
                    *config = Config::default();
                    ui::print_success("Configuration reset to defaults");
                }
            }
            7 => {
                config.save()?;
                ui::print_success("Configuration saved!");
                break;
            }
            8 => {
                if Confirm::with_theme(&ColorfulTheme::default())
                    .with_prompt("Exit without saving changes?")
                    .default(false)
                    .interact()?
                {
                    break;
                }
            }
            _ => {}
        }
        println!();
    }

    Ok(())
}

fn configure_mode(config: &mut Config) -> Result<()> {
    println!();
    println!(
        "  {} Current mode: {}",
        style("→").dim(),
        style(&config.mode).cyan()
    );
    println!();

    let modes = vec!["server (recommended)", "local"];
    let current = if config.mode == Mode::Server { 0 } else { 1 };

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select mode")
        .items(&modes)
        .default(current)
        .interact()?;

    config.mode = if selection == 0 {
        Mode::Server
    } else {
        Mode::Local
    };

    ui::print_success(&format!("Mode set to: {}", config.mode));
    Ok(())
}

fn configure_server(config: &mut Config) -> Result<()> {
    println!();
    println!(
        "  {} Current: {}:{}",
        style("→").dim(),
        style(&config.server_host).cyan(),
        style(&config.server_port).cyan()
    );
    println!();

    let host: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Server host")
        .default(config.server_host.clone())
        .interact_text()?;

    let port: u16 = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Server port")
        .default(config.server_port)
        .interact_text()?;

    config.server_host = host;
    config.server_port = port;

    ui::print_success(&format!(
        "Server configured: {}:{}",
        config.server_host, config.server_port
    ));
    Ok(())
}

fn configure_models(config: &mut Config) -> Result<()> {
    println!();
    println!(
        "  {} Models directory: {}",
        style("→").dim(),
        style(config.get_models_dir()?.display()).cyan()
    );
    println!(
        "  {} Embedding: {} {}",
        style("→").dim(),
        style(&config.embedding_model).cyan(),
        if config.has_embedding_model() {
            style("[OK]").green()
        } else {
            style("[Missing]").red()
        }
    );
    println!(
        "  {} Reranker: {} {} ({})",
        style("→").dim(),
        style(&config.reranker_model).cyan(),
        if config.has_reranker_model() {
            style("[OK]").green()
        } else {
            style("[Missing]").red()
        },
        if config.use_reranker {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!();

    let items = vec![
        "Set embedding model name",
        "Set reranker model name",
        "Toggle reranker (on/off)",
        "Set custom models directory",
        "Back",
    ];

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Model settings")
        .items(&items)
        .default(0)
        .interact()?;

    match selection {
        0 => {
            let model: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Embedding model filename")
                .default(config.embedding_model.clone())
                .interact_text()?;
            config.embedding_model = model;
            ui::print_success("Embedding model updated");
        }
        1 => {
            let model: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Reranker model filename")
                .default(config.reranker_model.clone())
                .interact_text()?;
            config.reranker_model = model;
            ui::print_success("Reranker model updated");
        }
        2 => {
            config.use_reranker = !config.use_reranker;
            ui::print_success(&format!(
                "Reranker {}",
                if config.use_reranker {
                    "enabled"
                } else {
                    "disabled"
                }
            ));
        }
        3 => {
            let dir: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Models directory (empty for default)")
                .default(
                    config
                        .models_dir
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default(),
                )
                .allow_empty(true)
                .interact_text()?;

            if dir.is_empty() {
                config.models_dir = None;
                ui::print_success("Using default models directory");
            } else {
                config.models_dir = Some(std::path::PathBuf::from(dir));
                ui::print_success("Custom models directory set");
            }
        }
        _ => {}
    }

    Ok(())
}

fn configure_search(config: &mut Config) -> Result<()> {
    println!();
    println!(
        "  {} Max results: {}",
        style("→").dim(),
        style(&config.max_results).cyan()
    );
    println!(
        "  {} Show content: {}",
        style("→").dim(),
        style(&config.show_content).cyan()
    );
    println!();

    let max_results: usize = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Default max results")
        .default(config.max_results)
        .interact_text()?;

    let show_content = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Show content in results by default?")
        .default(config.show_content)
        .interact()?;

    config.max_results = max_results;
    config.show_content = show_content;

    ui::print_success("Search settings updated");
    Ok(())
}

fn configure_indexing(config: &mut Config) -> Result<()> {
    println!();
    println!(
        "  {} Max file size: {} KB",
        style("→").dim(),
        style(config.max_file_size / 1024).cyan()
    );
    println!(
        "  {} Chunk size: {}",
        style("→").dim(),
        style(&config.chunk_size).cyan()
    );
    println!(
        "  {} Batch size: {}",
        style("→").dim(),
        style(&config.batch_size).cyan()
    );
    println!(
        "  {} Chunk overlap: {}",
        style("→").dim(),
        style(&config.chunk_overlap).cyan()
    );
    println!(
        "  {} Watch debounce: {} ms",
        style("→").dim(),
        style(&config.watch_debounce_ms).cyan()
    );
    println!();

    let max_file_size: u64 = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Max file size (KB)")
        .default(config.max_file_size / 1024)
        .interact_text()?;

    let chunk_size: usize = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Chunk size (characters)")
        .default(config.chunk_size)
        .interact_text()?;

    let batch_size: usize = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Batch size (server requests)")
        .default(config.batch_size)
        .interact_text()?;

    let chunk_overlap: usize = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Chunk overlap (characters)")
        .default(config.chunk_overlap)
        .interact_text()?;

    let watch_debounce: u64 = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Watch debounce (ms)")
        .default(config.watch_debounce_ms)
        .interact_text()?;

    config.max_file_size = max_file_size * 1024;
    config.chunk_size = chunk_size;
    config.batch_size = batch_size;
    config.chunk_overlap = chunk_overlap;
    config.watch_debounce_ms = watch_debounce;

    ui::print_success("Indexing settings updated");
    Ok(())
}

fn configure_performance(config: &mut Config) -> Result<()> {
    println!();
    println!(
        "  {} Threads: {} {}",
        style("→").dim(),
        style(&config.n_threads).cyan(),
        if config.n_threads == 0 {
            format!("(auto: {})", config.get_n_threads())
        } else {
            String::new()
        }
    );
    println!(
        "  {} Context size: {}",
        style("→").dim(),
        style(&config.context_size).cyan()
    );
    println!();

    let n_threads: usize = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Number of threads (0 = auto)")
        .default(config.n_threads)
        .interact_text()?;

    let context_size: usize = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Context size")
        .default(config.context_size)
        .interact_text()?;

    config.n_threads = n_threads;
    config.context_size = context_size;

    ui::print_success("Performance settings updated");
    Ok(())
}
