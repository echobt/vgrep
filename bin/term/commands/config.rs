//! Config command - show challenge configuration

use crate::print_banner;
use crate::style::*;
use anyhow::{anyhow, Result};

pub async fn run(platform_url: &str) -> Result<()> {
    print_banner();
    print_header("Challenge Configuration");

    let config = fetch_config(platform_url).await?;

    print_section("General");
    print_key_value("Challenge ID", &config.challenge_id);
    print_key_value("Dataset", &config.dataset);
    print_key_value("Dataset Version", &config.dataset_version);
    println!();

    print_section("Submission Requirements");
    print_key_value("Min Stake", &format!("{} TAO", config.min_stake_tao));
    println!();

    print_section("Evaluation");
    print_key_value(
        "Tasks per Evaluation",
        &config.tasks_per_evaluation.to_string(),
    );
    print_key_value("Max Steps per Task", &config.max_steps_per_task.to_string());
    print_key_value(
        "Max Concurrent Tasks",
        &config.max_concurrent_tasks.to_string(),
    );
    print_key_value(
        "Max Cost per Task",
        &format!("${:.2}", config.max_cost_per_task_usd),
    );
    print_key_value(
        "Max Total Cost",
        &format!("${:.2}", config.max_total_cost_usd),
    );
    print_key_value("Test Mode", &config.test_mode.to_string());
    println!();

    print_section("Scoring");
    print_key_value("Task Completion", "100%");
    print_key_value("Formula", "tasks_passed / total_tasks");
    println!();

    print_section("Network");
    print_key_value("Platform URL", platform_url);
    println!();

    Ok(())
}

struct ChallengeConfig {
    challenge_id: String,
    dataset: String,
    dataset_version: String,
    min_stake_tao: u64,
    tasks_per_evaluation: u32,
    max_steps_per_task: u32,
    max_concurrent_tasks: u32,
    max_cost_per_task_usd: f64,
    max_total_cost_usd: f64,
    test_mode: bool,
}

async fn fetch_config(platform_url: &str) -> Result<ChallengeConfig> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let url = format!("{}/api/v1/challenges/term-challenge/config", platform_url);

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to connect to platform: {}", e))?;

    if !resp.status().is_success() {
        return Err(anyhow!(
            "Failed to fetch config: HTTP {} from {}",
            resp.status(),
            url
        ));
    }

    let data: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow!("Invalid response: {}", e))?;

    Ok(ChallengeConfig {
        challenge_id: data["challenge_id"]
            .as_str()
            .unwrap_or("unknown")
            .to_string(),
        dataset: data["dataset"]
            .as_str()
            .unwrap_or("terminal-bench")
            .to_string(),
        dataset_version: data["dataset_version"]
            .as_str()
            .unwrap_or("unknown")
            .to_string(),
        min_stake_tao: data["min_stake_tao"].as_u64().unwrap_or(0),
        tasks_per_evaluation: data["tasks_per_evaluation"].as_u64().unwrap_or(0) as u32,
        max_steps_per_task: data["max_steps_per_task"].as_u64().unwrap_or(0) as u32,
        max_concurrent_tasks: data["max_concurrent_tasks"].as_u64().unwrap_or(0) as u32,
        max_cost_per_task_usd: data["max_cost_per_task_usd"].as_f64().unwrap_or(0.0),
        max_total_cost_usd: data["max_total_cost_usd"].as_f64().unwrap_or(0.0),
        test_mode: data["test_mode"].as_bool().unwrap_or(false),
    })
}
