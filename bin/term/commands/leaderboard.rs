//! Leaderboard command - show top agents

use crate::print_banner;
use crate::style::*;
use anyhow::{anyhow, Result};

pub async fn run(platform_url: &str, limit: usize) -> Result<()> {
    print_banner();
    print_header("Leaderboard");

    let entries = fetch_leaderboard(platform_url, limit).await?;

    if entries.is_empty() {
        println!("  {} No agents on the leaderboard yet.", style_dim("─"));
        println!();
        print_info("Be the first to submit an agent!");
        println!(
            "  Run: {}",
            style_cyan("term submit -a your_agent.py -k YOUR_KEY")
        );
        println!();
        return Ok(());
    }

    // Table header
    println!(
        "  {:<4} {:<10} {:<18} {:<8} {:<12} {}",
        style_bold("Rank"),
        style_bold("Score"),
        style_bold("Agent"),
        style_bold("Evals"),
        style_bold("Updated"),
        style_bold("Miner")
    );
    println!("  {}", style_dim(&"─".repeat(75)));

    // Table rows
    for entry in &entries {
        let rank_icon = match entry.rank {
            1 => "🥇",
            2 => "🥈",
            3 => "🥉",
            _ => "  ",
        };

        let score_color = if entry.score >= 0.8 {
            colors::GREEN
        } else if entry.score >= 0.6 {
            colors::YELLOW
        } else {
            colors::RED
        };

        let updated = if entry.updated_at.len() >= 10 {
            &entry.updated_at[..10]
        } else {
            &entry.updated_at
        };

        let miner_short = if entry.miner.len() > 8 {
            format!("{}...", &entry.miner[..8])
        } else {
            entry.miner.clone()
        };

        println!(
            "  {}{:<2} {}{:>6.2}%{}  {:<18} {:<8} {:<12} {}",
            rank_icon,
            entry.rank,
            score_color,
            entry.score * 100.0,
            colors::RESET,
            truncate(&entry.name, 16),
            entry.evaluation_count,
            updated,
            style_gray(&miner_short)
        );
    }

    println!();

    // Summary
    let total = entries.len();
    let avg_score: f64 = entries.iter().map(|e| e.score).sum::<f64>() / total as f64;

    print_section("Summary");
    print_key_value("Total Agents", &total.to_string());
    print_key_value("Average Score", &format!("{:.2}%", avg_score * 100.0));

    if let Some(best) = entries.first() {
        print_key_value_colored(
            "Best Score",
            &format!("{:.2}%", best.score * 100.0),
            colors::GREEN,
        );
    }

    println!();
    Ok(())
}

struct LeaderboardEntry {
    rank: u32,
    name: String,
    score: f64,
    evaluation_count: u32,
    updated_at: String,
    miner: String,
}

async fn fetch_leaderboard(platform_url: &str, limit: usize) -> Result<Vec<LeaderboardEntry>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let url = format!("{}/api/v1/leaderboard?limit={}", platform_url, limit);

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to connect to platform: {}", e))?;

    if !resp.status().is_success() {
        return Err(anyhow!(
            "Failed to fetch leaderboard: HTTP {} from {}",
            resp.status(),
            url
        ));
    }

    let data: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| anyhow!("Invalid response: {}", e))?;

    Ok(data
        .iter()
        .map(|v| LeaderboardEntry {
            rank: v["rank"].as_u64().unwrap_or(0) as u32,
            name: v["name"].as_str().unwrap_or("unnamed").to_string(),
            score: v["consensus_score"].as_f64().unwrap_or(0.0),
            evaluation_count: v["evaluation_count"].as_u64().unwrap_or(0) as u32,
            updated_at: v["updated_at"]
                .as_str()
                .or_else(|| v["updated_at"].as_i64().map(|_| ""))
                .unwrap_or("")
                .to_string(),
            miner: v["miner_hotkey"].as_str().unwrap_or("").to_string(),
        })
        .collect())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}

use crate::style::colors;
