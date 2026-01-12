//! Time-Based Reward Decay System
//!
//! Implements a decay mechanism based on time since submission:
//! - Grace period: 48 hours after submission = no decay
//! - After grace period: Rewards decay by 50% each day (24 hours)
//!
//! Formula: multiplier = 0.5 ^ (days_past_grace)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for time-based decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeDecayConfig {
    /// Whether time decay is enabled
    pub enabled: bool,
    /// Grace period in hours before decay starts (default: 48 hours)
    pub grace_period_hours: u64,
    /// Half-life in hours - time for weight to decay by 50% (default: 24 hours = 1 day)
    pub half_life_hours: u64,
    /// Minimum multiplier (weight never goes below this, default: 0.01 = 1%)
    pub min_multiplier: f64,
}

impl Default for TimeDecayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            grace_period_hours: 48, // 48 hours = 2 days grace period
            half_life_hours: 24,    // 24 hours = 50% decay per day
            min_multiplier: 0.01,
        }
    }
}

impl TimeDecayConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("TIME_DECAY_ENABLED")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            grace_period_hours: std::env::var("TIME_DECAY_GRACE_HOURS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(48),
            half_life_hours: std::env::var("TIME_DECAY_HALF_LIFE_HOURS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(24),
            min_multiplier: std::env::var("TIME_DECAY_MIN_MULTIPLIER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.01),
        }
    }
}

/// Result of decay calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayInfo {
    /// The decay multiplier to apply to weight (0.0 to 1.0)
    pub multiplier: f64,
    /// Age of submission in hours
    pub age_hours: f64,
    /// Hours remaining in grace period (0 if grace period expired)
    pub grace_period_remaining_hours: f64,
    /// Whether decay is currently active
    pub decay_active: bool,
    /// Days since grace period ended (for display)
    pub days_decaying: f64,
}

/// Calculate decay multiplier based on time since submission
///
/// Formula:
/// - If hours_elapsed <= grace_period_hours: multiplier = 1.0
/// - Otherwise: multiplier = 0.5 ^ (hours_past_grace / half_life_hours)
///
/// The multiplier is clamped to min_multiplier to prevent complete decay.
pub fn calculate_decay_multiplier(submission_time: DateTime<Utc>, config: &TimeDecayConfig) -> f64 {
    if !config.enabled {
        return 1.0;
    }

    let now = Utc::now();
    let hours_elapsed = (now - submission_time).num_minutes() as f64 / 60.0;

    if hours_elapsed <= config.grace_period_hours as f64 {
        return 1.0;
    }

    let hours_past_grace = hours_elapsed - config.grace_period_hours as f64;
    let half_lives = hours_past_grace / config.half_life_hours as f64;

    // multiplier = 0.5 ^ half_lives
    let multiplier = 0.5_f64.powf(half_lives);

    // Clamp to minimum
    multiplier.max(config.min_multiplier)
}

/// Calculate full decay info for a submission
pub fn calculate_decay_info(submission_time: DateTime<Utc>, config: &TimeDecayConfig) -> DecayInfo {
    let now = Utc::now();
    let hours_elapsed = (now - submission_time).num_minutes() as f64 / 60.0;

    if !config.enabled {
        return DecayInfo {
            multiplier: 1.0,
            age_hours: hours_elapsed,
            grace_period_remaining_hours: 0.0,
            decay_active: false,
            days_decaying: 0.0,
        };
    }

    let grace_remaining = (config.grace_period_hours as f64 - hours_elapsed).max(0.0);
    let decay_active = hours_elapsed > config.grace_period_hours as f64;

    let (multiplier, days_decaying) = if decay_active {
        let hours_past_grace = hours_elapsed - config.grace_period_hours as f64;
        let half_lives = hours_past_grace / config.half_life_hours as f64;
        let mult = 0.5_f64.powf(half_lives).max(config.min_multiplier);
        (mult, hours_past_grace / 24.0)
    } else {
        (1.0, 0.0)
    };

    DecayInfo {
        multiplier,
        age_hours: hours_elapsed,
        grace_period_remaining_hours: grace_remaining,
        decay_active,
        days_decaying,
    }
}

/// Decay status response for API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayStatusResponse {
    pub winner: Option<WinnerDecayStatus>,
    pub config: TimeDecayConfigResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinnerDecayStatus {
    pub agent_hash: String,
    pub miner_hotkey: String,
    pub name: Option<String>,
    pub submitted_at: String,
    pub age_hours: f64,
    pub grace_period_remaining_hours: f64,
    pub decay_active: bool,
    pub decay_multiplier: f64,
    pub effective_weight: f64,
    pub days_decaying: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeDecayConfigResponse {
    pub enabled: bool,
    pub grace_period_hours: u64,
    pub half_life_hours: u64,
    pub min_multiplier: f64,
}

impl From<&TimeDecayConfig> for TimeDecayConfigResponse {
    fn from(config: &TimeDecayConfig) -> Self {
        Self {
            enabled: config.enabled,
            grace_period_hours: config.grace_period_hours,
            half_life_hours: config.half_life_hours,
            min_multiplier: config.min_multiplier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn default_config() -> TimeDecayConfig {
        TimeDecayConfig {
            enabled: true,
            grace_period_hours: 48,
            half_life_hours: 24,
            min_multiplier: 0.01,
        }
    }

    #[test]
    fn test_no_decay_during_grace_period() {
        let config = default_config();

        // 24 hours ago - in grace period
        let submission_time = Utc::now() - Duration::hours(24);
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert_eq!(multiplier, 1.0);

        // 48 hours ago - exactly at grace period boundary
        let submission_time = Utc::now() - Duration::hours(48);
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert_eq!(multiplier, 1.0);
    }

    #[test]
    fn test_decay_after_grace_period() {
        let config = default_config();

        // 72 hours ago - 24 hours past grace (1 half-life = 50%)
        let submission_time = Utc::now() - Duration::hours(72);
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert!(
            (multiplier - 0.5).abs() < 0.01,
            "After 24 hours past grace should be ~0.5, got {}",
            multiplier
        );

        // 96 hours ago - 48 hours past grace (2 half-lives = 25%)
        let submission_time = Utc::now() - Duration::hours(96);
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert!(
            (multiplier - 0.25).abs() < 0.01,
            "After 48 hours past grace should be ~0.25, got {}",
            multiplier
        );

        // 120 hours ago - 72 hours past grace (3 half-lives = 12.5%)
        let submission_time = Utc::now() - Duration::hours(120);
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert!(
            (multiplier - 0.125).abs() < 0.01,
            "After 72 hours past grace should be ~0.125, got {}",
            multiplier
        );
    }

    #[test]
    fn test_min_multiplier_cap() {
        let config = TimeDecayConfig {
            enabled: true,
            grace_period_hours: 48,
            half_life_hours: 24,
            min_multiplier: 0.1, // 10% minimum
        };

        // Many days past grace - would be very small without cap
        let submission_time = Utc::now() - Duration::hours(500);
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert_eq!(multiplier, 0.1, "Should be capped at min_multiplier");
    }

    #[test]
    fn test_decay_disabled() {
        let config = TimeDecayConfig {
            enabled: false,
            ..default_config()
        };

        // Even after long time, no decay when disabled
        let submission_time = Utc::now() - Duration::hours(500);
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert_eq!(multiplier, 1.0);
    }

    #[test]
    fn test_decay_info_in_grace() {
        let config = default_config();

        // 24 hours ago - in grace period
        let submission_time = Utc::now() - Duration::hours(24);
        let info = calculate_decay_info(submission_time, &config);

        assert!(!info.decay_active);
        assert!(info.grace_period_remaining_hours > 20.0);
        assert_eq!(info.multiplier, 1.0);
        assert_eq!(info.days_decaying, 0.0);
    }

    #[test]
    fn test_decay_info_after_grace() {
        let config = default_config();

        // 72 hours ago (24 hours past grace)
        let submission_time = Utc::now() - Duration::hours(72);
        let info = calculate_decay_info(submission_time, &config);

        assert!(info.decay_active);
        assert_eq!(info.grace_period_remaining_hours, 0.0);
        assert!(
            (info.multiplier - 0.5).abs() < 0.02,
            "Expected ~0.5, got {}",
            info.multiplier
        );
        assert!((info.days_decaying - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_half_decay_per_day() {
        let config = default_config();

        // Verify that after 1 day past grace, we have 50% decay
        let submission_time = Utc::now() - Duration::hours(48 + 24); // Grace + 1 day
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert!(
            (multiplier - 0.5).abs() < 0.01,
            "1 day past grace should be 50%, got {}",
            multiplier
        );

        // After 2 days past grace, we have 25% decay
        let submission_time = Utc::now() - Duration::hours(48 + 48); // Grace + 2 days
        let multiplier = calculate_decay_multiplier(submission_time, &config);
        assert!(
            (multiplier - 0.25).abs() < 0.01,
            "2 days past grace should be 25%, got {}",
            multiplier
        );
    }
}
