//! Term Challenge API Client
//!
//! Centralized client that handles routing to the correct endpoints.
//! All requests go through /api/v1/bridge/term-challenge/...

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Serialize;
use std::time::Duration;

const CHALLENGE_ID: &str = "term-challenge";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Term Challenge API client
pub struct TermClient {
    client: Client,
    base_url: String,
}

impl TermClient {
    /// Create a new client pointing to platform server
    pub fn new(platform_url: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(DEFAULT_TIMEOUT)
                .build()
                .expect("Failed to create HTTP client"),
            base_url: platform_url.trim_end_matches('/').to_string(),
        }
    }

    /// Create client with custom timeout
    pub fn with_timeout(platform_url: &str, timeout: Duration) -> Self {
        Self {
            client: Client::builder()
                .timeout(timeout)
                .build()
                .expect("Failed to create HTTP client"),
            base_url: platform_url.trim_end_matches('/').to_string(),
        }
    }

    /// Get the bridge URL for term-challenge endpoints
    fn bridge_url(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{}/api/v1/bridge/{}/{}", self.base_url, CHALLENGE_ID, path)
    }

    /// Get network state URL (not bridged)
    fn network_url(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{}/api/v1/{}", self.base_url, path)
    }

    // =========================================================================
    // PUBLIC API - Submission
    // =========================================================================

    /// Submit an agent
    pub async fn submit(&self, request: &impl Serialize) -> Result<serde_json::Value> {
        self.post_bridge("submit", request).await
    }

    // =========================================================================
    // PUBLIC API - Leaderboard
    // =========================================================================

    /// Get leaderboard
    pub async fn get_leaderboard(&self, limit: usize) -> Result<serde_json::Value> {
        self.get_bridge(&format!("leaderboard?limit={}", limit))
            .await
    }

    /// Get agent details by hash
    pub async fn get_agent(&self, agent_hash: &str) -> Result<serde_json::Value> {
        self.get_bridge(&format!("leaderboard/{}", agent_hash))
            .await
    }

    // =========================================================================
    // PUBLIC API - My Agents (authenticated)
    // =========================================================================

    /// List my agents
    pub async fn list_my_agents(&self, request: &impl Serialize) -> Result<serde_json::Value> {
        self.post_bridge("my/agents", request).await
    }

    /// Get source code of my agent
    pub async fn get_my_agent_source(
        &self,
        agent_hash: &str,
        request: &impl Serialize,
    ) -> Result<serde_json::Value> {
        self.post_bridge(&format!("my/agents/{}/source", agent_hash), request)
            .await
    }

    // =========================================================================
    // PUBLIC API - Validator endpoints
    // =========================================================================

    /// Claim jobs for validation
    pub async fn claim_jobs(&self, request: &impl Serialize) -> Result<serde_json::Value> {
        self.post_bridge("validator/claim_jobs", request).await
    }

    /// Log a task result
    pub async fn log_task(&self, request: &impl Serialize) -> Result<serde_json::Value> {
        self.post_bridge("validator/log_task", request).await
    }

    /// Submit evaluation result
    pub async fn submit_result(&self, request: &impl Serialize) -> Result<serde_json::Value> {
        self.post_bridge("validator/submit_result", request).await
    }

    /// Get my jobs
    pub async fn get_my_jobs(&self, request: &impl Serialize) -> Result<serde_json::Value> {
        self.post_bridge("validator/my_jobs", request).await
    }

    /// Get agent evaluation status
    pub async fn get_agent_eval_status(&self, agent_hash: &str) -> Result<serde_json::Value> {
        self.get_bridge(&format!("validator/agent_status/{}", agent_hash))
            .await
    }

    // =========================================================================
    // PUBLIC API - Status
    // =========================================================================

    /// Get challenge status
    pub async fn get_status(&self) -> Result<serde_json::Value> {
        self.get_bridge("status").await
    }

    // =========================================================================
    // PUBLIC API - Network (not bridged)
    // =========================================================================

    /// Get network state
    pub async fn get_network_state(&self) -> Result<serde_json::Value> {
        self.get_network("network/state").await
    }

    // =========================================================================
    // Internal HTTP methods
    // =========================================================================

    async fn get_bridge(&self, path: &str) -> Result<serde_json::Value> {
        let url = self.bridge_url(path);
        let resp = self.client.get(&url).send().await?;
        self.handle_response(resp, &url).await
    }

    async fn post_bridge(&self, path: &str, body: &impl Serialize) -> Result<serde_json::Value> {
        let url = self.bridge_url(path);
        let resp = self.client.post(&url).json(body).send().await?;
        self.handle_response(resp, &url).await
    }

    async fn get_network(&self, path: &str) -> Result<serde_json::Value> {
        let url = self.network_url(path);
        let resp = self.client.get(&url).send().await?;
        self.handle_response(resp, &url).await
    }

    async fn handle_response(
        &self,
        resp: reqwest::Response,
        url: &str,
    ) -> Result<serde_json::Value> {
        let status = resp.status();

        if status.is_success() {
            Ok(resp.json().await?)
        } else {
            let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".into());
            Err(anyhow!(
                "Request failed: {} {} - {}",
                status.as_u16(),
                url,
                error_text
            ))
        }
    }
}
