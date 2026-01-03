//! In-Container Agent Execution
//!
//! Runs agent code INSIDE the task container (like Harbor).
//! The agent uses term-sdk and LLM calls go through platform-server bridge.
//!
//! Architecture:
//! ```text
//! Task Container
//! ├── Task environment (Dockerfile from task)
//! ├── Agent code (copied at runtime)
//! ├── term-sdk (pre-installed)
//! └── LLM calls → Platform-Server Bridge → Provider
//! ```
//!
//! Environment variables injected:
//! - LLM_API_URL: Platform-server bridge endpoint
//! - LLM_API_KEY: Agent's API key (from submission)
//! - LLM_PROVIDER: Provider name (openrouter, chutes, etc.)
//! - TERM_AGENT_HASH: Agent hash for tracking
//! - TERM_PLATFORM_URL: Platform server URL

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::environment::DockerEnvironment;
use super::runner::Agent;
use super::session::{AgentResponse, CommandSpec, TmuxSession};

/// Agent that runs inside the task container
pub struct InContainerAgent {
    /// Agent source code
    source_code: String,
    /// Agent name
    name: String,
    /// API key for LLM calls (optional)
    api_key: Option<String>,
    /// API provider (openrouter, chutes, etc.)
    api_provider: String,
    /// Platform server URL for LLM bridge
    platform_url: String,
    /// Agent hash for tracking
    agent_hash: String,
    /// Validator hotkey for audit
    validator_hotkey: String,
    /// Cost limit per validator (USD)
    cost_limit_usd: f64,
    /// Whether agent has been installed
    installed: bool,
}

impl InContainerAgent {
    /// Create new in-container agent
    pub fn new(
        source_code: String,
        name: String,
        agent_hash: String,
        platform_url: String,
        validator_hotkey: String,
    ) -> Self {
        Self {
            source_code,
            name,
            api_key: None,
            api_provider: "openrouter".to_string(),
            platform_url,
            agent_hash,
            validator_hotkey,
            cost_limit_usd: 10.0,
            installed: false,
        }
    }

    /// Set API key for LLM calls
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Set API provider
    pub fn with_provider(mut self, provider: Option<String>) -> Self {
        self.api_provider = provider.unwrap_or_else(|| "openrouter".to_string());
        self
    }

    /// Set cost limit
    pub fn with_cost_limit(mut self, limit: f64) -> Self {
        self.cost_limit_usd = limit;
        self
    }

    /// Install the agent in the container
    async fn install_agent(&mut self, env: &DockerEnvironment) -> Result<()> {
        if self.installed {
            return Ok(());
        }

        info!("Installing agent '{}' in container", self.name);

        // Create agent directory
        env.exec(&["mkdir", "-p", "/agent"]).await?;

        // Write agent source code
        let escaped_code = self.source_code.replace("'", "'\"'\"'");
        let write_cmd = format!(
            "cat > /agent/agent.py << 'AGENT_EOF'\n{}\nAGENT_EOF",
            self.source_code
        );
        env.exec_shell(&write_cmd).await?;

        // Write runner script that uses term-sdk
        let runner_script = self.generate_runner_script();
        let escaped_runner = runner_script.replace("'", "'\"'\"'");
        let write_runner = format!(
            "cat > /agent/run.py << 'RUNNER_EOF'\n{}\nRUNNER_EOF",
            runner_script
        );
        env.exec_shell(&write_runner).await?;

        // Make executable
        env.exec(&["chmod", "+x", "/agent/run.py"]).await?;

        // Install term-sdk if not present
        let check_sdk = env.exec(&["python3", "-c", "import term_sdk"]).await;
        if check_sdk.is_err() {
            info!("Installing term-sdk...");
            env.exec(&["pip3", "install", "--quiet", "term-sdk"])
                .await
                .context("Failed to install term-sdk")?;
        }

        self.installed = true;
        info!("Agent installed successfully");

        Ok(())
    }

    /// Generate the runner script that wraps the agent with term-sdk
    fn generate_runner_script(&self) -> String {
        r#"#!/usr/bin/env python3
"""
Agent runner - wraps user agent with term-sdk HTTP server for step execution.
"""
import os
import sys
import json
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add agent directory to path
sys.path.insert(0, '/agent')

# Import term-sdk
try:
    from term_sdk import Request, Response
except ImportError:
    print("ERROR: term-sdk not installed", file=sys.stderr)
    sys.exit(1)

# Import user's agent
try:
    from agent import agent_instance, MyAgent
except ImportError:
    # Try to find Agent class
    import agent as user_agent
    agent_classes = [
        obj for name, obj in vars(user_agent).items()
        if isinstance(obj, type) and hasattr(obj, 'solve') and name != 'Agent'
    ]
    if agent_classes:
        agent_instance = agent_classes[0]()
    else:
        print("ERROR: No agent class found in agent.py", file=sys.stderr)
        print("Define a class with solve(self, request) method", file=sys.stderr)
        sys.exit(1)

# Setup agent
try:
    if hasattr(agent_instance, 'setup'):
        agent_instance.setup()
except Exception as e:
    print(f"WARNING: Agent setup failed: {e}", file=sys.stderr)

class AgentHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/step':
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            # Create Request object
            req = Request(
                instruction=data.get('instruction', ''),
                step=data.get('step', 1),
                output=data.get('output'),
                exit_code=data.get('exit_code'),
                cwd=data.get('cwd', '/app'),
            )

            # Call agent's solve method
            response = agent_instance.solve(req)

            # Convert to JSON
            result = {
                'command': response.command,
                'task_complete': response.task_complete,
                'message': response.message,
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            error_msg = f"Agent error: {e}\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

if __name__ == '__main__':
    port = int(os.environ.get('AGENT_PORT', '8765'))
    server = HTTPServer(('0.0.0.0', port), AgentHandler)
    print(f"Agent server listening on port {port}", file=sys.stderr)
    server.serve_forever()
"#
        .to_string()
    }

    /// Get environment variables for the agent
    fn get_env_vars(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        // Platform LLM bridge configuration
        env.insert(
            "LLM_API_URL".to_string(),
            format!("{}/api/v1/llm/chat", self.platform_url),
        );

        if let Some(ref key) = self.api_key {
            env.insert("LLM_API_KEY".to_string(), key.clone());
        }

        env.insert("LLM_PROVIDER".to_string(), self.api_provider.clone());
        env.insert("TERM_AGENT_HASH".to_string(), self.agent_hash.clone());
        env.insert("TERM_PLATFORM_URL".to_string(), self.platform_url.clone());
        env.insert(
            "TERM_VALIDATOR_HOTKEY".to_string(),
            self.validator_hotkey.clone(),
        );
        env.insert(
            "TERM_COST_LIMIT_USD".to_string(),
            self.cost_limit_usd.to_string(),
        );
        env.insert("AGENT_PORT".to_string(), "8765".to_string());

        env
    }

    /// Start the agent server in the container
    async fn start_agent_server(&self, env: &DockerEnvironment) -> Result<()> {
        info!("Starting agent server in container");

        // Set environment variables and start server in background
        let env_vars: Vec<String> = self
            .get_env_vars()
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();

        let env_str = env_vars.join(" ");
        let start_cmd = format!(
            "cd /agent && {} python3 run.py > /agent/server.log 2>&1 &",
            env_str
        );

        env.exec_shell(&start_cmd).await?;

        // Wait for server to be ready
        for i in 0..30 {
            tokio::time::sleep(Duration::from_millis(200)).await;

            let check = env
                .exec(&["curl", "-s", "http://localhost:8765/health"])
                .await;
            if check.is_ok() {
                info!("Agent server ready after {}ms", (i + 1) * 200);
                return Ok(());
            }
        }

        // Get logs for debugging
        let logs = env.exec(&["cat", "/agent/server.log"]).await.ok();
        let log_content = logs
            .as_ref()
            .map(|l| l.stdout.as_str())
            .unwrap_or("No logs");

        bail!("Agent server failed to start. Logs:\n{}", log_content);
    }

    /// Send a step request to the agent server
    async fn send_step(
        &self,
        env: &DockerEnvironment,
        instruction: &str,
        output: Option<&str>,
        exit_code: Option<i32>,
        step: u32,
    ) -> Result<AgentStepResponse> {
        let request = serde_json::json!({
            "instruction": instruction,
            "step": step,
            "output": output,
            "exit_code": exit_code,
            "cwd": "/app",
        });

        let request_json = serde_json::to_string(&request)?;
        let escaped_json = request_json.replace("'", "'\"'\"'");

        let curl_cmd = format!(
            "curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://localhost:8765/step",
            escaped_json
        );

        let result = env
            .exec_shell(&curl_cmd)
            .await
            .context("Failed to send step to agent")?;

        let response: AgentStepResponse = serde_json::from_str(&result.stdout)
            .context(format!("Failed to parse agent response: {}", result.stdout))?;

        Ok(response)
    }
}

#[derive(Debug, Deserialize)]
struct AgentStepResponse {
    command: Option<String>,
    task_complete: bool,
    message: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[async_trait::async_trait]
impl Agent for InContainerAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn setup(&self, session: &TmuxSession) -> Result<()> {
        // Get mutable reference through interior mutability pattern
        // Note: In practice, we'd need to restructure this
        // For now, installation happens in step() if needed
        Ok(())
    }

    async fn step(&self, instruction: &str, screen: &str, step: u32) -> Result<AgentResponse> {
        // Parse previous output from screen
        let (output, exit_code) = if step > 1 && !screen.is_empty() {
            // Extract exit code if present
            let exit_code = if screen.contains("[exit code: ") {
                screen
                    .lines()
                    .find(|l| l.contains("[exit code: "))
                    .and_then(|l| {
                        l.trim_start_matches("[exit code: ")
                            .trim_end_matches(']')
                            .parse()
                            .ok()
                    })
            } else {
                Some(0)
            };
            (Some(screen), exit_code)
        } else {
            (None, None)
        };

        // This is a simplified version - in production, we'd need to:
        // 1. Get the DockerEnvironment from the session
        // 2. Install agent if not done
        // 3. Start server if not running
        // 4. Send step request

        // For now, return a placeholder that indicates we need the full integration
        bail!("InContainerAgent requires full integration with DockerEnvironment. Use InContainerRunner instead.");
    }
}

/// Runner that executes agent inside the task container
pub struct InContainerRunner {
    agent: InContainerAgent,
}

impl InContainerRunner {
    pub fn new(agent: InContainerAgent) -> Self {
        Self { agent }
    }

    /// Run the agent in the container
    pub async fn run(
        &mut self,
        env: &DockerEnvironment,
        instruction: &str,
        max_steps: u32,
        timeout_secs: u64,
    ) -> Result<InContainerResult> {
        // Install agent
        self.agent.install_agent(env).await?;

        // Start agent server
        self.agent.start_agent_server(env).await?;

        let mut steps = 0u32;
        let mut last_output: Option<String> = None;
        let mut last_exit_code: Option<i32> = None;
        let mut task_complete = false;
        let mut commands_executed: Vec<String> = vec![];

        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);

        while steps < max_steps && !task_complete {
            if start.elapsed() > timeout {
                warn!("Agent timeout after {} steps", steps);
                break;
            }

            steps += 1;
            debug!("Step {}", steps);

            // Send step to agent
            let response = self
                .agent
                .send_step(
                    env,
                    instruction,
                    last_output.as_deref(),
                    last_exit_code,
                    steps,
                )
                .await?;

            if let Some(ref error) = response.error {
                error!("Agent error: {}", error);
                break;
            }

            task_complete = response.task_complete;

            // Execute command if provided
            if let Some(ref cmd) = response.command {
                if !cmd.is_empty() {
                    info!(
                        ">>> [{}] $ {}",
                        steps,
                        cmd.chars().take(100).collect::<String>()
                    );
                    commands_executed.push(cmd.clone());

                    let exec_result = env.exec_shell(cmd).await;
                    match exec_result {
                        Ok(result) => {
                            last_output = Some(format!(
                                "$ {}\n{}{}",
                                cmd,
                                result.stdout,
                                if !result.stderr.is_empty() {
                                    format!("\nSTDERR: {}", result.stderr)
                                } else {
                                    String::new()
                                }
                            ));
                            last_exit_code = result.exit_code.map(|c| c as i32);
                        }
                        Err(e) => {
                            last_output = Some(format!("$ {}\nError: {}", cmd, e));
                            last_exit_code = Some(1);
                        }
                    }
                }
            }
        }

        Ok(InContainerResult {
            steps,
            task_complete,
            commands_executed,
            duration_secs: start.elapsed().as_secs_f64(),
        })
    }
}

#[derive(Debug)]
pub struct InContainerResult {
    pub steps: u32,
    pub task_complete: bool,
    pub commands_executed: Vec<String>,
    pub duration_secs: f64,
}
