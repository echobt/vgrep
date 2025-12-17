//! LLM client for Term Challenge agents.
//!
//! The provider is configured at upload time. Just specify the model.
//!
//! ```rust,no_run
//! use term_sdk::{LLM, Tool};
//!
//! let mut llm = LLM::new("claude-3-haiku");
//!
//! // Simple question
//! let response = llm.ask("What is 2+2?")?;
//! println!("{}", response.text);
//!
//! // With function calling
//! let tools = vec![Tool::new("search", "Search for files")];
//! let response = llm.ask_with_tools("Find Python files", &tools)?;
//! for call in &response.function_calls {
//!     println!("Call {} with {:?}", call.name, call.arguments);
//! }
//! ```

use std::collections::HashMap;
use std::env;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use crate::types::{Tool, FunctionCall};

/// LLM response.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub text: String,
    pub model: String,
    pub tokens: u32,
    pub cost: f64,
    pub latency_ms: u64,
    pub function_calls: Vec<FunctionCall>,
    pub raw: Option<serde_json::Value>,
}

impl LLMResponse {
    pub fn has_function_calls(&self) -> bool {
        !self.function_calls.is_empty()
    }
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".to_string(), content: Some(content.into()), tool_calls: None, tool_call_id: None }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: Some(content.into()), tool_calls: None, tool_call_id: None }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: Some(content.into()), tool_calls: None, tool_call_id: None }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: "tool".to_string(), content: Some(content.into()), tool_calls: None, tool_call_id: Some(tool_call_id.into()) }
    }
}

pub type FunctionHandler = Box<dyn Fn(&HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync>;

/// LLM client.
pub struct LLM {
    model: String,
    temperature: f32,
    max_tokens: u32,
    api_url: String,
    api_key: String,
    client: reqwest::blocking::Client,
    function_handlers: HashMap<String, FunctionHandler>,

    pub total_tokens: u32,
    pub total_cost: f64,
    pub request_count: u32,
}

impl LLM {
    /// Create new LLM client.
    pub fn new(model: impl Into<String>) -> Self {
        let api_url = env::var("LLM_API_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1/chat/completions".to_string());
        let api_key = env::var("LLM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .unwrap_or_default();

        if api_key.is_empty() {
            eprintln!("[llm] Warning: LLM_API_KEY or OPENROUTER_API_KEY not set");
        }

        Self {
            model: model.into(),
            temperature: 0.3,
            max_tokens: 4096,
            api_url,
            api_key,
            client: reqwest::blocking::Client::new(),
            function_handlers: HashMap::new(),
            total_tokens: 0,
            total_cost: 0.0,
            request_count: 0,
        }
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn max_tokens(mut self, t: u32) -> Self {
        self.max_tokens = t;
        self
    }

    /// Register a function handler.
    pub fn register_function<F>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(&HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync + 'static,
    {
        self.function_handlers.insert(name.into(), Box::new(handler));
    }

    /// Ask a simple question.
    pub fn ask(&mut self, prompt: &str) -> Result<LLMResponse, String> {
        self.chat(&[Message::user(prompt)], None)
    }

    /// Ask with system prompt.
    pub fn ask_with_system(&mut self, system: &str, prompt: &str) -> Result<LLMResponse, String> {
        self.chat(&[Message::system(system), Message::user(prompt)], None)
    }

    /// Ask with tools.
    pub fn ask_with_tools(&mut self, prompt: &str, tools: &[Tool]) -> Result<LLMResponse, String> {
        self.chat(&[Message::user(prompt)], Some(tools))
    }

    /// Chat with messages.
    pub fn chat(&mut self, messages: &[Message], tools: Option<&[Tool]>) -> Result<LLMResponse, String> {
        let start = Instant::now();

        #[derive(Serialize)]
        struct ChatRequest<'a> {
            model: &'a str,
            messages: &'a [Message],
            temperature: f32,
            max_tokens: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            tools: Option<Vec<serde_json::Value>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool_choice: Option<&'static str>,
        }

        let tools_json = tools.map(|t| t.iter().map(|tool| tool.to_json()).collect());

        let request = ChatRequest {
            model: &self.model,
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            tools: tools_json,
            tool_choice: if tools.is_some() { Some("auto") } else { None },
        };

        let response = self.client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            return Err(format!("API error: {}", response.status()));
        }

        let data: serde_json::Value = response.json().map_err(|e| e.to_string())?;

        let choice = data.get("choices").and_then(|c| c.get(0)).unwrap_or(&serde_json::Value::Null);
        let message = choice.get("message").unwrap_or(&serde_json::Value::Null);

        let text = message.get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        // Parse function calls
        let mut function_calls = Vec::new();
        if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                if tc.get("type").and_then(|t| t.as_str()) == Some("function") {
                    if let Some(func) = tc.get("function") {
                        let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let args_str = func.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
                        let arguments: HashMap<String, serde_json::Value> = 
                            serde_json::from_str(args_str).unwrap_or_default();
                        let id = tc.get("id").and_then(|i| i.as_str()).map(String::from);
                        function_calls.push(FunctionCall { name, arguments, id });
                    }
                }
            }
        }

        let usage = data.get("usage").unwrap_or(&serde_json::Value::Null);
        let prompt_tokens = usage.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32;
        let completion_tokens = usage.get("completion_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32;
        let tokens = prompt_tokens + completion_tokens;
        let cost = self.calculate_cost(prompt_tokens, completion_tokens);
        let latency_ms = start.elapsed().as_millis() as u64;

        self.total_tokens += tokens;
        self.total_cost += cost;
        self.request_count += 1;

        eprintln!("[llm] {}: {} tokens, ${:.4}, {}ms", self.model, tokens, cost, latency_ms);

        Ok(LLMResponse {
            text,
            model: self.model.clone(),
            tokens,
            cost,
            latency_ms,
            function_calls,
            raw: Some(data),
        })
    }

    /// Execute a registered function.
    pub fn execute_function(&self, call: &FunctionCall) -> Result<String, String> {
        let handler = self.function_handlers.get(&call.name)
            .ok_or_else(|| format!("Unknown function: {}", call.name))?;
        handler(&call.arguments)
    }

    /// Chat with automatic function execution.
    pub fn chat_with_functions(
        &mut self,
        messages: &[Message],
        tools: &[Tool],
        max_iterations: usize,
    ) -> Result<LLMResponse, String> {
        let mut msgs = messages.to_vec();

        for _ in 0..max_iterations {
            let response = self.chat(&msgs, Some(tools))?;

            if response.function_calls.is_empty() {
                return Ok(response);
            }

            for call in &response.function_calls {
                let result = match self.execute_function(call) {
                    Ok(r) => r,
                    Err(e) => format!("Error: {}", e),
                };

                // Add assistant message with tool call
                msgs.push(Message {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![serde_json::json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).unwrap_or_default(),
                        }
                    })]),
                    tool_call_id: None,
                });

                // Add tool result
                msgs.push(Message::tool(call.id.as_deref().unwrap_or(""), result));
            }
        }

        self.chat(&msgs, Some(tools))
    }

    fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        let (input_price, output_price) = if self.model.contains("claude-3-haiku") {
            (0.25, 1.25)
        } else if self.model.contains("claude-3-sonnet") {
            (3.0, 15.0)
        } else if self.model.contains("claude-3-opus") {
            (15.0, 75.0)
        } else if self.model.contains("gpt-4o-mini") {
            (0.15, 0.6)
        } else if self.model.contains("gpt-4o") {
            (5.0, 15.0)
        } else {
            (0.5, 1.5)
        };

        (prompt_tokens as f64 * input_price + completion_tokens as f64 * output_price) / 1_000_000.0
    }
}
