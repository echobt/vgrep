# Term Challenge SDK

Build agents that solve terminal tasks.

## Features

- **Simple Protocol** - Request/Response JSON communication
- **LLM Integration** - Any model, configured at upload time
- **Function Calling** - Define and execute custom functions
- **Text + Commands** - Send messages and execute commands

## Quick Start

### Python

```python
from term_sdk import Agent, Request, Response, run

class MyAgent(Agent):
    def solve(self, req: Request) -> Response:
        if req.step == 1:
            return Response.cmd("ls -la")
        return Response.done()

if __name__ == "__main__":
    run(MyAgent())
```

### TypeScript

```typescript
import { Agent, Request, Response, run } from 'term-sdk';

class MyAgent implements Agent {
  solve(req: Request): Response {
    if (req.step === 1) return Response.cmd("ls -la");
    return Response.done();
  }
}

run(new MyAgent());
```

### Rust

```rust
use term_sdk::{Agent, Request, Response, run};

struct MyAgent;

impl Agent for MyAgent {
    fn solve(&mut self, req: &Request) -> Response {
        if req.step == 1 { return Response::cmd("ls -la"); }
        Response::done()
    }
}

fn main() { run(&mut MyAgent); }
```

## LLM Integration

Use any model - the provider is configured at upload time.

### Python

```python
from term_sdk import Agent, Request, Response, LLM, run

class LLMAgent(Agent):
    def setup(self):
        # Just specify the model
        self.llm = LLM(model="claude-3-haiku")
    
    def solve(self, req: Request) -> Response:
        result = self.llm.ask(f"Task: {req.instruction}")
        return Response.from_llm(result.text)
```

### TypeScript

```typescript
import { Agent, Request, Response, LLM, run } from 'term-sdk';

class LLMAgent implements Agent {
  private llm = new LLM({ model: "claude-3-haiku" });

  async solve(req: Request): Promise<Response> {
    const result = await this.llm.ask(`Task: ${req.instruction}`);
    return Response.fromLLM(result.text);
  }
}
```

### Rust

```rust
use term_sdk::{Agent, Request, Response, LLM, run};

struct LLMAgent { llm: LLM }

impl Agent for LLMAgent {
    fn solve(&mut self, req: &Request) -> Response {
        match self.llm.ask(&format!("Task: {}", req.instruction)) {
            Ok(r) => Response::from_llm(&r.text),
            Err(_) => Response::done(),
        }
    }
}
```

## Function Calling

Define custom functions the LLM can call.

### Python

```python
from term_sdk import Agent, Request, Response, LLM, Tool, run

class ToolAgent(Agent):
    def setup(self):
        self.llm = LLM(model="claude-3-haiku")
        self.llm.register_function("search", self.search)
    
    def search(self, query: str) -> str:
        return f"Found: {query}"
    
    def solve(self, req: Request) -> Response:
        tools = [Tool(
            name="search",
            description="Search for files",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        )]
        result = self.llm.chat_with_functions(
            [{"role": "user", "content": req.instruction}],
            tools
        )
        return Response.from_llm(result.text)
```

### TypeScript

```typescript
import { Agent, Request, Response, LLM, Tool, run } from 'term-sdk';

class ToolAgent implements Agent {
  private llm = new LLM({ model: "claude-3-haiku" });

  setup() {
    this.llm.registerFunction("search", (args) => `Found: ${args.query}`);
  }

  async solve(req: Request): Promise<Response> {
    const tools = [new Tool("search", "Search for files", {
      type: "object",
      properties: { query: { type: "string" } }
    })];
    const result = await this.llm.chatWithFunctions(
      [{ role: "user", content: req.instruction }],
      tools
    );
    return Response.fromLLM(result.text);
  }
}
```

## Response Types

### Command Only
```python
Response.cmd("ls -la")
```

### Text Only
```python
Response.say("Analyzing the output...")
```

### Command + Text
```python
Response.cmd("make build").with_text("Building project...")
```

### Done with Summary
```python
Response.done("Task completed successfully!")
```

## Protocol

### Request (harness → agent)

```json
{
  "instruction": "Create hello.txt with 'Hello World'",
  "step": 2,
  "last_command": "ls -la",
  "output": "total 0\ndrwxr-xr-x...",
  "exit_code": 0,
  "cwd": "/app"
}
```

### Response (agent → harness)

```json
{
  "command": "echo 'Hello' > hello.txt",
  "text": "Creating file...",
  "task_complete": false
}
```

## Installation

### Python
```bash
pip install -e sdk/python
```

### TypeScript
```bash
cd sdk/typescript && npm install && npm run build
```

### Rust
```toml
[dependencies]
term-sdk = { path = "sdk/rust" }
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key for LLM provider |
| `LLM_API_URL` | Custom API endpoint |
| `OPENROUTER_API_KEY` | OpenRouter API key (fallback) |

The provider is configured when you upload your agent to the chain.
