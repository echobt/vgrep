# Term Challenge SDK

Build agents with streaming LLM support.

## Providers

- **OpenRouter** (default) - Access to Claude, GPT-4, Llama, Mixtral, etc.
- **Chutes** - Alternative provider

## Features

- **Streaming** - See LLM responses in real-time
- **Multi-Model** - Use different models per call
- **Function Calling** - Define custom tools
- **Early Stop** - Stop streaming based on content

## Quick Start

### Python

```python
from term_sdk import Agent, Request, Response, LLM, run

class MyAgent(Agent):
    def setup(self):
        self.llm = LLM()  # Uses OpenRouter
    
    def solve(self, req: Request) -> Response:
        # Streaming - see response in real-time
        for chunk in self.llm.stream("Solve this", model="claude-3-haiku"):
            print(chunk, end="", flush=True)
        
        # Or stream with early stop
        result = self.llm.ask_stream(
            "Write solution",
            model="gpt-4o",
            on_chunk=lambda text: "DONE" not in text
        )
        
        return Response.from_llm(result.text)

if __name__ == "__main__":
    run(MyAgent())
```

### TypeScript

```typescript
import { Agent, Request, Response, LLM, run } from 'term-sdk';

class MyAgent implements Agent {
  private llm = new LLM();

  async solve(req: Request): Promise<Response> {
    // Streaming
    for await (const chunk of this.llm.stream("Solve", { model: "claude-3-haiku" })) {
      process.stdout.write(chunk);
    }

    // Or with callback
    const result = await this.llm.askStream("Write", {
      model: "gpt-4o",
      onChunk: (text) => !text.includes("DONE")
    });

    return Response.fromLLM(result.text);
  }
}

run(new MyAgent());
```

### Rust

```rust
use term_sdk::{Agent, Request, Response, LLM, run};

struct MyAgent { llm: LLM }

impl Agent for MyAgent {
    fn solve(&mut self, req: &Request) -> Response {
        // Streaming with callback
        let result = self.llm.ask_stream(
            "Solve this",
            "claude-3-haiku",
            |chunk| {
                print!("{}", chunk);
                !chunk.contains("DONE")  // Stop on DONE
            }
        );

        match result {
            Ok(r) => Response::from_llm(&r.text),
            Err(_) => Response::done(),
        }
    }
}

fn main() {
    run(&mut MyAgent { llm: LLM::new() });
}
```

## Streaming API

### Python

```python
llm = LLM()

# Iterator - yields chunks
for chunk in llm.stream("Question", model="claude-3-haiku"):
    print(chunk, end="")

# With callback - returns full response
result = llm.ask_stream(
    "Question",
    model="claude-3-opus",
    on_chunk=lambda chunk: True  # Return False to stop
)
```

### TypeScript

```typescript
const llm = new LLM();

// Async iterator
for await (const chunk of llm.stream("Question", { model: "claude-3-haiku" })) {
  process.stdout.write(chunk);
}

// With callback
const result = await llm.askStream("Question", {
  model: "claude-3-opus",
  onChunk: (chunk) => true  // Return false to stop
});
```

### Rust

```rust
let mut llm = LLM::new();

// With callback
let result = llm.ask_stream("Question", "claude-3-haiku", |chunk| {
    print!("{}", chunk);
    true  // Return false to stop
})?;
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key (primary) |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `CHUTES_API_KEY` | Chutes API key |
| `LLM_API_URL` | Custom API endpoint |

## Models

| Model | Provider | Speed | Cost |
|-------|----------|-------|------|
| `claude-3-haiku` | OpenRouter | Fast | $ |
| `claude-3-sonnet` | OpenRouter | Medium | $$ |
| `claude-3-opus` | OpenRouter | Slow | $$$ |
| `gpt-4o` | OpenRouter | Medium | $$ |
| `gpt-4o-mini` | OpenRouter | Fast | $ |
| `llama-3-70b` | OpenRouter/Chutes | Medium | $ |
| `mixtral-8x7b` | OpenRouter/Chutes | Fast | $ |
| `qwen-72b` | Chutes | Medium | $ |

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
