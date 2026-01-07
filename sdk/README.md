# Term Challenge SDK

Build agents with streaming LLM support.

## Providers

- **OpenRouter** (default) - Claude, GPT-4, Llama, Mixtral
- **Chutes** - Llama, Qwen, Mixtral

## Features

- **Streaming** - See LLM responses in real-time
- **Multi-Model** - Use different models per call
- **Function Calling** - Define custom tools
- **Structured Errors** - JSON error responses with codes

## Quick Start

```python
from term_sdk import Agent, Request, Response, LLM, LLMError, run

class MyAgent(Agent):
    def setup(self):
        self.llm = LLM()
    
    def solve(self, req: Request) -> Response:
        try:
            full_text = ""
            for chunk in self.llm.stream(req.instruction, model="anthropic/claude-3.5-sonnet"):
                print(chunk, end="", flush=True)
                full_text += chunk
            return Response.from_llm(full_text)
        except LLMError as e:
            print(f"Error {e.code}: {e.message}")
            return Response.done()

if __name__ == "__main__":
    run(MyAgent())
```

## Streaming

```python
# Iterator
for chunk in llm.stream("Question", model="anthropic/claude-3.5-sonnet"):
    print(chunk, end="")

# With callback
result = llm.ask_stream("Question", model="anthropic/claude-3.5-sonnet", on_chunk=lambda c: True)
```

## Error Handling

The SDK returns structured JSON errors:

```json
{
  "error": {
    "code": "rate_limit",
    "message": "Rate limit exceeded",
    "details": {
      "http_status": 429,
      "model": "anthropic/claude-3.5-sonnet",
      "provider": "openrouter"
    }
  }
}
```

### Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `invalid_provider` | - | Unknown provider |
| `no_model` | - | No model specified |
| `authentication_error` | 401 | Invalid API key |
| `payment_required` | 402 | Insufficient credits |
| `permission_denied` | 403 | Access denied |
| `not_found` | 404 | Model not found |
| `rate_limit` | 429 | Rate limit exceeded |
| `server_error` | 500 | Provider error |
| `service_unavailable` | 503 | Service unavailable |
| `proxy_error` | 502 | Platform proxy error |
| `unknown_function` | - | Function not registered |

### Python Error Handling

```python
from term_sdk import LLMError

try:
    result = llm.ask("Q", model="anthropic/claude-3.5-sonnet")
except LLMError as e:
    print(e.code, e.message, e.details)
    # e.details["http_status"] contains the original HTTP status code
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key (primary) |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `CHUTES_API_KEY` | Chutes API key |
| `LLM_API_URL` | Custom API endpoint |

## Installation

```bash
pip install -e sdk/python
```

## Documentation

- [Python SDK](./python/README.md)
- [Protocol Specification](./PROTOCOL.md)
