"""
LLM Client for Term Challenge agents.

The provider is configured at upload time. In your agent code,
just specify the model you want to use.

Example:
    ```python
    from term_sdk import LLM
    
    # Just specify the model - provider is configured at upload
    llm = LLM(model="claude-3-haiku")
    
    # Simple question
    response = llm.ask("What is 2+2?")
    print(response.text)
    
    # With function calling
    tools = [
        Tool(
            name="search",
            description="Search for files",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}}
        )
    ]
    response = llm.ask("Find Python files", tools=tools)
    if response.function_calls:
        for call in response.function_calls:
            print(f"Call {call.name} with {call.arguments}")
    ```
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import httpx

from .types import Tool, FunctionCall


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    function_calls: List[FunctionCall] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None
    
    def json(self) -> Optional[Dict]:
        """Parse response text as JSON."""
        try:
            text = self.text.strip()
            if "```" in text:
                import re
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if match:
                    text = match.group(1)
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
        except:
            pass
        return None
    
    def has_function_calls(self) -> bool:
        """Check if response has function calls."""
        return len(self.function_calls) > 0


def _log(msg: str):
    print(f"[llm] {msg}", file=sys.stderr)


class LLM:
    """
    LLM client for inference.
    
    The provider is determined at runtime based on environment configuration.
    Just specify the model you want to use.
    
    Args:
        model: Model name (e.g., "claude-3-haiku", "gpt-4o")
        temperature: Sampling temperature (0.0 - 2.0)
        max_tokens: Maximum response tokens
        timeout: Request timeout in seconds
    
    Example:
        ```python
        llm = LLM(model="claude-3-haiku")
        
        # Simple question
        response = llm.ask("What is Python?")
        print(response.text)
        
        # With system prompt
        response = llm.ask("Write hello world", system="You are a Python expert.")
        
        # With function calling
        response = llm.ask(
            "What's the weather?",
            tools=[Tool(name="get_weather", description="Get weather", parameters={})]
        )
        
        # Chat with history
        response = llm.chat([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ])
        ```
    """
    
    def __init__(
        self,
        model: str = "claude-3-haiku",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 120,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Get API configuration from environment
        self._api_url = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
        self._api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        
        if not self._api_key:
            _log("Warning: LLM_API_KEY or OPENROUTER_API_KEY not set")
        
        # Stats
        self.total_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        
        # HTTP client
        self._client = httpx.Client(timeout=timeout)
        
        # Function handlers
        self._function_handlers: Dict[str, Callable] = {}
    
    def register_function(self, name: str, handler: Callable):
        """Register a function handler for function calling."""
        self._function_handlers[name] = handler
    
    def ask(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Ask a question.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            tools: Optional list of tools/functions
            **kwargs: Override model, temperature, max_tokens
        
        Returns:
            LLMResponse with text, function_calls, tokens, cost
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, tools=tools, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Chat with message history.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            tools: Optional list of tools/functions for function calling
            **kwargs: Override model, temperature, max_tokens
        
        Returns:
            LLMResponse with text, function_calls, tokens, cost
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        start = time.time()
        
        try:
            # Build request
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add tools if provided
            if tools:
                payload["tools"] = [t.to_dict() for t in tools]
                payload["tool_choice"] = "auto"
            
            # Make request
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            response = self._client.post(
                self._api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            text = message.get("content", "") or ""
            
            # Parse function calls
            function_calls = []
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except:
                        args = {}
                    function_calls.append(FunctionCall(
                        name=func.get("name", ""),
                        arguments=args,
                        id=tc.get("id"),
                    ))
            
            # Calculate tokens and cost
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
            latency_ms = int((time.time() - start) * 1000)
            
            # Update stats
            self.total_tokens += total_tokens
            self.total_cost += cost
            self.request_count += 1
            
            _log(f"{model}: {total_tokens} tokens, ${cost:.4f}, {latency_ms}ms")
            
            return LLMResponse(
                text=text,
                model=model,
                tokens=total_tokens,
                cost=cost,
                latency_ms=latency_ms,
                function_calls=function_calls,
                raw=data,
            )
            
        except Exception as e:
            _log(f"Error: {e}")
            raise
    
    def execute_function(self, call: FunctionCall) -> Any:
        """Execute a registered function."""
        if call.name not in self._function_handlers:
            raise ValueError(f"Unknown function: {call.name}")
        return self._function_handlers[call.name](**call.arguments)
    
    def chat_with_functions(
        self,
        messages: List[Dict[str, str]],
        tools: List[Tool],
        max_iterations: int = 10,
        **kwargs
    ) -> LLMResponse:
        """
        Chat with automatic function execution.
        
        Automatically executes function calls and continues conversation
        until the model returns a text response.
        
        Args:
            messages: Initial messages
            tools: Available tools
            max_iterations: Max function call iterations
            **kwargs: Model parameters
        
        Returns:
            Final LLMResponse
        """
        messages = list(messages)  # Copy
        
        for _ in range(max_iterations):
            response = self.chat(messages, tools=tools, **kwargs)
            
            if not response.function_calls:
                return response
            
            # Execute functions and add results
            for call in response.function_calls:
                try:
                    result = self.execute_function(call)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"Error: {e}",
                    })
        
        return response
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model pricing."""
        # Pricing per 1M tokens (input, output)
        pricing = {
            "claude-3-haiku": (0.25, 1.25),
            "claude-3-sonnet": (3.0, 15.0),
            "claude-3-opus": (15.0, 75.0),
            "claude-3.5-sonnet": (3.0, 15.0),
            "gpt-4o": (5.0, 15.0),
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-3.5-turbo": (0.5, 1.5),
        }
        
        # Find matching pricing
        input_price, output_price = 0.5, 1.5  # Default
        for key, prices in pricing.items():
            if key in model.lower():
                input_price, output_price = prices
                break
        
        return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    
    def close(self):
        """Close HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
