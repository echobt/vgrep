#!/usr/bin/env python3
"""Tests for term_sdk."""

import json
import pytest
from term_sdk import Agent, Request, Response, run


class TestResponse:
    """Test Response class."""
    
    def test_cmd(self):
        r = Response.cmd("ls -la")
        assert r.command == "ls -la"
        assert r.task_complete is False
    
    def test_done(self):
        r = Response.done()
        assert r.command is None
        assert r.task_complete is True
    
    def test_from_llm_valid_json(self):
        """Test parsing valid JSON from LLM."""
        text = '{"command": "ls -la", "task_complete": false}'
        r = Response.from_llm(text)
        assert r.command == "ls -la"
        assert r.task_complete is False
    
    def test_from_llm_with_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        text = '''Here's the command:
```json
{"command": "cat file.txt", "task_complete": false}
```
'''
        r = Response.from_llm(text)
        assert r.command == "cat file.txt"
        assert r.task_complete is False
    
    def test_from_llm_task_complete(self):
        """Test task_complete response."""
        text = '{"command": null, "task_complete": true}'
        r = Response.from_llm(text)
        assert r.command is None
        assert r.task_complete is True
    
    def test_from_llm_empty_string(self):
        """Empty string should NOT return task_complete."""
        r = Response.from_llm("")
        assert r.task_complete is False
        assert r.command is not None  # Should return an error command
    
    def test_from_llm_invalid_json(self):
        """Invalid JSON should NOT return task_complete."""
        r = Response.from_llm("This is not JSON at all")
        assert r.task_complete is False
        assert r.command is not None  # Should return an error command
    
    def test_from_llm_partial_json(self):
        """Partial JSON should NOT return task_complete."""
        r = Response.from_llm('{"command": "ls"')  # Missing closing brace
        assert r.task_complete is False
    
    def test_from_llm_command_with_complete(self):
        """Command + task_complete should run command first."""
        text = '{"command": "echo done", "task_complete": true}'
        r = Response.from_llm(text)
        # Should NOT complete yet - run command first
        assert r.command == "echo done"
        assert r.task_complete is False


class TestRequest:
    """Test Request class."""
    
    def test_parse(self):
        data = {
            "instruction": "Create a file",
            "step": 1,
            "output": None,
        }
        req = Request.parse(data)
        assert req.instruction == "Create a file"
        assert req.step == 1
        assert req.first is True
    
    def test_first_property(self):
        req = Request.parse({"instruction": "test", "step": 1})
        assert req.first is True
        
        req = Request.parse({"instruction": "test", "step": 2})
        assert req.first is False
    
    def test_get_output_empty(self):
        req = Request.parse({"instruction": "test", "step": 1, "output": None})
        assert req.get_output() == ""
        assert req.get_output(100) == ""
    
    def test_get_output_truncate(self):
        req = Request.parse({
            "instruction": "test",
            "step": 2,
            "output": "a" * 10000
        })
        assert len(req.get_output(100)) == 100
        assert len(req.get_output(3000)) == 3000
    
    def test_has_pattern(self):
        req = Request.parse({
            "instruction": "test",
            "step": 2,
            "output": "Hello World"
        })
        assert req.has("hello") is True
        assert req.has("WORLD") is True
        assert req.has("foo") is False
    
    def test_ok_failed(self):
        req = Request.parse({"instruction": "test", "step": 2, "exit_code": 0})
        assert req.ok is True
        assert req.failed is False
        
        req = Request.parse({"instruction": "test", "step": 2, "exit_code": 1})
        assert req.ok is False
        assert req.failed is True


class TestAgent:
    """Test Agent class."""
    
    def test_simple_agent(self):
        """Test creating a simple agent."""
        class SimpleAgent(Agent):
            def solve(self, req: Request) -> Response:
                if req.first:
                    return Response.cmd("ls -la")
                return Response.done()
        
        agent = SimpleAgent()
        
        # First request
        req1 = Request.parse({"instruction": "test", "step": 1})
        resp1 = agent.solve(req1)
        assert resp1.command == "ls -la"
        assert resp1.task_complete is False
        
        # Second request
        req2 = Request.parse({"instruction": "test", "step": 2, "output": "file1\nfile2"})
        resp2 = agent.solve(req2)
        assert resp2.task_complete is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
