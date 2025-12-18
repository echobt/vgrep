"""
Agent runner for Term Challenge.
"""

import sys
import json
import traceback
from .types import Request, Response
from .agent import Agent


def log(msg: str) -> None:
    """Log to stderr (stdout is reserved for protocol)."""
    print(f"[agent] {msg}", file=sys.stderr)


def run(agent: Agent) -> None:
    """
    Run an agent in the Term Challenge harness.
    
    This reads requests from stdin (line by line) and writes responses to stdout.
    The agent process stays alive between steps, preserving memory/state.
    
    Args:
        agent: Your agent instance
    
    Example:
        ```python
        from term_sdk import Agent, Request, Response, run
        
        class MyAgent(Agent):
            def solve(self, req: Request) -> Response:
                return Response.cmd("ls")
        
        if __name__ == "__main__":
            run(MyAgent())
        ```
    """
    try:
        # Setup once at start
        agent.setup()
        
        # Read requests line by line (allows persistent process)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse request
                request = Request.parse(line)
                log(f"Step {request.step}: {request.instruction[:50]}...")
                
                # Solve
                response = agent.solve(request)
                
                # Output response (single line JSON)
                print(response.to_json(), flush=True)
                
                # If task complete, we can exit
                if response.task_complete:
                    break
                    
            except json.JSONDecodeError as e:
                log(f"Invalid JSON: {e}")
                print(Response.done().to_json(), flush=True)
                break
            except Exception as e:
                log(f"Error in step: {e}")
                traceback.print_exc(file=sys.stderr)
                print(Response.done().to_json(), flush=True)
                break
        
        # Cleanup when done
        agent.cleanup()
        
    except KeyboardInterrupt:
        log("Interrupted")
        agent.cleanup()
    except Exception as e:
        log(f"Fatal error: {e}")
        traceback.print_exc(file=sys.stderr)
        print(Response.done().to_json(), flush=True)


def run_loop(agent: Agent) -> None:
    """
    Run agent in continuous loop mode (for testing).
    
    Reads multiple requests, one per line.
    """
    try:
        agent.setup()
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = Request.parse(line)
                response = agent.solve(request)
                print(response.to_json(), flush=True)
                
                if response.task_complete:
                    break
            except Exception as e:
                log(f"Error: {e}")
                print(Response.done().to_json(), flush=True)
                break
        
        agent.cleanup()
        
    except KeyboardInterrupt:
        log("Interrupted")
    except Exception as e:
        log(f"Fatal: {e}")
        traceback.print_exc(file=sys.stderr)
