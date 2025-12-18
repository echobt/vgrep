//! Agent runner for Term Challenge.

use std::io::{self, BufRead, Write};
use crate::{Agent, Request, Response};

/// Log to stderr.
fn log(msg: &str) {
    eprintln!("[agent] {}", msg);
}

/// Run an agent in the Term Challenge harness.
///
/// Reads requests from stdin (line by line), calls agent.solve(), writes response to stdout.
/// The agent process stays alive between steps, preserving memory/state.
///
/// ```rust,no_run
/// use term_sdk::{Agent, Request, Response, run};
///
/// struct MyAgent;
///
/// impl Agent for MyAgent {
///     fn solve(&mut self, req: &Request) -> Response {
///         Response::cmd("ls")
///     }
/// }
///
/// fn main() {
///     run(&mut MyAgent);
/// }
/// ```
pub fn run(agent: &mut impl Agent) {
    // Setup once at start
    agent.setup();
    
    // Read requests line by line (allows persistent process)
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        // Parse request
        let request = match Request::parse(line) {
            Ok(req) => req,
            Err(e) => {
                log(&format!("Invalid JSON: {}", e));
                println!("{}", Response::done().to_json());
                io::stdout().flush().ok();
                break;
            }
        };
        
        log(&format!("Step {}: {}...", request.step, &request.instruction.chars().take(50).collect::<String>()));
        
        // Solve
        let response = agent.solve(&request);
        
        // Output (single line JSON)
        println!("{}", response.to_json());
        io::stdout().flush().ok();
        
        // If task complete, exit
        if response.task_complete {
            break;
        }
    }
    
    // Cleanup when done
    agent.cleanup();
}

/// Run agent in loop mode (for testing) - alias for run().
pub fn run_loop(agent: &mut impl Agent) {
    run(agent);
}
