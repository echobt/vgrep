/**
 * Term SDK for TypeScript
 * 
 * Build agents for Term Challenge.
 * 
 * @example Quick Start
 * ```typescript
 * import { Agent, Request, Response, run } from 'term-sdk';
 * 
 * class MyAgent implements Agent {
 *   solve(req: Request): Response {
 *     if (req.step === 1) return Response.cmd("ls -la");
 *     return Response.done();
 *   }
 * }
 * 
 * run(new MyAgent());
 * ```
 * 
 * @example With LLM
 * ```typescript
 * import { Agent, Request, Response, LLM, run } from 'term-sdk';
 * 
 * class LLMAgent implements Agent {
 *   private llm = new LLM({ model: "claude-3-haiku" });
 * 
 *   async solve(req: Request): Promise<Response> {
 *     const result = await this.llm.ask(`Task: ${req.instruction}`);
 *     return Response.fromLLM(result.text);
 *   }
 * }
 * 
 * run(new LLMAgent());
 * ```
 * 
 * @example With Function Calling
 * ```typescript
 * import { Agent, Request, Response, LLM, Tool, run } from 'term-sdk';
 * 
 * class ToolAgent implements Agent {
 *   private llm = new LLM({ model: "claude-3-haiku" });
 * 
 *   setup() {
 *     this.llm.registerFunction("search", (args) => `Found: ${args.query}`);
 *   }
 * 
 *   async solve(req: Request): Promise<Response> {
 *     const tools = [new Tool("search", "Search for files", { ... })];
 *     const result = await this.llm.chatWithFunctions(
 *       [{ role: "user", content: req.instruction }],
 *       tools
 *     );
 *     return Response.fromLLM(result.text);
 *   }
 * }
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface RequestData {
  instruction: string;
  step: number;
  last_command: string | null;
  output: string | null;
  exit_code: number | null;
  cwd: string;
}

export class Request {
  readonly instruction: string;
  readonly step: number;
  readonly lastCommand: string | null;
  readonly output: string | null;
  readonly exitCode: number | null;
  readonly cwd: string;

  constructor(data: RequestData) {
    this.instruction = data.instruction;
    this.step = data.step;
    this.lastCommand = data.last_command;
    this.output = data.output;
    this.exitCode = data.exit_code;
    this.cwd = data.cwd || "/app";
  }

  static parse(json: string): Request {
    return new Request(JSON.parse(json));
  }

  get first(): boolean { return this.step === 1; }
  get ok(): boolean { return this.exitCode === 0; }
  get failed(): boolean { return this.exitCode !== null && this.exitCode !== 0; }

  has(...patterns: string[]): boolean {
    if (!this.output) return false;
    const lower = this.output.toLowerCase();
    return patterns.some(p => lower.includes(p.toLowerCase()));
  }
}

export class Response {
  command: string | null;
  text: string | null;
  taskComplete: boolean;
  data: Record<string, any> | null;

  constructor(
    command: string | null = null,
    text: string | null = null,
    taskComplete = false,
    data: Record<string, any> | null = null
  ) {
    this.command = command;
    this.text = text;
    this.taskComplete = taskComplete;
    this.data = data;
  }

  static cmd(command: string, text?: string): Response {
    return new Response(command, text ?? null, false);
  }

  static say(text: string): Response {
    return new Response(null, text, false);
  }

  static done(text?: string): Response {
    return new Response(null, text ?? null, true);
  }

  withText(text: string): Response {
    this.text = text;
    return this;
  }

  withData(data: Record<string, any>): Response {
    this.data = data;
    return this;
  }

  complete(): Response {
    this.taskComplete = true;
    return this;
  }

  toJSON(): string {
    const obj: any = {
      command: this.command,
      task_complete: this.taskComplete,
    };
    if (this.text) obj.text = this.text;
    if (this.data) obj.data = this.data;
    return JSON.stringify(obj);
  }

  static fromLLM(text: string): Response {
    text = text.trim();
    const codeMatch = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
    if (codeMatch) text = codeMatch[1];

    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');

    if (start >= 0 && end > start) {
      try {
        const data = JSON.parse(text.slice(start, end + 1));
        return new Response(
          data.command ?? null,
          data.text ?? null,
          data.task_complete ?? false,
          data.data ?? null
        );
      } catch { }
    }
    return Response.done();
  }
}

// ============================================================================
// Function Calling
// ============================================================================

export interface FunctionCall {
  name: string;
  arguments: Record<string, any>;
  id?: string;
}

export class Tool {
  name: string;
  description: string;
  parameters: Record<string, any>;

  constructor(name: string, description: string, parameters: Record<string, any> = {}) {
    this.name = name;
    this.description = description;
    this.parameters = parameters;
  }

  toJSON(): any {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: this.parameters,
      }
    };
  }
}

// ============================================================================
// Agent
// ============================================================================

export interface Agent {
  setup?(): void | Promise<void>;
  solve(request: Request): Response | Promise<Response>;
  cleanup?(): void | Promise<void>;
}

// ============================================================================
// Runner
// ============================================================================

function log(msg: string): void {
  console.error(`[agent] ${msg}`);
}

export async function run(agent: Agent): Promise<void> {
  try {
    if (agent.setup) await agent.setup();

    const input = await readStdin();
    if (!input) {
      log("No input received");
      console.log(Response.done().toJSON());
      return;
    }

    let request: Request;
    try {
      request = Request.parse(input);
    } catch (e) {
      log(`Invalid JSON: ${e}`);
      console.log(Response.done().toJSON());
      return;
    }

    log(`Step ${request.step}: ${request.instruction.slice(0, 50)}...`);
    const response = await agent.solve(request);
    console.log(response.toJSON());

    if (agent.cleanup) await agent.cleanup();
  } catch (e) {
    log(`Error: ${e}`);
    console.log(Response.done().toJSON());
  }
}

async function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', () => resolve(data.trim()));
    setTimeout(() => { if (!data) resolve(''); }, 100);
  });
}

// ============================================================================
// LLM Client
// ============================================================================

export interface LLMOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  timeout?: number;
}

export interface LLMResponse {
  text: string;
  model: string;
  tokens: number;
  cost: number;
  latencyMs: number;
  functionCalls: FunctionCall[];
  raw?: any;
}

export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: any[];
  tool_call_id?: string;
}

type FunctionHandler = (args: Record<string, any>) => any | Promise<any>;

export class LLM {
  private model: string;
  private temperature: number;
  private maxTokens: number;
  private timeout: number;
  private apiUrl: string;
  private apiKey: string;
  private functionHandlers: Map<string, FunctionHandler> = new Map();

  totalTokens = 0;
  totalCost = 0;
  requestCount = 0;

  constructor(options: LLMOptions = {}) {
    this.model = options.model || 'claude-3-haiku';
    this.temperature = options.temperature ?? 0.3;
    this.maxTokens = options.maxTokens ?? 4096;
    this.timeout = options.timeout ?? 120000;
    this.apiUrl = process.env.LLM_API_URL || 'https://openrouter.ai/api/v1/chat/completions';
    this.apiKey = process.env.LLM_API_KEY || process.env.OPENROUTER_API_KEY || '';

    if (!this.apiKey) {
      console.error('[llm] Warning: LLM_API_KEY or OPENROUTER_API_KEY not set');
    }
  }

  registerFunction(name: string, handler: FunctionHandler): void {
    this.functionHandlers.set(name, handler);
  }

  async ask(prompt: string, system?: string, tools?: Tool[]): Promise<LLMResponse> {
    const messages: Message[] = [];
    if (system) messages.push({ role: 'system', content: system });
    messages.push({ role: 'user', content: prompt });
    return this.chat(messages, tools);
  }

  async chat(messages: Message[], tools?: Tool[]): Promise<LLMResponse> {
    const start = Date.now();

    const payload: any = {
      model: this.model,
      messages,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
    };

    if (tools && tools.length > 0) {
      payload.tools = tools.map(t => t.toJSON());
      payload.tool_choice = "auto";
    }

    const response = await fetch(this.apiUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json() as any;
    const choice = data.choices?.[0] || {};
    const message = choice.message || {};

    const text = message.content || '';
    const functionCalls: FunctionCall[] = [];

    for (const tc of message.tool_calls || []) {
      if (tc.type === 'function') {
        let args = {};
        try { args = JSON.parse(tc.function?.arguments || '{}'); } catch { }
        functionCalls.push({
          name: tc.function?.name || '',
          arguments: args,
          id: tc.id,
        });
      }
    }

    const promptTokens = data.usage?.prompt_tokens || 0;
    const completionTokens = data.usage?.completion_tokens || 0;
    const tokens = promptTokens + completionTokens;
    const cost = this.calculateCost(promptTokens, completionTokens);
    const latencyMs = Date.now() - start;

    this.totalTokens += tokens;
    this.totalCost += cost;
    this.requestCount++;

    console.error(`[llm] ${this.model}: ${tokens} tokens, $${cost.toFixed(4)}, ${latencyMs}ms`);

    return { text, model: this.model, tokens, cost, latencyMs, functionCalls, raw: data };
  }

  async executeFunction(call: FunctionCall): Promise<any> {
    const handler = this.functionHandlers.get(call.name);
    if (!handler) throw new Error(`Unknown function: ${call.name}`);
    return handler(call.arguments);
  }

  async chatWithFunctions(
    messages: Message[],
    tools: Tool[],
    maxIterations = 10
  ): Promise<LLMResponse> {
    const msgs = [...messages];

    for (let i = 0; i < maxIterations; i++) {
      const response = await this.chat(msgs, tools);

      if (response.functionCalls.length === 0) {
        return response;
      }

      for (const call of response.functionCalls) {
        try {
          const result = await this.executeFunction(call);
          msgs.push({
            role: 'assistant',
            content: null,
            tool_calls: [{
              id: call.id,
              type: 'function',
              function: { name: call.name, arguments: JSON.stringify(call.arguments) }
            }]
          });
          msgs.push({
            role: 'tool',
            tool_call_id: call.id,
            content: typeof result === 'string' ? result : JSON.stringify(result),
          });
        } catch (e) {
          msgs.push({
            role: 'tool',
            tool_call_id: call.id,
            content: `Error: ${e}`,
          });
        }
      }
    }

    return this.chat(msgs, tools);
  }

  private calculateCost(promptTokens: number, completionTokens: number): number {
    const pricing: Record<string, [number, number]> = {
      'claude-3-haiku': [0.25, 1.25],
      'claude-3-sonnet': [3.0, 15.0],
      'claude-3-opus': [15.0, 75.0],
      'gpt-4o': [5.0, 15.0],
      'gpt-4o-mini': [0.15, 0.6],
    };

    let [inputPrice, outputPrice] = [0.5, 1.5];
    for (const [key, prices] of Object.entries(pricing)) {
      if (this.model.toLowerCase().includes(key)) {
        [inputPrice, outputPrice] = prices;
        break;
      }
    }

    return (promptTokens * inputPrice + completionTokens * outputPrice) / 1_000_000;
  }
}
