import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import cliInvocation from './cli-invocation.cjs';

const { codexInvocation } = cliInvocation;

// App Server owns native threads and persistence. This client translates only
// transport, turn completion and cancellation; it never reconstructs history.
export class AppServerClient {
  constructor({ binary = process.env.BRAINX_CODEX_BIN || 'codex', prefixArgs = [],
    timeoutMs = 30 * 60 * 1000 } = {}) {
    this.binary = binary;
    this.prefixArgs = prefixArgs;
    this.timeoutMs = timeoutMs;
    this.pending = new Map();
    this.turns = new Map();
    this.nextId = 1;
  }

  initialize() {
    if (!this.ready) this.ready = this.connect();
    return this.ready;
  }

  async connect() {
    const invocation = codexInvocation(this.binary, this.prefixArgs);
    this.child = spawn(invocation.binary, [...invocation.prefixArgs, 'app-server', '--listen', 'stdio://'], {
      env: process.env, stdio: ['pipe', 'pipe', 'inherit'],
    });
    this.child.on('error', (error) => this.fail(new Error(`Cannot start Codex App Server: ${error.message}`)));
    this.child.on('exit', () => this.fail(new Error('Codex App Server exited. Check that the CLI supports codex app-server.')));
    this.child.stdin.on('error', () => this.fail(new Error('Codex App Server input closed')));
    this.lines = createInterface({ input: this.child.stdout, crlfDelay: Infinity });
    this.lines.on('line', (line) => {
      try { this.receive(JSON.parse(line)); }
      catch { this.fail(new Error('Invalid Codex App Server response')); }
    });
    await this.request('initialize', {
      clientInfo: { name: 'brainx_reviewer', title: 'BrainX reviewer', version: '2.0.0' },
    }, 20_000);
    this.send({ method: 'initialized', params: {} });
    this.mcpServerNames = await this.readMcpServerNames();
  }

  async readMcpServerNames(cwd) {
    const { config } = await this.request('config/read', { cwd, includeLayers: false }, 20_000);
    return Object.keys(config.mcp_servers || {});
  }

  send(message) {
    if (this.failure) throw this.failure;
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  request(method, params, timeoutMs = this.timeoutMs) {
    if (this.failure) return Promise.reject(this.failure);
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Codex App Server timed out: ${method}`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try { this.send({ id, method, params }); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }

  receive(message) {
    if (message.method && message.id !== undefined) {
      // The reviewer never approves commands, file changes or permission
      // escalation. Unknown server requests fail explicitly instead of hanging.
      if (['item/commandExecution/requestApproval', 'item/fileChange/requestApproval'].includes(message.method)) {
        this.send({ id: message.id, result: { decision: 'decline' } });
      } else {
        this.send({ id: message.id, error: { code: -32601, message: 'Reviewer does not support interactive requests' } });
      }
      return;
    }
    if (message.id !== undefined) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(message.error.message || 'Codex App Server request failed'));
      else pending.resolve(message.result);
      return;
    }
    const turn = this.turns.get(message.params?.threadId);
    if (turn) {
      if (!turn.id) turn.buffer.push(message);
      else this.turnEvent(turn, message);
    }
  }

  turnEvent(turn, { method, params }) {
    if (params.turnId !== turn.id && params.turn?.id !== turn.id) return;
    if (method === 'item/completed' && params.item?.type === 'agentMessage'
      && params.item.phase !== 'commentary') {
      turn.messages.set(params.item.id, params.item.text);
    }
    if (method === 'turn/completed') {
      if (params.turn.status !== 'completed') {
        turn.reject(new Error(params.turn.error?.message || `Review turn ${params.turn.status}`));
      } else {
        const text = [...turn.messages.values()].join('\n\n');
        if (!text.trim()) turn.reject(new Error('Reviewer completed without a report'));
        else turn.resolve(text);
      }
    }
  }

  async runTurn(threadId, prompt, signal) {
    if (signal?.aborted) throw new Error('Review cancelled');
    if (this.turns.has(threadId)) throw new Error('A review is already active on this thread');
    const turn = { buffer: [], messages: new Map() };
    const completion = new Promise((resolve, reject) => Object.assign(turn, { resolve, reject }));
    // A process exit may reject completion while turn/start is still pending.
    completion.catch(() => {});
    this.turns.set(threadId, turn);
    const interrupt = () => {
      if (turn.id) this.request('turn/interrupt', { threadId, turnId: turn.id }, 10_000)
        .catch(() => turn.reject(new Error('Review cancellation failed')));
    };
    const abort = () => {
      interrupt();
      turn.cancelTimer = setTimeout(() => {
        this.fail(new Error('Review cancellation timed out'));
        this.child.kill();
      }, 10_000);
    };
    signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(() => {
      interrupt();
      this.fail(new Error('Review turn timed out'));
      this.child.kill();
    }, this.timeoutMs);
    try {
      const result = await this.request('turn/start', {
        threadId, input: [{ type: 'text', text: prompt, text_elements: [] }],
        approvalPolicy: 'never',
      });
      turn.id = result.turn.id;
      for (const message of turn.buffer) this.turnEvent(turn, message);
      turn.buffer = [];
      if (signal?.aborted) interrupt();
      return await completion;
    } finally {
      clearTimeout(timer);
      clearTimeout(turn.cancelTimer);
      signal?.removeEventListener('abort', abort);
      this.turns.delete(threadId);
    }
  }

  fail(error) {
    this.failure ||= error;
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(this.failure);
    }
    this.pending.clear();
    for (const turn of this.turns.values()) turn.reject(this.failure);
  }

  close() {
    this.fail(new Error('Reviewer connection closed'));
    this.lines?.close();
    const child = this.child;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    // Let the CLI exit on EOF. An npm shim can have a native child of its own.
    child.stdin.end();
    const timer = setTimeout(() => child.kill(), 2_000);
    timer.unref();
    child.once('close', () => clearTimeout(timer));
  }
}
