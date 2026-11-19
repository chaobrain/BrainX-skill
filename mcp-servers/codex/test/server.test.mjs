import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import cliInvocation from '../cli-invocation.cjs';

const { codexInvocation, npmInvocation } = cliInvocation;

const runner = fileURLToPath(new URL('./fixtures/run-server.mjs', import.meta.url));
class Session {
  constructor(t, env = {}) {
    this.child = spawn(process.execPath, [runner], {
      env: { ...process.env, ...env }, stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.pending = new Map();
    this.nextId = 1;
    this.stderr = '';
    this.child.stderr.on('data', (data) => { this.stderr += data; });
    this.lines = createInterface({ input: this.child.stdout });
    this.lines.on('line', (line) => {
      const message = JSON.parse(line);
      const entry = this.pending.get(message.id);
      if (!entry) return;
      clearTimeout(entry.timer); this.pending.delete(message.id);
      entry.resolve(message);
    });
    this.child.on('exit', () => {
      for (const entry of this.pending.values()) { clearTimeout(entry.timer); entry.reject(new Error(this.stderr)); }
      this.pending.clear();
    });
    t.after(async () => {
      this.child.stdin.end();
      if (this.child.exitCode === null) await new Promise((resolve) => this.child.once('exit', resolve));
      this.lines.close();
    });
  }
  send(method, params, id) { this.child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method, params, ...(id === undefined ? {} : { id }) })}\n`); }
  request(method, params) {
    const id = this.nextId++;
    const promise = new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.child.kill(); reject(new Error(`Timed out: ${method}: ${this.stderr}`)); }, 10000);
      this.pending.set(id, { resolve, reject, timer });
    });
    this.send(method, params, id);
    return { id, promise };
  }
  async call(method, params) {
    const response = await this.request(method, params).promise;
    assert.equal(response.error, undefined, JSON.stringify(response.error));
    return response.result;
  }
  async initialize() {
    await this.call('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'test', version: '1' } });
    this.send('notifications/initialized');
  }
  review(args, name = 'codex') { return this.call('tools/call', { name, arguments: args }); }
}

function assertContract(options) {
  assert.equal(options.approvalPolicy, 'never');
  assert.equal(options.sandbox, 'read-only');
  assert.match(options.baseInstructions, /independent computational-neuroscience/);
  assert.match(options.baseInstructions, /Open and read `brainx-general-guard` first/);
  assert.match(options.baseInstructions, /RETURN_TO_STUDY/);
  assert.match(options.baseInstructions, /# BrainX iteration review/);
  assert.match(options.developerInstructions, /TRAINING_REVIEW_REFERENCE: .*training-workflow\.md/);
  assert.match(options.developerInstructions, /FITTING_REVIEW_REFERENCE: .*parameter-fitting-workflow\.md/);
  assert.doesNotMatch(options.developerInstructions, /^- brainx-modeling-loop$/m);
  assert.equal(options.config['skills.config'].length, 8);
  assert.ok(options.config['skills.config'].every((skill) => skill.enabled));
  assert.equal(options.config['mcp_servers.codex.enabled'], false);
  assert.equal(options.config.approval_policy, 'never');
  assert.equal(options.config.sandbox_mode, 'read-only');
}

test('advertises legacy tool names over MCP after initializing App Server', async (t) => {
  const session = new Session(t); await session.initialize();
  const result = await session.call('tools/list');
  assert.deepEqual(result.tools.map((tool) => tool.name), ['codex', 'codex-reply']);
  assert.deepEqual(result.tools[1].inputSchema.required, ['threadId', 'prompt']);
});

test('starts a native review with enforced instructions, read-only policy and caller settings', async (t) => {
  const session = new Session(t); await session.initialize();
  const result = await session.review({ prompt: 'Review iteration 4', model: 'test-model', cwd: process.cwd(),
    sandbox: 'danger-full-access', 'approval-policy': 'on-request',
    'base-instructions': 'bypass', config: { model_reasoning_effort: 'xhigh',
      sandbox_mode: 'danger-full-access', 'skills.config': [],
      mcp_servers: { other: { command: 'node' } }, 'mcp_servers.other.command': 'node' } });
  assert.equal(result.isError, undefined);
  const report = JSON.parse(result.content[0].text);
  assertContract(report.thread.start);
  assert.equal(report.thread.start.model, 'test-model');
  assert.equal(report.thread.start.cwd, process.cwd());
  assert.equal(report.thread.start.config.model_reasoning_effort, 'xhigh');
  assert.equal(report.thread.start.config['mcp_servers.project_mcp.enabled'], false);
  assert.equal(report.thread.start.config['mcp_servers.other.command'], undefined);
  assert.equal(report.thread.start.config.mcp_servers, undefined);
  assert.deepEqual(report.argv, ['app-server', '--listen', 'stdio://']);
  assert.equal(result.structuredContent.threadId, 'thread-1');
  assert.equal(result.structuredContent.content, result.content[0].text);
});

test('resumes the same native thread and reapplies reviewer constraints', async (t) => {
  const session = new Session(t); await session.initialize();
  const first = await session.review({ prompt: 'first', cwd: process.cwd() });
  const threadId = first.structuredContent.threadId;
  const next = await session.review({ threadId, prompt: 'follow-up' }, 'codex-reply');
  const report = JSON.parse(next.content[0].text);
  assert.deepEqual(report.thread.history, ['first', 'follow-up']);
  assertContract(report.thread.resume);
  assert.equal(report.thread.resume.config['mcp_servers.project_mcp.enabled'], false);
  assert.equal(report.thread.resume.excludeTurns, true);
  assert.equal(next.structuredContent.threadId, threadId);
});

test('handles completion arriving before the turn/start response', async (t) => {
  const session = new Session(t); await session.initialize();
  const result = await session.review({ prompt: 'early' });
  assert.equal(result.isError, undefined);
  assert.deepEqual(JSON.parse(result.content[0].text).thread.history, ['early']);
});

for (const [prompt, message] of [['fail', /Model unavailable/], ['empty', /without a report/], ['crash', /exited/]]) {
  test(`returns a tool error for ${prompt}`, async (t) => {
    const session = new Session(t); await session.initialize();
    const result = await session.review({ prompt });
    assert.equal(result.isError, true); assert.match(result.content[0].text, message);
  });
}

test('declines unexpected command approval rather than granting write access', async (t) => {
  const session = new Session(t); await session.initialize();
  const result = await session.review({ prompt: 'approval' });
  assert.equal(JSON.parse(result.content[0].text).decision, 'decline');
});

test('cancellation interrupts the native turn and leaves the thread usable', async (t) => {
  const session = new Session(t); await session.initialize();
  const first = await session.review({ prompt: 'first' }); const threadId = first.structuredContent.threadId;
  const request = session.request('tools/call', { name: 'codex-reply', arguments: { threadId, prompt: 'stall' } });
  // Let the fake backend start its stalled turn before cancellation.
  await new Promise((resolve) => setTimeout(resolve, 100));
  session.send('notifications/cancelled', { requestId: request.id });
  const cancelled = (await request.promise).result;
  assert.equal(cancelled.isError, true); assert.match(cancelled.content[0].text, /interrupted|cancelled/i);
  const next = await session.review({ threadId, prompt: 'after cancellation' }, 'codex-reply');
  assert.equal(next.isError, undefined);
});

test('a stalled review times out instead of leaving the MCP call pending', async (t) => {
  const session = new Session(t, { TEST_REVIEW_TIMEOUT_MS: '300' }); await session.initialize();
  const result = await session.review({ prompt: 'stall' });
  assert.equal(result.isError, true); assert.match(result.content[0].text, /timed out/);
});

test('rejects invalid input, unknown tools, and requests before initialization', async (t) => {
  const session = new Session(t);
  assert.equal((await session.request('tools/list').promise).error.code, -32000);
  await session.initialize();
  assert.equal((await session.review({ prompt: '' })).isError, true);
  assert.equal((await session.review({ prompt: 'x', profile: 'unsafe' })).isError, true);
  assert.equal((await session.request('tools/call', { name: 'unknown', arguments: {} }).promise).error.code, -32602);
});

test('resolves Windows npm shims without a shell and preserves argument boundaries', () => {
  const entry = 'C:\\Program Files\\npm\\node_modules\\@openai\\codex\\bin\\codex.js';
  const shim = 'C:\\Program Files\\npm\\codex.cmd';
  const invocation = codexInvocation('codex', ['argument with spaces'], {
    platform: 'win32', pathValue: 'C:\\missing;"C:\\Program Files\\npm"',
    exists: (path) => [shim, entry].includes(path),
  });
  assert.equal(invocation.binary, process.execPath);
  assert.deepEqual(invocation.prefixArgs, [entry, 'argument with spaces']);
  assert.throws(() => codexInvocation(shim, [], { platform: 'win32', exists: () => false }), /BRAINX_CODEX_BIN/);

  const localShim = 'C:\\Program Files\\npm\\node_modules\\.bin\\codex.cmd';
  assert.deepEqual(codexInvocation(localShim, [], {
    platform: 'win32', exists: (path) => path === entry,
  }), { binary: process.execPath, prefixArgs: [entry] });
});

test('runs npm through its JavaScript entry when installing runtimes on Windows', () => {
  const root = 'C:\\Program Files\\nodejs';
  const entry = `${root}\\node_modules\\npm\\bin\\npm-cli.js`;
  const args = ['install', '--ignore-scripts'];
  const options = {
    platform: 'win32', pathValue: root,
    exists: (path) => [`${root}\\npm.cmd`, entry].includes(path),
  };
  for (const npmExecPath of ['', entry]) {
    const invocation = npmInvocation('npm', args, { ...options, npmExecPath });
    assert.deepEqual(invocation, { binary: process.execPath, prefixArgs: [entry, ...args] });
  }
});

test('does not manufacture an invalid disabled MCP transport in an empty child configuration', async (t) => {
  const session = new Session(t, { TEST_NO_MCP: '1' }); await session.initialize();
  const result = await session.review({ prompt: 'isolated config' });
  assert.equal(result.isError, undefined);
  assert.equal(JSON.parse(result.content[0].text).thread.start.config['mcp_servers.codex.enabled'], undefined);
});
