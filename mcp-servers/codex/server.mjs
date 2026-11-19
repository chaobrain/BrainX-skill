#!/usr/bin/env node

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { AppServerClient } from './app-server.mjs';

const serverDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(serverDirectory, '../..');
const systemPrompt = readFileSync(resolve(serverDirectory, 'system-prompt.md'), 'utf8').trim();
const skillNames = JSON.parse(
  readFileSync(resolve(serverDirectory, 'skills.json'), 'utf8'),
);

function resolveSkillDirectory(name) {
  const candidates = [
    resolve(repositoryRoot, 'skills', name),
    resolve(repositoryRoot, 'skills', 'package-skills', name),
  ];
  for (const candidate of candidates) {
    try {
      readFileSync(resolve(candidate, 'SKILL.md'));
      return candidate;
    } catch {
      // Try the next supported repository layout.
    }
  }
  throw new Error(`Configured BrainX reviewer skill is missing: ${name}`);
}

const exposedSkills = skillNames.map((name) => ({
  path: resolveSkillDirectory(name),
  enabled: true,
}));

const reviewReferences = {
  training: resolve(
    repositoryRoot,
    'skills/brainx-modeling-loop/references/training-workflow.md',
  ),
  fitting: resolve(
    repositoryRoot,
    'skills/brainx-modeling-loop/references/parameter-fitting-workflow.md',
  ),
};

for (const [name, path] of Object.entries(reviewReferences)) {
  try {
    readFileSync(path);
  } catch {
    throw new Error(`Configured BrainX ${name} review reference is missing: ${path}`);
  }
}

const skillInstructions = [
  'Use only the BrainX skills listed below as domain guidance for this review.',
  'Open a skill only when its scope participates in the supplied model or evidence.',
  'Do not invoke workflow behavior from these skills; use them to verify BrainX APIs and invariants.',
  ...skillNames.map((name) => `- ${name}`),
  `TRAINING_REVIEW_REFERENCE: ${reviewReferences.training}`,
  `FITTING_REVIEW_REFERENCE: ${reviewReferences.fitting}`,
].join('\n');

const tools = [
  {
    name: 'codex', description: 'Start an independent read-only BrainX scientific review.',
    inputSchema: {
      type: 'object', required: ['prompt'], additionalProperties: false,
      properties: {
        prompt: { type: 'string' }, model: { type: 'string' }, cwd: { type: 'string' },
        config: { type: 'object', additionalProperties: true },
        'approval-policy': { type: 'string', description: 'Ignored: reviews always use never.' },
        sandbox: { type: 'string', description: 'Ignored: reviews always use read-only.' },
        'base-instructions': { type: 'string', description: 'Ignored: the BrainX reviewer contract is mandatory.' },
        'developer-instructions': { type: 'string', description: 'Ignored: BrainX skill guidance is mandatory.' },
        profile: { type: 'string', description: 'Unsupported; use model, cwd or config overrides.' },
      },
    },
  },
  {
    name: 'codex-reply', description: 'Continue a BrainX review in its native Codex thread.',
    inputSchema: {
      type: 'object', required: ['threadId', 'prompt'], additionalProperties: false,
      properties: { threadId: { type: 'string' }, prompt: { type: 'string' } },
    },
  },
].map((tool) => ({ ...tool, outputSchema: {
  type: 'object', properties: { threadId: { type: 'string' }, content: { type: 'string' } },
  required: ['threadId', 'content'],
} }));

function reviewOptions(args = {}, mcpServerNames = []) {
  const overrides = Object.fromEntries(Object.entries(args.config || {}).filter(
    ([key]) => key !== 'mcp_servers' && !key.startsWith('mcp_servers.'),
  ));
  const config = { ...overrides,
    'skills.config': exposedSkills,
    approval_policy: 'never', sandbox_mode: 'read-only',
    base_instructions: systemPrompt, developer_instructions: skillInstructions,
  };
  // Disable only registrations that actually exist. A lone enabled=false
  // override creates an invalid transport when the child has no such server.
  for (const name of mcpServerNames) config[`mcp_servers.${name}.enabled`] = false;
  return {
    ...(args.model ? { model: args.model } : {}),
    ...(args.cwd ? { cwd: args.cwd } : {}),
    approvalPolicy: 'never', sandbox: 'read-only',
    baseInstructions: systemPrompt, developerInstructions: skillInstructions, config,
  };
}

export function serve(client = new AppServerClient()) {
  let initialization;
  let initialized = false;
  const active = new Map();
  const locks = new Set();
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const write = (message) => process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', ...message })}\n`);
  const error = (id, code, message) => write({ id, error: { code, message } });

  async function review(name, args, signal) {
    if (!args || typeof args.prompt !== 'string' || !args.prompt.trim()) {
      throw new Error('prompt must be a non-empty string');
    }
    if (name === 'codex' && args.profile !== undefined) {
      throw new Error('profile is not supported; use explicit model, cwd or config overrides');
    }
    let threadId;
    if (name === 'codex') {
      const servers = args.cwd ? await client.readMcpServerNames(args.cwd) : client.mcpServerNames;
      const result = await client.request('thread/start', reviewOptions(args, servers));
      threadId = result.thread.id;
    } else {
      if (typeof args.threadId !== 'string' || !args.threadId) throw new Error('threadId is required');
      threadId = args.threadId;
    }
    if (locks.has(threadId)) throw new Error('A review is already active on this thread');
    locks.add(threadId);
    try {
      if (signal.aborted) throw new Error('Review cancelled');
      if (name === 'codex-reply') {
        const { thread } = await client.request('thread/read', { threadId, includeTurns: false });
        const servers = thread.cwd ? await client.readMcpServerNames(thread.cwd) : client.mcpServerNames;
        await client.request('thread/resume', {
          threadId, excludeTurns: true, ...reviewOptions({}, servers),
        });
      }
      const content = await client.runTurn(threadId, args.prompt, signal);
      return { content: [{ type: 'text', text: content }], structuredContent: { threadId, content } };
    } finally { locks.delete(threadId); }
  }

  async function dispatch(message, signal) {
    const { id, method, params } = message;
    if (method === 'initialize') {
      if (initialization) throw Object.assign(new Error('Already initialized'), { code: -32600 });
      initialization = client.initialize();
      await initialization;
      initialized = true;
      return {
        protocolVersion: ['2024-11-05', '2025-03-26', '2025-06-18', '2025-11-25'].includes(params?.protocolVersion)
          ? params.protocolVersion : '2024-11-05',
        capabilities: { tools: {} }, serverInfo: { name: 'brainx-codex-reviewer', version: '2.0.0' },
      };
    }
    if (method === 'ping') return {};
    if (!initialized) throw Object.assign(new Error('Not initialized'), { code: -32000 });
    if (method === 'tools/list') return { tools };
    if (method === 'tools/call') {
      if (!tools.some((tool) => tool.name === params?.name)) {
        throw Object.assign(new Error('Unknown reviewer tool'), { code: -32602 });
      }
      try { return await review(params.name, params.arguments, signal); }
      catch (failure) { return { isError: true, content: [{ type: 'text', text: failure.message }] }; }
    }
    throw Object.assign(new Error('Method not found'), { code: -32601 });
  }

  input.on('line', (line) => {
    let message;
    try { message = JSON.parse(line); }
    catch { error(null, -32700, 'Parse error'); return; }
    if (!message || Array.isArray(message) || message.jsonrpc !== '2.0'
      || typeof message.method !== 'string') {
      error(message?.id ?? null, -32600, 'Invalid request'); return;
    }
    if (message.method === 'notifications/cancelled') {
      active.get(message.params?.requestId)?.controller.abort(); return;
    }
    if (message.id === undefined) return;
    if (active.has(message.id)) { error(message.id, -32600, 'Duplicate active request id'); return; }
    const controller = new AbortController();
    const task = dispatch(message, controller.signal)
      .then((result) => write({ id: message.id, result }))
      .catch((failure) => error(message.id, failure.code || -32603, failure.message))
      .finally(() => active.delete(message.id));
    active.set(message.id, { controller, task });
  });
  input.on('close', () => {
    for (const { controller } of active.values()) controller.abort();
    Promise.allSettled([...active.values()].map(({ task }) => task)).finally(() => client.close());
  });
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, () => { client.close(); input.close(); process.stdin.pause(); });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) serve();
