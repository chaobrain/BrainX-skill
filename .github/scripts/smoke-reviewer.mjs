import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const server = process.argv[2] ? resolve(process.argv[2])
  : fileURLToPath(new URL('../../mcp-servers/codex/server.mjs', import.meta.url));
const codexHome = await mkdtemp(join(tmpdir(), 'brainx-reviewer-smoke-'));
const requests = [];
const responses = createServer(async (req, res) => {
  if (req.method !== 'POST' || req.url !== '/v1/responses') {
    res.writeHead(404).end();
    return;
  }
  let body = '';
  for await (const chunk of req) body += chunk;
  requests.push(body);
  const n = requests.length;
  const item = {
    id: `msg_${n}`, type: 'message', status: 'completed', role: 'assistant',
    content: [{ type: 'output_text', text: `# Test review ${n}`, annotations: [] }],
  };
  const response = { id: `resp_${n}`, status: 'completed', output: [item] };
  const events = [
    { type: 'response.created', response: { ...response, status: 'in_progress', output: [] } },
    { type: 'response.output_item.done', output_index: 0, item },
    { type: 'response.completed', response },
  ];
  res.writeHead(200, { 'Content-Type': 'text/event-stream' });
  res.end(events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''));
});

async function connect() {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [server],
    cwd: codexHome,
    env: {
      CODEX_HOME: codexHome,
      BRAINX_SMOKE_KEY: 'local-test-key',
      ...(process.env.BRAINX_CODEX_BIN ? { BRAINX_CODEX_BIN: process.env.BRAINX_CODEX_BIN } : {}),
    },
  });
  const client = new Client({ name: 'brainx-reviewer-smoke', version: '1.0.0' });
  try {
    await client.connect(transport, { timeout: 50_000 });
    assert.equal(client.getServerVersion()?.name, 'brainx-codex-reviewer');
    const { tools } = await client.listTools();
    assert.deepEqual(tools.map(({ name }) => name), ['codex', 'codex-reply']);
    await client.ping();
    return client;
  } catch (error) {
    await client.close();
    throw error;
  }
}

async function review(name, args, expected) {
  const client = await connect();
  try {
    const result = await client.callTool({ name, arguments: args }, undefined, { timeout: 45_000 });
    assert.equal(result.isError, undefined, JSON.stringify(result.content));
    assert.equal(result.content[0].text, expected);
    assert.equal(result.structuredContent?.content, expected);
    assert.ok(result.structuredContent.threadId);
    return result.structuredContent.threadId;
  } finally {
    await client.close();
  }
}

try {
  await new Promise((listening, reject) => {
    responses.once('error', reject);
    responses.listen(0, '127.0.0.1', listening);
  });
  await writeFile(join(codexHome, 'config.toml'), `
model = "gpt-5.4"
model_provider = "brainx_smoke"
[model_providers.brainx_smoke]
name = "BrainX smoke fixture"
base_url = "http://127.0.0.1:${responses.address().port}/v1"
wire_api = "responses"
env_key = "BRAINX_SMOKE_KEY"
[mcp_servers.codex]
command = "brainx-smoke-missing-executable"
required = true
`);
  const threadId = await review('codex', { prompt: 'BrainX smoke first turn' }, '# Test review 1');
  assert.equal(await review('codex-reply', { threadId, prompt: 'BrainX smoke follow-up' }, '# Test review 2'), threadId);
  assert.equal(requests.length, 2);
  for (const request of requests) {
    assert.match(request, /independent computational-neuroscience/);
    assert.match(request, /TRAINING_REVIEW_REFERENCE/);
  }
  assert.match(requests[1], /BrainX smoke first turn/);
  console.log('PASS: real Codex MCP initialize, tools/list, fresh review and persisted-thread resume.');
} finally {
  responses.closeAllConnections();
  await new Promise((closed) => responses.close(closed));
  await rm(codexHome, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
}
