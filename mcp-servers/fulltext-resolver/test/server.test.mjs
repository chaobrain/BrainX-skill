import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const testDirectory = dirname(fileURLToPath(import.meta.url));
const serverPath = resolve(testDirectory, '..', 'server.mjs');

function listTools() {
  return new Promise((resolveTools, reject) => {
    const child = spawn(process.execPath, [serverPath], { stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error(`Timed out waiting for MCP tools/list: ${stderr}`));
    }, 5000);

    child.stdout.on('data', (chunk) => {
      stdout += chunk;
      const messages = stdout.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line));
      const toolsResponse = messages.find((message) => message.id === 2);
      if (!toolsResponse) return;
      clearTimeout(timeout);
      child.stdin.end();
      child.kill();
      resolveTools(toolsResponse.result.tools);
    });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    const messages = [
      {
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {
          protocolVersion: '2025-06-18',
          capabilities: {},
          clientInfo: { name: 'test-client', version: '1.0.0' },
        },
      },
      { jsonrpc: '2.0', method: 'notifications/initialized', params: {} },
      { jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} },
    ];
    for (const message of messages) child.stdin.write(`${JSON.stringify(message)}\n`);
  });
}

test('launches over stdio and exposes only the three resolver tools', async () => {
  const tools = await listTools();
  assert.deepEqual(tools.map((tool) => tool.name).sort(), [
    'get_fulltext',
    'list_versions',
    'resolve_fulltext',
  ]);
  assert.ok(tools.every((tool) => tool.annotations.readOnlyHint));
  assert.ok(tools.every((tool) => tool.annotations.openWorldHint));
});
