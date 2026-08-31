import assert from 'node:assert/strict';
import { chmodSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { spawn } from 'node:child_process';

const testDirectory = dirname(fileURLToPath(import.meta.url));
const server = resolve(testDirectory, '..', 'server.mjs');
const fakeCodex = resolve(testDirectory, 'fixtures', 'fake-codex.mjs');
chmodSync(fakeCodex, 0o755);

function runProxy(messages) {
  return new Promise((resolveOutput, reject) => {
    const child = spawn(process.execPath, [server], {
      env: { ...process.env, BRAINX_CODEX_BIN: fakeCodex },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`proxy exited ${code}: ${stderr}`));
        return;
      }
      resolveOutput(stdout.trimEnd().split('\n').map((line) => JSON.parse(line)));
    });
    for (const message of messages) {
      child.stdin.write(`${JSON.stringify(message)}\n`);
    }
    child.stdin.end();
  });
}

test('injects the reviewer prompt and configured BrainX skills into fresh sessions', async () => {
  const [message] = await runProxy([{
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/call',
    params: {
      name: 'codex',
      arguments: {
        prompt: 'Review iteration 4',
        'approval-policy': 'on-request',
        sandbox: 'danger-full-access',
        config: { model_reasoning_effort: 'xhigh' },
      },
    },
  }]);

  const args = message.params.arguments;
  assert.match(args['base-instructions'], /independent computational-neuroscience/);
  assert.match(args['base-instructions'], /Open and read `brainx-general-guard` first/);
  assert.match(args['base-instructions'], /Is the result scientifically backed/);
  assert.match(args['base-instructions'], /Is the code minimal and BrainX-native/);
  assert.match(args['base-instructions'], /Is training or fitting good enough/);
  assert.match(args['base-instructions'], /\*\*Loss closure:\*\*/);
  assert.match(args['base-instructions'], /\*\*Architecture:\*\*/);
  assert.match(args['base-instructions'], /\*\*Hyperparameters and method:\*\*/);
  assert.match(args['base-instructions'], /BrainState's prebuilt `brainstate\.nn` layers/);
  assert.match(args['base-instructions'], /braintools\.optim\.Adam/);
  assert.match(args['base-instructions'], /BrainTools API gap/);
  assert.match(args['base-instructions'], /RETURN_TO_STUDY/);
  assert.match(args['base-instructions'], /Return a Markdown document as the tool response/);
  assert.match(args['base-instructions'], /# BrainX iteration review/);
  assert.match(args['base-instructions'], /- \*\*OUTCOME:\*\* `PASS \| REFUSE`/);
  assert.match(args['base-instructions'], /- \*\*LOSS_CLOSURE:\*\* `NOT_APPLICABLE/);
  assert.match(args['base-instructions'], /## Findings/);
  assert.doesNotMatch(args['base-instructions'], /```text\s+OUTCOME:/);
  assert.match(args['base-instructions'], /Keep plotting code minimal and clean/);
  assert.match(args['developer-instructions'], /brainx-general-guard/);
  assert.match(args['developer-instructions'], /TRAINING_REVIEW_REFERENCE: \/.*\/training-workflow\.md/);
  assert.match(args['developer-instructions'], /FITTING_REVIEW_REFERENCE: \/.*\/parameter-fitting-workflow\.md/);
  assert.doesNotMatch(args['developer-instructions'], /^- brainx-modeling-loop$/m);
  assert.equal(args.config.model_reasoning_effort, 'xhigh');
  assert.equal(args['approval-policy'], 'never');
  assert.equal(args.sandbox, 'read-only');
  assert.equal(args.config['skills.config'].length, 8);
  assert.ok(args.config['skills.config'].every((skill) => skill.enabled));
  assert.ok(args.config['skills.config'].every((skill) => skill.path.startsWith('/')));
});

test('passes continuation and unrelated MCP messages through unchanged', async () => {
  const reply = {
    jsonrpc: '2.0',
    id: 2,
    method: 'tools/call',
    params: {
      name: 'codex-reply',
      arguments: { threadId: 'thread-1', prompt: 'Recheck finding BX-2' },
    },
  };
  const list = { jsonrpc: '2.0', id: 3, method: 'tools/list', params: {} };

  assert.deepEqual(await runProxy([reply, list]), [reply, list]);
});
