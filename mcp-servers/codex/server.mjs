#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

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

function transformMessage(message) {
  if (Array.isArray(message)) {
    return message.map(transformMessage);
  }
  if (
    message === null
    || typeof message !== 'object'
    || message.method !== 'tools/call'
    || message.params?.name !== 'codex'
  ) {
    return message;
  }

  const originalArguments = message.params.arguments;
  const args = originalArguments && typeof originalArguments === 'object'
    ? { ...originalArguments }
    : {};
  const config = args.config && typeof args.config === 'object'
    ? { ...args.config }
    : {};

  config['skills.config'] = exposedSkills;

  return {
    ...message,
    params: {
      ...message.params,
      arguments: {
        ...args,
        config,
        'approval-policy': 'never',
        'base-instructions': systemPrompt,
        'developer-instructions': skillInstructions,
        sandbox: 'read-only',
      },
    },
  };
}

const codexBinary = process.env.BRAINX_CODEX_BIN || 'codex';
const child = spawn(codexBinary, ['mcp-server'], {
  env: process.env,
  stdio: ['pipe', 'pipe', 'inherit'],
});

child.on('error', (error) => {
  process.stderr.write(`Failed to start ${codexBinary} mcp-server: ${error.message}\n`);
  process.exitCode = 1;
});
child.stdin.on('error', (error) => {
  if (error.code !== 'EPIPE') {
    process.stderr.write(`Codex MCP stdin failed: ${error.message}\n`);
  }
});

child.stdout.pipe(process.stdout);

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on('line', (line) => {
  try {
    const message = JSON.parse(line);
    const transformed = transformMessage(message);
    child.stdin.write(`${JSON.stringify(transformed)}\n`);
  } catch {
    child.stdin.write(`${line}\n`);
  }
});
input.on('close', () => child.stdin.end());

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}

child.on('exit', (code, signal) => {
  input.close();
  process.stdin.pause();
  process.exitCode = code ?? (signal ? 1 : 0);
});
