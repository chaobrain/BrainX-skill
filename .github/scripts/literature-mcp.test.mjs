import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import cliModule from '../../installation/lib/cli.js';
import mcpModule from '../../installation/lib/literature-mcp.js';

const { parseCommand, runCli } = cliModule;
const { expectedServers, runLiteratureMcp, runtimePaths } = mcpModule;

function stream() {
  return {
    text: '',
    write(chunk) {
      this.text += chunk;
    },
  };
}

function createFixture(t) {
  const root = mkdtempSync(path.join(tmpdir(), 'brainx-mcp-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const homeDir = path.join(root, 'home');
  const packageRoot = path.join(root, 'package');
  mkdirSync(path.join(packageRoot, 'mcp-servers', 'fulltext-resolver'), { recursive: true });
  mkdirSync(path.join(packageRoot, 'mcp-servers', 'codex'), { recursive: true });
  mkdirSync(path.join(packageRoot, 'skills', 'brainx-general-guard'), { recursive: true });
  writeFileSync(path.join(packageRoot, 'mcp-servers', 'fulltext-resolver', 'server.mjs'), '// resolver\n');
  writeFileSync(path.join(packageRoot, 'mcp-servers', 'codex', 'server.mjs'), '// reviewer\n');
  writeFileSync(path.join(packageRoot, 'mcp-servers', 'codex', 'system-prompt.md'), '# Review\n');
  writeFileSync(path.join(packageRoot, 'mcp-servers', 'codex', 'skills.json'), '[]\n');
  writeFileSync(path.join(packageRoot, 'skills', 'brainx-general-guard', 'SKILL.md'), '# Guard\n');
  writeFileSync(path.join(packageRoot, 'package.json'), JSON.stringify({
    dependencies: {
      '@modelcontextprotocol/sdk': '^1.0.0',
      linkedom: '^0.18.0',
      turndown: '^7.0.0',
      zod: '^4.0.0',
    },
  }));
  return { homeDir, packageRoot };
}

function configuredServer(spec) {
  return {
    transport: {
      type: 'stdio',
      command: spec.command,
      args: spec.args,
    },
  };
}

function createRunner(homeDir, options = {}) {
  const servers = new Map(options.servers || []);
  const calls = [];
  const runner = async (command, args, commandOptions = {}) => {
    calls.push({ command, args, options: commandOptions });
    if (command === 'codex' && args[0] === '--version') {
      return { status: 0, stdout: 'codex-cli test\n', stderr: '' };
    }
    if (command === 'codex' && args[0] === 'mcp' && args[1] === 'get') {
      const server = servers.get(args[2]);
      return server
        ? { status: 0, stdout: JSON.stringify(server), stderr: '' }
        : { status: 1, stdout: '', stderr: `Error: No MCP server named '${args[2]}' found.` };
    }
    if (command === 'codex' && args[0] === 'mcp' && args[1] === 'add') {
      const name = args[2];
      if (options.failAdd === name) {
        return { status: 1, stdout: '', stderr: 'simulated registration failure' };
      }
      const separator = args.indexOf('--');
      servers.set(name, configuredServer({
        command: args[separator + 1],
        args: args.slice(separator + 2),
      }));
      return { status: 0, stdout: '', stderr: '' };
    }
    if (command === 'codex' && args[0] === 'mcp' && args[1] === 'remove') {
      servers.delete(args[2]);
      return { status: 0, stdout: '', stderr: '' };
    }
    if (command === 'npm') {
      await mkdir(path.join(commandOptions.cwd, 'node_modules'), { recursive: true });
      return { status: 0, stdout: '', stderr: '' };
    }
    if (command === 'uv') {
      const executable = path.join(
        commandOptions.env.UV_TOOL_BIN_DIR,
        process.platform === 'win32' ? 'europepmc-mcp-server.exe' : 'europepmc-mcp-server',
      );
      await mkdir(path.dirname(executable), { recursive: true });
      await writeFile(executable, '#!/bin/sh\n');
      return { status: 0, stdout: '', stderr: '' };
    }
    return { status: 1, stdout: '', stderr: `unexpected command: ${command} ${args.join(' ')}` };
  };
  return { calls, runner, servers };
}

function installOptions(fixture, mock) {
  return {
    ...fixture,
    runner: mock.runner,
    codexBin: 'codex',
    npmBin: 'npm',
    uvBin: 'uv',
    nodeBin: '/test/node',
    stdout: stream(),
    stderr: stream(),
  };
}

test('CLI routes one-command MCP install and remove operations', async () => {
  assert.equal(parseCommand(['mcp', 'install']), 'mcp:install');
  assert.equal(parseCommand(['mcp', 'remove']), 'mcp:remove');
  const commands = [];
  const result = await runCli(['mcp', 'install'], {
    mcpInstaller: async (command) => {
      commands.push(command);
      return 0;
    },
  });
  assert.equal(result, 0);
  assert.deepEqual(commands, ['install']);
});

test('installs all three MCP servers and repeated installation is idempotent', async (t) => {
  const fixture = createFixture(t);
  const mock = createRunner(fixture.homeDir);
  const options = installOptions(fixture, mock);

  assert.equal(await runLiteratureMcp('install', options), 0);
  assert.deepEqual([...mock.servers.keys()], ['europepmc', 'fulltext_resolver', 'codex']);

  const paths = runtimePaths(fixture.homeDir, path, process.platform);
  assert.ok(existsSync(paths.europePmcExecutable));
  assert.ok(existsSync(paths.resolverServer));
  assert.ok(existsSync(paths.codexReviewerServer));
  assert.ok(existsSync(path.join(paths.reviewerSkillsRoot, 'brainx-general-guard', 'SKILL.md')));
  const receipt = JSON.parse(readFileSync(paths.receiptPath, 'utf8'));
  assert.deepEqual(Object.keys(receipt.servers), ['europepmc', 'fulltext_resolver', 'codex']);

  const mutatingCalls = mock.calls.filter(({ args }) => ['add', 'install'].includes(args[1]) || args[0] === 'tool').length;
  assert.equal(await runLiteratureMcp('install', options), 0);
  assert.equal(
    mock.calls.filter(({ args }) => ['add', 'install'].includes(args[1]) || args[0] === 'tool').length,
    mutatingCalls,
  );
  assert.match(options.stdout.text, /already registered/);
});

test('keeps the exact legacy BrainX reviewer registration and its settings', async (t) => {
  const fixture = createFixture(t);
  const legacyPath = path.join(fixture.packageRoot, 'mcp-servers', 'codex', 'server.mjs');
  const legacy = {
    ...configuredServer({ command: 'node', args: [legacyPath] }),
    tool_timeout_sec: 1800,
  };
  const mock = createRunner(fixture.homeDir, { servers: [['codex', legacy]] });
  const options = installOptions(fixture, mock);

  assert.equal(await runLiteratureMcp('install', options), 0);
  assert.equal(mock.servers.get('codex'), legacy);
  assert.equal(
    mock.calls.some(({ args }) => args[0] === 'mcp' && args[1] === 'remove' && args[2] === 'codex'),
    false,
  );
  assert.match(options.stdout.text, /Keeping the existing BrainX Codex reviewer/);

  const receipt = JSON.parse(readFileSync(
    runtimePaths(fixture.homeDir, path, process.platform).receiptPath,
    'utf8',
  ));
  assert.deepEqual(receipt.servers.codex, { command: 'node', args: [legacyPath] });
});

test('refuses to overwrite a conflicting existing registration', async (t) => {
  const fixture = createFixture(t);
  const mock = createRunner(fixture.homeDir, {
    servers: [['europepmc', configuredServer({ command: '/user/server', args: [] })]],
  });
  const options = installOptions(fixture, mock);

  assert.equal(await runLiteratureMcp('install', options), 1);
  assert.match(options.stderr.text, /Refusing to overwrite.*europepmc/);
  assert.equal(mock.calls.some(({ command }) => command === 'npm' || command === 'uv'), false);
});

test('refuses to adopt exact registrations without a matching receipt', async (t) => {
  const fixture = createFixture(t);
  const paths = runtimePaths(fixture.homeDir, path, process.platform);
  const expected = expectedServers(paths, '/test/node');
  const mock = createRunner(fixture.homeDir, {
    servers: [['europepmc', configuredServer(expected.europepmc)]],
  });
  const options = installOptions(fixture, mock);

  assert.equal(await runLiteratureMcp('install', options), 1);
  assert.match(options.stderr.text, /without a BrainX receipt: europepmc/);
});

test('removes only receipt-owned registrations and managed runtime', async (t) => {
  const fixture = createFixture(t);
  const mock = createRunner(fixture.homeDir);
  const options = installOptions(fixture, mock);
  assert.equal(await runLiteratureMcp('install', options), 0);

  assert.equal(await runLiteratureMcp('remove', options), 0);
  assert.equal(mock.servers.size, 0);
  assert.equal(existsSync(runtimePaths(fixture.homeDir, path, process.platform).root), false);
});

test('rolls back registrations when a later registration fails', async (t) => {
  const fixture = createFixture(t);
  const mock = createRunner(fixture.homeDir, { failAdd: 'codex' });
  const options = installOptions(fixture, mock);

  assert.equal(await runLiteratureMcp('install', options), 1);
  assert.equal(mock.servers.size, 0);
  assert.equal(existsSync(runtimePaths(fixture.homeDir, path, process.platform).receiptPath), false);
  const removals = mock.calls
    .filter(({ args }) => args[0] === 'mcp' && args[1] === 'remove')
    .map(({ args }) => args[2]);
  assert.deepEqual(removals, ['fulltext_resolver', 'europepmc']);
});
