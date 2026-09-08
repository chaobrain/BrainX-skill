'use strict';

const childProcess = require('node:child_process');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');

const EUROPE_PMC_NAME = 'europepmc';
const FULLTEXT_NAME = 'fulltext_resolver';
const CODEX_REVIEWER_NAME = 'codex';
const EUROPE_PMC_SOURCE = 'git+https://github.com/CoChatAI/europepmc-mcp-server@a1af9d06ec7f5f2caef56005fdc1d9d4f12cb115';
const RUNTIME_SCHEMA_VERSION = 1;

function writeLine(stream, text = '') {
  stream.write(`${text}\n`);
}

function defaultRunner(command, args, options = {}) {
  return childProcess.spawnSync(command, args, {
    cwd: options.cwd,
    env: options.env || process.env,
    encoding: 'utf8',
    stdio: options.stdio || 'pipe',
  });
}

function resultText(result) {
  return [result.stdout, result.stderr, result.error && result.error.message]
    .filter(Boolean)
    .join('\n')
    .trim();
}

async function runChecked(runner, command, args, options, label) {
  const result = await runner(command, args, options);
  if (result.error || result.status !== 0) {
    const detail = resultText(result);
    throw new Error(`${label} failed${detail ? `: ${detail}` : ''}`);
  }
  return result;
}

function executableName(name, platform) {
  return platform === 'win32' ? `${name}.exe` : name;
}

function runtimePaths(homeDir, pathApi, platform) {
  const root = pathApi.resolve(homeDir, '.brainx', 'mcp');
  const europePmcRoot = pathApi.join(root, 'europepmc');
  const europePmcBinDir = pathApi.join(europePmcRoot, 'bin');
  const bundleRoot = pathApi.join(root, 'bundle');
  return {
    root,
    receiptPath: pathApi.join(root, 'receipt.json'),
    packageJsonPath: pathApi.join(root, 'package.json'),
    resolverRoot: pathApi.join(root, 'fulltext-resolver'),
    resolverServer: pathApi.join(root, 'fulltext-resolver', 'server.mjs'),
    bundleRoot,
    codexReviewerRoot: pathApi.join(bundleRoot, 'mcp-servers', 'codex'),
    codexReviewerServer: pathApi.join(bundleRoot, 'mcp-servers', 'codex', 'server.mjs'),
    reviewerSkillsRoot: pathApi.join(bundleRoot, 'skills'),
    europePmcRoot,
    europePmcBinDir,
    europePmcExecutable: pathApi.join(
      europePmcBinDir,
      executableName('europepmc-mcp-server', platform),
    ),
  };
}

function expectedServers(paths, nodeBin) {
  return {
    [EUROPE_PMC_NAME]: {
      command: paths.europePmcExecutable,
      args: [],
    },
    [FULLTEXT_NAME]: {
      command: nodeBin,
      args: [paths.resolverServer],
    },
    [CODEX_REVIEWER_NAME]: {
      command: nodeBin,
      args: [paths.codexReviewerServer],
    },
  };
}

function sameStringArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function matchesExpected(server, expected) {
  return Boolean(
    server
      && server.transport
      && server.transport.type === 'stdio'
      && server.transport.command === expected.command
      && sameStringArray(server.transport.args || [], expected.args),
  );
}

async function getCodexServer(runner, codexBin, name) {
  const result = await runner(codexBin, ['mcp', 'get', name, '--json'], {});
  if (!result.error && result.status === 0) {
    try {
      return JSON.parse(result.stdout);
    } catch (error) {
      throw new Error(`Codex returned invalid MCP configuration for ${name}: ${error.message}`);
    }
  }

  const detail = resultText(result);
  if (/No MCP server named|not found/i.test(detail)) {
    return null;
  }
  throw new Error(`Cannot inspect Codex MCP server ${name}${detail ? `: ${detail}` : ''}`);
}

async function inspectServers(runner, codexBin, expected) {
  const inspected = {};
  for (const [name, spec] of Object.entries(expected)) {
    const existing = await getCodexServer(runner, codexBin, name);
    inspected[name] = {
      existing,
      matches: matchesExpected(existing, spec),
    };
  }
  return inspected;
}

function conflictingNames(inspected) {
  return Object.entries(inspected)
    .filter(([, state]) => state.existing && !state.matches)
    .map(([name]) => name);
}

function asServerSpec(server) {
  if (!server || !server.transport || server.transport.type !== 'stdio') {
    return null;
  }
  return {
    command: server.transport.command,
    args: server.transport.args || [],
  };
}

function legacyReviewerSpec(server, packageRoot, pathApi) {
  const spec = asServerSpec(server);
  if (!spec || spec.args.length !== 1) {
    return null;
  }
  const commandName = pathApi.basename(spec.command).toLowerCase();
  if (commandName !== 'node' && commandName !== 'node.exe') {
    return null;
  }
  const legacyServer = pathApi.resolve(packageRoot, 'mcp-servers', 'codex', 'server.mjs');
  if (pathApi.resolve(spec.args[0]) !== legacyServer) {
    return null;
  }
  const environment = server.transport.env;
  if (environment && Object.keys(environment).length > 0) {
    return null;
  }
  return spec;
}

async function pathExists(fsApi, target) {
  try {
    await fsApi.access(target);
    return true;
  } catch {
    return false;
  }
}

async function readRuntimeReceipt(paths, fsApi) {
  if (!(await pathExists(fsApi, paths.receiptPath))) {
    return null;
  }
  let receipt;
  try {
    receipt = JSON.parse(await fsApi.readFile(paths.receiptPath, 'utf8'));
  } catch (error) {
    throw new Error(`Cannot read the BrainX MCP receipt: ${error.message}`);
  }
  const validServers = receipt.servers
    && typeof receipt.servers === 'object'
    && Object.values(receipt.servers).every((spec) => (
      spec
      && typeof spec.command === 'string'
      && spec.command.length > 0
      && Array.isArray(spec.args)
      && spec.args.every((arg) => typeof arg === 'string')
    ));
  if (receipt.schemaVersion !== RUNTIME_SCHEMA_VERSION || !validServers) {
    throw new Error('The BrainX MCP receipt has an unsupported format');
  }
  return receipt;
}

function receiptOwns(receipt, expected) {
  return Boolean(
    receipt
      && Object.entries(expected).every(([name, spec]) => (
        receipt.servers[name]
        && receipt.servers[name].command === spec.command
        && sameStringArray(receipt.servers[name].args, spec.args)
      )),
  );
}

async function readPackageDependencies(packageRoot, fsApi, pathApi) {
  const packagePath = pathApi.join(packageRoot, 'package.json');
  const packageJson = JSON.parse(await fsApi.readFile(packagePath, 'utf8'));
  const names = [
    '@modelcontextprotocol/sdk',
    'linkedom',
    'turndown',
    'zod',
  ];
  const dependencies = {};
  for (const name of names) {
    const version = packageJson.dependencies && packageJson.dependencies[name];
    if (!version) {
      throw new Error(`package.json is missing the resolver dependency ${name}`);
    }
    dependencies[name] = version;
  }
  return dependencies;
}

async function installRuntimes(options) {
  const {
    fsApi,
    pathApi,
    packageRoot,
    paths,
    runner,
    npmBin,
    uvBin,
    stdout,
  } = options;

  const resolverSource = pathApi.join(packageRoot, 'mcp-servers', 'fulltext-resolver');
  const codexReviewerSource = pathApi.join(packageRoot, 'mcp-servers', 'codex');
  const skillsSource = pathApi.join(packageRoot, 'skills');
  if (!(await pathExists(fsApi, pathApi.join(resolverSource, 'server.mjs')))) {
    throw new Error(`Bundled Full-Text Resolver is missing: ${resolverSource}`);
  }
  if (!(await pathExists(fsApi, pathApi.join(codexReviewerSource, 'server.mjs')))) {
    throw new Error(`Bundled BrainX Codex reviewer is missing: ${codexReviewerSource}`);
  }
  if (!(await pathExists(fsApi, skillsSource))) {
    throw new Error(`Bundled BrainX skills are missing: ${skillsSource}`);
  }

  await fsApi.mkdir(paths.root, { recursive: true });
  await fsApi.cp(resolverSource, paths.resolverRoot, {
    recursive: true,
    force: true,
  });
  await fsApi.rm(paths.bundleRoot, { recursive: true, force: true });
  await fsApi.mkdir(pathApi.dirname(paths.codexReviewerRoot), { recursive: true });
  await fsApi.cp(codexReviewerSource, paths.codexReviewerRoot, {
    recursive: true,
    force: true,
  });
  await fsApi.cp(skillsSource, paths.reviewerSkillsRoot, {
    recursive: true,
    force: true,
  });

  const runtimePackage = {
    private: true,
    type: 'module',
    dependencies: await readPackageDependencies(packageRoot, fsApi, pathApi),
  };
  await fsApi.writeFile(
    paths.packageJsonPath,
    `${JSON.stringify(runtimePackage, null, 2)}\n`,
    'utf8',
  );

  writeLine(stdout, 'Installing the Full-Text Resolver runtime...');
  await runChecked(
    runner,
    npmBin,
    ['install', '--omit=dev', '--ignore-scripts', '--no-audit', '--no-fund'],
    { cwd: paths.root },
    'Full-Text Resolver dependency installation',
  );

  await fsApi.mkdir(paths.europePmcBinDir, { recursive: true });
  writeLine(stdout, 'Installing the pinned Europe PMC MCP runtime...');
  await runChecked(
    runner,
    uvBin,
    ['tool', 'install', '--force', '--from', EUROPE_PMC_SOURCE, 'europepmc-mcp-server'],
    {
      env: {
        ...process.env,
        UV_TOOL_DIR: pathApi.join(paths.europePmcRoot, 'tools'),
        UV_TOOL_BIN_DIR: paths.europePmcBinDir,
      },
    },
    'Europe PMC MCP installation',
  );

  if (!(await pathExists(fsApi, paths.europePmcExecutable))) {
    throw new Error(`Europe PMC executable was not created: ${paths.europePmcExecutable}`);
  }
  if (!(await pathExists(fsApi, paths.resolverServer))) {
    throw new Error(`Full-Text Resolver server was not created: ${paths.resolverServer}`);
  }
  if (!(await pathExists(fsApi, paths.codexReviewerServer))) {
    throw new Error(`BrainX Codex reviewer server was not created: ${paths.codexReviewerServer}`);
  }
}

async function addServer(runner, codexBin, name, spec) {
  await runChecked(
    runner,
    codexBin,
    ['mcp', 'add', name, '--', spec.command, ...spec.args],
    {},
    `Codex MCP registration for ${name}`,
  );
}

async function removeServer(runner, codexBin, name) {
  await runChecked(
    runner,
    codexBin,
    ['mcp', 'remove', name],
    {},
    `Codex MCP removal for ${name}`,
  );
}

async function installLiteratureMcp(options) {
  const {
    fsApi,
    pathApi,
    paths,
    runner,
    codexBin,
    stdout,
    packageRoot,
  } = options;

  let expected = options.expected;
  const inspected = await inspectServers(runner, codexBin, expected);
  const legacyReviewer = legacyReviewerSpec(
    inspected[CODEX_REVIEWER_NAME].existing,
    packageRoot,
    pathApi,
  );
  if (legacyReviewer) {
    expected = { ...expected, [CODEX_REVIEWER_NAME]: legacyReviewer };
    inspected[CODEX_REVIEWER_NAME].matches = true;
  }
  const conflicts = conflictingNames(inspected);
  if (conflicts.length) {
    throw new Error(
      `Refusing to overwrite existing Codex MCP configuration: ${conflicts.join(', ')}`,
    );
  }

  const receipt = await readRuntimeReceipt(paths, fsApi);
  if (legacyReviewer && !receipt) {
    writeLine(stdout, 'Keeping the existing BrainX Codex reviewer registration and settings.');
  }
  if (!receiptOwns(receipt, expected)) {
    const exactButUnowned = Object.entries(inspected)
      .filter(([name, state]) => (
        state.matches && (name !== CODEX_REVIEWER_NAME || !legacyReviewer)
      ))
      .map(([name]) => name);
    if (exactButUnowned.length) {
      throw new Error(
        `Refusing to adopt existing Codex MCP configuration without a BrainX receipt: ${exactButUnowned.join(', ')}`,
      );
    }
  }

  const runtimesExist = await pathExists(fsApi, paths.europePmcExecutable)
    && await pathExists(fsApi, paths.resolverServer)
    && await pathExists(fsApi, paths.codexReviewerServer)
    && await pathExists(fsApi, pathApi.join(paths.root, 'node_modules'));
  if (
    runtimesExist
    && receiptOwns(receipt, expected)
    && Object.values(inspected).every((state) => state.matches)
  ) {
    writeLine(stdout, 'BrainX MCP servers are already registered.');
    return 0;
  }

  await installRuntimes(options);

  const added = [];
  try {
    for (const [name, spec] of Object.entries(expected)) {
      if (!inspected[name].matches) {
        await addServer(runner, codexBin, name, spec);
        added.push(name);
      }
    }

    const nextReceipt = {
      schemaVersion: RUNTIME_SCHEMA_VERSION,
      servers: expected,
    };
    await fsApi.writeFile(
      paths.receiptPath,
      `${JSON.stringify(nextReceipt, null, 2)}\n`,
      'utf8',
    );
  } catch (error) {
    for (const name of added.reverse()) {
      try {
        await removeServer(runner, codexBin, name);
      } catch {
        // Preserve the original failure; the user can remove a residual entry manually.
      }
    }
    throw error;
  }

  writeLine(stdout, 'Registered Codex MCP servers: europepmc, fulltext_resolver, codex');
  writeLine(stdout, 'Restart Codex or reload MCP configuration before using the new tools.');
  return 0;
}

async function removeLiteratureMcp(options) {
  const {
    fsApi,
    paths,
    runner,
    codexBin,
    stdout,
  } = options;

  const receipt = await readRuntimeReceipt(paths, fsApi);
  const expectedNames = Object.keys(options.expected);
  const receiptHasEveryServer = receipt
    && expectedNames.every((name) => receipt.servers[name])
    && Object.keys(receipt.servers).every((name) => expectedNames.includes(name));
  const expected = receiptHasEveryServer ? receipt.servers : options.expected;
  const inspected = await inspectServers(runner, codexBin, expected);
  const conflicts = conflictingNames(inspected);
  if (conflicts.length) {
    throw new Error(
      `Refusing to remove MCP configuration not owned by BrainX: ${conflicts.join(', ')}`,
    );
  }

  const hasRegistrations = Object.values(inspected).some((state) => state.existing);
  const hasRuntime = await pathExists(fsApi, paths.root);
  if (!receiptHasEveryServer || !receiptOwns(receipt, expected)) {
    if (!hasRegistrations && !hasRuntime) {
      writeLine(stdout, 'No BrainX-managed MCP servers are installed.');
      return 0;
    }
    throw new Error('Refusing to remove MCP state without a valid BrainX receipt');
  }

  for (const name of Object.keys(expected)) {
    if (inspected[name].matches) {
      await removeServer(runner, codexBin, name);
    }
  }
  await fsApi.rm(paths.root, { recursive: true, force: true });
  writeLine(stdout, 'Removed the BrainX MCP registrations and managed runtime.');
  return 0;
}

async function runLiteratureMcp(command, options = {}) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  const homeDir = options.homeDir || os.homedir();
  const packageRoot = options.packageRoot || path.resolve(__dirname, '../..');
  const runner = options.runner || defaultRunner;
  const stdout = options.stdout || process.stdout;
  const stderr = options.stderr || process.stderr;
  const codexBin = options.codexBin || 'codex';
  const npmBin = options.npmBin || 'npm';
  const uvBin = options.uvBin || 'uv';
  const platform = options.platform || process.platform;
  const paths = runtimePaths(homeDir, pathApi, platform);
  const expected = expectedServers(paths, options.nodeBin || process.execPath);

  try {
    await runChecked(runner, codexBin, ['--version'], {}, 'Codex CLI check');
    const shared = {
      ...options,
      fsApi,
      pathApi,
      homeDir,
      packageRoot,
      runner,
      stdout,
      stderr,
      codexBin,
      npmBin,
      uvBin,
      platform,
      paths,
      expected,
    };
    if (command === 'install') {
      return await installLiteratureMcp(shared);
    }
    if (command === 'remove') {
      return await removeLiteratureMcp(shared);
    }
    throw new Error(`Unknown literature MCP command: ${command}`);
  } catch (error) {
    writeLine(stderr, `x BrainX MCP ${command} failed: ${error.message}`);
    return 1;
  }
}

module.exports = {
  CODEX_REVIEWER_NAME,
  EUROPE_PMC_NAME,
  EUROPE_PMC_SOURCE,
  FULLTEXT_NAME,
  expectedServers,
  matchesExpected,
  runLiteratureMcp,
  runtimePaths,
};
