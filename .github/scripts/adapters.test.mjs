import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const repoRoot = fileURLToPath(new URL('../..', import.meta.url));
const adaptersDir = path.join(repoRoot, 'adapters');

const adapterFiles = readdirSync(adaptersDir).filter((name) => name.endsWith('.js'));
const adapters = adapterFiles.map((file) => require(path.join(adaptersDir, file)));

const { DEFAULT_ADAPTERS } = require(path.join(repoRoot, 'installation/lib/installer.js'));
const {
  groupAdaptersByDestination,
  resolveLocations,
  resolveDestinationRoot,
} = require(path.join(repoRoot, 'installation/lib/paths.js'));

test('every adapters/*.js file exports a frozen, well-formed adapter', () => {
  for (const [index, adapter] of adapters.entries()) {
    const file = adapterFiles[index];
    assert.ok(Object.isFrozen(adapter), `${file} adapter must be frozen`);
    assert.equal(adapter.id, path.basename(file, '.js'), `${file} id must match its filename`);
    assert.match(adapter.id, /^[a-z][a-z0-9-]*$/, `${file} id must be lowercase kebab-case`);
    assert.equal(typeof adapter.label, 'string');
    assert.ok(adapter.label.length > 0, `${file} label must be non-empty`);
    assert.ok(Array.isArray(adapter.homePath), `${file} homePath must be an array`);
    assert.ok(adapter.homePath.length > 0, `${file} homePath must not be empty`);
    for (const segment of adapter.homePath) {
      assert.equal(typeof segment, 'string');
      assert.ok(segment.length > 0, `${file} homePath segments must be non-empty`);
    }
    if (adapter.projectPath !== undefined) {
      assert.ok(Array.isArray(adapter.projectPath), `${file} projectPath must be an array`);
      assert.ok(adapter.projectPath.length > 0, `${file} projectPath must not be empty`);
      for (const segment of adapter.projectPath) {
        assert.equal(typeof segment, 'string');
        assert.ok(segment.length > 0, `${file} projectPath segments must be non-empty`);
      }
    }
  }
});

test('DEFAULT_ADAPTERS registers exactly one entry per adapters/*.js file, no duplicates', () => {
  const registeredIds = DEFAULT_ADAPTERS.map((adapter) => adapter.id);
  const fileIds = adapters.map((adapter) => adapter.id);

  assert.deepEqual(
    [...registeredIds].sort(),
    [...fileIds].sort(),
    'DEFAULT_ADAPTERS must contain exactly the adapters defined under adapters/',
  );
  assert.equal(
    new Set(registeredIds).size,
    registeredIds.length,
    'adapter ids registered in DEFAULT_ADAPTERS must be unique',
  );
});

test('resolveLocations resolves each registered adapter to its own home path', () => {
  const homeDir = path.resolve('/home/tester');
  const { destinations } = resolveLocations(homeDir, DEFAULT_ADAPTERS, path);

  for (const adapter of DEFAULT_ADAPTERS) {
    const expected = path.resolve(homeDir, ...adapter.homePath);
    assert.equal(destinations[adapter.id], expected);
  }
});

test('resolveLocations uses projectPath for project scope when an adapter declares one', () => {
  const homeDir = path.resolve('/home/tester');
  const cwd = path.resolve('/home/tester/repo');
  const { destinations } = resolveLocations(homeDir, DEFAULT_ADAPTERS, path, {
    scope: 'project',
    cwd,
  });

  for (const adapter of DEFAULT_ADAPTERS) {
    const expected = path.resolve(cwd, ...(adapter.projectPath || adapter.homePath));
    assert.equal(destinations[adapter.id], expected);
  }
});

test('resolveDestinationRoot round-trips every path an adapter can install to', () => {
  const homeDir = path.resolve('/home/tester');

  for (const adapter of DEFAULT_ADAPTERS) {
    for (const skillPath of [adapter.homePath, adapter.projectPath].filter(Boolean)) {
      const destination = path.resolve(homeDir, ...skillPath);
      assert.equal(resolveDestinationRoot(destination, adapter, path), homeDir);
    }
  }
});

test('groupAdaptersByDestination merges harnesses that share one skill directory', () => {
  const cwd = path.resolve('/home/tester/repo');
  const { destinations } = resolveLocations(path.resolve('/home/tester'), DEFAULT_ADAPTERS, path, {
    scope: 'project',
    cwd,
  });
  const groups = groupAdaptersByDestination(DEFAULT_ADAPTERS, destinations, path);

  assert.equal(
    groups.reduce((total, group) => total + group.adapters.length, 0),
    DEFAULT_ADAPTERS.length,
    'every adapter must belong to exactly one group',
  );
  for (const group of groups) {
    for (const adapter of group.adapters) {
      assert.equal(destinations[adapter.id], group.destination);
    }
  }

  const shared = groups.find((group) => group.destination === path.resolve(cwd, '.agents', 'skills'));
  assert.deepEqual(
    shared.adapters.map((adapter) => adapter.id).sort(),
    ['antigravity', 'codex'],
    'Codex and Antigravity share <cwd>/.agents/skills in project scope',
  );
});
