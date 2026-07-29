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
const { resolveLocations, resolveDestinationRoot } = require(path.join(repoRoot, 'installation/lib/paths.js'));

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

test('resolveDestinationRoot round-trips each registered adapter home path', () => {
  const homeDir = path.resolve('/home/tester');

  for (const adapter of DEFAULT_ADAPTERS) {
    const destination = path.resolve(homeDir, ...adapter.homePath);
    assert.equal(resolveDestinationRoot(destination, adapter, path), homeDir);
  }
});
