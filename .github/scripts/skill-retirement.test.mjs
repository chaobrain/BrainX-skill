import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import transaction from '../../installation/lib/adapter-transaction.js';
import hash from '../../installation/lib/hash.js';

test('updating an owned installation retires brainpy-state and preserves unrelated skills', async (t) => {
  const root = mkdtempSync(join(tmpdir(), 'brainx-retirement-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const destination = join(root, 'installed');
  const oldPath = join(destination, 'brainpy-state');
  const unrelated = join(destination, 'personal-skill');
  const source = join(root, 'source', 'brainspike');
  for (const folder of [oldPath, unrelated, source]) mkdirSync(folder, { recursive: true });
  writeFileSync(join(oldPath, 'SKILL.md'), 'old managed skill');
  writeFileSync(join(unrelated, 'SKILL.md'), 'personal skill');
  writeFileSync(join(source, 'SKILL.md'), '---\nname: brainspike\n---\n');
  const skill = { name: 'brainspike', sourcePath: source, digest: await hash.digestTree(source) };
  const previous = { packageVersion: '1.0.1', installedAt: '2026-11-09T00:00:00Z',
    skills: [{ name: 'brainpy-state', directory: oldPath, digest: await hash.digestTree(oldPath) }] };
  const operation = await transaction.prepareAdapter(
    { id: 'codex', label: 'Codex' }, destination,
    { version: '1.0.1', skills: [skill] }, previous, { homeDir: root },
  );
  await operation.commit();
  await operation.finalize();
  assert.equal(existsSync(oldPath), false);
  assert.equal(existsSync(join(destination, 'brainspike', 'SKILL.md')), true);
  assert.equal(existsSync(join(unrelated, 'SKILL.md')), true);
  assert.deepEqual(operation.record.skills.map((entry) => entry.name), ['brainspike']);
});
