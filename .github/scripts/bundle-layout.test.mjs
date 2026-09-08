import assert from 'node:assert/strict';
import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import bundle from '../../installation/lib/bundle.js';

const { validateBundle } = bundle;
const repositoryRoot = fileURLToPath(new URL('../..', import.meta.url));

test('validates the repository with grouped package skills', async () => {
  const result = await validateBundle(repositoryRoot);

  assert.equal(result.skills.length, 13);
  assert.match(
    result.skills.find((skill) => skill.name === 'brainstate').sourcePath,
    /skills[/\\]package-skills[/\\]brainstate$/,
  );
});

test('discovers grouped source skills while preserving flat skill names', async (t) => {
  const root = mkdtempSync(join(tmpdir(), 'brainx-bundle-layout-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));

  writeFileSync(
    join(root, 'package.json'),
    JSON.stringify({ name: 'brainx-skill', version: '1.0.0' }),
  );
  writeFileSync(
    join(root, 'manifest.json'),
    JSON.stringify({ schemaVersion: 1, skills: ['brainstate', 'brainx-install'] }),
  );

  const grouped = join(root, 'skills', 'package-skills', 'brainstate');
  const direct = join(root, 'skills', 'brainx-install');
  mkdirSync(grouped, { recursive: true });
  mkdirSync(direct, { recursive: true });
  writeFileSync(join(grouped, 'SKILL.md'), '---\nname: brainstate\n---\n');
  writeFileSync(join(direct, 'SKILL.md'), '---\nname: brainx-install\n---\n');

  const result = await validateBundle(root);

  assert.deepEqual(result.skills.map((skill) => skill.name), [
    'brainstate',
    'brainx-install',
  ]);
  assert.equal(result.skills[0].sourcePath, grouped);
  assert.equal(result.skills[1].sourcePath, direct);
});
