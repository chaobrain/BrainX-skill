import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const prepareRelease = fileURLToPath(
  new URL('./prepare-release.mjs', import.meta.url),
);

function git(cwd, ...args) {
  return execFileSync('git', args, {
    cwd,
    encoding: 'utf8',
  }).trim();
}

function commit(cwd, subject, body = '') {
  git(cwd, 'add', '.');
  git(
    cwd,
    'commit',
    '--allow-empty',
    '-m',
    subject,
    ...(body ? ['-m', body] : []),
  );
}

test('prepare-release creates the combined npm catch-up as 1.0.6', (t) => {
  const cwd = mkdtempSync(join(tmpdir(), 'brainx-release-'));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));

  git(cwd, 'init', '--quiet');
  git(cwd, 'config', 'user.name', 'Release Test');
  git(cwd, 'config', 'user.email', 'release-test@example.test');

  writeFileSync(
    join(cwd, 'package.json'),
    `${JSON.stringify({ name: 'brainx-skill', version: '1.0.5' }, null, 2)}\n`,
  );
  writeFileSync(
    join(cwd, 'CHANGELOG.md'),
    '# Changelog\n\nRelease history.\n\n## 1.0.3\n\nBaseline.\n',
  );
  writeFileSync(join(cwd, 'RELEASE_NOTES.md'), 'Previous notes.\n');
  writeFileSync(
    join(cwd, 'release-notes.config.json'),
    JSON.stringify({
      repository: 'chaobrain/BrainX-skill',
      npmCatchUp: {
        publishedVersion: '1.0.3',
        previousRef: 'BASELINE',
      },
    }),
  );
  writeFileSync(join(cwd, 'baseline.txt'), 'npm 1.0.3\n');
  commit(cwd, 'chore: npm baseline');
  const baseline = git(cwd, 'rev-parse', 'HEAD');

  writeFileSync(
    join(cwd, 'release-notes.config.json'),
    `${JSON.stringify(
      {
        repository: 'chaobrain/BrainX-skill',
        npmCatchUp: {
          publishedVersion: '1.0.3',
          previousRef: baseline,
        },
      },
      null,
      2,
    )}\n`,
  );
  writeFileSync(join(cwd, 'feature.txt'), 'GitHub 1.0.4\n');
  commit(cwd, 'feat: GitHub 1.0.4 change', 'Feature details.');
  commit(cwd, 'chore(release): v1.0.4');
  git(cwd, 'tag', '-a', 'v1.0.4', '-m', 'v1.0.4');

  writeFileSync(join(cwd, 'refactor.txt'), 'GitHub 1.0.5\n');
  commit(cwd, 'refactor: GitHub 1.0.5 change', 'Refactor details.');
  commit(cwd, 'chore(release): v1.0.5');
  git(cwd, 'tag', '-a', 'v1.0.5', '-m', 'v1.0.5');

  writeFileSync(join(cwd, 'publishing.txt'), 'npm publishing\n');
  commit(cwd, 'ci: add synchronized npm publishing', 'Publishing details.');

  const outputPath = join(cwd, 'github-output.txt');
  execFileSync(process.execPath, [prepareRelease], {
    cwd,
    env: {
      ...process.env,
      GITHUB_OUTPUT: outputPath,
      GITHUB_REPOSITORY: 'chaobrain/BrainX-skill',
      NPM_PUBLISHED_VERSION: '1.0.3',
    },
  });

  const packageJson = JSON.parse(
    readFileSync(join(cwd, 'package.json'), 'utf8'),
  );
  const notes = readFileSync(join(cwd, 'RELEASE_NOTES.md'), 'utf8');
  const output = readFileSync(outputPath, 'utf8');

  assert.equal(packageJson.version, '1.0.6');
  assert.match(output, /^skip=false$/m);
  assert.match(output, /^version=1\.0\.6$/m);
  assert.match(output, /^commit_count=3$/m);
  assert.match(notes, /GitHub 1\.0\.4 change/);
  assert.match(notes, /GitHub 1\.0\.5 change/);
  assert.match(notes, /synchronized npm publishing/);
  assert.doesNotMatch(notes, /chore\(release\):/);
  assert.match(notes, new RegExp(`Previous release:\\*\\* \`${baseline}\``));
  assert.match(notes, new RegExp(`compare/${baseline}\\.\\.\\.v1\\.0\\.6`));
});
