import assert from 'node:assert/strict';
import test from 'node:test';

import {
  bumpPatch,
  compareVersions,
  parseCommitLog,
  renderReleaseNotes,
  selectNextVersion,
  selectPreviousRef,
  updateChangelog,
} from './prepare-release.mjs';

test('bumpPatch increments only the patch component', () => {
  assert.equal(bumpPatch('1.0.3'), '1.0.4');
  assert.equal(bumpPatch('12.8.99'), '12.8.100');
  assert.throws(() => bumpPatch('1.0'), /Expected a semantic version/);
});

test('compareVersions orders semantic versions numerically', () => {
  assert.equal(compareVersions('1.0.3', '1.0.5'), -1);
  assert.equal(compareVersions('1.10.0', '1.9.9'), 1);
  assert.equal(compareVersions('2.4.6', '2.4.6'), 0);
  assert.throws(() => compareVersions('latest', '1.0.5'), /semantic version/);
});

test('selectNextVersion skips versions below the configured release floor', () => {
  assert.equal(selectNextVersion('1.0.6', '1.0.8'), '1.0.8');
  assert.equal(selectNextVersion('1.0.8', '1.0.8'), '1.0.9');
  assert.equal(selectNextVersion('1.0.8'), '1.0.9');
});

test('selectPreviousRef uses the latest tag when npm and Git are synchronized', () => {
  assert.equal(
    selectPreviousRef({
      packageVersion: '1.0.5',
      npmPublishedVersion: '1.0.5',
      latestTag: 'v1.0.5',
      npmVersionTagExists: true,
    }),
    'v1.0.5',
  );
});

test('selectPreviousRef uses the configured baseline for the npm catch-up', () => {
  assert.equal(
    selectPreviousRef({
      packageVersion: '1.0.5',
      npmPublishedVersion: '1.0.3',
      latestTag: 'v1.0.5',
      npmVersionTagExists: false,
      catchUp: {
        publishedVersion: '1.0.3',
        previousRef: 'e2b723e489a645fab5d42044a1b73ab21f69b38e',
      },
    }),
    'e2b723e489a645fab5d42044a1b73ab21f69b38e',
  );
});

test('selectPreviousRef prefers an npm version tag when it exists', () => {
  assert.equal(
    selectPreviousRef({
      packageVersion: '1.0.5',
      npmPublishedVersion: '1.0.4',
      latestTag: 'v1.0.5',
      npmVersionTagExists: true,
      catchUp: {
        publishedVersion: '1.0.3',
        previousRef: 'unused',
      },
    }),
    'v1.0.4',
  );
});

test('selectPreviousRef rejects inconsistent or unknown version state', () => {
  assert.throws(
    () =>
      selectPreviousRef({
        packageVersion: '1.0.5',
        npmPublishedVersion: '1.0.4',
        latestTag: 'v1.0.4',
        npmVersionTagExists: true,
      }),
    /does not match/,
  );
  assert.throws(
    () =>
      selectPreviousRef({
        packageVersion: '1.0.5',
        npmPublishedVersion: '1.0.6',
        latestTag: 'v1.0.5',
        npmVersionTagExists: false,
      }),
    /is ahead/,
  );
  assert.throws(
    () =>
      selectPreviousRef({
        packageVersion: '1.0.5',
        npmPublishedVersion: '1.0.3',
        latestTag: 'v1.0.5',
        npmVersionTagExists: false,
      }),
    /No Git comparison baseline/,
  );
});

test('parseCommitLog preserves detailed commit bodies', () => {
  const log =
    'abc123\u001ffix(cli): repair update\u001fExplain the repair.\u001e' +
    'def456\u001fdocs: add guide\u001fFirst line.\n\nSecond line.\u001e';

  assert.deepEqual(parseCommitLog(log), [
    {
      sha: 'abc123',
      subject: 'fix(cli): repair update',
      body: 'Explain the repair.',
    },
    {
      sha: 'def456',
      subject: 'docs: add guide',
      body: 'First line.\n\nSecond line.',
    },
  ]);
});

test('renderReleaseNotes includes details, files, commits, and comparison link', () => {
  const notes = renderReleaseNotes({
    version: '1.0.4',
    repository: 'chaobrain/BrainX-skill',
    previousRef: 'v1.0.3',
    commits: [
      {
        sha: 'abcdef0123456789',
        subject: 'feat(brainstate): add diagnostics',
        body: 'Adds probe-aware state diagnostics and regression coverage.',
        files: ['skills/package-skills/brainstate/SKILL.md', 'test/diagnostics.test.js'],
      },
    ],
  });

  assert.match(notes, /## New features/);
  assert.match(notes, /Adds probe-aware state diagnostics/);
  assert.match(notes, /skills\/package-skills\/brainstate\/SKILL\.md/);
  assert.match(notes, /commit\/abcdef0123456789/);
  assert.match(notes, /compare\/v1\.0\.3\.\.\.v1\.0\.4/);
});

test('updateChangelog inserts the newest version before the baseline', () => {
  const original = '# Changelog\n\nRelease history.\n\n## 1.0.3\n\nBaseline.\n';
  const updated = updateChangelog(
    original,
    '1.0.4',
    '2026-07-24',
    'Release details.',
  );

  assert.match(updated, /## 1\.0\.4 - 2026-07-24\n\nRelease details\./);
  assert.ok(updated.indexOf('## 1.0.4') < updated.indexOf('## 1.0.3'));
});
