import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const workflow = readFileSync(
  new URL('../workflows/release-notes.yml', import.meta.url),
  'utf8',
);
const packageJson = JSON.parse(
  readFileSync(new URL('../../package.json', import.meta.url), 'utf8'),
);

test('release workflow remains manual and uses the npm environment', () => {
  assert.match(workflow, /^on:\n  workflow_dispatch:\n/m);
  assert.doesNotMatch(workflow, /^  (push|schedule):/m);
  assert.match(workflow, /environment:\n      name: npm/);
  assert.match(workflow, /id-token: write/);
  assert.match(workflow, /github\.ref == 'refs\/heads\/main'/);
});

test('release workflow publishes npm before the GitHub Release', () => {
  const npmPublish = workflow.indexOf('- name: Publish npm package');
  const githubRelease = workflow.indexOf(
    '- name: Publish detailed GitHub Release',
  );

  assert.ok(npmPublish > 0);
  assert.ok(githubRelease > npmPublish);
  assert.match(workflow, /npm publish --provenance --access public/);
  assert.doesNotMatch(workflow, /NPM_TOKEN|NODE_AUTH_TOKEN/);
});

test('release recovery only runs when the GitHub Release is missing', () => {
  const recovery = workflow.slice(
    workflow.indexOf('- name: Recover an incomplete tagged release'),
    workflow.indexOf('- name: Read latest npm version'),
  );
  const afterRecovery = workflow.slice(
    workflow.indexOf('- name: Read latest npm version'),
  );

  assert.match(recovery, /id: recovery/);
  assert.match(recovery, /! gh release view/);
  assert.match(recovery, /npm-registry\.mjs exists/);
  assert.ok(recovery.indexOf('npm publish') < recovery.indexOf('gh release create'));
  assert.match(recovery, /recovered=true/);
  assert.equal(
    (
      afterRecovery.match(
        /steps\.recovery\.outputs\.recovered != 'true'/g,
      ) ?? []
    ).length,
    7,
  );
});

test('package metadata targets this repository with a semantic version', () => {
  assert.match(packageJson.version, /^\d+\.\d+\.\d+$/);
  assert.equal(
    packageJson.repository.url,
    'git+https://github.com/chaobrain/BrainX-skill.git',
  );
  assert.equal(
    packageJson.homepage,
    'https://github.com/chaobrain/BrainX-skill#readme',
  );
  assert.equal(
    packageJson.bugs.url,
    'https://github.com/chaobrain/BrainX-skill/issues',
  );
  assert.deepEqual(packageJson.publishConfig, {
    access: 'public',
    provenance: true,
  });
});
