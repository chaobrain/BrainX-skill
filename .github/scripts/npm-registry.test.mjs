import assert from 'node:assert/strict';
import test from 'node:test';

import {
  fetchPackageDocument,
  latestVersion,
  packageDocumentUrl,
  versionExists,
} from './npm-registry.mjs';

const packageDocument = {
  'dist-tags': {
    latest: '1.0.3',
  },
  versions: {
    '1.0.2': {},
    '1.0.3': {},
  },
};

test('packageDocumentUrl encodes scoped and unscoped package names', () => {
  assert.equal(
    packageDocumentUrl('brainx-skill'),
    'https://registry.npmjs.org/brainx-skill',
  );
  assert.equal(
    packageDocumentUrl('@brainx/skill', 'https://registry.example.test'),
    'https://registry.example.test/%40brainx%2Fskill',
  );
});

test('fetchPackageDocument requests compact npm metadata', async () => {
  let request;
  const document = await fetchPackageDocument('brainx-skill', {
    fetchImpl: async (url, options) => {
      request = { url, options };
      return {
        ok: true,
        async json() {
          return packageDocument;
        },
      };
    },
  });

  assert.deepEqual(document, packageDocument);
  assert.equal(request.url, 'https://registry.npmjs.org/brainx-skill');
  assert.equal(
    request.options.headers.accept,
    'application/vnd.npm.install-v1+json',
  );
});

test('fetchPackageDocument fails closed on registry and metadata errors', async () => {
  await assert.rejects(
    fetchPackageDocument('brainx-skill', {
      fetchImpl: async () => ({
        ok: false,
        status: 503,
      }),
    }),
    /HTTP 503/,
  );
  await assert.rejects(
    fetchPackageDocument('brainx-skill', {
      fetchImpl: async () => ({
        ok: true,
        async json() {
          return null;
        },
      }),
    }),
    /invalid metadata/,
  );
  await assert.rejects(
    fetchPackageDocument('brainx-skill', { fetchImpl: null }),
    /fetch support/,
  );
});

test('latestVersion and versionExists read package state', () => {
  assert.equal(latestVersion(packageDocument), '1.0.3');
  assert.equal(versionExists(packageDocument, '1.0.3'), true);
  assert.equal(versionExists(packageDocument, '1.0.4'), false);
  assert.equal(versionExists({}, '1.0.3'), false);
  assert.throws(() => latestVersion({}), /no latest dist-tag/);
});
