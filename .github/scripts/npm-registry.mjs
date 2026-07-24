import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const DEFAULT_REGISTRY = 'https://registry.npmjs.org/';

export function packageDocumentUrl(
  packageName,
  registry = DEFAULT_REGISTRY,
) {
  const base = registry.endsWith('/') ? registry : `${registry}/`;
  return new URL(encodeURIComponent(packageName), base).toString();
}

export async function fetchPackageDocument(
  packageName,
  {
    fetchImpl = globalThis.fetch,
    registry = DEFAULT_REGISTRY,
  } = {},
) {
  if (typeof fetchImpl !== 'function') {
    throw new Error('This command requires a runtime with fetch support');
  }

  const response = await fetchImpl(packageDocumentUrl(packageName, registry), {
    headers: {
      accept: 'application/vnd.npm.install-v1+json',
    },
  });
  if (!response.ok) {
    throw new Error(
      `npm registry returned HTTP ${response.status} for ${packageName}`,
    );
  }

  const document = await response.json();
  if (!document || typeof document !== 'object') {
    throw new Error(`npm registry returned invalid metadata for ${packageName}`);
  }
  return document;
}

export function latestVersion(document) {
  const version = document['dist-tags']?.latest;
  if (typeof version !== 'string' || version.length === 0) {
    throw new Error('npm package metadata has no latest dist-tag');
  }
  return version;
}

export function versionExists(document, version) {
  return Object.hasOwn(document.versions ?? {}, version);
}

export async function main(args = process.argv.slice(2)) {
  const [command, packageName, version] = args;
  if (!['latest', 'exists'].includes(command) || !packageName) {
    throw new Error(
      'Usage: npm-registry.mjs <latest|exists> <package-name> [version]',
    );
  }
  if (command === 'exists' && !version) {
    throw new Error('The exists command requires a version');
  }

  const document = await fetchPackageDocument(packageName);
  if (command === 'latest') {
    process.stdout.write(`${latestVersion(document)}\n`);
  } else {
    process.stdout.write(`${versionExists(document, version)}\n`);
  }
}

const entryPoint = process.argv[1] ? realpathSync(process.argv[1]) : null;
if (entryPoint === fileURLToPath(import.meta.url)) {
  try {
    await main();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
