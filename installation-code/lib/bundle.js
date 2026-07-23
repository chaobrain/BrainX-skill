'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { PACKAGE_NAME, SKILL_NAME_PATTERN } = require('./constants');
const { digestTree } = require('./hash');

class BundleValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BundleValidationError';
  }
}

async function readJson(filePath, fsApi) {
  let contents;
  try {
    contents = await fsApi.readFile(filePath, 'utf8');
  } catch (error) {
    throw new BundleValidationError(`Cannot read ${filePath}: ${error.message}`);
  }

  try {
    return JSON.parse(contents);
  } catch (error) {
    throw new BundleValidationError(`Invalid JSON in ${filePath}: ${error.message}`);
  }
}

async function validatePortableTree(root, fsApi, pathApi) {
  const entries = await fsApi.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const entryPath = pathApi.join(root, entry.name);
    if (entry.isDirectory()) {
      await validatePortableTree(entryPath, fsApi, pathApi);
    } else if (!entry.isFile()) {
      throw new BundleValidationError(
        `Skill bundles may contain only regular files and directories: ${entryPath}`,
      );
    }
  }
}

async function validateBundle(packageRoot, options = {}) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  const packageJsonPath = pathApi.join(packageRoot, 'package.json');
  const manifestPath = pathApi.join(packageRoot, 'manifest.json');
  const skillsRoot = pathApi.join(packageRoot, 'skills');
  const packageJson = await readJson(packageJsonPath, fsApi);
  const manifest = await readJson(manifestPath, fsApi);

  if (packageJson.name !== PACKAGE_NAME) {
    throw new BundleValidationError(
      `Expected package name ${PACKAGE_NAME}, found ${packageJson.name || 'missing'}`,
    );
  }
  if (typeof packageJson.version !== 'string' || packageJson.version.length === 0) {
    throw new BundleValidationError('package.json must contain a package version');
  }
  if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.skills)) {
    throw new BundleValidationError('manifest.json must use schemaVersion 1 and contain a skills array');
  }

  const skillNames = [];
  const seen = new Set();
  for (const name of manifest.skills) {
    if (typeof name !== 'string' || !SKILL_NAME_PATTERN.test(name)) {
      throw new BundleValidationError(`Invalid or unsafe skill name in manifest: ${name}`);
    }
    if (seen.has(name)) {
      throw new BundleValidationError(`Duplicate skill in manifest: ${name}`);
    }
    seen.add(name);
    skillNames.push(name);
  }
  if (skillNames.length === 0) {
    throw new BundleValidationError('manifest.json must declare at least one skill');
  }

  let topLevelEntries;
  try {
    topLevelEntries = await fsApi.readdir(skillsRoot, { withFileTypes: true });
  } catch (error) {
    throw new BundleValidationError(`Cannot read skills directory: ${error.message}`);
  }

  const actualNames = [];
  for (const entry of topLevelEntries) {
    if (!entry.isDirectory()) {
      throw new BundleValidationError(
        `The skills directory may contain only declared skill directories: ${entry.name}`,
      );
    }
    actualNames.push(entry.name);
  }
  const missing = skillNames.filter((name) => !actualNames.includes(name));
  if (missing.length) {
    throw new BundleValidationError(
      `Manifest declares missing skill directories: ${missing.join(', ')}`,
    );
  }

  const skills = [];
  for (const name of skillNames) {
    const sourcePath = pathApi.join(skillsRoot, name);
    const skillFile = pathApi.join(sourcePath, 'SKILL.md');
    let stat;
    try {
      stat = await fsApi.lstat(skillFile);
    } catch (error) {
      throw new BundleValidationError(`Missing SKILL.md for ${name}: ${error.message}`);
    }
    if (!stat.isFile() || stat.size === 0) {
      throw new BundleValidationError(`SKILL.md for ${name} must be a nonempty regular file`);
    }
    await validatePortableTree(sourcePath, fsApi, pathApi);
    skills.push({
      name,
      sourcePath,
      digest: await digestTree(sourcePath, { fsApi, pathApi }),
    });
  }

  return {
    packageName: packageJson.name,
    version: packageJson.version,
    packageRoot,
    skills,
  };
}

module.exports = {
  BundleValidationError,
  validateBundle,
};
