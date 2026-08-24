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

async function discoverSkillDirectories(skillsRoot, fsApi, pathApi) {
  const discovered = new Map();

  async function walk(container) {
    const entries = await fsApi.readdir(container, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        throw new BundleValidationError(
          `Skill source groups may contain only skill directories: ${pathApi.join(container, entry.name)}`,
        );
      }

      const entryPath = pathApi.join(container, entry.name);
      const skillFile = pathApi.join(entryPath, 'SKILL.md');
      let skillStat;
      try {
        skillStat = await fsApi.lstat(skillFile);
      } catch (error) {
        if (error.code !== 'ENOENT') {
          throw new BundleValidationError(`Cannot inspect ${skillFile}: ${error.message}`);
        }
      }

      if (skillStat) {
        if (discovered.has(entry.name)) {
          throw new BundleValidationError(`Duplicate skill source directory: ${entry.name}`);
        }
        discovered.set(entry.name, entryPath);
      } else {
        await walk(entryPath);
      }
    }
  }

  await walk(skillsRoot);
  return discovered;
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

  let sourceByName;
  try {
    sourceByName = await discoverSkillDirectories(skillsRoot, fsApi, pathApi);
  } catch (error) {
    if (error instanceof BundleValidationError) {
      throw error;
    }
    throw new BundleValidationError(`Cannot read skills directory: ${error.message}`);
  }

  const missing = skillNames.filter((name) => !sourceByName.has(name));
  if (missing.length) {
    throw new BundleValidationError(
      `Manifest declares missing skill directories: ${missing.join(', ')}`,
    );
  }
  const undeclared = [...sourceByName.keys()].filter((name) => !seen.has(name));
  if (undeclared.length) {
    throw new BundleValidationError(
      `Skill directories missing from manifest: ${undeclared.join(', ')}`,
    );
  }

  const skills = [];
  for (const name of skillNames) {
    const sourcePath = sourceByName.get(name);
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
