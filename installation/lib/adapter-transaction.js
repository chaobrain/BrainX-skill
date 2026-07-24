'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const { digestTree } = require('./hash');
const { samePath } = require('./paths');

class AdapterOperationError extends Error {
  constructor(adapter, affectedPath, action, resolution, cause) {
    super(cause ? cause.message : action);
    this.name = 'AdapterOperationError';
    this.adapter = adapter;
    this.affectedPath = affectedPath;
    this.action = action;
    this.resolution = resolution;
    this.cause = cause;
  }
}

async function pathExists(targetPath, fsApi) {
  try {
    await fsApi.lstat(targetPath);
    return true;
  } catch (error) {
    if (error.code === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function assertNoSymlinkComponents(targetPath, boundaryPath, fsApi, pathApi) {
  let current = pathApi.resolve(targetPath);
  const boundary = pathApi.resolve(boundaryPath);

  while (true) {
    try {
      const stat = await fsApi.lstat(current);
      if (stat.isSymbolicLink()) {
        throw new Error(`Symbolic-link destinations are not supported: ${current}`);
      }
    } catch (error) {
      if (error.code !== 'ENOENT') {
        throw error;
      }
    }

    if (samePath(current, boundary, pathApi)) {
      return;
    }
    const parent = pathApi.dirname(current);
    if (samePath(parent, current, pathApi)) {
      throw new Error(`Destination is outside the expected home directory: ${targetPath}`);
    }
    current = parent;
  }
}

async function isCurrentInstallation(record, bundle, destination, options) {
  if (!record || record.packageVersion !== bundle.version) {
    return false;
  }
  if (record.skills.length !== bundle.skills.length) {
    return false;
  }

  const ownedByName = new Map(record.skills.map((skill) => [skill.name, skill]));
  for (const skill of bundle.skills) {
    const owned = ownedByName.get(skill.name);
    if (!owned || owned.digest !== skill.digest) {
      return false;
    }
    const targetPath = options.pathApi.join(destination, skill.name);
    try {
      const stat = await options.fsApi.lstat(targetPath);
      if (!stat.isDirectory()) {
        return false;
      }
      const digest = await digestTree(targetPath, options);
      if (digest !== skill.digest) {
        return false;
      }
    } catch (error) {
      if (error.code === 'ENOENT') {
        return false;
      }
      throw error;
    }
  }
  return true;
}

function createReceiptRecord(adapter, destination, bundle, previousRecord, timestamp, pathApi) {
  return {
    label: adapter.label,
    destination,
    packageVersion: bundle.version,
    skills: bundle.skills.map((skill) => ({
      name: skill.name,
      directory: pathApi.join(destination, skill.name),
      digest: skill.digest,
    })),
    installedAt: previousRecord ? previousRecord.installedAt : timestamp,
    updatedAt: timestamp,
  };
}

async function prepareAdapter(adapter, destination, bundle, previousRecord, options = {}) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  const uuid = options.uuid || crypto.randomUUID;
  const copySkill = options.copySkill || (async (source, target) => {
    await fsApi.cp(source, target, {
      recursive: true,
      errorOnExist: true,
      force: false,
      dereference: false,
    });
  });
  const transactionOptions = { fsApi, pathApi };

  try {
    await assertNoSymlinkComponents(
      destination,
      options.homeDir || pathApi.dirname(destination),
      fsApi,
      pathApi,
    );
  } catch (error) {
    throw new AdapterOperationError(
      adapter,
      destination,
      'The destination was not changed because it uses an unsafe filesystem path.',
      'Use a real directory under the user home instead of a symbolic link, then retry.',
      error,
    );
  }

  if (await isCurrentInstallation(previousRecord, bundle, destination, transactionOptions)) {
    return {
      status: 'current',
      adapter,
      destination,
      record: previousRecord,
    };
  }

  for (const skill of bundle.skills) {
    const targetPath = pathApi.join(destination, skill.name);
    const isOwned = previousRecord && previousRecord.skills.some(
      (owned) => samePath(owned.directory, targetPath, pathApi),
    );
    if (await pathExists(targetPath, fsApi) && !isOwned) {
      throw new AdapterOperationError(
        adapter,
        targetPath,
        'The existing directory was not replaced because BrainX cannot prove ownership.',
        'Move or remove the conflicting directory, then run the command again.',
      );
    }
  }

  const operationId = uuid();
  const stageRoot = pathApi.join(destination, `.brainx-stage-${operationId}`);
  const backupRoot = pathApi.join(destination, `.brainx-backup-${operationId}`);

  try {
    await fsApi.mkdir(destination, { recursive: true });
    await fsApi.mkdir(stageRoot);
    for (const skill of bundle.skills) {
      const stagedPath = pathApi.join(stageRoot, skill.name);
      await copySkill(skill.sourcePath, stagedPath, { adapter, skill });
      const stagedDigest = await digestTree(stagedPath, transactionOptions);
      if (stagedDigest !== skill.digest) {
        throw new Error(`Staged validation failed for ${skill.name}`);
      }
    }
  } catch (error) {
    await fsApi.rm(stageRoot, { recursive: true, force: true }).catch(() => {});
    await fsApi.rm(backupRoot, { recursive: true, force: true }).catch(() => {});
    throw new AdapterOperationError(
      adapter,
      destination,
      'The staged BrainX installation was not applied.',
      'Check the destination permissions and available disk space, then run the command again.',
      error,
    );
  }

  const timestamp = options.timestamp;
  const movedBackups = [];
  const installedPaths = [];
  let committed = false;

  async function rollback() {
    const failures = [];
    for (const installedPath of [...installedPaths].reverse()) {
      try {
        await fsApi.rm(installedPath, { recursive: true, force: true });
      } catch (error) {
        failures.push(error);
      }
    }
    for (const moved of [...movedBackups].reverse()) {
      try {
        if (await pathExists(moved.targetPath, fsApi)) {
          throw new Error(`Cannot restore backup because the target exists: ${moved.targetPath}`);
        }
        await fsApi.rename(moved.backupPath, moved.targetPath);
      } catch (error) {
        failures.push(error);
      }
    }
    await fsApi.rm(stageRoot, { recursive: true, force: true }).catch((error) => failures.push(error));
    await fsApi.rm(backupRoot, { recursive: true, force: true }).catch((error) => failures.push(error));
    committed = false;
    if (failures.length) {
      throw new Error(`Rollback failed: ${failures.map((error) => error.message).join('; ')}`);
    }
  }

  async function commit() {
    try {
      await fsApi.mkdir(backupRoot);
      for (const owned of previousRecord ? previousRecord.skills : []) {
        if (!await pathExists(owned.directory, fsApi)) {
          continue;
        }
        const backupPath = pathApi.join(backupRoot, owned.name);
        await fsApi.rename(owned.directory, backupPath);
        movedBackups.push({ targetPath: owned.directory, backupPath });
      }
      for (const skill of bundle.skills) {
        const targetPath = pathApi.join(destination, skill.name);
        if (await pathExists(targetPath, fsApi)) {
          throw new Error(`Target appeared during installation: ${targetPath}`);
        }
        await fsApi.rename(pathApi.join(stageRoot, skill.name), targetPath);
        installedPaths.push(targetPath);
      }
      committed = true;
    } catch (error) {
      let rollbackError;
      try {
        await rollback();
      } catch (caught) {
        rollbackError = caught;
      }
      const cause = rollbackError
        ? new Error(`${error.message}; ${rollbackError.message}`)
        : error;
      throw new AdapterOperationError(
        adapter,
        destination,
        'The BrainX installation was rolled back and not applied.',
        'Check destination permissions and conflicting files, then run the command again.',
        cause,
      );
    }
  }

  async function finalize() {
    if (!committed) {
      return;
    }
    await fsApi.rm(stageRoot, { recursive: true, force: true });
    await fsApi.rm(backupRoot, { recursive: true, force: true });
  }

  return {
    status: 'prepared',
    adapter,
    destination,
    record: createReceiptRecord(
      adapter,
      destination,
      bundle,
      previousRecord,
      timestamp,
      pathApi,
    ),
    commit,
    finalize,
    rollback,
  };
}

module.exports = {
  AdapterOperationError,
  assertNoSymlinkComponents,
  isCurrentInstallation,
  pathExists,
  prepareAdapter,
};
