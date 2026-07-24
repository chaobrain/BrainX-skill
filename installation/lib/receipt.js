'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const {
  PACKAGE_NAME,
  RECEIPT_SCHEMA_VERSION,
  SKILL_NAME_PATTERN,
} = require('./constants');
const { resolveDestinationRoot, samePath } = require('./paths');

class ReceiptValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ReceiptValidationError';
  }
}

function validateReceipt(receipt, adapters, destinations, pathApi = path) {
  if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt)) {
    throw new ReceiptValidationError('BrainX receipt must be a JSON object');
  }
  if (receipt.schemaVersion !== RECEIPT_SCHEMA_VERSION) {
    throw new ReceiptValidationError(`Unsupported BrainX receipt schema: ${receipt.schemaVersion}`);
  }
  if (receipt.packageName !== PACKAGE_NAME) {
    throw new ReceiptValidationError(`Receipt belongs to a different package: ${receipt.packageName}`);
  }
  if (!receipt.adapters || typeof receipt.adapters !== 'object' || Array.isArray(receipt.adapters)) {
    throw new ReceiptValidationError('BrainX receipt must contain adapter records');
  }

  const adapterById = new Map(adapters.map((adapter) => [adapter.id, adapter]));
  for (const id of Object.keys(receipt.adapters)) {
    if (!adapterById.has(id)) {
      throw new ReceiptValidationError(`Unknown adapter in BrainX receipt: ${id}`);
    }
  }

  for (const adapter of adapters) {
    const record = receipt.adapters[adapter.id];
    if (record === undefined) {
      continue;
    }
    if (!record || typeof record !== 'object' || Array.isArray(record)) {
      throw new ReceiptValidationError(`Invalid ${adapter.label} receipt record`);
    }
    if (record.label !== adapter.label || typeof record.destination !== 'string') {
      throw new ReceiptValidationError(`Invalid ${adapter.label} receipt identity`);
    }
    const expectedDestination = destinations && destinations[adapter.id];
    if (expectedDestination && !samePath(record.destination, expectedDestination, pathApi)) {
      throw new ReceiptValidationError(
        `${adapter.label} receipt destination does not match ${expectedDestination}`,
      );
    }
    if (!pathApi.isAbsolute(record.destination)
      || !resolveDestinationRoot(record.destination, adapter, pathApi)) {
      throw new ReceiptValidationError(
        `Unsafe ${adapter.label} receipt destination: ${record.destination}`,
      );
    }
    if (typeof record.packageVersion !== 'string' || record.packageVersion.length === 0) {
      throw new ReceiptValidationError(`Invalid ${adapter.label} package version in receipt`);
    }
    if (!Array.isArray(record.skills) || record.skills.length === 0) {
      throw new ReceiptValidationError(`Invalid ${adapter.label} skill ownership list`);
    }

    const seen = new Set();
    for (const skill of record.skills) {
      if (!skill || typeof skill !== 'object' || !SKILL_NAME_PATTERN.test(skill.name || '')) {
        throw new ReceiptValidationError(`Invalid ${adapter.label} skill ownership entry`);
      }
      if (seen.has(skill.name)) {
        throw new ReceiptValidationError(`Duplicate ${adapter.label} skill in receipt: ${skill.name}`);
      }
      seen.add(skill.name);
      const expectedPath = pathApi.join(record.destination, skill.name);
      if (typeof skill.directory !== 'string' || !samePath(skill.directory, expectedPath, pathApi)) {
        throw new ReceiptValidationError(
          `Unsafe ${adapter.label} owned path for ${skill.name}: ${skill.directory}`,
        );
      }
      if (typeof skill.digest !== 'string' || !/^[a-f0-9]{64}$/.test(skill.digest)) {
        throw new ReceiptValidationError(`Invalid ${adapter.label} digest for ${skill.name}`);
      }
    }
  }

  return receipt;
}

async function readReceipt(receiptPath, adapters, destinations, options = {}) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  let contents;
  try {
    contents = await fsApi.readFile(receiptPath, 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT') {
      return null;
    }
    throw new ReceiptValidationError(`Cannot read BrainX receipt at ${receiptPath}: ${error.message}`);
  }

  let receipt;
  try {
    receipt = JSON.parse(contents);
  } catch (error) {
    throw new ReceiptValidationError(`Invalid JSON in BrainX receipt at ${receiptPath}: ${error.message}`);
  }
  return validateReceipt(receipt, adapters, destinations, pathApi);
}

async function writeReceiptAtomic(receiptPath, receipt, options = {}) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  const uuid = options.uuid || crypto.randomUUID;
  const stateDir = pathApi.dirname(receiptPath);
  const temporaryPath = pathApi.join(stateDir, `.receipt-${uuid()}.tmp`);

  await fsApi.mkdir(stateDir, { recursive: true });
  try {
    await fsApi.writeFile(temporaryPath, `${JSON.stringify(receipt, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
      mode: 0o600,
    });
    await fsApi.rename(temporaryPath, receiptPath);
  } catch (error) {
    await fsApi.rm(temporaryPath, { force: true }).catch(() => {});
    throw error;
  }
}

module.exports = {
  ReceiptValidationError,
  readReceipt,
  validateReceipt,
  writeReceiptAtomic,
};
