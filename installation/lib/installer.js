'use strict';

const os = require('node:os');
const fs = require('node:fs/promises');
const path = require('node:path');
const codex = require('../../adapters/codex');
const claude = require('../../adapters/claude');
const cursor = require('../../adapters/cursor');
const windsurf = require('../../adapters/windsurf');
const gemini = require('../../adapters/gemini');
const opencode = require('../../adapters/opencode');
const { prepareAdapter, AdapterOperationError } = require('./adapter-transaction');
const { validateBundle } = require('./bundle');
const { PACKAGE_NAME, RECEIPT_SCHEMA_VERSION } = require('./constants');
const { resolveDestinationRoot, resolveLocations } = require('./paths');
const { readReceipt, writeReceiptAtomic } = require('./receipt');

const DEFAULT_ADAPTERS = [claude, codex, cursor, windsurf, gemini, opencode];

function writeLine(stream, text = '') {
  stream.write(`${text}\n`);
}

function adapterLabel(adapter) {
  return adapter.promptLabel || adapter.label;
}

function formatAdapterFailure(error, stderr) {
  writeLine(stderr, `x ${adapterLabel(error.adapter)} installation failed`);
  writeLine(stderr, `  Path: ${error.affectedPath}`);
  writeLine(stderr, `  Action not performed: ${error.action}`);
  writeLine(stderr, `  Resolve: ${error.resolution}`);
  if (error.cause && error.cause.message) {
    writeLine(stderr, `  Detail: ${error.cause.message}`);
  }
}

function asAdapterError(adapter, destination, error) {
  if (error instanceof AdapterOperationError) {
    return error;
  }
  return new AdapterOperationError(
    adapter,
    destination,
    'BrainX skills were not installed at this destination.',
    'Check the path and its permissions, then run the command again.',
    error,
  );
}

async function runInstaller(command, options = {}) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  const availableAdapters = options.adapters || DEFAULT_ADAPTERS;
  const stdout = options.stdout || process.stdout;
  const stderr = options.stderr || process.stderr;
  const packageRoot = options.packageRoot || path.resolve(__dirname, '../..');
  const homeDir = options.homeDir || os.homedir();
  const cwd = options.cwd || process.cwd();
  const now = options.now || (() => new Date().toISOString());
  const stateLocations = resolveLocations(homeDir, [], pathApi);
  const readReceiptFn = options.readReceipt || readReceipt;
  const writeReceiptFn = options.writeReceipt || writeReceiptAtomic;

  let receipt;
  try {
    receipt = await readReceiptFn(
      stateLocations.receiptPath,
      availableAdapters,
      null,
      { fsApi, pathApi },
    );
  } catch (error) {
    writeLine(stderr, `x Cannot inspect BrainX-managed installation state: ${error.message}`);
    writeLine(stderr, `  Receipt: ${stateLocations.receiptPath}`);
    writeLine(stderr, '  No skill directories were changed.');
    return 1;
  }

  const hasManagedInstallation = Boolean(
    receipt && availableAdapters.some((adapter) => receipt.adapters[adapter.id]),
  );
  if (command === 'update' && !hasManagedInstallation) {
    writeLine(stdout, 'No BrainX-managed skills are currently installed.');
    writeLine(stdout, 'Run: npx brainx-skill install');
    return 0;
  }

  let adapters;
  let destinations;
  if (command === 'update') {
    adapters = availableAdapters.filter((adapter) => receipt.adapters[adapter.id]);
    destinations = Object.fromEntries(adapters.map((adapter) => [
      adapter.id,
      receipt.adapters[adapter.id].destination,
    ]));
  } else {
    const selectedIds = options.selectedHarnessIds || availableAdapters.map((adapter) => adapter.id);
    const selected = new Set(selectedIds);
    if (selected.size === 0 || selectedIds.some((id) => (
      !availableAdapters.some((adapter) => adapter.id === id)
    ))) {
      writeLine(stderr, 'x Invalid harness selection.');
      return 1;
    }
    adapters = availableAdapters.filter((adapter) => selected.has(adapter.id));
    const installLocations = resolveLocations(homeDir, adapters, pathApi, {
      scope: options.scope || 'global',
      cwd,
    });
    destinations = installLocations.destinations;
  }

  let bundle;
  try {
    bundle = await validateBundle(packageRoot, { fsApi, pathApi });
  } catch (error) {
    writeLine(stderr, `x BrainX skill bundle validation failed: ${error.message}`);
    writeLine(stderr, '  No skill directories were changed.');
    return 1;
  }

  const timestamp = now();
  const results = [];
  const committedTransactions = [];
  const failures = [];

  for (const adapter of adapters) {
    const destination = destinations[adapter.id];
    const previousRecord = receipt ? receipt.adapters[adapter.id] : undefined;
    try {
      const transaction = await prepareAdapter(
        adapter,
        destination,
        bundle,
        previousRecord,
        {
          fsApi,
          pathApi,
          uuid: options.uuid,
          timestamp,
          copySkill: options.copySkill,
          homeDir: resolveDestinationRoot(destination, adapter, pathApi),
        },
      );
      if (transaction.status === 'current') {
        results.push(transaction);
        continue;
      }
      await transaction.commit();
      committedTransactions.push(transaction);
      results.push({ ...transaction, status: 'installed' });
    } catch (error) {
      failures.push(asAdapterError(adapter, destination, error));
    }
  }

  if (committedTransactions.length > 0) {
    const nextAdapters = receipt ? { ...receipt.adapters } : {};
    for (const transaction of committedTransactions) {
      nextAdapters[transaction.adapter.id] = transaction.record;
    }
    const nextReceipt = {
      schemaVersion: RECEIPT_SCHEMA_VERSION,
      packageName: PACKAGE_NAME,
      lastOperation: {
        command,
        at: timestamp,
      },
      adapters: nextAdapters,
    };

    try {
      await writeReceiptFn(stateLocations.receiptPath, nextReceipt, {
        fsApi,
        pathApi,
        uuid: options.uuid,
      });
    } catch (error) {
      const rollbackFailures = [];
      for (const transaction of [...committedTransactions].reverse()) {
        try {
          await transaction.rollback();
        } catch (rollbackError) {
          rollbackFailures.push(rollbackError.message);
        }
      }
      writeLine(stderr, `x BrainX receipt could not be written: ${error.message}`);
      writeLine(stderr, `  Receipt: ${stateLocations.receiptPath}`);
      writeLine(stderr, '  Installed changes were rolled back.');
      if (rollbackFailures.length) {
        writeLine(stderr, `  Rollback detail: ${rollbackFailures.join('; ')}`);
      }
      for (const failure of failures) {
        formatAdapterFailure(failure, stderr);
      }
      return 1;
    }

    for (const transaction of committedTransactions) {
      try {
        await transaction.finalize();
      } catch (error) {
        failures.push(new AdapterOperationError(
          transaction.adapter,
          transaction.destination,
          'The installation succeeded, but its temporary backup could not be removed.',
          'Check permissions and remove the reported .brainx-backup directory after verifying the installation.',
          error,
        ));
      }
    }
  }

  const allCurrent = failures.length === 0
    && results.length === adapters.length
    && results.every((result) => result.status === 'current');
  if (allCurrent) {
    writeLine(stdout, `\u2713 BrainX skills ${bundle.version} are already installed`);
  } else {
    for (const adapter of adapters) {
      const result = results.find((candidate) => candidate.adapter.id === adapter.id);
      if (!result) {
        continue;
      }
      const state = result.status === 'current' ? 'are already installed' : 'installed';
      writeLine(stdout, `\u2713 BrainX skills ${bundle.version} ${state} for ${adapterLabel(adapter)}`);
      writeLine(stdout, `  ${result.destination}`);
      writeLine(stdout);
    }
  }

  for (const failure of failures) {
    formatAdapterFailure(failure, stderr);
  }
  return failures.length > 0 ? 1 : 0;
}

module.exports = {
  DEFAULT_ADAPTERS,
  runInstaller,
};
