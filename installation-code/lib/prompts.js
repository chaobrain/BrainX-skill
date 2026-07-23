'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');

class NonInteractiveInstallError extends Error {
  constructor() {
    super('Interactive installation requires a terminal.');
    this.name = 'NonInteractiveInstallError';
  }
}

class InstallCancelledError extends Error {
  constructor() {
    super('Installation cancelled.');
    this.name = 'InstallCancelledError';
  }
}

function harnessLabel(adapter) {
  return adapter.promptLabel || adapter.label;
}

function harnessRoot(adapter) {
  return adapter.homePath.slice(0, -1);
}

async function isDirectory(directory, fsApi) {
  try {
    return (await fsApi.stat(directory)).isDirectory();
  } catch (error) {
    if (error.code === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function detectHarnesses(options) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  const detected = [];
  const seen = new Set();

  for (const adapter of options.adapters) {
    if (seen.has(adapter.id)) {
      continue;
    }
    seen.add(adapter.id);
    const root = harnessRoot(adapter);
    const globalRoot = pathApi.resolve(options.homeDir, ...root);
    const projectRoot = pathApi.resolve(options.cwd, ...root);
    const globalDetected = await isDirectory(globalRoot, fsApi);
    const projectDetected = globalRoot === projectRoot
      ? globalDetected
      : await isDirectory(projectRoot, fsApi);

    if (globalDetected || projectDetected) {
      detected.push({
        id: adapter.id,
        label: harnessLabel(adapter),
        displayPath: globalDetected ? `~/${root.join('/')}` : projectRoot,
      });
    }
  }

  return detected;
}

function createPromptApi(input, output) {
  const { AutoComplete, Select } = require('enquirer');

  async function run(PromptType, options) {
    const prompt = new PromptType({ ...options, stdin: input, stdout: output });
    const cancel = prompt.cancel.bind(prompt);
    prompt.cancel = async function cancelSafely(error) {
      if (typeof this.stop === 'function') {
        const stop = this.stop;
        this.removeListener('close', stop);
        try {
          stop();
        } catch (cleanupError) {
          if (cleanupError.code !== 'ERR_USE_AFTER_CLOSE') {
            throw cleanupError;
          }
        }
        this.stop = undefined;
      }
      return cancel(error);
    };

    try {
      return await prompt.run();
    } catch (error) {
      if (prompt.state.cancelled) {
        throw new InstallCancelledError();
      }
      throw error;
    }
  }

  return {
    select(options) {
      return run(Select, options);
    },
    multiSelect(options) {
      return run(AutoComplete, {
        ...options,
        multiple: true,
      });
    },
  };
}

function writeDetectedHarnesses(output, detected) {
  output.write('Detected harnesses\n\n');
  if (detected.length === 0) {
    output.write('  None\n\n');
    return;
  }
  const width = Math.max(...detected.map((item) => item.label.length));
  for (const item of detected) {
    output.write(`  ${item.label.padEnd(width)}  ${item.displayPath}\n`);
  }
  output.write('\n');
}

async function promptForInstall(options) {
  const input = options.stdin || process.stdin;
  const output = options.stdout || process.stdout;
  if (!input.isTTY) {
    throw new NonInteractiveInstallError();
  }

  const detected = options.detected || await detectHarnesses(options);
  const detectedIds = detected.map((item) => item.id);
  const prompt = options.promptApi || createPromptApi(input, output);
  writeDetectedHarnesses(output, detected);

  let selectedHarnessIds;
  if (detectedIds.length > 0) {
    const mode = await prompt.select({
      name: 'mode',
      message: 'Install for detected harnesses only, or add more?',
      initial: 0,
      choices: [
        {
          name: 'detected',
          message: `Detected only (${detectedIds.join(', ')})`,
        },
        { name: 'customize', message: 'Customize...' },
      ],
    });
    if (mode === 'detected') {
      selectedHarnessIds = detectedIds;
    }
  }

  if (!selectedHarnessIds) {
    const initiallySelected = detectedIds.length > 0
      ? []
      : options.adapters.map((adapter) => adapter.id);
    selectedHarnessIds = await prompt.multiSelect({
      name: 'harnesses',
      message: 'Select harnesses',
      hint: 'Type to filter, space to select, enter to confirm',
      emptyError: ' ',
      initial: initiallySelected,
      symbols: {
        indicator: { on: '\u25cf', off: '\u25cb' },
      },
      choices: options.adapters.map((adapter) => ({
        name: adapter.id,
        message: harnessLabel(adapter),
        hint: `(${adapter.homePath.join('/')})`,
      })),
      validate(value) {
        return value.length > 0 || 'Select at least one harness.';
      },
      footer() {
        const labels = this.selected.map((choice) => choice.message);
        return labels.length > 0 ? `Selected: ${labels.join(', ')}` : '';
      },
    });
  }

  const scope = await prompt.select({
    name: 'scope',
    message: 'Install location',
    initial: 1,
    choices: [
      { name: 'project', message: 'Project (current repository)' },
      { name: 'global', message: 'Global (~)' },
    ],
  });

  return { selectedHarnessIds, scope };
}

module.exports = {
  InstallCancelledError,
  NonInteractiveInstallError,
  detectHarnesses,
  promptForInstall,
  writeDetectedHarnesses,
};
