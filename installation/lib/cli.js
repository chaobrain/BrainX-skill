'use strict';

const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { DEFAULT_ADAPTERS, runInstaller } = require('./installer');
const {
  InstallCancelledError,
  NonInteractiveInstallError,
  promptForInstall,
} = require('./prompts');

const USAGE = `Usage:
  npx brainx-skill install
  npx brainx-skill update`;

function parseCommand(args) {
  if (args.length !== 1 || (args[0] !== 'install' && args[0] !== 'update')) {
    return null;
  }
  return args[0];
}

async function runCli(args, options = {}) {
  const stderr = options.stderr || process.stderr;
  const command = parseCommand(args);
  if (!command) {
    stderr.write(`${USAGE}\n`);
    return 1;
  }
  const installer = options.installer || runInstaller;
  if (command === 'update') {
    return installer(command, options);
  }

  const selectInstall = options.promptForInstall || promptForInstall;
  const homeDir = options.homeDir || os.homedir();
  const cwd = options.cwd || process.cwd();
  let selection;
  try {
    selection = await selectInstall({
      adapters: options.adapters || DEFAULT_ADAPTERS,
      fsApi: options.fsApi || fs,
      pathApi: options.pathApi || path,
      homeDir,
      cwd,
      stdin: options.stdin || process.stdin,
      stdout: options.stdout || process.stdout,
    });
  } catch (error) {
    if (error instanceof InstallCancelledError) {
      stderr.write('Installation cancelled.\n');
      return 130;
    }
    if (error instanceof NonInteractiveInstallError) {
      stderr.write('Interactive installation requires a terminal.\n');
      stderr.write('Run this command in an interactive shell.\n');
      return 1;
    }
    throw error;
  }

  return installer(command, {
    ...options,
    ...selection,
    homeDir,
    cwd,
  });
}

module.exports = {
  USAGE,
  parseCommand,
  runCli,
};
