'use strict';

const { existsSync } = require('node:fs');
const { win32 } = require('node:path');

function windowsInvocation(binary, args, packageEntry, { pathValue, exists }) {
  let executable = binary;
  if (!/[\\/]/.test(binary)) {
    const names = win32.extname(binary) ? [binary] : [`${binary}.exe`, `${binary}.cmd`, binary];
    for (const directory of pathValue.split(';')) {
      const candidate = names.map((name) => win32.join(directory.replace(/^"|"$/g, ''), name)).find(exists);
      if (candidate) {
        executable = candidate;
        break;
      }
    }
  }
  if (!/\.(cmd|bat|ps1)$/i.test(executable)) return { binary: executable, prefixArgs: args };

  const directory = win32.dirname(executable);
  const modules = win32.basename(directory) === '.bin'
    ? win32.dirname(directory) : win32.join(directory, 'node_modules');
  const entry = win32.join(modules, ...packageEntry);
  if (!exists(entry)) throw new Error(`Cannot find the JavaScript entry for ${executable}. Use the installed executable or a standard npm shim.`);
  return { binary: process.execPath, prefixArgs: [entry, ...args] };
}

function codexInvocation(binary, prefixArgs = [], {
  platform = process.platform, pathValue = process.env.PATH || '', exists = existsSync,
} = {}) {
  if (platform !== 'win32') return { binary, prefixArgs };
  try {
    return windowsInvocation(binary, prefixArgs, ['@openai', 'codex', 'bin', 'codex.js'], { pathValue, exists });
  } catch (error) {
    throw new Error(`${error.message} Set BRAINX_CODEX_BIN to a supported Codex executable.`, { cause: error });
  }
}

function npmInvocation(binary, prefixArgs, {
  platform = process.platform, pathValue = process.env.PATH || '', exists = existsSync,
  npmExecPath = process.env.npm_execpath,
} = {}) {
  if (platform !== 'win32') return { binary, prefixArgs };
  if (binary === 'npm' && npmExecPath && win32.basename(npmExecPath) === 'npm-cli.js' && exists(npmExecPath)) {
    return { binary: process.execPath, prefixArgs: [npmExecPath, ...prefixArgs] };
  }
  return windowsInvocation(binary, prefixArgs, ['npm', 'bin', 'npm-cli.js'], { pathValue, exists });
}

module.exports = { codexInvocation, npmInvocation };
