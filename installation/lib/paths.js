'use strict';

const path = require('node:path');

function adapterSkillPath(adapter, scope = 'global') {
  if (scope === 'project' && adapter.projectPath) {
    return adapter.projectPath;
  }
  return adapter.homePath;
}

function adapterSkillPaths(adapter) {
  return adapter.projectPath ? [adapter.homePath, adapter.projectPath] : [adapter.homePath];
}

function resolveLocations(homeDir, adapters, pathApi = path, options = {}) {
  if (typeof homeDir !== 'string' || !pathApi.isAbsolute(homeDir)) {
    throw new Error(`Home directory must be an absolute path: ${homeDir}`);
  }

  const scope = options.scope || 'global';
  if (scope !== 'global' && scope !== 'project') {
    throw new Error(`Unknown installation scope: ${scope}`);
  }
  const baseDir = scope === 'global' ? homeDir : options.cwd;
  if (typeof baseDir !== 'string' || !pathApi.isAbsolute(baseDir)) {
    throw new Error(`Installation base directory must be an absolute path: ${baseDir}`);
  }

  const destinations = {};
  for (const adapter of adapters) {
    destinations[adapter.id] = pathApi.resolve(baseDir, ...adapterSkillPath(adapter, scope));
  }

  return {
    homeDir: pathApi.resolve(homeDir),
    baseDir: pathApi.resolve(baseDir),
    stateDir: pathApi.resolve(homeDir, '.brainx'),
    receiptPath: pathApi.resolve(homeDir, '.brainx', 'receipt.json'),
    destinations,
  };
}

function stripSkillPath(destination, skillPath, pathApi) {
  let current = pathApi.resolve(destination);
  for (const part of [...skillPath].reverse()) {
    const name = pathApi.basename(current);
    const matches = pathApi === path.win32
      ? name.toLowerCase() === part.toLowerCase()
      : name === part;
    if (!matches) {
      return null;
    }
    current = pathApi.dirname(current);
  }
  return current;
}

function resolveDestinationRoot(destination, adapter, pathApi = path) {
  for (const skillPath of adapterSkillPaths(adapter)) {
    const root = stripSkillPath(destination, skillPath, pathApi);
    if (root) {
      return root;
    }
  }
  return null;
}

function samePath(left, right, pathApi = path) {
  const normalizedLeft = pathApi.normalize(pathApi.resolve(left));
  const normalizedRight = pathApi.normalize(pathApi.resolve(right));
  if (pathApi === path.win32) {
    return normalizedLeft.toLowerCase() === normalizedRight.toLowerCase();
  }
  return normalizedLeft === normalizedRight;
}

function groupAdaptersByDestination(adapters, destinations, pathApi = path) {
  const groups = [];
  for (const adapter of adapters) {
    const destination = destinations[adapter.id];
    const existing = groups.find((group) => samePath(group.destination, destination, pathApi));
    if (existing) {
      existing.adapters.push(adapter);
      continue;
    }
    groups.push({ destination, adapters: [adapter] });
  }
  return groups;
}

module.exports = {
  adapterSkillPath,
  adapterSkillPaths,
  groupAdaptersByDestination,
  resolveDestinationRoot,
  resolveLocations,
  samePath,
};
