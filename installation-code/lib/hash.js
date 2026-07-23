'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');

async function digestTree(root, options = {}) {
  const fsApi = options.fsApi || fs;
  const pathApi = options.pathApi || path;
  const hash = crypto.createHash('sha256');

  async function walk(directory, relativeDirectory) {
    const entries = await fsApi.readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));

    for (const entry of entries) {
      const absolutePath = pathApi.join(directory, entry.name);
      const relativePath = relativeDirectory
        ? `${relativeDirectory}/${entry.name}`
        : entry.name;

      if (entry.isDirectory()) {
        hash.update(`D\0${relativePath}\0`);
        await walk(absolutePath, relativePath);
      } else if (entry.isFile()) {
        const contents = await fsApi.readFile(absolutePath);
        hash.update(`F\0${relativePath}\0${contents.length}\0`);
        hash.update(contents);
      } else {
        throw new Error(`Unsupported filesystem entry: ${absolutePath}`);
      }
    }
  }

  await walk(root, '');
  return hash.digest('hex');
}

module.exports = {
  digestTree,
};
