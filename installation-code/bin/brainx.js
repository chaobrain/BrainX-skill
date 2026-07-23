#!/usr/bin/env node
'use strict';

const { runCli } = require('../lib/cli');

runCli(process.argv.slice(2))
  .then((exitCode) => {
    process.exitCode = exitCode;
  })
  .catch((error) => {
    process.stderr.write(`x Unexpected BrainX installer failure: ${error.message}\n`);
    process.exitCode = 1;
  });
