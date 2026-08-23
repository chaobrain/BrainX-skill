#!/usr/bin/env node

import { createInterface } from 'node:readline';

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on('line', (line) => process.stdout.write(`${line}\n`));
