# BrainX Skill

[![Supported BrainX release](https://img.shields.io/badge/BrainX-v2026.7.9-007EC6?style=flat)](skills/brainx-install/references/compatibility-and-release-matching.md#version-compatibility-matrix)
[![License](https://img.shields.io/badge/License-Apache%202.0-4C1?style=flat)](LICENSE)
[![npm downloads](https://img.shields.io/npm/dm/brainx-skill?label=npm&color=007EC6&style=flat)](https://www.npmjs.com/package/brainx-skill)
[![Skills](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fchaobrain%2FBrainX-skill%2Fmain%2Fmanifest.json&query=%24.skills.length&label=Skills&color=007EC6&style=flat)](manifest.json)
[![Agent Skills validation](https://img.shields.io/github/actions/workflow/status/chaobrain/BrainX-skill/agent-skills-validation.yml?branch=main&label=Agent%20Skills&style=flat)](https://github.com/chaobrain/BrainX-skill/actions/workflows/agent-skills-validation.yml)
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/chaobrain/BrainX-skill?label=OpenSSF&style=flat)](https://scorecard.dev/viewer/?uri=github.com/chaobrain/BrainX-skill)

[![Works with Claude Code, Codex, and Cursor](https://img.shields.io/badge/Works%20with-Claude%20Code%20%7C%20Codex%20%7C%20Cursor-007EC6?style=flat)](#installation-locations)

make your coding agent understand the power of BrainX

## Requirements

- Node.js 18 or newer
- macOS, Linux, or Windows
## Install

```bash
npx brainx-skill install
```

## Update

```bash
npx brainx-skill update
```


## Install when the internet connection is low

```bash
npx --registry=https://registry.npmmirror.com brainx-skill install
```

## Installation locations

Depending on the selected scope, the canonical `brainx-install` skill is installed into:

```text
Claude Code: ~/.claude/skills/<skill-name> or <cwd>/.claude/skills/<skill-name>
Codex:       ~/.agents/skills/<skill-name> or <cwd>/.agents/skills/<skill-name>
Cursor:      ~/.cursor/skills/<skill-name> or <cwd>/.cursor/skills/<skill-name>
```

Installation ownership is recorded in `~/.brainx/receipt.json`.
