# BrainX Skill

[![Supported BrainX release](https://img.shields.io/badge/Supported_BrainX-v2026.7.9-0b7285?style=flat-square)](skills/brainx-install/references/compatibility-and-release-matching.md#version-compatibility-matrix)
[![License](https://img.shields.io/github/license/chaobrain/BrainX-skill?style=flat-square)](LICENSE)
[![npm downloads](https://img.shields.io/npm/dm/brainx-skill?style=flat-square&logo=npm)](https://www.npmjs.com/package/brainx-skill)
[![Skills](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fchaobrain%2FBrainX-skill%2Fmain%2Fmanifest.json&query=%24.skills.length&label=Skills&color=2ea44f&style=flat-square)](manifest.json)
[![Agent Skills validation](https://img.shields.io/github/actions/workflow/status/chaobrain/BrainX-skill/agent-skills-validation.yml?branch=main&label=Agent%20Skills&style=flat-square&logo=githubactions)](https://github.com/chaobrain/BrainX-skill/actions/workflows/agent-skills-validation.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/chaobrain/BrainX-skill/badge)](https://scorecard.dev/viewer/?uri=github.com/chaobrain/BrainX-skill)

[![Works with Claude Code, Codex, and Cursor](https://img.shields.io/badge/Works_with-Claude_Code_%7C_Codex_%7C_Cursor-1f6feb?style=flat-square)](#installation-locations)

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
