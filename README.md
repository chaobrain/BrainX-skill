<p align="center">
  <img src="https://brainx.chaobrain.com/images/BrainX-skill.webp" alt="BrainX Skill" width="240">
</p>

# BrainX Skill

[![Supported BrainX release](https://img.shields.io/badge/BrainX-v2026.7.9-007EC6?style=flat)](skills/brainx-install/references/compatibility-and-release-matching.md#version-compatibility-matrix) [![License](https://img.shields.io/badge/License-Apache%202.0-4C1?style=flat)](LICENSE) [![npm downloads](https://img.shields.io/npm/dm/brainx-skill?label=npm&color=007EC6&style=flat)](https://www.npmjs.com/package/brainx-skill) [![Skills](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fchaobrain%2FBrainX-skill%2Fmain%2Fmanifest.json&query=%24.skills.length&label=Skills&color=007EC6&style=flat)](manifest.json) [![Agent Skills validation](https://img.shields.io/github/actions/workflow/status/chaobrain/BrainX-skill/agent-skills-validation.yml?branch=main&label=Agent%20Skills&style=flat)](https://github.com/chaobrain/BrainX-skill/actions/workflows/agent-skills-validation.yml) [![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/chaobrain/BrainX-skill?label=OpenSSF&style=flat)](https://scorecard.dev/viewer/?uri=github.com/chaobrain/BrainX-skill)

[![Works with Claude Code, Codex, Cursor, Windsurf, Gemini CLI, and OpenCode](https://img.shields.io/badge/Works%20with-Claude%20Code%20%7C%20Codex%20%7C%20Cursor%20%7C%20Windsurf%20%7C%20Gemini%20CLI%20%7C%20OpenCode-007EC6?style=flat)](#installation-locations)

make your coding agent understand the power of BrainX

## Requirements

- Node.js 18 or newer
- macOS, Linux, or Windows
## Install

```bash
npx brainx-skill install
```

```bash
# Claude Code, current project
npx skills add chaobrain/BrainX-skill --agent claude-code

# Codex, current project
npx skills add chaobrain/BrainX-skill --agent codex

# Claude Code, globally
npx skills add chaobrain/BrainX-skill --agent claude-code --global

# Specific project
cd /actual/path/to/project
npx skills add chaobrain/BrainX-skill --agent codex
```

## Update

```bash
npx brainx-skill update
```


## When the internet connection is low

```bash
# install
npx --registry=https://registry.npmmirror.com brainx-skill install

# update
npx --registry=https://registry.npmmirror.com brainx-skill update
```

## MCP servers

Install and activate every BrainX MCP integration in Codex with one command:

```bash
npx brainx-skill mcp install
```

This installs durable runtimes under `~/.brainx/mcp` and registers:

| Server | Registration | Purpose |
|---|---|---|
| Europe PMC search | `europepmc` | Search biology and neuroscience literature and retrieve structured article metadata. |
| Full-text resolver | `fulltext_resolver` | Resolve PMC, OpenAlex, and bioRxiv/medRxiv full-text versions with provenance. |
| BrainX Codex reviewer | `codex` | Run independent BrainX iteration reviews. |

The command requires the Codex CLI, a working Codex login, and `uv`. It refuses to overwrite unrelated registrations. An exact reviewer registration from this repository is retained so its existing approval and timeout settings remain intact. Restart Codex or reload MCP configuration after installation.

Remove only the registrations and runtime owned by this installer:

```bash
npx brainx-skill mcp remove
```

See `mcp-servers/fulltext-resolver/README.md` and `mcp-servers/codex/README.md` for tool behavior, configuration, and development setup.

## Installation locations

Depending on the selected scope, the canonical `brainx-install` skill is installed into:

```text
Claude Code: ~/.claude/skills/<skill-name> or <cwd>/.claude/skills/<skill-name>
Codex:       ~/.agents/skills/<skill-name> or <cwd>/.agents/skills/<skill-name>
Cursor:      ~/.cursor/skills/<skill-name> or <cwd>/.cursor/skills/<skill-name>
Windsurf:    ~/.codeium/windsurf/skills/<skill-name> or <cwd>/.codeium/windsurf/skills/<skill-name>
Gemini CLI:  ~/.gemini/skills/<skill-name> or <cwd>/.gemini/skills/<skill-name>
Antigravity: ~/.gemini/config/skills/<skill-name> or <cwd>/.agents/skills/<skill-name>
OpenCode:    ~/.config/opencode/skills/<skill-name> or <cwd>/.config/opencode/skills/<skill-name>
```

Antigravity reads global skills from `~/.gemini/config/skills` and workspace skills
from `<cwd>/.agents/skills`. In project scope that workspace directory is the same one
Codex uses, so selecting both harnesses installs the skills once and records them for
both.

Installation ownership is recorded in `~/.brainx/receipt.json`.
