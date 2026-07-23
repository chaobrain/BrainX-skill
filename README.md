# BrainX Agent Skills

## Requirements

- Node.js 18 or newer
- macOS, Linux, or Windows
## Install

```bash
npx brainx-skill install
```

If npm resolves a stale cached package, request the latest release explicitly:

```bash
npx brainx-skill@latest install
```

## Update

```bash
npx brainx-skill update
```


## Install when the internet connection is low

```bash
npx --registry=https://registry.npmmirror.com brainx-skill install
```
## Update when the internet connection is low
```bash
npx --registry=https://registry.npmmirror.com brainx-skill@latest install
```

## Installation locations

Depending on the selected scope, the canonical `brainx-install` skill is installed into:

```text
Claude Code: ~/.claude/skills/<skill-name> or <cwd>/.claude/skills/<skill-name>
Codex:       ~/.agents/skills/<skill-name> or <cwd>/.agents/skills/<skill-name>
Cursor:      ~/.cursor/skills/<skill-name> or <cwd>/.cursor/skills/<skill-name>
```

Installation ownership is recorded in `~/.brainx/receipt.json`.

