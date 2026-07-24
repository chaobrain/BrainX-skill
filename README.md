# BrainX Skill
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

## Releases

Every commit merged into `main` updates the automated release notes. Write a
Conventional Commit subject and a detailed body so the change is represented
clearly:

```text
feat(brainstate): add transformed-state diagnostics

Describe what changed, who it affects, any compatibility impact, and how the
change was verified.
```

Use `feat:` for a feature, `fix:` for a bug fix, and add `!` before the colon
for a breaking change. Release Please collects these commits in a release pull
request. Merging that pull request updates `package.json` and `CHANGELOG.md`,
creates the version tag, and publishes the GitHub Release in
[`chaobrain/BrainX-skill`](https://github.com/chaobrain/BrainX-skill/releases).


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
