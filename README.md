# BrainX Agent Skills

Portable BrainX skills for Claude Code, Codex, and Cursor.

## Requirements

- Node.js 18 or newer
- macOS, Linux, or Windows

## Included Skills

- `brainx-install`: install, remove, configure, and reconcile BrainX packages and device targets.
- `brainunit`: work with physical quantities, units, array operations, and mathematical functions in BrainUnit.

## Install

Install or update both skills interactively:

```bash
npx brainx-skill@latest install
```

```bash
npx brainx-skill@latest update
```

For manual installation, copy either directory from `skills/` into the skill directory used by your coding agent:

```text
Claude Code: ~/.claude/skills/<skill-name> or <project>/.claude/skills/<skill-name>
Codex:       ~/.agents/skills/<skill-name> or <project>/.agents/skills/<skill-name>
Cursor:      ~/.cursor/skills/<skill-name> or <project>/.cursor/skills/<skill-name>
```

Keep the complete selected skill directory, including its `references/` files.

The files in `adapters/` declare the default user-level skill location for each supported agent.
The `installation-code/` directory contains the command-line installer implementation.
