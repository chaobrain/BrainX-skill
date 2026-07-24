# BrainX skills

Guidance for any agent working in this repository. These instructions override
default behavior. Follow them exactly.

## Project overview

This repository is a **skill-authoring workspace**. It produces agent skills that
teach coding agents (Claude, Codex, Cursor) the BrainX ecosystem, plus the npm
installer that deploys them.

- `skills/<name>/` — one skill per BrainX package (`brainunit`, `brainstate`,
  `braincell`, `brainevent`, `brainmass`, `brainpy`, `braintrace`,
  `brainx-acceleration-audit`, `brainx-install`). Each holds a `SKILL.md` core,
  a `references/` tree, and optional `scripts/` and `agents/`.
- `adapters/` — per-agent install adapters (`claude.js`, `codex.js`, `cursor.js`).
- `installation/` — the `brainx-skill` CLI (`bin/`, `lib/`).
- `plan.md` — the design source of truth for skill scope, layering, and routing.
  Read it before adding or changing a skill; keep it in sync when the design shifts.

The mission (see `plan.md`): let researchers use a coding agent and BrainX
together without prior expertise, by giving the agent clean, correct,
high-performance BrainX knowledge through **progressive disclosure** — a compact
core skill plus deep, on-demand Markdown references.


## Skill-authoring principles

Match the conventions already established in `skills/brainunit/` and
`skills/brainstate/`. Read the target skill's `SKILL.md` before editing it.

### Progressive disclosure

- The `SKILL.md` is the **compact core**: essential concepts, the canonical
  workflow, and a routing table. Keep it token-light so the agent preserves its
  own reasoning and general ability.
- Deep, specialized, or variant knowledge lives in `references/*.md`, opened only
  when needed. Prefer adding a new reference file over growing the core.
- Runnable, canonical patterns go in `scripts/` — real code the agent can copy,
  not prose.


### Section skeleton

Follow the established order: **Purpose and boundary → core concepts / API
table → canonical workflow → reference routing → boundaries and common
failures**. Keep the canonical path in the core; route everything else out.

### Reference routing discipline

- Every reference must be reachable through a routing table whose rows read as
  **"Open when …"** triggers.
- Keep nesting shallow. When a reference may open a deeper one, state that inbound
  route explicitly (e.g. "only X may open Y").
- Do not route to references outside the architecture the skill owns; hand off to
  the owning skill instead.

### Correctness and voice

- Preserve scientific meaning. Encode the safe path and forbid silent shortcuts
  (e.g. never strip units, never read `.mantissa` in place of conversion, never
  guess a unit from a name).
- Write in an imperative, directive voice — "Use …", "Do not …", present tense.
- Keep terminology and structure consistent across skills; a pattern shown in one
  skill should look the same in the next.

### Heading style

Section headings use **sentence case**: capitalize only the first letter of the
heading; keep the rest lowercase. Proper nouns and API/identifier names keep
their own casing (e.g. `brainunit`, `Quantity`, JAX). Example: `## Purpose and
boundary`, not `## Purpose And Boundary`.

## Installer and CLI conventions

- Node.js 18+; must run on macOS, Linux, and Windows.
- Installs the canonical skill per agent scope via the `adapters/`; ownership is
  recorded in a receipt (`~/.brainx/receipt.json`). Keep adapter behavior
  symmetric across Claude, Codex, and Cursor.
- Verify against `README.md`'s documented install/update flows before changing
  CLI behavior.
