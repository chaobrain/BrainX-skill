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

Match the conventions in `skills/brainunit/` and `skills/brainstate/`. Read the target `SKILL.md` before editing it.

A BrainX skill is not a textbook or exhaustive API catalogue. It is a compact guide that gives an agent the mental model and operational instructions required to use a library correctly.

> **Preserve essential understanding while removing everything that does not improve a decision or implementation.**

### 1. Teach the irreducible mental model

Explain the system only deeply enough for the agent to predict its behavior and make correct decisions.

Focus on:

* the main abstractions;
* how they relate;
* how data, state, or control flows;
* the lifecycle or execution order;
* the invariants that later workflows depend on.

Remove implementation history, broad background, and descriptive commentary.

### 2. Keep the canonical path in the root skill

Present the shortest complete route from user intent to a correct implementation:

> **understand the model → choose the abstraction → construct the workflow → execute it → verify the result**

Keep common decisions, essential constraints, and likely failure points in `SKILL.md`. Route uncommon variants, advanced controls, and exhaustive API coverage to references.

### 3. Explain each concept and action once

Give each important concept one authoritative definition, then apply it without re-explaining it.

Use one code example for one operational lesson. Each example should demonstrate exact syntax, object relationships, execution order, or a non-obvious result with the smallest code block that makes the rule unambiguous.

> **One concept, one explanation; one example, one lesson.**

### 4. Write for decisions

Every sentence should answer at least one question:

* What should I use?
* When should I use it?
* How should I use it?
* What must I avoid?
* Where should I look next?

Use concise, imperative language. State the action first and its consequence only when needed.

Prefer direct decision boundaries:

> **Use X when Y. Use Z when W.**

Compare similar APIs by purpose, lifecycle, input shape, side effects, performance, or compatibility rather than describing each one separately.

### 5. Keep code and prose complementary

Use code to show mechanics:

* exact syntax;
* execution order;
* object relationships;
* expected inputs and outputs.

Use prose to explain:

* when to choose the pattern;
* why it is correct;
* which invariant it depends on;
* which failure it prevents.

Do not narrate obvious lines of code.

### 6. Preserve critical exceptions and route everything else

Keep a warning in the root skill when the failure is likely, difficult to diagnose, silent, misleading, or fundamental to correct use. State it once beside the workflow where it matters.

Route non-canonical cases precisely. Every routing instruction should state:

* when to open the reference;
* the exact file to open;
* what decisions or details it contains.

Avoid vague directions such as “see the advanced guide.”

### Editing test

For every sentence, ask:

> Would removing this make the agent choose or implement something incorrectly?

For every code block, ask:

> What unique behavior does this example teach?

For every section, ask:

> Does this section have one clear instructional purpose?

Remove, merge, divide, or relocate anything that fails these tests.

## Final rule

> **Teach the mental model sharply. Define each invariant once. Show each canonical action once. State each critical exception once. Route everything else precisely.**


## Other general principles

### Progressive disclosure

- The `SKILL.md` is the **compact core**: essential concepts, the canonical
  workflow, and a routing table. Keep it token-light so the agent preserves its
  own reasoning and general ability.
- Deep, specialized, or variant knowledge lives in `references/*.md`, opened only
  when needed. Prefer adding a new reference file over growing the core.
- Runnable, canonical patterns go in `scripts/` — real code the agent can copy,
  not prose.

### Section structure

Organize the root `SKILL.md` in this order:

> **Purpose and boundary → underlying mental model → operational sections → reference routing → boundaries and common failures**

Structure each operational section as follows:

1. **State the mental model in one sentence.**
   Explain what the mechanism represents, how it behaves, and why that behavior matters when using it.

2. **List the APIs that implement the workflow.**
   Use a two-column `API | Description` table and give each important API its own row.

3. **List the essential API rows**
   Describe the API in this order:

   > **when to use it → important behavior → result **

4. **Show one canonical example.**
   Demonstrate the normal end-to-end pattern with the smallest code block that makes the workflow unambiguous. Remove unrelated setup, optional variants, repeated outputs, and assertions that do not verify an important invariant.

5. **Route non-canonical cases precisely.**
   End with the exact reference file to open for advanced variants, edge cases, configuration details, uncommon failures, or more complex compositions.

Use this default shape:

````markdown
### Operational concept

One sentence explaining the mechanism and why it matters.

| API | Description |
|---|---|
| `api()` | When to use it, what it does, its important behavior, and its result or relevant failure. |

```python
# Canonical workflow
```

Open `references/example.md` when [specific condition], for [specific decisions or details].
````

Include closely related APIs only when they are required to understand or complete the canonical workflow. Do not expand the root skill into an exhaustive API catalogue.

Keep an API in the root skill only when it:

is directly required by the canonical workflow;
is required to choose between canonical workflows;
represents a major direction of variation that the agent must recognize;
protects against a likely and costly failure; or
is required to route the task correctly.

For a major variation family, include only the representative API needed to establish the decision boundary.

An API does not belong in the root merely because it is useful, common, or part of the same package.


### Correctness and voice

- Write in an imperative, directive voice — "Use …", "Do not …", present tense.
- Keep terminology and structure consistent across skills; a pattern shown in one
  skill should look the same in the next.

### Heading style

Section headings use **sentence case**: capitalize only the first letter of the
heading; keep the rest lowercase. Proper nouns and API/identifier names keep
their own casing (e.g. `brainunit`, `Quantity`, JAX). Example: `## Purpose and
boundary`, not `## Purpose And Boundary`.
