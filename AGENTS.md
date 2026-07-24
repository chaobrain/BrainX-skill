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
# BrainX Skill Writing Philosophy

A BrainX skill is not a textbook or an API catalog. It is a compact guide that gives an agent the mental model and operational instructions needed to use a library correctly.

The goal is:

> **Preserve essential understanding while removing every sentence that does not improve a decision or implementation.**

## 1. Write the underlying principles sharply

Explain the system deeply enough to support correct decisions, but no further.

The underlying-principles section must remain because individual APIs are difficult to use reliably without understanding the system they belong to. However, it should contain only the **irreducible mental model**.

Explain relationships and consequences rather than implementation history or broad background. Once the reader can predict how the system behaves, move to operational guidance.

## 2. Keep decisions, remove commentary

Retain information that changes an action; remove information that merely describes the library.

Every sentence should answer at least one question:

* What should I use?
* When should I use it?
* How should I use it?
* What must I avoid?
* Where should I look next?


## 3. Explain each concept once

Give every important idea one authoritative definition, then reuse it without re-explaining it.

Later sections should apply previously established principles rather than restating them in different words. Brief reminders are acceptable only when they prevent a likely mistake.

> **One concept, one explanation, many applications.**

## 4. Keep the main skill as the canonical path

The root skill should present the shortest complete route from user intent to a correct implementation.

Organize it in the order an agent actually works:

> understand the model → choose the abstraction → construct the workflow → execute it → verify the result

Keep common decisions and essential constraints in the root skill. Move uncommon variants, advanced controls, and exhaustive API coverage to references.

## 5. One example, one lesson

Each code block should prove one operational pattern clearly.

An example should show exact syntax, object relationships, execution order, or a non-obvious result. Do not make one example teach several unrelated features.

> **Use the smallest example that makes the rule unambiguous.**

## 6. Prefer contrasts over description

Explain similar APIs by showing the decision boundary between them.

Do not write separate descriptive paragraphs when a direct comparison can reveal the choice more clearly.

Use this pattern:

> **Use X when Y. Use Z when W.**

Comparisons should focus on practical differences such as purpose, lifecycle, input shape, side effects, performance, or compatibility.

## 7. Route precisely and sharply

Tell the agent exactly when to leave the root skill and exactly where to go.

Each reference should own a distinct class of problems. A routing instruction should identify:

* the condition that triggers the reference;
* the exact reference to open;
* the decisions or details contained there.

Avoid vague instructions such as “see the advanced guide.”

## 8. Use concise, imperative language

State the required action first, followed by the consequence when it is not obvious.

Prefer:

> Assign the object before initialization so it becomes part of the managed structure.

Over:

> The framework provides a mechanism through which assigned objects can become part of its managed structure.

Use direct verbs such as **choose, create, assign, initialize, wrap, collect, preserve, verify, avoid,** and **open**.

## 9. Keep code and prose complementary

Let code show the mechanics; use prose only for meaning that the code cannot communicate safely.

Code should show:

* exact syntax;
* operation order;
* object relationships;
* expected inputs and outputs.

Prose should explain:

* why the pattern is correct;
* when to choose it;
* which invariant it depends on;
* which failure it prevents.

Do not narrate obvious lines of code.

## 10. Preserve critical exceptions

Remove general commentary, but retain exceptions that prevent common or costly failures.

A warning belongs in the root skill when the failure is:

* likely to occur;
* difficult to diagnose;
* silent or misleading;
* fundamental to correct library use.

State each warning once, beside the workflow where it matters.

## 11. Editing test

Evaluate every sentence and example by the decision it protects.

For every sentence, ask:

> Would removing this sentence make the agent choose or implement something incorrectly?

If not, remove it or relocate it.

For every code block, ask:

> What unique behavior does this example teach?

If it teaches nothing new, merge or delete it.

For every section, ask:

> Does this section have one clear instructional purpose?

If not, divide it, sharpen it, or move part of it elsewhere.

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


### Section skeleton

Follow the established order: **Purpose and boundary → core concepts / API
table → canonical workflow → reference routing → boundaries and common
failures**. Keep the canonical path in the core; route everything else out.

### Correctness and voice

- Write in an imperative, directive voice — "Use …", "Do not …", present tense.
- Keep terminology and structure consistent across skills; a pattern shown in one
  skill should look the same in the next.

### Heading style

Section headings use **sentence case**: capitalize only the first letter of the
heading; keep the rest lowercase. Proper nouns and API/identifier names keep
their own casing (e.g. `brainunit`, `Quantity`, JAX). Example: `## Purpose and
boundary`, not `## Purpose And Boundary`.
