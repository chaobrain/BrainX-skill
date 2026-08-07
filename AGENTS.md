# BrainX skills

Guidance for any agent working in this repository. These instructions override
default behavior. Follow them exactly.

---

# Rules for skill creation

Rules 1–10 are mechanical and enforced by CI or the installer; violating one
breaks the build. Rules 11–22 are authoring rules; they decide whether the skill
actually works. Rules 23–26 govern how you finish.

## Layout and naming

### Rule 1 — One skill is one directory under `skills/`

A skill is exactly `skills/<name>/` containing `SKILL.md` at its root. Nothing
else may sit directly inside `skills/`; `installation/lib/bundle.js` rejects any
loose file there.

Permitted subdirectories:

| Subdirectory | Contents |
|---|---|
| `references/` | Deep, on-demand Markdown. May nest up to two levels further for grouping (`references/brainstate/`, `references/nest-compatible/scripts/`). |
| `references/scripts/` or `scripts/` | Runnable `.py` files the agent can copy. |
| `agents/` | Optional per-platform interface metadata (e.g. `agents/openai.yaml`). |

Ship only regular files and directories. Symlinks, sockets, and other special
files fail bundle validation.

### Rule 2 — The directory name is the skill name

`skills/<name>/SKILL.md` must declare `name: <name>`, byte-identical to the
directory. The validator fails on any mismatch.

Names must satisfy the installer pattern `^[a-z0-9]+(?:-[a-z0-9]+)*$`: lowercase
ASCII letters and digits, single hyphens between segments, no leading or trailing
hyphen, no consecutive hyphens, 64 characters maximum. Name the skill after the
BrainX package or the cross-cutting concern it teaches (`brainunit`,
`brainpy-state`, `brainx-general-guard`).

## Frontmatter

### Rule 3 — Emit only the six allowed frontmatter fields

`SKILL.md` opens with YAML frontmatter delimited by `---`. Only these keys are
accepted; any other key fails validation:

| Field | Required | Constraint |
|---|---|---|
| `name` | Yes | Matches the directory name; see rule 2. |
| `description` | Yes | Non-empty; 1024 characters maximum. |
| `license` | No | SPDX identifier. Omit unless the skill differs from the repository's Apache-2.0. |
| `allowed-tools` | No | Tool patterns the skill requires. Experimental; omit by default. |
| `metadata` | No | Free-form map. |
| `compatibility` | No | 500 characters maximum. |

Do not add `version`, `author`, `tags`, or any other field.

### Rule 4 — Write the description as a routing decision, not a summary

`description` is the only text an agent sees before loading the skill. It must
let the agent decide, without opening the file, whether this skill applies.

State in one block:

1. what the package or concern is, in one clause;
2. **"Use this skill for …"** followed by the concrete triggers — user-visible
   nouns, symptoms, and API names that should route here;
3. **"Do not use for …"** whenever an adjacent skill owns the neighbouring
   territory.

Prefer the vocabulary a researcher would type ("dimensional mismatch",
"suspicious bare numbers", "spike-driven postsynaptic input") over internal
jargon. Existing descriptions run 225–430 characters; that is the working range.

### Rule 5 — Keep skill boundaries disjoint

Two skills must not claim the same trigger. When scope overlaps, one skill owns
the trigger and the other explicitly disclaims it in its `description` and in its
`## Purpose and boundary` section. `brainx-install` versus the modelling skills is
the reference example: package installation versus everything else.

## Registration

### Rule 6 — Register every skill directory in `manifest.json`

CI compares the sorted `manifest.json` `skills` array against the sorted list of
directories under `skills/` and **fails on any difference in either direction**.
Adding a directory without a manifest entry, or removing a directory without
removing its entry, breaks `Agent Skills validation`.

Keep `schemaVersion: 1`. Entries must be unique strings.

> A skill that is not ready to ship does not belong in `skills/`. Park it in a
> branch or in `archive/`, not as an unregistered directory.

### Rule 7 — Add the skill to the published `files` list

`package.json` `files` enumerates what npm ships. A new skill needs its own
`"skills/<name>/"` entry, or the installer will resolve a directory that is
absent from the published tarball.

### Rule 8 — Never hand-edit the version

`package.json` `version` is owned by the release workflow. Releases are manual:
GitHub Actions → **Release npm package and notes** → Run workflow. Do not bump
the version, write `CHANGELOG.md`, or create tags by hand.

### Rule 9 — Do not change installer behavior from a skill change

`installation/`, `adapters/`, and `.github/` are separate concerns. A skill
authoring change touches `skills/`, `manifest.json`, `package.json` `files`, and
`plan.md` — nothing else. If a skill genuinely requires an installer change, say
so and treat it as its own piece of work.

### Rule 10 — Validate before claiming the skill is done

Run these checks and paste the output. Validate every skill, not just the one you
touched — a routing change in one skill can strand a file in another:

```bash
python -m pip install --disable-pip-version-check skills-ref==0.1.1
for d in skills/*/; do
  python -c 'from skills_ref.cli import main; main()' validate "$d"
done
node --test .github/scripts/*.test.mjs
```

Neither check reads routing tables, so neither catches a dangling target or an
unrouted file. Verify rule 20's two invariants yourself before claiming the skill
is done.

---

# Skill-authoring principles

Match the conventions in `skills/brainunit/` and `skills/brainstate/`. Read the
target `SKILL.md` before editing it.

A BrainX skill is not a textbook or an exhaustive API catalogue. It is a compact
guide that gives an agent the mental model and operational instructions required
to use a library correctly.

> **Preserve essential understanding while removing everything that does not
> improve a decision or implementation.**

### Rule 11 — Teach the irreducible mental model

Explain the system only deeply enough for the agent to predict its behavior and
make correct decisions.

Focus on:

* the main abstractions;
* how they relate;
* how data, state, or control flows;
* the lifecycle or execution order;
* the invariants that later workflows depend on.

Remove implementation history, broad background, and descriptive commentary.

### Rule 12 — Keep the canonical path in the root skill

Present the shortest complete route from user intent to a correct implementation:

> **understand the model → choose the abstraction → construct the workflow →
> execute it → verify the result**

Keep common decisions, essential constraints, and likely failure points in
`SKILL.md`. Route uncommon variants, advanced controls, and exhaustive API
coverage to references.

### Rule 13 — Explain each concept and action once

Give each important concept one authoritative definition, then apply it without
re-explaining it.

Use one code example for one operational lesson. Each example should demonstrate
exact syntax, object relationships, execution order, or a non-obvious result with
the smallest code block that makes the rule unambiguous.

> **One concept, one explanation; one example, one lesson.**

### Rule 14 — Write for decisions

Every sentence should answer at least one question:

* What should I use?
* When should I use it?
* How should I use it?
* What must I avoid?
* Where should I look next?

Use concise, imperative language. State the action first and its consequence only
when needed.

Prefer direct decision boundaries:

> **Use X when Y. Use Z when W.**

Compare similar APIs by purpose, lifecycle, input shape, side effects,
performance, or compatibility rather than describing each one separately.

### Rule 15 — Keep code and prose complementary

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

### Rule 16 — Preserve critical exceptions and route everything else

Keep a warning in the root skill when the failure is likely, difficult to
diagnose, silent, misleading, or fundamental to correct use. State it once beside
the workflow where it matters.

Route non-canonical cases precisely. Every routing instruction must state:

* when to open the reference;
* the exact file to open;
* what decisions or details it contains.

Avoid vague directions such as "see the advanced guide."

---

# Structure of `SKILL.md`

### Rule 17 — Follow the fixed section order

Organize the root `SKILL.md` in this order:

> **Purpose and boundary → underlying mental model → operational sections →
> reference routing → application script examples → boundaries and common
> failures**

The first three are required. The last three are the tail: keep them in this
relative order, and omit any that the skill does not need — a skill that ships no
references has no routing table, one that ships no scripts has no script table,
and `## Boundaries and common failures` is optional when the operational sections
already state each failure beside the workflow it threatens.

`## Purpose and boundary` is always the first `##` section, and it states both
what the skill covers and what it refuses. An H1 title may precede it; nothing
else may.

### Rule 18 — Give every operational section the same five-part shape

1. **State the mental model in one sentence.**
   Explain what the mechanism represents, how it behaves, and why that behavior
   matters when using it.

2. **List the APIs that implement the workflow.**
   Use a two-column `API | Description` table and give each important API its own
   row.

3. **Order each API row consistently.**

   > **when to use it → important behavior → result**

4. **Show one canonical example.**
   Demonstrate the normal end-to-end pattern with the smallest code block that
   makes the workflow unambiguous. Remove unrelated setup, optional variants,
   repeated outputs, and assertions that do not verify an important invariant.

5. **Route non-canonical cases precisely.**
   End with the exact reference file to open for advanced variants, edge cases,
   configuration details, uncommon failures, or more complex compositions.

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

### Rule 19 — Apply a hard admission test to every root-skill API

Include closely related APIs only when they are required to understand or
complete the canonical workflow. Keep an API in the root skill only when it:

* is directly required by the canonical workflow;
* is required to choose between canonical workflows;
* represents a major direction of variation that the agent must recognize;
* protects against a likely and costly failure; or
* is required to route the task correctly.

For a major variation family, include only the representative API needed to
establish the decision boundary.

> An API does not belong in the root merely because it is useful, common, or part
> of the same package.

### Rule 20 — Route every shipped file from a routing table

Open the tail of `SKILL.md` with a `## Reference routing` table:

```markdown
| Reference | Open when |
|---|---|
| `references/<file>.md` | Open when <specific condition>; it contains <the decisions and details inside>. |
```

Give runnable examples their own `## Application script examples` table in the
same form, placed after the routing table. See rule 17 for what may follow.

Two invariants hold across the whole skill:

* **Every routing row points at a file that exists.** Write the path relative to
  the skill root (`references/brainstate/optimizer.md`), not relative to the
  citing file, and never as a bare filename. To route into another skill, give
  the full repository path (`skills/brainunit/references/array-mechanics.md`).
* **Every shipped `references/` and script file has exactly one inbound routing
  row.** That row does not have to live in `SKILL.md`. A grouped family may be
  delegated: `SKILL.md` routes to the family's workflow reference, and that
  reference's own routing table owns its leaves. When you delegate, say so
  explicitly and forbid the root from opening the leaves directly, as
  `skills/braincell/SKILL.md` does for `references/multicompartment/`.

This invariant covers `references/` and script files. `agents/` holds per-platform
interface metadata that the harness reads, not content an agent routes to; it is
not part of any routing table.

## References and scripts

### Rule 21 — Make each reference a self-contained answer

A reference file opens with an H1 title and a one-paragraph statement of when to
open it and what it settles — mirroring the routing row that points here. It then
answers that question completely, including the imports needed to run its code.

Prefer adding a new reference over growing `SKILL.md`. A reference may repeat a
definition from the root skill only when an agent reading the reference alone
would otherwise get it wrong.

### Rule 22 — Ship scripts as working, copyable code

Files under `scripts/` are real programs, not prose. Each one runs end to end on
its own, uses unit-aware BrainX APIs throughout, and demonstrates one complete
application pattern. Do not fragment a script into snippets that only make sense
next to the SKILL.md text.

---

# Style, verification, and workflow

### Rule 23 — Write in an imperative, directive voice

Use "Use …", "Do not …", present tense. Keep terminology and structure consistent
across skills; a pattern shown in one skill must look the same in the next.

Section headings use **sentence case**: capitalize only the first letter of the
heading and keep the rest lowercase. Proper nouns and API or identifier names keep
their own casing (e.g. `brainunit`, `Quantity`, JAX). Write `## Purpose and
boundary`, not `## Purpose And Boundary`.

### Rule 24 — Apply the editing test before you finish

For every sentence, ask:

> Would removing this make the agent choose or implement something incorrectly?

For every code block, ask:

> What unique behavior does this example teach?

For every section, ask:

> Does this section have one clear instructional purpose?

Remove, merge, divide, or relocate anything that fails these tests.

### Rule 25 — Never invent BrainX API surface

Every symbol, signature, keyword argument, and default in a skill must come from
the real BrainX source, its documentation, or `plan.md`. If you cannot verify an
API, say so and leave it out. A confidently wrong signature is the most expensive
possible defect in this repository — it teaches every downstream agent the same
mistake.

### Rule 26 — Report what you actually ran

State which validators you ran and paste their output. If a check failed or was
skipped, say so plainly. Do not describe a skill as complete on the strength of
reading it.

Follow the repository's commit conventions: Conventional Commits with a scope
(`feat(brainevent): …`, `docs: …`, `chore(release): …`), and no
`Co-Authored-By` trailer.

---

## Final rule

> **Teach the mental model sharply. Define each invariant once. Show each
> canonical action once. State each critical exception once. Route everything else
> precisely.**
