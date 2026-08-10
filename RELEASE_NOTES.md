BrainX Skill 1.0.11 records 6 commits merged into `main`.

## Release record

- **Version:** `1.0.11`
- **Previous release:** `v1.0.10`
- **Commits included:** 6

## New features

### feat(skills): add gated legacy BrainPy workflow ([`3706cbe`](https://github.com/chaobrain/BrainX-skill/commit/3706cbed7d0a17a332872d1481ccc35e8250c72f))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/brainPy(legacy)/analysis.md`, `skills/brainpy-state/references/brainPy(legacy)/brainpy legacy workflow.md`, `skills/brainpy-state/references/brainPy(legacy)/built-in dynamic neuron model.md`, `skills/brainpy-state/references/brainPy(legacy)/connecting neurons.md`, `skills/brainpy-state/references/brainPy(legacy)/customize neuron and synpase.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/Input generation.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/More about simulation.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/Multi-device array sharding.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/Parallel experiment execution.md`, `skills/brainpy-state/references/brainPy(legacy)/infrastructure/array creation and mechanics.md`, and 14 more

### feat(adapters): support Google Antigravity and repair skills validation CI (#10) ([`5abe23f`](https://github.com/chaobrain/BrainX-skill/commit/5abe23f6b67d02ebe5c70f95218e06120a8a53a5))

* feat(adapters): support Google Antigravity

Antigravity reads global skills from ~/.gemini/config/skills and workspace
skills from <cwd>/.agents/skills, so an adapter needs a scope-dependent
destination for the first time.

Add an optional projectPath to the adapter shape, resolve it for project
scope, and accept either path when validating a receipt destination.

In project scope Antigravity shares <cwd>/.agents/skills with Codex.
Install once per resolved destination and record that ownership for every
harness in the group, and fall back to any receipt record at the same
destination so adding Antigravity after Codex proves ownership instead of
failing.

* fix(ci): register braintrace and close validation gaps

skills/braintrace was shipped without a manifest.json or package.json files
entry, which broke Agent Skills validation on main.

Register it, and close the gaps that let the drift through: verify the
published files list against the skill directories, reject loose files
directly under skills/, check manifest names against the installer name
pattern, and compare the lists with diff so mismatches are readable and
names containing spaces cannot mask a difference.

**Changed files:** `.github/scripts/adapters.test.mjs`, `.github/workflows/agent-skills-validation.yml`, `README.md`, `adapters/antigravity.js`, `installation/lib/installer.js`, `installation/lib/paths.js`, `installation/lib/prompts.js`, `manifest.json`, `package.json`

## Documentation

### docs(braintrace): add planned source references ([`7830770`](https://github.com/chaobrain/BrainX-skill/commit/7830770d8b759a044435357bcc389fbae0d2e129))

No additional commit description was provided.

**Changed files:** `skills/braintrace/SKILL.md`, `skills/braintrace/references/Drtrl.md`, `skills/braintrace/references/ETP operators.md`, `skills/braintrace/references/algorithm selection.md`, `skills/braintrace/references/batching.md`, `skills/braintrace/references/compiler_internal.md`, `skills/braintrace/references/custom ETP primitives.md`, `skills/braintrace/references/custom algorithms.md`, `skills/braintrace/references/customizing_primitive_transforms.md`, `skills/braintrace/references/pp_pprop workflow.md`, `skills/braintrace/references/pre-built-braintrace-layer.md`

### docs(braintrace): complete online learning guidance ([`d92f3e9`](https://github.com/chaobrain/BrainX-skill/commit/d92f3e98c7efb56b7a84bfc623f9b9f670e5b98d))

No additional commit description was provided.

**Changed files:** `skills/braintrace/SKILL.md`, `skills/braintrace/references/Drtrl.md`, `skills/braintrace/references/ETP operators.md`, `skills/braintrace/references/algorithm selection.md`, `skills/braintrace/references/batching.md`, `skills/braintrace/references/compiler_internal.md`, `skills/braintrace/references/custom ETP primitives.md`, `skills/braintrace/references/custom algorithms.md`, `skills/braintrace/references/customizing_primitive_transforms.md`, `skills/braintrace/references/pp_pprop workflow.md`, `skills/braintrace/references/pre-built-braintrace-layer.md`

### docs: repair broken skill routing and resolve AGENTS.md contradictions (#11) ([`16eed45`](https://github.com/chaobrain/BrainX-skill/commit/16eed450d87f915a8320863ed93a76dd0312a573))

* docs(skills): repair broken routing and align skills with AGENTS.md

An audit of all nine skills found defects that CI does not catch: skills-ref
validation, the node test suite, and the manifest/package checks were all green
while four routing targets pointed at files that do not exist and seven of
braincell's eight shipped scripts were unreachable.

Dangling routing targets:

- brainstate cited references/braintools-optimizer-reference.md from SKILL.md
  and transformation-grad-expansion.md, but the file was never written. Add
  references/braintools/optimizer.md, derived from the verified brainmass and
  brainpy-state copies and scoped to BrainState training, and retarget both
  citations. No API surface is invented.
- brainstate's only inbound route to parameter-transforms-regularizers-catalog.md
  used bare filenames that resolve against the skill root and miss. Rewrite both
  as references/brainstate/ paths.
- brainpy-state's array-creation.md routed array_split(), backend extraction, and
  as_numpy() to array-mechanics.md, which exists only in brainunit. Retarget to
  skills/brainunit/references/array-mechanics.md, matching the cross-skill path
  form already used elsewhere.

Unreachable content:

- braincell had no script table at all. Add Application script examples routing
  its seven previously unroutable scripts, each row derived from the script's own
  purpose docstring. cell_multicompartment_reference.py keeps its single existing
  route from the multicompartment workflow.

Rule 20 form:

- brainx-install's Reference Routing was three prose paragraphs; convert to the
  required two-column table.
- brainstate's Script references was a bullet list placed before the routing
  table; rename to Application script examples, convert to a table, and move it
  after the routing table.

Rules 17 and 23:

- brainx-install and brainx-general-guard had no Purpose and boundary section.
  Add one to each; all nine skills now open with it. Sentence-case the fourteen
  Title Case headings in brainx-install, and repair the guard heading that ran
  two sentences together with a shouted ONLY.
- brainx-install's description was the only quoted one in the repo and began with
  a literal leading space and a lowercase sentence start.
- Fix the # Brainstate H1 casing, a double space in brainmass, and two missing
  blank lines before script tables.

The guard routed only the three scale skills, leaving braintrace, brainevent,
brainunit, and brainstate with no guard-level entry point. Add a cross-cutting
concern table rather than forcing them into the deliberately scale-based one.

Verified: skills-ref 0.1.1 reports Valid skill for all nine; node --test passes
26/26; a link scan reports zero dangling targets and zero unrouted shipped files
(was four and eight); all 40 scripts compile.

Noted but not changed: skills/brainstate/agents/openai.yaml is valid and Rule
1-compliant but nothing in adapters/ or installation/ reads it, and it is the
only agents/ directory in the repo. AGENTS.md Rule 17 orders reference routing
before boundaries and common failures while Rule 20 requires the file to end on
the routing table; the two contradict and skills follow both readings. Neither
is resolved here.

* docs(skills): drop unconsumed brainstate openai.yaml

skills/brainstate/agents/openai.yaml declared display_name, short_description,
and default_prompt, but nothing read it: no adapter in adapters/ and no code in
installation/ references agents/ or openai.yaml, and it was the only agents/
directory in the repo. Remove it rather than leave unreplicated dead metadata.

Rule 1 still permits agents/ for per-platform interface metadata; this deletes
one unused instance, not the allowance.

* docs: resolve contradictions in AGENTS.md

Six rules contradicted each other or contradicted what the nine shipped skills
actually do. In every case the rule was wrong, not the skills: after these edits
all nine pass unchanged.

Rule 17 vs Rule 20 — the tail order. Rule 17 ordered reference routing before
boundaries and common failures; Rule 20 told you to close the file with the
routing table. Rule 20 then contradicted itself by requiring a separate script
table, which has to sit somewhere. Skills split across all three readings: four
end on a script table, three on the routing table, one on boundaries. Rule 17 now
states the full tail — reference routing, application script examples, boundaries
and common failures — keeps them in that relative order, and marks each optional.
Rule 20 is renamed from "End with a reference routing table" accordingly.

Rule 17's first heading. It required Purpose and boundary to be the first heading
after the frontmatter, which seven of nine skills violate by carrying an H1 title.
Now: first `##` section, H1 title permitted.

Rule 20's routing invariant. Read strictly it required every shipped file to have
a row in SKILL.md, which braincell, brainpy-state, and brainmass all break on
purpose by delegating a grouped family to its workflow reference. The invariant is
now exactly one inbound row anywhere in the skill, with delegation described and
braincell named as the pattern. Also states the path form — skill-root-relative,
never bare, full repository path across skills — which is what the four dangling
targets fixed in the previous commit got wrong.

Rule 20 vs Rule 1. Rule 1 permits agents/ while Rule 20 demanded every shipped
file appear in a routing row. Scope the invariant to references/ and script files.

Rule 1's nesting limit. It allowed one level below references/; brainpy-state
ships 22 files at two levels under nest-compatible/scripts/ and
brainPy(legacy)/infrastructure/. Raised to two.

Rule 4's stated range. Descriptions were said to run 240-580 characters; they
actually run 227-426. Corrected.

Rule 10 said "run both checks" above three commands and validated a single skill
while CI validates all of them. It now validates every skill, and says plainly
that neither check reads routing tables — so neither would have caught any of the
routing defects fixed in the previous commit.

**Changed files:** `AGENTS.md`, `skills/braincell/SKILL.md`, `skills/brainevent/SKILL.md`, `skills/brainmass/SKILL.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/array-creation.md`, `skills/brainstate/SKILL.md`, `skills/brainstate/agents/openai.yaml`, `skills/brainstate/references/brainstate/transformation-grad-expansion.md`, `skills/brainstate/references/braintools/optimizer.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-install/SKILL.md`

### docs(braintrace): center memory-efficient training ([`c379905`](https://github.com/chaobrain/BrainX-skill/commit/c37990583c27b3080131cab420ccffe50565ad75))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/braintrace/SKILL.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.10...v1.0.11
