BrainX Skill 1.0.8 records 16 commits merged into `main`.

## Release record

- **Version:** `1.0.8`
- **Previous release:** `v1.0.6`
- **Commits included:** 16

## New features

### feat(brainevent): add event-driven connectivity skill ([`67ea94b`](https://github.com/chaobrain/BrainX-skill/commit/67ea94b429665cbc3e1c417176d567efc99fe0af))

No additional commit description was provided.

**Changed files:** `manifest.json`, `package.json`, `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/connectivity-variants.md`, `skills/brainevent/references/custom-operators.md`, `skills/brainevent/references/scripts/102_EI_net_1996.py`, `skills/brainevent/references/scripts/204_joglekar_2018_propagation.py`, `skills/brainevent/references/sparse-formats.md`, `skills/brainevent/references/synaptic-plasticity.md`

### feat(installer): add Windsurf, Gemini CLI, and OpenCode adapters (#8) ([`9fb1c03`](https://github.com/chaobrain/BrainX-skill/commit/9fb1c03dc4defee657bcb502dcc4689097281a16))

Extend the install-destination adapter registry beyond Claude Code, Codex,
and Cursor. The consumer pipeline (installer, paths, prompts, receipt) is
already generic over the adapters array, so this is purely additive plus
docs and the first test coverage for the adapter registry.

**Changed files:** `.github/scripts/adapters.test.mjs`, `README.md`, `adapters/gemini.js`, `adapters/opencode.js`, `adapters/windsurf.js`, `installation/lib/installer.js`, `package.json`

## Fixes

### fix(release): stop after recovering a tagged release ([`211ff21`](https://github.com/chaobrain/BrainX-skill/commit/211ff214951566a50dca02c9b56ef18c6bfc8ee5))

Mark successful recovery in the workflow outputs and skip version preparation, tagging, and publishing steps after the missing npm or GitHub release has been restored.

**Changed files:** `.github/scripts/release-workflow.test.mjs`, `.github/workflows/release-notes.yml`

### fix(release): remove accidental v1.0.7 ([`231d749`](https://github.com/chaobrain/BrainX-skill/commit/231d7493ccd4ea2e83c7db1aa8056fb100e57676))

No additional commit description was provided.

**Changed files:** `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `package.json`, `release-notes.config.json`

### fix(manifest): register brainpy-state and brainx-general-guard skills (#7) ([`46a51c4`](https://github.com/chaobrain/BrainX-skill/commit/46a51c4fca20038f8248989f1dc9562719889d55))

Register skills/brainpy-state/ and skills/brainx-general-guard/ in manifest.json
and package.json files, fixing the Agent Skills validation workflow, which failed
on every run because the sorted manifest array did not match the sorted skill
directories.

Relocate skills/brainpy-state/NEST-compatible/ to references/nest-compatible/ to
satisfy AGENTS.md rule 1, and normalize the branch's routing links to
skill-root-relative paths.

Add the npmmirror update command to the low-bandwidth README section.

**Changed files:** `README.md`, `manifest.json`, `package.json`, `plan.md`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/NEST-compatible/references/devices.md`, `skills/brainpy-state/NEST-compatible/references/divergence-and-parity.md`, `skills/brainpy-state/NEST-compatible/references/integration-categories.md`, `skills/brainpy-state/NEST-compatible/references/model-library.md`, `skills/brainpy-state/NEST-compatible/references/network-building.md`, `skills/brainpy-state/NEST-compatible/references/synapse-and-connectivity.md`, `skills/brainpy-state/NEST-compatible/scripts/brette_et_al_2007.py`, and 22 more

## Documentation

### docs: refresh BrainX skill guidance ([`acefb6d`](https://github.com/chaobrain/BrainX-skill/commit/acefb6dfd97a584c17e6f9727a4a7aa41fdb8290))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `archive/solver-library-with-effects.md`, `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/custom-operators-cpu.md`, `skills/brainevent/references/custom-operators-gpu.md`, `skills/brainevent/references/custom-operators.md`, `skills/brainevent/references/scripts/coba_ei_teaching.py`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/NEST-compatible/references/devices.md`, `skills/brainpy-state/NEST-compatible/references/divergence-and-parity.md`, `skills/brainpy-state/NEST-compatible/references/integration-categories.md`, and 36 more

### docs: add application script table headers ([`b508f5d`](https://github.com/chaobrain/BrainX-skill/commit/b508f5d2e71e1572707db4ed1e2201bf18932d19))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`, `skills/brainpy-state/SKILL.md`

### docs: refine BrainEvent API structure ([`097d2be`](https://github.com/chaobrain/BrainX-skill/commit/097d2be61ccb0330c47cf3ac776f296fdb034bd7))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/sparse-formats.md`

### docs: sharpen BrainEvent neuroscience mappings ([`06138a5`](https://github.com/chaobrain/BrainX-skill/commit/06138a567db0e47f6766250fbbc18e31e46994b8))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`

### docs: clarify BrainEvent plasticity mechanism ([`4903635`](https://github.com/chaobrain/BrainX-skill/commit/4903635d390733b1f51a3a719054d8bca4f54d2d))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`

### docs: add AGENTS.md skill-creation rules and CLAUDE.md pointer (#5) ([`90b1a23`](https://github.com/chaobrain/BrainX-skill/commit/90b1a23139337857c8c39831ad22457bb1634f31))

Give agents a concrete, numbered rulebook for authoring BrainX skills,
covering layout, frontmatter, manifest/package registration, the
progressive-disclosure authoring principles, SKILL.md structure, and
verification steps. CLAUDE.md imports it via @AGENTS.md.

**Changed files:** `AGENTS.md`, `CLAUDE.md`

### docs(brainpy-state): remove unrouted event-driven operators stub (#9) ([`39d271a`](https://github.com/chaobrain/BrainX-skill/commit/39d271a382ed23dcfe31098b7b022f44093d4f24))

references/brainstate-dynamics/brain-dynamics-event-driven-operators.md was
draft scaffolding that shipped: its body is "Should eventually cover",
"Common mistakes to document", and "Placeholder examples" rather than a
reference.

Nothing routed to it, making it the only shipped file under skills/brainpy-state
absent from every routing row. Its own "Used by" block pointed at two files
that have never existed in this repository. Its subject is owned by
skills/brainevent/, which SKILL.md already routes to.

plan.md places this reference under skills/brainstate/, not skills/brainpy-state/,
which is why no brainpy-state routing row ever pointed at it. That planned
brainstate reference is untouched and remains unbuilt.

**Changed files:** `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-event-driven-operators.md`

### docs: add BrainCell and reorganize skill references ([`ef3a39d`](https://github.com/chaobrain/BrainX-skill/commit/ef3a39d35edbd7e9b4920a19833ed66f13145354))

No additional commit description was provided.

**Changed files:** `archive/braincell-multicompartment-parent-2026-07-29.md`, `archive/braincell-skill-single-and-multicompartment-2026-07-29.md`, `braintools-references/braintools-cogtask.md`, `braintools-references/braintools-connectivity.md`, `braintools-references/braintools-data-preprocessing.md`, `braintools-references/braintools-input-current.md`, `braintools-references/braintools-metric.md`, `braintools-references/braintools-optimizer.md`, `braintools-references/braintools-parameter-initializer.md`, `braintools-references/braintools-surrogate.md`, `braintools-references/braintools-visualize.md`, `manifest.json`, and 44 more

## Other changes

### put draft ([`7ed22b3`](https://github.com/chaobrain/BrainX-skill/commit/7ed22b3bb6d0166f11bb6139d6cf98bc26693b04))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-delay-protocol.md`, `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-event-driven-operators.md`, `skills/brainpy-state/references/brainstate-dynamics/scripts/training-snn.py`, `skills/brainpy-state/references/braintools-optimizer.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/advanced-randomness.md`, `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`

### updates ([`44f811e`](https://github.com/chaobrain/BrainX-skill/commit/44f811e8d97edb11356ed382496f566b694f8e57))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/component-selection.md`, `skills/brainpy-state/references/projection-patterns.md`, `skills/brainpy-state/references/training-variations.md`

### chore: stop ignoring AGENTS.md and CLAUDE.md (#6) ([`8173882`](https://github.com/chaobrain/BrainX-skill/commit/8173882d9ddebdc36a3232270ad003ce54d0f2a1))

Both files are now tracked and committed (#5); the ignore rule was
vestigial and only obscured that they're shared, not local-only.

**Changed files:** `.gitignore`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.6...v1.0.8
