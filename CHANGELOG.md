# Changelog

All notable changes to BrainX Skill are recorded here. The release workflow
updates this file from every unreleased commit on `main`.

## 1.0.10 - 2026-07-31

BrainX Skill 1.0.10 records 1 commit merged into `main`.

### Release record

- **Version:** `1.0.10`
- **Previous release:** `v1.0.9`
- **Commits included:** 1

### New features

#### feat(skills): tighten BrainX guard and install handoff ([`4df5ebc`](https://github.com/chaobrain/BrainX-skill/commit/4df5ebc94063433bf2dc88a2b12d9c2a212942d8))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-install/SKILL.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.9...v1.0.10

## 1.0.9 - 2026-07-30

BrainX Skill 1.0.9 records 3 commits merged into `main`.

### Release record

- **Version:** `1.0.9`
- **Previous release:** `v1.0.8`
- **Commits included:** 3

### New features

#### feat(skills): add BrainMass and refine model routing ([`f5426d6`](https://github.com/chaobrain/BrainX-skill/commit/f5426d646b09c6f159fb5b44cceb63ddda165c41))

No additional commit description was provided.

**Changed files:** `README.md`, `manifest.json`, `package.json`, `plan.md`, `skills/braincell/SKILL.md`, `skills/brainmass/SKILL.md`, `skills/brainmass/references/batch-transform-acceleration.md`, `skills/brainmass/references/brainstate/parameter-constraints-regularization.md`, `skills/brainmass/references/brainstate/parameter-transforms-regularizers-catalog.md`, `skills/brainmass/references/braintools/cogtask.md`, `skills/brainmass/references/braintools/data-preprocessing.md`, `skills/brainmass/references/braintools/metric.md`, and 28 more

### Documentation

#### docs(braincell): refine authoring and visualization guides ([`54dee95`](https://github.com/chaobrain/BrainX-skill/commit/54dee95b94f0d3af23a837dc7cd36180d9396148))

No additional commit description was provided.

**Changed files:** `skills/braincell/SKILL.md`, `skills/braincell/references/braincell-custom-ion-channel-authoring.md`, `skills/braincell/references/channel-library.md`, `skills/braincell/references/ion-library.md`, `skills/braincell/references/multicompartment/filter-function-library.md`, `skills/braincell/references/multicompartment/topology-building-and-visualization.md`

#### docs(brainmass): refine reference routing categories ([`909833a`](https://github.com/chaobrain/BrainX-skill/commit/909833a67c35199a045f68b666e1c5c487e3a1de))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainmass/SKILL.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.8...v1.0.9

## 1.0.8 - 2026-07-29

BrainX Skill 1.0.8 records 16 commits merged into `main`.

### Release record

- **Version:** `1.0.8`
- **Previous release:** `v1.0.6`
- **Commits included:** 16

### New features

#### feat(brainevent): add event-driven connectivity skill ([`67ea94b`](https://github.com/chaobrain/BrainX-skill/commit/67ea94b429665cbc3e1c417176d567efc99fe0af))

No additional commit description was provided.

**Changed files:** `manifest.json`, `package.json`, `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/connectivity-variants.md`, `skills/brainevent/references/custom-operators.md`, `skills/brainevent/references/scripts/102_EI_net_1996.py`, `skills/brainevent/references/scripts/204_joglekar_2018_propagation.py`, `skills/brainevent/references/sparse-formats.md`, `skills/brainevent/references/synaptic-plasticity.md`

#### feat(installer): add Windsurf, Gemini CLI, and OpenCode adapters (#8) ([`9fb1c03`](https://github.com/chaobrain/BrainX-skill/commit/9fb1c03dc4defee657bcb502dcc4689097281a16))

Extend the install-destination adapter registry beyond Claude Code, Codex,
and Cursor. The consumer pipeline (installer, paths, prompts, receipt) is
already generic over the adapters array, so this is purely additive plus
docs and the first test coverage for the adapter registry.

**Changed files:** `.github/scripts/adapters.test.mjs`, `README.md`, `adapters/gemini.js`, `adapters/opencode.js`, `adapters/windsurf.js`, `installation/lib/installer.js`, `package.json`

### Fixes

#### fix(release): stop after recovering a tagged release ([`211ff21`](https://github.com/chaobrain/BrainX-skill/commit/211ff214951566a50dca02c9b56ef18c6bfc8ee5))

Mark successful recovery in the workflow outputs and skip version preparation, tagging, and publishing steps after the missing npm or GitHub release has been restored.

**Changed files:** `.github/scripts/release-workflow.test.mjs`, `.github/workflows/release-notes.yml`

#### fix(release): remove accidental v1.0.7 ([`231d749`](https://github.com/chaobrain/BrainX-skill/commit/231d7493ccd4ea2e83c7db1aa8056fb100e57676))

No additional commit description was provided.

**Changed files:** `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `package.json`, `release-notes.config.json`

#### fix(manifest): register brainpy-state and brainx-general-guard skills (#7) ([`46a51c4`](https://github.com/chaobrain/BrainX-skill/commit/46a51c4fca20038f8248989f1dc9562719889d55))

Register skills/brainpy-state/ and skills/brainx-general-guard/ in manifest.json
and package.json files, fixing the Agent Skills validation workflow, which failed
on every run because the sorted manifest array did not match the sorted skill
directories.

Relocate skills/brainpy-state/NEST-compatible/ to references/nest-compatible/ to
satisfy AGENTS.md rule 1, and normalize the branch's routing links to
skill-root-relative paths.

Add the npmmirror update command to the low-bandwidth README section.

**Changed files:** `README.md`, `manifest.json`, `package.json`, `plan.md`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/NEST-compatible/references/devices.md`, `skills/brainpy-state/NEST-compatible/references/divergence-and-parity.md`, `skills/brainpy-state/NEST-compatible/references/integration-categories.md`, `skills/brainpy-state/NEST-compatible/references/model-library.md`, `skills/brainpy-state/NEST-compatible/references/network-building.md`, `skills/brainpy-state/NEST-compatible/references/synapse-and-connectivity.md`, `skills/brainpy-state/NEST-compatible/scripts/brette_et_al_2007.py`, and 22 more

### Documentation

#### docs: refresh BrainX skill guidance ([`acefb6d`](https://github.com/chaobrain/BrainX-skill/commit/acefb6dfd97a584c17e6f9727a4a7aa41fdb8290))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `archive/solver-library-with-effects.md`, `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/custom-operators-cpu.md`, `skills/brainevent/references/custom-operators-gpu.md`, `skills/brainevent/references/custom-operators.md`, `skills/brainevent/references/scripts/coba_ei_teaching.py`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/NEST-compatible/references/devices.md`, `skills/brainpy-state/NEST-compatible/references/divergence-and-parity.md`, `skills/brainpy-state/NEST-compatible/references/integration-categories.md`, and 36 more

#### docs: add application script table headers ([`b508f5d`](https://github.com/chaobrain/BrainX-skill/commit/b508f5d2e71e1572707db4ed1e2201bf18932d19))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`, `skills/brainpy-state/SKILL.md`

#### docs: refine BrainEvent API structure ([`097d2be`](https://github.com/chaobrain/BrainX-skill/commit/097d2be61ccb0330c47cf3ac776f296fdb034bd7))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainevent/SKILL.md`, `skills/brainevent/references/sparse-formats.md`

#### docs: sharpen BrainEvent neuroscience mappings ([`06138a5`](https://github.com/chaobrain/BrainX-skill/commit/06138a567db0e47f6766250fbbc18e31e46994b8))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`

#### docs: clarify BrainEvent plasticity mechanism ([`4903635`](https://github.com/chaobrain/BrainX-skill/commit/4903635d390733b1f51a3a719054d8bca4f54d2d))

No additional commit description was provided.

**Changed files:** `skills/brainevent/SKILL.md`

#### docs: add AGENTS.md skill-creation rules and CLAUDE.md pointer (#5) ([`90b1a23`](https://github.com/chaobrain/BrainX-skill/commit/90b1a23139337857c8c39831ad22457bb1634f31))

Give agents a concrete, numbered rulebook for authoring BrainX skills,
covering layout, frontmatter, manifest/package registration, the
progressive-disclosure authoring principles, SKILL.md structure, and
verification steps. CLAUDE.md imports it via @AGENTS.md.

**Changed files:** `AGENTS.md`, `CLAUDE.md`

#### docs(brainpy-state): remove unrouted event-driven operators stub (#9) ([`39d271a`](https://github.com/chaobrain/BrainX-skill/commit/39d271a382ed23dcfe31098b7b022f44093d4f24))

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

#### docs: add BrainCell and reorganize skill references ([`ef3a39d`](https://github.com/chaobrain/BrainX-skill/commit/ef3a39d35edbd7e9b4920a19833ed66f13145354))

No additional commit description was provided.

**Changed files:** `archive/braincell-multicompartment-parent-2026-07-29.md`, `archive/braincell-skill-single-and-multicompartment-2026-07-29.md`, `braintools-references/braintools-cogtask.md`, `braintools-references/braintools-connectivity.md`, `braintools-references/braintools-data-preprocessing.md`, `braintools-references/braintools-input-current.md`, `braintools-references/braintools-metric.md`, `braintools-references/braintools-optimizer.md`, `braintools-references/braintools-parameter-initializer.md`, `braintools-references/braintools-surrogate.md`, `braintools-references/braintools-visualize.md`, `manifest.json`, and 44 more

### Other changes

#### put draft ([`7ed22b3`](https://github.com/chaobrain/BrainX-skill/commit/7ed22b3bb6d0166f11bb6139d6cf98bc26693b04))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `skills/brainpy-state/NEST-compatible/nest-workflow.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-delay-protocol.md`, `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-event-driven-operators.md`, `skills/brainpy-state/references/brainstate-dynamics/scripts/training-snn.py`, `skills/brainpy-state/references/braintools-optimizer.md`, `skills/brainx-general-guard/SKILL.md`, `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/advanced-randomness.md`, `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`

#### updates ([`44f811e`](https://github.com/chaobrain/BrainX-skill/commit/44f811e8d97edb11356ed382496f566b694f8e57))

No additional commit description was provided.

**Changed files:** `AGENTS.md`, `skills/brainpy-state/SKILL.md`, `skills/brainpy-state/references/component-selection.md`, `skills/brainpy-state/references/projection-patterns.md`, `skills/brainpy-state/references/training-variations.md`

#### chore: stop ignoring AGENTS.md and CLAUDE.md (#6) ([`8173882`](https://github.com/chaobrain/BrainX-skill/commit/8173882d9ddebdc36a3232270ad003ce54d0f2a1))

Both files are now tracked and committed (#5); the ignore rule was
vestigial and only obscured that they're shared, not local-only.

**Changed files:** `.gitignore`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.6...v1.0.8

## 1.0.6 - 2026-07-25

BrainX Skill 1.0.6 records 13 commits merged into `main`.

### Release record

- **Version:** `1.0.6`
- **Previous release:** `e2b723e489a645fab5d42044a1b73ab21f69b38e`
- **Commits included:** 13

### New features

#### feat(skills): expand BrainUnit and BrainState guidance ([`cc434f6`](https://github.com/chaobrain/BrainX-skill/commit/cc434f69150cc49fc8cbb79571a286bccd70f0af))

Improve BrainUnit unit-aware transformations with explicit guidance for JAX transforms, state-aware gradients, custom JVP/VJP rules, and unit-boundary validation.

Rebuild the BrainState core workflow around state roles, environments, exp_euler_step, state-aware transforms, randomness, parameter constraints, diagnostics, and canonical reference routing. Consolidate randomness and diagnostics, move runnable examples into scripts, and remove dynamics and training references now owned by other skills.

Add the main-branch release-note workflow. It increments patch versions, records package, changelog, and latest notes, and publishes detailed commit bodies and changed files only to chaobrain/BrainX-skill.

Verification:
- 39 Node tests passed (35 existing and 4 release tests)
- 19 BrainState Markdown files have valid local links
- 4 Python scripts parse successfully
- workflow YAML and JSON configuration parse successfully

**Changed files:** `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/workflows/release-notes.yml`, `CHANGELOG.md`, `README.md`, `RELEASE_NOTES.md`, `plan.md`, `release-notes.config.json`, `skills/brainstate/SKILL.md`, `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md`, `skills/brainstate/references/brainstate/brainstate-transformed-diagnostics.md`, `skills/brainstate/references/brainstate/parameter-constraints-regularization.md`, and 16 more

#### feat(release): synchronize npm and GitHub publishing ([`ddaa37a`](https://github.com/chaobrain/BrainX-skill/commit/ddaa37a2ae06b96b11d1aee4ee3802f6cc1f7bca))

No additional commit description was provided.

**Changed files:** `.github/scripts/npm-registry.mjs`, `.github/scripts/npm-registry.test.mjs`, `.github/scripts/prepare-release.integration.test.mjs`, `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/scripts/release-workflow.test.mjs`, `.github/workflows/release-notes.yml`, `package.json`, `release-notes.config.json`

#### feat(skills): refine BrainState and BrainUnit guidance ([`a9025df`](https://github.com/chaobrain/BrainX-skill/commit/a9025df8c8f961796e500b353a7e611955cfe313))

Condense BrainState reference routing, restructure its reproducible randomness guidance, and identify BrainState and BrainUnit as the central stateful and physical-quantity infrastructure for BrainX modeling and simulation.

**Changed files:** `skills/brainstate/SKILL.md`, `skills/brainunit/SKILL.md`

### Fixes

#### fix(ci): invoke Agent Skills validator through Python ([`e1705e6`](https://github.com/chaobrain/BrainX-skill/commit/e1705e66f3a4009bd5947013d3db16dbac746b25))

No additional commit description was provided.

**Changed files:** `.github/scripts/agent-skills-workflow.test.mjs`, `.github/workflows/agent-skills-validation.yml`

### Documentation

#### docs: add trust and compatibility badges ([`8fde8af`](https://github.com/chaobrain/BrainX-skill/commit/8fde8af1dea8d9f9c56585d46eb7a424b4c73480))

No additional commit description was provided.

**Changed files:** `.github/workflows/agent-skills-validation.yml`, `.github/workflows/scorecard.yml`, `README.md`

#### docs: polish README badge styling ([`75e5320`](https://github.com/chaobrain/BrainX-skill/commit/75e5320452c7ca635ed417946b764e162f5fa502))

No additional commit description was provided.

**Changed files:** `.impeccable.md`, `README.md`

#### docs: add agent-specific install commands ([`9e8ef51`](https://github.com/chaobrain/BrainX-skill/commit/9e8ef51bf3ced7a7567dcd3ca4ba2fcadb06b4e2))

No additional commit description was provided.

**Changed files:** `README.md`

### Other changes

#### refactor: rename installation-code/ to installation/ (#3) ([`44192b6`](https://github.com/chaobrain/BrainX-skill/commit/44192b68e0f5b5c582b9d55ea3d1c039c69f9be9))

- git mv installation-code -> installation (history preserved for all 10 files)
- update package.json bin entry and files[] to the new path
- update AGENTS.md CLI-location reference
- set Apache LICENSE copyright to 2026 BrainX Team

**Changed files:** `AGENTS.md`, `LICENSE`, `installation-code/bin/brainx.js`, `installation-code/lib/adapter-transaction.js`, `installation-code/lib/bundle.js`, `installation-code/lib/cli.js`, `installation-code/lib/constants.js`, `installation-code/lib/hash.js`, `installation-code/lib/installer.js`, `installation-code/lib/paths.js`, `installation-code/lib/prompts.js`, `installation-code/lib/receipt.js`, and 11 more

#### ci: make release-notes workflow manual-only via workflow_dispatch ([`06677b8`](https://github.com/chaobrain/BrainX-skill/commit/06677b8b25953d6cc924d26c1ff69bac72c7d163))

Releases no longer fire on every push/merge to main. The workflow now
runs only when triggered manually from the Actions tab. The head_commit
guard (which existed only to prevent the release commit from
re-triggering the push workflow) is dropped since manual dispatch cannot
recurse; the fork-repository guard is kept.

**Changed files:** `.github/workflows/release-notes.yml`

#### Merge pull request #4 from chaobrain/codex/npm-publish-sync ([`447a47b`](https://github.com/chaobrain/BrainX-skill/commit/447a47b9abe8641211c5e1febd6976b552e6cb50))

Synchronize npm and GitHub releases

#### small update on gitignore ([`7517ac0`](https://github.com/chaobrain/BrainX-skill/commit/7517ac0efb0e3501670429f690ea3697d6e285b2))

No additional commit description was provided.

**Changed files:** `.gitignore`, `.impeccable.md`, `AGENTS.md`

#### chore: ignore local agent instruction files ([`30ef962`](https://github.com/chaobrain/BrainX-skill/commit/30ef96236581ea11b18fa135f15e8ac2bea45eeb))

No additional commit description was provided.

**Changed files:** `.gitignore`, `CLAUDE.md`

#### refine brainstate skill guidance ([`08f3336`](https://github.com/chaobrain/BrainX-skill/commit/08f33364a74758b8bca38843365f1faf1d975b84))

No additional commit description was provided.

**Changed files:** `plan.md`, `skills/brainstate/SKILL.md`, `skills/brainstate/references/simulation-environment.md`, `skills/brainstate/references/size-inference-variations.md`

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/e2b723e489a645fab5d42044a1b73ab21f69b38e...v1.0.6

## 1.0.5 - 2026-07-24

BrainX Skill 1.0.5 records 1 commit merged into `main`.

### Release record

- **Version:** `1.0.5`
- **Previous release:** `v1.0.4`
- **Commits included:** 1

### Other changes

#### refactor: rename installation-code/ to installation/ (#3) ([`44192b6`](https://github.com/chaobrain/BrainX-skill/commit/44192b68e0f5b5c582b9d55ea3d1c039c69f9be9))

- git mv installation-code -> installation (history preserved for all 10 files)
- update package.json bin entry and files[] to the new path
- update AGENTS.md CLI-location reference
- set Apache LICENSE copyright to 2026 BrainX Team

**Changed files:** `AGENTS.md`, `LICENSE`, `installation-code/bin/brainx.js`, `installation-code/lib/adapter-transaction.js`, `installation-code/lib/bundle.js`, `installation-code/lib/cli.js`, `installation-code/lib/constants.js`, `installation-code/lib/hash.js`, `installation-code/lib/installer.js`, `installation-code/lib/paths.js`, `installation-code/lib/prompts.js`, `installation-code/lib/receipt.js`, and 11 more

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/v1.0.4...v1.0.5

## 1.0.4 - 2026-07-24

BrainX Skill 1.0.4 records 1 commit merged into `main`.

### Release record

- **Version:** `1.0.4`
- **Previous release:** `e2b723e489a645fab5d42044a1b73ab21f69b38e`
- **Commits included:** 1

### New features

#### feat(skills): expand BrainUnit and BrainState guidance ([`cc434f6`](https://github.com/chaobrain/BrainX-skill/commit/cc434f69150cc49fc8cbb79571a286bccd70f0af))

Improve BrainUnit unit-aware transformations with explicit guidance for JAX transforms, state-aware gradients, custom JVP/VJP rules, and unit-boundary validation.

Rebuild the BrainState core workflow around state roles, environments, exp_euler_step, state-aware transforms, randomness, parameter constraints, diagnostics, and canonical reference routing. Consolidate randomness and diagnostics, move runnable examples into scripts, and remove dynamics and training references now owned by other skills.

Add the main-branch release-note workflow. It increments patch versions, records package, changelog, and latest notes, and publishes detailed commit bodies and changed files only to chaobrain/BrainX-skill.

Verification:
- 39 Node tests passed (35 existing and 4 release tests)
- 19 BrainState Markdown files have valid local links
- 4 Python scripts parse successfully
- workflow YAML and JSON configuration parse successfully

**Changed files:** `.github/scripts/prepare-release.mjs`, `.github/scripts/prepare-release.test.mjs`, `.github/workflows/release-notes.yml`, `CHANGELOG.md`, `README.md`, `RELEASE_NOTES.md`, `plan.md`, `release-notes.config.json`, `skills/brainstate/SKILL.md`, `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md`, `skills/brainstate/references/brainstate/brainstate-transformed-diagnostics.md`, `skills/brainstate/references/brainstate/parameter-constraints-regularization.md`, and 16 more

**Full Changelog:** https://github.com/chaobrain/BrainX-skill/compare/e2b723e489a645fab5d42044a1b73ab21f69b38e...v1.0.4

## 1.0.3

Version `1.0.3` is the baseline for automated release tracking. Earlier changes
were released before this changelog was introduced.
