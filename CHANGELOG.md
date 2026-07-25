# Changelog

All notable changes to BrainX Skill are recorded here. The release workflow
updates this file from every unreleased commit on `main`.

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
