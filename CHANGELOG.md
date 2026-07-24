# Changelog

All notable changes to BrainX Skill are recorded here. The release workflow
updates this file from every unreleased commit on `main`.

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
