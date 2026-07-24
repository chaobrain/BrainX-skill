BrainX Skill 1.0.4 records 1 commit merged into `main`.

## Release record

- **Version:** `1.0.4`
- **Previous release:** `e2b723e489a645fab5d42044a1b73ab21f69b38e`
- **Commits included:** 1

## New features

### feat(skills): expand BrainUnit and BrainState guidance ([`cc434f6`](https://github.com/chaobrain/BrainX-skill/commit/cc434f69150cc49fc8cbb79571a286bccd70f0af))

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
