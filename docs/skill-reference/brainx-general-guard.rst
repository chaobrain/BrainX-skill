.. rst-class:: skill-reference-page

brainx-general-guard
====================

Complete skill: :skill-source:`source <skills/brainx-general-guard/SKILL.md>`

Purpose and boundary
--------------------

Open ``brainx-general-guard`` first for every BrainX modeling, simulation, training, debugging, review, or optimization task. It identifies every explicitly represented modeling scale, opens only the package skills that own those scales, and keeps the implementation BrainX-native.

Major contents
--------------

- Route point neurons to ``brainpy-state``, cellular mechanisms to ``braincell``, and aggregate populations to ``brainmass``.
- Compose multiple owning skills only when the scientific model is genuinely multiscale.
- Treat installed packages as execution dependencies, not substitute documentation.
- Study the selected skill and its relevant scripts before adapting an implementation.
- Prefer the highest-level owning-package orchestrator that preserves the scientific operation.
- Use BrainState transformations only when the owning package cannot express the required execution.
- Keep code minimal while preserving units, State effects, reproducibility, controls, observables, and scientific claims.

Reference Markdown
------------------

This skill has no separate ``references/*.md`` files. Its routing, implementation, and validation rules live in ``skills/brainx-general-guard/SKILL.md``.
