.. rst-class:: skill-reference-page

brainx-acceleration-audit
=========================

Purpose and boundary
--------------------

Use the planned ``brainx-acceleration-audit`` skill for evidence-based performance work in BrainX and BrainState simulations: vectorization, batching, sweeps, compilation boundaries, memory, warm runtime, GPU use, and multi-device scaling. Optimize only after preserving model semantics, State effects, RNG behavior, shapes, and numerical outputs.

Major contents
--------------

- Inventory hot-path shapes, State, loops, transformations, RNG, connectivity, and host interaction.
- Classify inefficiencies and rank expected impact, semantic risk, and confidence.
- Establish a deterministic baseline and rewrite one execution axis at a time.
- Prefer stable compositions such as compiled time scans, mapped trials, batched State, and compiled batch gradients.
- Compare outputs, final State, RNG, gradients, shapes, warm runtime, and memory after every rewrite.
- Complete single-device cleanup before introducing multi-device execution.

Implementation status
---------------------

``plan.md`` defines this skill, but the current bundle does not contain ``skills/brainx-acceleration-audit/``. The planned reference Markdown below is therefore not yet provided by the repository.

Planned reference Markdown
--------------------------

- ``skills/brainx-acceleration-audit/references/brainstate/brainstate-control-flow-patterns.md``
- ``skills/brainx-acceleration-audit/references/brainstate/transformation-jit-expansion.md``
- ``skills/brainx-acceleration-audit/references/brainstate/transformation-vmap-expansion.md``
- ``skills/brainx-acceleration-audit/references/brainstate/transformation-grad-expansion.md``
- ``skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md``
- ``skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/advanced-randomness.md``
