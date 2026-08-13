.. rst-class:: skill-reference-page

braintrace
==========

Complete skill: :skill-source:`source <skills/braintrace/SKILL.md>`

Purpose and boundary
--------------------

Use ``braintrace`` when BPTT activation memory limits recurrent or spiking training. It replaces the sequence-length-dependent backward graph with forward eligibility traces. Use it for temporal learning memory pressure, not general simulation speed or model dynamics.

Major contents
--------------

- Choose a supported eligibility-trace estimator from the scientific and memory requirements.
- Prefer prebuilt BrainTrace layers before custom ETP operations, primitives, or algorithms.
- Compile a correctly shaped example step once, then reuse the learner in temporal training.
- Map batches and models without applying mapping twice or bypassing the mapped learner.
- Reset recurrent and eligibility State correctly between independent sequences.
- Customize parameter transforms and primitive rules only after built-in paths are exhausted.
- Inspect learner reports and graphs before opening compiler internals.

Reference Markdown
------------------

- ``skills/braintrace/references/Drtrl.md`` - :skill-source:`source <skills/braintrace/references/Drtrl.md>`
- ``skills/braintrace/references/ETP operators.md`` - :skill-source:`source <skills/braintrace/references/ETP operators.md>`
- ``skills/braintrace/references/algorithm selection.md`` - :skill-source:`source <skills/braintrace/references/algorithm selection.md>`
- ``skills/braintrace/references/batching.md`` - :skill-source:`source <skills/braintrace/references/batching.md>`
- ``skills/braintrace/references/compiler_internal.md`` - :skill-source:`source <skills/braintrace/references/compiler_internal.md>`
- ``skills/braintrace/references/custom ETP primitives.md`` - :skill-source:`source <skills/braintrace/references/custom ETP primitives.md>`
- ``skills/braintrace/references/custom algorithms.md`` - :skill-source:`source <skills/braintrace/references/custom algorithms.md>`
- ``skills/braintrace/references/customizing_primitive_transforms.md`` - :skill-source:`source <skills/braintrace/references/customizing_primitive_transforms.md>`
- ``skills/braintrace/references/pp_pprop workflow.md`` - :skill-source:`source <skills/braintrace/references/pp_pprop workflow.md>`
- ``skills/braintrace/references/pre-built-braintrace-layer.md`` - :skill-source:`source <skills/braintrace/references/pre-built-braintrace-layer.md>`
