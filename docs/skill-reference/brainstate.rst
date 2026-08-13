.. rst-class:: skill-reference-page

brainstate
==========

Complete skill: :skill-source:`source <skills/brainstate/SKILL.md>`

Purpose and boundary
--------------------

Use ``brainstate`` for general stateful execution: mutable ``State``, registered ``Module`` graphs, environment-scoped simulation, state initialization and collection, operational randomness, state-aware transformations, and training. It supplies shared infrastructure; the modeling package still owns neurons, cells, events, or neural masses.

Major contents
--------------

- Select State roles for parameters, hidden dynamics, batches, delays, and other lifecycles.
- Build reusable ``Module`` graphs and inspect, split, replace, or reconstruct their State.
- Compose size-aware neural-network layers and infer shapes through module graphs.
- Scope time, ``dt``, fit mode, precision, and platform in simulation environments.
- Seed and restore reproducible random streams.
- Apply state-aware ``jit``, ``grad``, ``vmap``, loops, branches, and checkpoints.
- Constrain and regularize parameters, debug transformed code, and execute paired perturbations.

Reference Markdown
------------------

- ``skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md`` - :skill-source:`source <skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md>`
- ``skills/brainstate/references/brainstate/brainstate-transformed-diagnostics.md`` - :skill-source:`source <skills/brainstate/references/brainstate/brainstate-transformed-diagnostics.md>`
- ``skills/brainstate/references/brainstate/parameter-constraints-regularization.md`` - :skill-source:`source <skills/brainstate/references/brainstate/parameter-constraints-regularization.md>`
- ``skills/brainstate/references/brainstate/parameter-transforms-regularizers-catalog.md`` - :skill-source:`source <skills/brainstate/references/brainstate/parameter-transforms-regularizers-catalog.md>`
- ``skills/brainstate/references/brainstate/randomness-and-reproducibility.md`` - :skill-source:`source <skills/brainstate/references/brainstate/randomness-and-reproducibility.md>`
- ``skills/brainstate/references/brainstate/transformation-grad-expansion.md`` - :skill-source:`source <skills/brainstate/references/brainstate/transformation-grad-expansion.md>`
- ``skills/brainstate/references/brainstate/transformation-jit-expansion.md`` - :skill-source:`source <skills/brainstate/references/brainstate/transformation-jit-expansion.md>`
- ``skills/brainstate/references/brainstate/transformation-vmap-expansion.md`` - :skill-source:`source <skills/brainstate/references/brainstate/transformation-vmap-expansion.md>`
- ``skills/brainstate/references/collective_model_operations.md`` - :skill-source:`source <skills/brainstate/references/collective_model_operations.md>`
- ``skills/brainstate/references/extension_mechanisms.md`` - :skill-source:`source <skills/brainstate/references/extension_mechanisms.md>`
- ``skills/brainstate/references/libraries/prebuilt-activation-library.md`` - :skill-source:`source <skills/brainstate/references/libraries/prebuilt-activation-library.md>`
- ``skills/brainstate/references/libraries/prebuilt-layer-library.md`` - :skill-source:`source <skills/brainstate/references/libraries/prebuilt-layer-library.md>`
- ``skills/brainstate/references/model-interop-and-migration.md`` - :skill-source:`source <skills/brainstate/references/model-interop-and-migration.md>`
- ``skills/brainstate/references/paired-perturbation-execution.md`` - :skill-source:`source <skills/brainstate/references/paired-perturbation-execution.md>`
- ``skills/brainstate/references/simulation-environment.md`` - :skill-source:`source <skills/brainstate/references/simulation-environment.md>`
- ``skills/brainstate/references/size-inference-variations.md`` - :skill-source:`source <skills/brainstate/references/size-inference-variations.md>`
- ``skills/brainstate/references/state-graph-operations.md`` - :skill-source:`source <skills/brainstate/references/state-graph-operations.md>`
- ``skills/brainstate/references/state_collections_and_utilities.md`` - :skill-source:`source <skills/brainstate/references/state_collections_and_utilities.md>`
