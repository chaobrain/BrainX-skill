.. rst-class:: skill-reference-page

brainevent
==========

Complete skill: :skill-source:`source <skills/brainevent/SKILL.md>`

Purpose and boundary
--------------------

Use ``brainevent`` for binary spikes and their communication through dense, explicit sparse, generated, or fixed-degree connectivity. It owns event operators and activity-dependent updates, not neuron dynamics or complete simulations.

Major contents
--------------

- Represent binary firing vectors with ``BinaryArray``.
- Choose dense storage, CSR/CSC, JIT-generated connectivity, or fixed fan-in/fan-out.
- Preserve matrix orientation, shape, seed stability, and connectivity semantics.
- Apply spike-driven postsynaptic operations and synaptic plasticity.
- Extend CPU or GPU event kernels only when built-in operators cannot express the computation.
- Keep custom operators compatible with JAX transformations and verify them against a reference implementation.

Reference Markdown
------------------

- ``skills/brainevent/references/connectivity-variants.md`` - :skill-source:`source <skills/brainevent/references/connectivity-variants.md>`
- ``skills/brainevent/references/custom-operators-cpu.md`` - :skill-source:`source <skills/brainevent/references/custom-operators-cpu.md>`
- ``skills/brainevent/references/custom-operators-gpu.md`` - :skill-source:`source <skills/brainevent/references/custom-operators-gpu.md>`
- ``skills/brainevent/references/sparse-formats.md`` - :skill-source:`source <skills/brainevent/references/sparse-formats.md>`
- ``skills/brainevent/references/synaptic-plasticity.md`` - :skill-source:`source <skills/brainevent/references/synaptic-plasticity.md>`
