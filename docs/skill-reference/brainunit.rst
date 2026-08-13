.. rst-class:: skill-reference-page

brainunit
=========

Complete skill: :skill-source:`source <skills/brainunit/SKILL.md>`

Purpose and boundary
--------------------

Use ``brainunit`` when physical meaning must remain attached to numerical values. It owns quantities, dimensions, unit conversion, dimensional validation, unit-aware JAX computation, and boundaries where a raw array enters or leaves a scientific model. Do not use it to choose model State, simulation architecture, or training structure.

Major contents
--------------

- Construct scalar and array quantities from predefined or custom units.
- Inspect dimensions, mantissas, units, compatibility, and conversions.
- Apply unit-preserving, dimensionless-input, unit-changing, reduction, and contraction math correctly.
- Reshape, index, update, broadcast, join, split, and transform unit-aware arrays.
- Use physical constants, SI prefixes, typing aliases, and runtime unit validation.
- Preserve dimensions through ``jit``, ``vmap``, and ``grad`` instead of stripping units.

Reference Markdown
------------------

- ``skills/brainunit/references/array-creation.md`` - :skill-source:`source <skills/brainunit/references/array-creation.md>`
- ``skills/brainunit/references/array-mechanics.md`` - :skill-source:`source <skills/brainunit/references/array-mechanics.md>`
- ``skills/brainunit/references/array-mechanics/functional-structural-api.md`` - :skill-source:`source <skills/brainunit/references/array-mechanics/functional-structural-api.md>`
- ``skills/brainunit/references/math-function-library.md`` - :skill-source:`source <skills/brainunit/references/math-function-library.md>`
- ``skills/brainunit/references/physical-constant-library.md`` - :skill-source:`source <skills/brainunit/references/physical-constant-library.md>`
- ``skills/brainunit/references/prefix-library.md`` - :skill-source:`source <skills/brainunit/references/prefix-library.md>`
- ``skills/brainunit/references/quantity-inspection-and-conversion.md`` - :skill-source:`source <skills/brainunit/references/quantity-inspection-and-conversion.md>`
- ``skills/brainunit/references/typing-and-runtime-validation.md`` - :skill-source:`source <skills/brainunit/references/typing-and-runtime-validation.md>`
- ``skills/brainunit/references/unit-structure-and-definition.md`` - :skill-source:`source <skills/brainunit/references/unit-structure-and-definition.md>`
