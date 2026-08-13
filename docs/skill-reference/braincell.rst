.. rst-class:: skill-reference-page

braincell
=========

Complete skill: :skill-source:`source <skills/braincell/SKILL.md>`

Purpose and boundary
--------------------

Use ``braincell`` when cellular biophysics is explicit: ions, channels, compartments, membrane area, concentration dynamics, or morphology. It owns both single-compartment conductance-based cells and morphology-based multicompartment cells. Route networks of cells to ``brainpy-state`` as well.

Major contents
--------------

- Choose between ``SingleCompartment`` and morphology-based ``Cell`` workflows.
- Select built-in ions and channels, match mechanism root types, and author custom mechanisms only when needed.
- Compose mixed-ion mechanisms for adaptation and controlled channel ablation.
- Convert density parameters to total capacitance, conductance, and current using membrane area.
- Select solvers and reason about their numerical effects.
- Load or construct morphology, choose CV policies, paint mechanisms, place point mechanisms, and record probes.
- Validate topology, discretization, mechanism placement, trace availability, and scientific controls.

Reference Markdown
------------------

- ``skills/braincell/references/area-scaled-hh-pattern.md`` - :skill-source:`source <skills/braincell/references/area-scaled-hh-pattern.md>`
- ``skills/braincell/references/braincell-custom-ion-channel-authoring.md`` - :skill-source:`source <skills/braincell/references/braincell-custom-ion-channel-authoring.md>`
- ``skills/braincell/references/channel-library.md`` - :skill-source:`source <skills/braincell/references/channel-library.md>`
- ``skills/braincell/references/ion-library.md`` - :skill-source:`source <skills/braincell/references/ion-library.md>`
- ``skills/braincell/references/mixions-for-adaptation.md`` - :skill-source:`source <skills/braincell/references/mixions-for-adaptation.md>`
- ``skills/braincell/references/solver-library-with-effects.md`` - :skill-source:`source <skills/braincell/references/solver-library-with-effects.md>`
- ``skills/braincell/references/multicompartment/braincell-manual-morphology-construction.md`` - :skill-source:`source <skills/braincell/references/multicompartment/braincell-manual-morphology-construction.md>`
- ``skills/braincell/references/multicompartment/cv-policy-reference.md`` - :skill-source:`source <skills/braincell/references/multicompartment/cv-policy-reference.md>`
- ``skills/braincell/references/multicompartment/filter-function-library.md`` - :skill-source:`source <skills/braincell/references/multicompartment/filter-function-library.md>`
- ``skills/braincell/references/multicompartment/morphology-io-loading-validation.md`` - :skill-source:`source <skills/braincell/references/multicompartment/morphology-io-loading-validation.md>`
- ``skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`` - :skill-source:`source <skills/braincell/references/multicompartment/multicompartment-cell-workflow.md>`
- ``skills/braincell/references/multicompartment/probe-reference.md`` - :skill-source:`source <skills/braincell/references/multicompartment/probe-reference.md>`
- ``skills/braincell/references/multicompartment/topology-building-and-visualization.md`` - :skill-source:`source <skills/braincell/references/multicompartment/topology-building-and-visualization.md>`
