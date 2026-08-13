.. rst-class:: skill-reference-page

brainpy-state
=============

Complete skill: :skill-source:`source <skills/brainpy-state/SKILL.md>`

Purpose and boundary
--------------------

Use ``brainpy-state`` for point neurons, synapses, projections, spiking networks, inputs, delays, plasticity, readouts, and trainable SNNs. The native ``brainpy.state`` path is canonical. Open the legacy or NEST-compatible branches only when the task explicitly requires them.

Major contents
--------------

- Select point-neuron, synapse, communication, output, and projection components.
- Build unit-aware networks whose projections separate communication, synaptic dynamics, output, and postsynaptic targets.
- Generate inputs, configure delays, initialize and reset State, and execute simulation rollouts.
- Validate controlled perturbations and causal comparisons.
- Train spiking models with losses, optimizers, initializers, surrogate gradients, metrics, and transform-safe control flow.
- Translate NEST devices, model names, connectivity, and workflow concepts with explicit parity limits.
- Preserve legacy BrainPy only for an existing legacy codebase that explicitly requires it.

Core and BrainTools reference Markdown
--------------------------------------

- ``skills/brainpy-state/references/array-creation.md`` - :skill-source:`source <skills/brainpy-state/references/array-creation.md>`
- ``skills/brainpy-state/references/brain-dynamics-delay-protocol.md`` - :skill-source:`source <skills/brainpy-state/references/brain-dynamics-delay-protocol.md>`
- ``skills/brainpy-state/references/component-selection.md`` - :skill-source:`source <skills/brainpy-state/references/component-selection.md>`
- ``skills/brainpy-state/references/perturbation-experiment-validity.md`` - :skill-source:`source <skills/brainpy-state/references/perturbation-experiment-validity.md>`
- ``skills/brainpy-state/references/projection-patterns.md`` - :skill-source:`source <skills/brainpy-state/references/projection-patterns.md>`
- ``skills/brainpy-state/references/braintools/brainstate-control-flow-patterns.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/brainstate-control-flow-patterns.md>`
- ``skills/brainpy-state/references/braintools/connectivity.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/connectivity.md>`
- ``skills/brainpy-state/references/braintools/data-preprocessing.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/data-preprocessing.md>`
- ``skills/brainpy-state/references/braintools/input-current.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/input-current.md>`
- ``skills/brainpy-state/references/braintools/metric.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/metric.md>`
- ``skills/brainpy-state/references/braintools/optimizer.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/optimizer.md>`
- ``skills/brainpy-state/references/braintools/parameter-initializer.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/parameter-initializer.md>`
- ``skills/brainpy-state/references/braintools/surrogate.md`` - :skill-source:`source <skills/brainpy-state/references/braintools/surrogate.md>`

NEST-compatible reference Markdown
----------------------------------

- ``skills/brainpy-state/references/nest-compatible/devices.md`` - :skill-source:`source <skills/brainpy-state/references/nest-compatible/devices.md>`
- ``skills/brainpy-state/references/nest-compatible/divergence-and-parity.md`` - :skill-source:`source <skills/brainpy-state/references/nest-compatible/divergence-and-parity.md>`
- ``skills/brainpy-state/references/nest-compatible/integration-categories.md`` - :skill-source:`source <skills/brainpy-state/references/nest-compatible/integration-categories.md>`
- ``skills/brainpy-state/references/nest-compatible/model-library.md`` - :skill-source:`source <skills/brainpy-state/references/nest-compatible/model-library.md>`
- ``skills/brainpy-state/references/nest-compatible/nest-workflow.md`` - :skill-source:`source <skills/brainpy-state/references/nest-compatible/nest-workflow.md>`
- ``skills/brainpy-state/references/nest-compatible/network-building.md`` - :skill-source:`source <skills/brainpy-state/references/nest-compatible/network-building.md>`
- ``skills/brainpy-state/references/nest-compatible/synapse-and-connectivity.md`` - :skill-source:`source <skills/brainpy-state/references/nest-compatible/synapse-and-connectivity.md>`

Legacy BrainPy reference Markdown
---------------------------------

- ``skills/brainpy-state/references/brainPy(legacy)/analysis.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/analysis.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/brainpy legacy workflow.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/brainpy legacy workflow.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/built-in dynamic neuron model.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/built-in dynamic neuron model.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/connecting neurons.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/connecting neurons.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/customize neuron and synpase.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/customize neuron and synpase.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/integrators.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/integrators.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/route activity through connectivity.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/route activity through connectivity.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/synaptic projections.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/synaptic projections.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/synpase properties.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/synpase properties.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/Input generation.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/Input generation.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/More about simulation.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/More about simulation.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/Multi-device array sharding.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/Multi-device array sharding.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/Parallel experiment execution.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/Parallel experiment execution.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/array creation and mechanics.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/array creation and mechanics.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/brainpy math environment setting.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/brainpy math environment setting.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/delays.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/delays.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/infrastructure/object oriented transformations and control flows.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/infrastructure/object oriented transformations and control flows.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/training/loss library.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/training/loss library.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/training/optimizers.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/training/optimizers.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/training/parameter initializers.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/training/parameter initializers.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/training/prebuilt neural network layers.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/training/prebuilt neural network layers.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/training/surrogate gradients.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/training/surrogate gradients.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/training/trainer library.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/training/trainer library.md>`
- ``skills/brainpy-state/references/brainPy(legacy)/training/trainingworkflows.md`` - :skill-source:`source <skills/brainpy-state/references/brainPy(legacy)/training/trainingworkflows.md>`
