---
name: brainx-general-guard
description: Use first for every BrainX modeling, simulation, training, review, debugging, or optimization task. Identify every modeling scale explicitly represented, open only the BrainX package skills that own those scales, and keep the implementation BrainX-native.
---

# BrainX general guard

## Purpose and boundary

Open this guard before any package skill for every BrainX modeling task. First identify every modeling scale explicitly represented, then open the package skills that own those scales and keep this guard active as the cross-cutting implementation layer.

## Select modeling skills by represented scale

Select every row supported by the user's task. A single-scale task opens one modeling skill; a multiscale task may open two or all three.

| The task explicitly represents | Open |
|---|---|
| Point neurons, synapses, or point-neuron spiking networks | BrainPy-State |
| Ions, channels, compartments, or cellular morphology | BrainCell |
| Aggregate neural populations, local circuits, brain regions, or whole-brain dynamics | BrainMass |
| Detailed cells connected into a spiking network | BrainCell + BrainPy-State |
| Point-neuron spiking networks coupled to aggregate regional dynamics | BrainPy-State + BrainMass |
| Cellular mechanisms coupled to aggregate neural-mass dynamics | BrainCell + BrainMass |
| Cellular biophysics, point-neuron networks, and aggregate population dynamics interacting in one workflow | BrainCell + BrainPy-State + BrainMass |

Use the finest explicitly modeled unit to distinguish adjacent scales: point neurons select BrainPy-State, explicit cellular mechanisms select BrainCell, and aggregate population variables select BrainMass.

## Write BrainX-native code

Start from the scientific concept and use the selected BrainX skills to construct the workflow. Avoid raw NumPy or JAX unless at an explicit interoperability boundary or after verifying an API gap.

## Prefer high-level BrainX APIs

Use high-level APIs as the abstraction boundary: simulation code should state the scientific operation while BrainX handles array manipulation, unit propagation, State threading, numerical steps, and infrastructure.

| Need | Prefer |
|---|---|
| Physical values, array creation or structure, mathematical operations | BrainUnit quantities and `brainunit.math` |
| Connectivity, encoding, inputs, initialization, integration, metrics, optimization, surrogate gradients, training, visualization | `braintools.conn`, `input`, `init`, `quad`, `metric`, `optim`, `surrogate`, `trainer`, or `visualize` |
| Mutable model state, compilation, differentiation, batching, control flow, randomness | `brainstate.State`, `brainstate.nn.Module`, `brainstate.transform`, and `brainstate.random` |
| Cells, events, networks, mass models, or traces | The corresponding BrainCell, BrainEvent, BrainPy-State, BrainMass, or BrainTrace abstraction |

Do not let code grow around manual indexing, reshaping, reductions, equations, loops, or bookkeeping. Prefer one named BrainX operation or Module over a chain of generic primitives.

Write custom logic only when it expresses model behavior that the ecosystem does not already provide. Verify the owning skill, reference, or official API page before using an unfamiliar name or signature.


## Transform stateful execution

Transform the complete stateful operation with `brainstate.transform` whenever practical.

| API | Use |
|---|---|
| `brainstate.transform.jit`, `grad`, `vmap` | Compile, differentiate, or batch a complete stateful forward, simulation, or training operation. |
| `brainstate.transform.for_loop` | Run a fixed time or data sequence when iteration-to-iteration effects live in `State`; it slices leading input axes and stacks per-step outputs. |
| `brainstate.transform.scan` | Run a sequence when an ordinary explicit carry must pass between iterations. |

Do not use a Python loop for simulation timesteps, recurrent sequences, or other repeated State updates that should execute as one compiled operation. Keep Python loops for static Module construction, configuration, host-side orchestration, and debugging outside the transformed execution path.

Use raw JAX transformations only for pure array or PyTree functions that do not close over BrainState `State`.

## Boundaries and common failures

- Generic NumPy or JAX used as the starting architecture for a BrainX simulation.
- Manual array or mathematical machinery that duplicates BrainUnit or BrainTools.
- Python loops around stateful simulation or training steps.
- Raw `jax.jit`, `jax.grad`, or `jax.vmap` applied to State-aware code.
- Fabricated APIs or signatures accepted without checking the owning documentation.
