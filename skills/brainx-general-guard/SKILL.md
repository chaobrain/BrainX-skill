---
name: brainx-general-guard
description: Use first for every BrainX modeling, simulation, training, review, debugging, or optimization task. Identify every modeling scale explicitly represented, open only the BrainX package skills that own those scales, and keep the implementation BrainX-native.
---

# BrainX general guard

## Purpose and boundary

Route every BrainX modeling, simulation, training, review, debugging, or optimization task to the package skills that own it, then enforce the ecosystem-wide rules that no single package skill owns: never mine the installed packages for knowledge, write BrainX-native code, prefer high-level APIs, and transform stateful execution.

This skill supplies no modeling APIs of its own. It selects the owning skills and hands the task to them; every signature, constructor, and workflow comes from those skills and their references.

## Never inspect BrainX packages in the venv for knowledge

Treat BrainX packages in the active virtual environment only as execution dependencies, never as sources of modeling knowledge.

- **ONLY check presence.** Determine whether each required BrainX package is present, then stop inspecting the environment.
- **NEVER investigate installed content.** Do not enumerate package files, modules, symbols, signatures, docstrings, runtime definitions, or object internals.
- **NEVER read installed source code.** Do not open or read any BrainX source code in the virtual environment.


## Study modeling skills and task-relevant scripts

1. Treat the selected modeling skills as the authoritative guide to BrainX modeling, dive deep into the skill.
2. Follow the skill's exact routing instructions. Open every reference that are likely relevant by the user's task.
3. Identify the example scripts referenced by the skill or its routed references. Open and study every script that is highly related to the user's task.
4. Trace each relevant script end to end.
5. Derive the implementation from those canonical patterns, then adapt only the parts required by the user's scientific model.



## Select modeling skills by represented scale

Select every row supported by the user's task. A single-scale task opens one modeling skill; a multiscale task may open two or all three of the scale skills.

| The task explicitly represents | Open |
|---|---|
| Point neurons, synapses, or point-neuron spiking networks | BrainPy-State |
| Ions, channels, compartments, or cellular morphology | BrainCell |
| Aggregate neural populations, local circuits, brain regions, or whole-brain dynamics | BrainMass |
| Detailed cells connected into a spiking network | BrainCell + BrainPy-State |
| Point-neuron spiking networks coupled to aggregate regional dynamics | BrainPy-State + BrainMass |
| Cellular mechanisms coupled to aggregate neural-mass dynamics | BrainCell + BrainMass |
| Cellular biophysics, point-neuron networks, and aggregate population dynamics interacting in one workflow | BrainCell + BrainPy-State + BrainMass |

## Also select skills by cross-cutting concern

Scale is not the only axis. Open each row whose concern the task contains, in addition to every scale skill selected above.

| The task also involves | Open |
|---|---|
| Physical quantities, units, dimensional mismatches, unit-aware math, or suspicious bare numbers | BrainUnit |
| Mutable State and `.value`, Module graphs, state initialization, environment scoping, or state-aware `jit`/`grad`/`vmap` | BrainState |
| Binary firing events, sparse or probabilistic connectivity, spike-driven postsynaptic input, or activity-dependent weight changes | BrainEvent |
| Eligibility-trace online learning, D-RTRL or pp-prop selection, or training a recurrent or spiking model without BPTT | BrainTrace |
| Installing, upgrading, pinning, migrating, or removing BrainX packages | BrainX-Install |

## Write BrainX-native code

Start from the scientific concept and use the selected BrainX skills to construct the workflow. Avoid raw NumPy or JAX unless at an explicit interoperability boundary or after verifying an API gap.

## Prefer high-level BrainX APIs

Use high-level APIs as the abstraction boundary: simulation code should state the scientific operation while BrainX handles array manipulation, unit propagation, State threading, numerical steps, and infrastructure.

| Need | Prefer |
|---|---|
| array creation or structure, mathematical operations | BrainUnit quantities and `brainunit.math` |
| Connectivity, encoding, inputs, initialization, integration, metrics, optimization, surrogate gradients, training, visualization | `braintools` |


Do not let code grow around manual indexing, reshaping, reductions, equations, loops, or bookkeeping. Prefer one named BrainX operation or Module over a chain of generic primitives.

Write custom logic only when it expresses model behavior that the ecosystem does not already provide. Verify the owning skill, reference, or official API page before using an unfamiliar name or signature.

## Keep plotting code short without lowering figure quality

Use high-level `matplotlib.pyplot` APIs for standard scientific plots. Prefer a compact sequence of `plt.figure()`, `plt.plot()`, labeling, layout, and display calls over low-level Figure, Axes, Artist, or styling machinery when the high-level API expresses the same figure clearly.


Brevity applies to the code, not the figure's scientific content or visual quality.


## Transform stateful execution

Transform the complete stateful operation with `brainstate.transform` whenever practical.

| API | Use |
|---|---|
| `brainstate.transform.jit`, `grad`, `vmap` | Compile, differentiate, or batch a complete stateful forward, simulation, or training operation. |
| `brainstate.transform.for_loop` | Run a fixed time or data sequence when iteration-to-iteration effects live in `State`; it slices leading input axes and stacks per-step outputs. |
| `brainstate.transform.scan` | Run a sequence when an ordinary explicit carry must pass between iterations. |

Do not add a transform only to construct parameter axes or satisfy a named-API checklist. When the owning package already represents independent conditions through a native batch or `size` axis, use that path and reserve `vmap` for a callable the owning package does not already batch.

Do not use a Python loop for simulation timesteps, recurrent sequences, or other repeated State updates that should execute as one compiled operation.

Use raw JAX transformations only for pure array or PyTree functions that do not close over BrainState `State`.

## Boundaries and common failures

- Generic NumPy or JAX used as the starting architecture for a BrainX simulation.
- Manual array or mathematical machinery that duplicates BrainUnit or BrainTools.
- Python loops around stateful simulation or training steps.
- Raw `jax.jit`, `jax.grad`, or `jax.vmap` applied to State-aware code.
- Fabricated APIs or signatures accepted without checking the owning documentation.
