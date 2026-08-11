---
name: brainx-general-guard
description: Use first for every BrainX modeling, simulation, training, review, debugging, or optimization task. Identify every modeling scale explicitly represented, open only the BrainX package skills that own those scales, and keep the implementation BrainX-native.
---

# BrainX general guard

## Purpose and boundary

Use this guard first to identify the represented modeling scales, open their owning package skills, and keep package orchestration ahead of lower-level infrastructure. Keep it active for cross-cutting API selection, execution, interoperability, and validation decisions.

## Check package presence without inspecting installed BrainX

Treat installed BrainX packages only as execution dependencies: check whether each required package is importable, then move to the owning skills or route an absent package to BrainX-install. Never inspect installed versions, metadata, files, modules, symbols, signatures, docstrings, runtime definitions, source code, or object internals for modeling knowledge.

## Study modeling skills and task-relevant scripts

1. Treat the selected modeling skills as the authoritative guides to BrainX modeling. Read each selected skill completely.
2. Follow each skill's exact routing instructions. Open every reference likely to affect the user's task.
3. Identify the example scripts referenced by the skill or its routed references. Open and study every script that is highly related to the user's task.
4. Trace each relevant script end to end.
5. Reconcile each script with the current root skill before copying infrastructure. Keep its scientific pattern, but replace superseded low-level execution with the owning package's current canonical API.
6. Derive the implementation from the reconciled patterns, then adapt only the parts required by the user's scientific model.

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

Keep the implementation at the selected scale. Do not introduce aggregate population-rate State into a point-neuron model, or point neurons into an aggregate model, merely to simplify execution. When both mechanisms are scientifically represented, treat the task as multiscale and open both owning skills.

## Write BrainX-native code

Start from the scientific concept and use the selected BrainX skills to construct the workflow. Keep ordinary Python, NumPy, or JAX at explicit boundaries for documented dimensionless model inputs, host-side statistics, serialization, timing, device reporting, or custom presentation logic. Preserve units and State until that boundary, and verify an API gap before writing generic numerical infrastructure.

## Enforce absolute coding simplicity

Invest substantial effort before coding. Study every routed skill, relevant example and reference, and authoritative BrainX API needed to understand the task and discover which BrainX operations remove custom logic. Then deliver absolutely simple code. Simplicity is an output criterion, not permission to skip study or guess an API.

Use the smallest clear composition that completely satisfies the scientific contract. Minimize concepts, layers, functions, classes, State objects, transforms, branches, conversions, dependencies, files, monitors, and outputs.

Keep an element only when removing it would change scientific meaning, correctness, required performance, verification of a costly failure, or requested output quality. Remove pass-through wrappers, duplicate representations, speculative configuration, redundant transforms, repeated calculations, defensive branches for impossible inputs, and unrequested artifacts. Prefer one direct data flow, one owner for each State, one stable transform boundary, and one canonical execution path.

Keep a one-off scientific demonstration concrete. Add a configuration object, result class, helper layer, command-line option, or extra artifact only when the request requires variation or reuse, or when that structure removes more complexity than it adds.

Do not simplify away units, explicit State lifecycles, numerical validity, scientifically necessary mechanisms, performance-critical compilation or batching, focused checks, or requested output quality. These are requirements, not optional complexity.

## Prefer high-level BrainX APIs

Use high-level APIs as the abstraction boundary: simulation code should state the scientific operation while BrainX handles array manipulation, unit propagation, State threading, numerical steps, and infrastructure.

| Need | Prefer |
|---|---|
| Package-owned simulation, fitting, training, or analysis | The selected modeling package's named orchestrator or Module, such as `brainmass.Simulator`, `Network`, `Fitter`, or `brainmass.viz` |
| Physical quantities, unit-aware arrays, or mathematical operations | BrainUnit quantities and `brainunit.math` |
| State, randomness, initialization, environments, or State-aware transforms | BrainState, after checking whether the selected package orchestrator already owns the operation |
| Connectivity, encoding, inputs, integration, metrics, optimization, surrogate gradients, training, or visualization | `braintools` |

Do not let code grow around manual indexing, reshaping, reductions, equations, loops, or bookkeeping. Prefer one named BrainX operation or Module over a chain of generic primitives.

Write custom logic only when it expresses model behavior that the ecosystem does not already provide. Verify the owning skill, reference, or official API page before using an unfamiliar name or signature.

Use the highest-level API in the selected owning package that preserves the scientific operation. Open lower-level BrainState control flow only when the package orchestrator cannot express the required inputs, monitors, State effects, or stable compilation boundary.

## Keep visualization simple without lowering figure quality

Use the simplest highest-level API that expresses the required scientific figure. Prefer the selected BrainX package's visualization API, such as `brainmass.viz`, then BrainTools visualization APIs, then high-level `matplotlib.pyplot`; use low-level Matplotlib `Figure`, `Axes`, `Artist`, or styling machinery only for a requirement those APIs cannot express.

Simplicity applies to implementation, not scientific content or figure quality. Preserve intentional size, units, readable labels, title, comparison styles, legend, unclipped layout, and sufficient output resolution.

## Transform stateful execution

Use the owning package's orchestrator for workflows it already implements. When a custom stateful operation is necessary, transform the complete operation with `brainstate.transform`.

| API | Use |
|---|---|
| `brainstate.transform.jit` | Compile a complete custom stateful operation. Construct one stable callable outside warm-up and timed repetitions when measuring compilation or steady execution. |
| `brainstate.transform.grad` | Differentiate a complete custom stateful forward, simulation, or training operation. |
| `brainstate.transform.vmap` | Batch a complete custom stateful operation when the owning package does not already expose the required batch axis. |
| `brainstate.transform.for_loop` | Run a fixed time or data sequence when iteration-to-iteration effects live in `State`; it slices leading input axes and stacks per-step outputs. |
| `brainstate.transform.scan` | Run a sequence when an ordinary explicit carry must pass between iterations. |

Do not add a transform only to construct parameter axes or satisfy a named-API checklist. When the owning package already represents independent conditions through a native batch or `size` axis, use that path and reserve `vmap` for a callable the owning package does not already batch.

Map only independent operations. If trial N mutates weights or other State that trial N+1 consumes, keep those trials sequential; batch only independent evaluation trials or ensembles, with separate dynamical State and deliberately shared read-only State. Mapping input construction or offline scoring does not batch the stateful simulation itself. Open `skills/brainstate/references/brainstate/transformation-vmap-expansion.md` when a complete mapped operation reads or writes State, for the State-axis contracts and independent-lane pattern.

A mapped complete per-step transition called inside one `for_loop` is a stateful simulation batch: the mapping owns independent conditions and the loop owns time. Select mapped State by semantic role, such as `Any(OfType(HiddenState), OfType(ShortTermState))`; do not infer State ownership from array rank, shape, or a coincidental leading size. Open `skills/brainstate/references/brainstate/transformation-vmap-expansion.md` for the canonical initialization, mapping, and loop composition.

Describe the transform that actually runs. Native batched State, vmapped input construction, and vmapped host scoring are not stateful `vmap`. When the user explicitly requires `vmap`, map the complete independent operation or report why the requirement cannot be met.

Separate dependency order from State lifetime. When only learned or long-term State must carry from trial N to trial N+1, preserve that State but reset membrane, refractory, delay, trace, and other per-trial State at the logical boundary unless continuous carryover is part of the model. A silent interval advances State; it does not reset it.

Do not use a Python loop for simulation timesteps, recurrent sequences, or repeated State updates inside one logical rollout. A small host loop is valid across causally sequential trials when each iteration calls one compiled trial rollout and the boundary must reset selected State while preserving learned State.

Use raw JAX transformations only for pure array or PyTree functions that do not close over BrainState `State`.

## Boundaries and common failures

- Generic NumPy or JAX used as the starting architecture for a BrainX simulation.
- Point-neuron and aggregate population mechanisms mixed without selecting both owning skills or declaring the multiscale boundary.
- A custom BrainState loop that duplicates the selected package's runner, inputs, monitoring, initialization, or sampling.
- Manual array or mathematical machinery that duplicates BrainUnit or BrainTools.
- Python timestep loops or host loops inside a stable logical rollout.
- Raw `jax.jit`, `jax.grad`, or `jax.vmap` applied to State-aware code.
- Host-side statistics, serialization, timing, device reporting, or custom presentation forced into BrainX without an owning API.
- Fabricated APIs or signatures accepted without checking the owning documentation.
