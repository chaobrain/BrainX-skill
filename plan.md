# BrainX Skill Workspace Implementation Plan

### Why BrainX skill?

BrainX is an differentiable, extensible, high performance and JAX-integrated infrastructure for brain simulation, and is designed around the practical workflow of modern computational neuroscience researchers. at the same time, the researchers are the target users of BrainX.

With the rise of AI coding agents such as Codex and Claude in 2026, agents are becoming increasingly embedded in researchers’ daily programming workflows. Researchers now expect an agent to use appropriate APIs, construct experiments, debug simulations, optimise performance, and modify code without changing its scientific meaning.

Generic coding agents, however, do not automatically understand the BrainX API cleanly, so it sometimes might produce messy, low performance code, then the user of brainX can't utilize the power of coding agent and BrainX at the same time easily.

### Mission
The skill is filling the gap: letting the codex understand the BrainX system, so that BrainX user utilize both powerful tools without effort or prior expertise.

The design of BrainX skill, mirror the guideline of Anthropic and other scientific computing package skills(e.g Nvidia scikit), follows the pattern of progressive disclosure: compact core skill defines the core concepts and standard workflows, while package-specific, variations of APIs, libraries knowledge lives in separate Markdown references.

This ensures two advantage: first, it avoids the agent from opening useless context, allowing the codex to perserve the original reasoning and generalized ability while integrating with BrainX. Second, extensibility, just like BrainX ecosystem itself, that future APIs built on the current core can be added directly as new Markdown files without rewriting the whole skill.



## 1. Primary Skill List

```text
skills/
├── brainunit/
├── brainstate/
├── braincell/
├── brainevent/
├── brainmass/
├── brainpy/
├── braintrace/
├── brainx-acceleration-audit/
└── brainx-install/
```


## 2. Skill Layer Design

### brainunit

#### Purpose

- Boundary: enforce physical-quantity, dimensional, conversion, and external-library boundary safety.
- Activate for voltage, current, time, conductance, capacitance, length, concentration, unit errors, dimensional mismatches, or suspicious bare values.


#### Essential Concepts
- brainunit provides physical units and unit-aware mathematical system in JAX for general AI-driven scientific computing.
- Deep integration with JAX, providing comprehensive support for modern AI framework features including automatic differentiation (autograd), just-in-time compilation (JIT), vectorization, and parallel computation
- Strict physical unit type checking and dimensional inference system, detecting unit inconsistencies during compilation
- Dimension matching, Units are tracked automatically. Incompatible operations raise errors.
- Creating quantities
- Arithmetic with units
- Unit conversion
- Quantity attributes
- `brainunit.math` functions
- Physical constants
- JAX transformations: `jit`, `vmap`, `grad`
- Unit validation with `@check_units`
- BrainUnit represents all units using seven irreducible SI dimensions—length, mass, time, electric current, temperature, amount of substance, and luminous intensity—and derives other units by combining their dimension exponents.
- brainunit generates standard names for units, combining the unit name (e.g. “siemens”) with a prefixes (e.g. “m”), and also generates squared and cubed versions by appending a number. For example, the units “msiemens”, “siemens2”, “usiemens3” are all predefined. look up the prefix in the prefix-library

#### Canonical Workflow Scripts Included in the Skill

1. Create different kinds of quantities: scalars, arrays, or direct construction.
2. Arithmetic with units: addition, multiplication, and division.
3. Arithmetic requires dimension matching.
4. Unit conversion with `to_decimal()`.
5. Quantity attributes from one initialization.
6. `brainunit.math` functions.
7. Physical constants.
8. JAX transformation examples.
9. Unit validation with `@check_units`.

#### Reference Routing

```text
brainunit/
├── quantity-inspection-and-conversion.md
├── array-creation.md
├── array-mechanics.md
├── math-function-library.md
├── unit-structure-and-definition.md
├── typing.md
├── prefix-library.md
└── physical-constant-library.md
```

| Reference | Scope | Sources |
|---|---|---|
| `skills/brainunit/references/array-creation.md` | Create unit-aware arrays from scalars, sequences, ranges, shapes, grids, and existing arrays; generate filled, identity, diagonal, triangular, and template-shaped arrays with explicit units and dtypes | [Array Creation](https://brainunit.readthedocs.io/unit_operations/array_creation.html), with array constructors from the [brainunit.math API](https://brainunit.readthedocs.io/apis/brainunit.math.html) |
| `skills/brainunit/references/array-mechanics.md` | Inspect array identity and metadata; index, slice, functionally update, reshape, flatten, squeeze, transpose, broadcast, concatenate, split, stack, repeat, and convert array backends; perform high-level named-axis transformations | Array properties, methods, functional updates, and backend conversion from the [Quantity API](https://brainunit.readthedocs.io/apis/generated/brainunit.Quantity.html), structural array operations from the [brainunit.math API](https://brainunit.readthedocs.io/apis/brainunit.math.html), and axis rearrangement and repetition from [Einstein Operations](https://brainunit.readthedocs.io/unit_operations/einstein_operations.html) |
| `skills/brainunit/references/quantity-inspection-and-conversion.md` | Inspect quantity mantissas, units, dimensions, compatibility, and convert or extract values in compatible units | [Unit Conversion](https://brainunit.readthedocs.io/physical_units/conversion.html), with selected inspection and conversion methods from the [Quantity API](https://brainunit.readthedocs.io/apis/generated/brainunit.Quantity.html) |
| `skills/brainunit/references/math-function-library.md` | Choose mathematical functions by unit semantics: dimensionless-input, unit-preserving, unit-changing, reduction, contraction, comparison, boolean, and index-returning operations | [brainunit.math API](https://brainunit.readthedocs.io/apis/brainunit.math.html), excluding array creation and structural array manipulation covered by `array-mechanics.md`, with reduction and contraction semantics from [Einstein Operations](https://brainunit.readthedocs.io/unit_operations/einstein_operations.html) |
| `skills/brainunit/references/unit-structure-and-definition.md` | Inspect unit structure, compare dimensions and scales, combine units, and define named, derived, or scaled custom units | [Unit API](https://brainunit.readthedocs.io/apis/generated/brainunit.Unit.html), with canonical unit-composition and custom-definition workflows from [Combining and Defining Unit](https://brainunit.readthedocs.io/advanced_tutorials/combining_and_defining.html) |
| `skills/brainunit/references/typing.md` | Use physical-type utilities, core quantity/unit/dimension type aliases, pre-built physical-dimension aliases, and runtime unit validation | [`brainunit.typing` API](https://brainx.chaobrain.com/brainunit/apis/brainunit.typing.html) |
| `skills/brainunit/references/physical-constant-library.md` | Find and use predefined unit-aware physical constants, including their names, values, dimensions, and canonical units | [Physical Constants](https://brainx.chaobrain.com/brainunit/physical_units/constants.html) |
| `skills/brainunit/references/prefix-library.md` | Find predefined SI base and derived units, understand BrainUnit unit naming, and apply supported prefix symbols and scales | [Standard Units](https://brainx.chaobrain.com/brainunit/physical_units/standard_units.html), including generated prefixed unit names |

#### Boundaries and Common Failures

- Bare physical values passed into BrainCell.
- Premature `.mantissa` extraction.
- `jnp` operations used where units must survive.
- Incompatible dimensions “fixed” by stripping units.
- Missing target unit at raw-array boundaries.
- Using this skill for State, simulation, or training architecture.

---

### brainstate

#### Purpose

- Boundary: own BrainState mutable State, Module graphs, state collection, initialization, transformations, randomness, and general stateful training structure.
- Activate for `State`, `.value`, `ParamState`, `HiddenState`, Modules, graph traversal, lifecycle operations, state-aware `jit`/`grad`/`vmap`, or BrainState training.
- Primary path: classify State roles → construct Modules → register State/children → initialize → transform → validate State and outputs.
- Advanced branches: training, dynamics, randomness, parameter constraints, model graphs, diagnostics, interop, layers, acceleration.

#### Essential Concepts

##### 1. States

- `State` is BrainState's mutation boundary. It encapsulates model values that change over time.
- A `State` can wrap Python scalars, arrays, `jax.Array` values, dictionaries, lists, or any stable PyTree structure. Its value remains mutable after compilation.
- Read and write State through `.value`.
- Core State features:
  1. **Mutable after compilation** — State values can be updated inside JIT-compiled functions.
  2. **Type and shape safety** — State enforces consistent types and shapes.
  3. **JAX integration** — State works with state-aware JAX transformations.
- Choose the appropriate subclass: `ParamState`, `HiddenState`, `ShortTermState`, or `LongTermState`.
- Distinguish `nn.Param` from `ParamState`; use `nn.Param` when parameter constraints are required.

##### 2. Modules

- `brainstate.nn.Module` is the base class for BrainState modules. A Module holds State and child Modules as attributes.
- A model forms a tree of nested Modules with State objects at the leaves.
- Assign State and child Modules to attributes for automatic registration, then collect trainable parameters with `model.states(brainstate.ParamState)`.
- Core Module features:
  1. **Automatic state management** — State and child Modules are registered and collected automatically.
  2. **Clean abstractions** — related State and computation stay encapsulated.
  3. **Reusability** — Modules can be constructed once and reused.
  4. **Composability** — simple Modules combine into larger model graphs.
- Modules also support State inspection, pretty printing, and integration with BrainState transformations.

##### 3. Size Inference and Composition

- Every `brainstate.nn.Module` has `in_size` and `out_size` properties describing data shape without the batch dimension.
- When `in_size` is known, the Module can infer `out_size` automatically.
- In `Sequential` composition, one layer's output size becomes the next layer's input size.
- Use `.desc()` to create a layer descriptor that is instantiated when its input size becomes available.

##### 4. Simulation environment

- Put run-level settings such as `dt`, `fit`, time, precision, and platform in `brainstate.environ`, not in model State or every Module signature.
- Prefer `context()` for scoped simulation, training, and evaluation settings; use `set()` only for intentional persistent defaults.
- Read settings with `get()` or typed accessors such as `get_dt()`.
- Use a separate `EnvironmentState` only when a configuration must be isolated from the default environment.

##### 5. State-Aware Transformations and Randomness

- Write ordinary code that reads and writes `.value`, then wrap the complete operation in `brainstate.transform`; do not apply raw `jax.jit`, `jax.grad`, or `jax.vmap` to stateful code.
- `brainstate.transform` mirrors the JAX transformation API with state-aware `jit`, `grad`, and `vmap` that track the State objects a model reads and writes.
- Use `brainstate.random.DEFAULT` and explicit seeding for reproducible stochastic workflows.
- Prefer whole-step JIT, gradient, and batching transformations over fragmented transforms.

#### Canonical Workflow Scripts Included in the Skill

1. Create State Values and PyTree State
2. State Subclasses and Parameter Choice
3. The Default `RandomState` illustration
4. Seed Management and Reproducibility
5. Add State to a Module
6. Using Basic Pre-built Neural-Network Layers
7. Environment-Scoped Time-Indexed Simulation
8. Minimal State-Aware JIT
9. Minimal Gradient and Parameter Update
10. Composed Training-Step Transform
11. Minimal `vmap`

#### Reference Routing

```text
brainstate/
├── state-graph-operations.md
├── state_utilities.md
├── collective_operations.md
├── extension_mechanisms_Mixin_Hooks.md
├── model-interop-and-migration.md
├── prebuilt-layer-library.md
├── prebuilt-activation-library.md
├── size-inference-variations.md
├── simulation-environment.md
├── parameter-constraints-regularization.md
│   └── parameter-transforms-regularizers-catalog.md
├── transformation-jit-expansion.md [shared]
├── transformation-grad-expansion.md [shared]
├── transformation-vmap-expansion.md
├── brainstate-control-flow-patterns.md [shared]
├── randomness-and-reproducibility.md
├── brainstate-transformed-diagnostics.md
└── braintools/optimizers.md [shared]
```

##### First-layer references

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainstate/references/state-graph-operations.md` | Find, extract, split, replace, and reconstruct State graphs | [JIT tutorial](https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html), [graph API](https://brainx.chaobrain.com/brainstate/apis/graph.html), [graph editing how-to](https://brainx.chaobrain.com/brainstate/how_to/inspect_and_edit_state_graph.html) |
| `skills/brainstate/references/model-interop-and-migration.md` | Flax/Equinox interop and PyTorch migration | [Interop API](https://brainx.chaobrain.com/brainstate/apis/interop.html), [Flax/Equinox how-to](https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html), [PyTorch migration](https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html) |
| `skills/brainstate/references/state_collections_and_utilities.md` | Filter, organize, freeze, flatten, configure, and print nested collections | [Utility Toolkit](https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html) |
| `skills/brainstate/references/collective_model_operations.md` | Initialize, reset, invoke methods, batch lifecycle operations, and restore model-wide State | [Collective Operations](https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html) |
| `skills/brainstate/references/extension_mechanisms.md` | Mixins, descriptors, runtime modes, and State hooks | [Mixin System](https://brainx.chaobrain.com/brainstate/how_to/custom_states_and_mixins.html), [State Hooks](https://brainx.chaobrain.com/brainstate/how_to/state_hooks.html) |
| `skills/brainstate/references/size-inference-variations.md` | Canonical `ComplexNet` composition with `Sequential` / `.desc()`, convolution size formulas and edge cases, pooling reduction, and flatten-size inference | [Common layers tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html) |
| `skills/brainstate/references/simulation-environment.md` | Scoped and nested run settings, persistent defaults, time and fit semantics, precision and platform controls, isolated `EnvironmentState` workflows, and environment-driven one-step exponential-Euler integration | [Time and Environment](https://brainx.chaobrain.com/brainstate/concepts/time_and_environment.html), [`brainstate.environ` API](https://brainx.chaobrain.com/brainstate/apis/environ.html), [`brainstate.nn.exp_euler_step` API](https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.exp_euler_step.html) |
| `skills/brainstate/references/brainstate/parameter-constraints-regularization.md` | Operational `nn.Param` workflow: constrained forward values, explicit loss integration, common transforms and penalties, prior reset, and `nn.Const` | [parameter model](https://brainx.chaobrain.com/brainstate/concepts/the_parameter_model.html), [parameters tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html), [constraint/regularization how-to](https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html) |
| `skills/brainstate/references/brainstate/randomness-and-reproducibility.md` | Independent streams, mapped randomness, direct key control, exact replay, dropout/noise, and checkpointed RNG State | [Randomness tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html), [random API](https://brainx.chaobrain.com/brainstate/apis/random.html) |
| `skills/brainstate/references/libraries/prebuilt-layer-library.md` | Full layer catalog | [Linear API](https://brainx.chaobrain.com/brainstate/apis/nn/linear.html), [convolution API](https://brainx.chaobrain.com/brainstate/apis/nn/conv.html), [normalization API](https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html), [pooling API](https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html), [padding API](https://brainx.chaobrain.com/brainstate/apis/nn/padding.html), [dropout API](https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html) |
| `skills/brainstate/references/libraries/prebuilt-activation-library.md` | Activation functions and normalization selection | [Activation API](https://brainx.chaobrain.com/brainstate/apis/nn/activation.html) |
| `skills/brainstate/references/brainstate/transformation-jit-expansion.md` | State write-back, cache/static args, compilation boundaries, and benchmarking | [JIT and Compilation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html), [Transformation Essentials](https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html) |
| `skills/brainstate/references/brainstate/transformation-grad-expansion.md` | Autodiff, differentiable simulation, fitting, `return_value`, and `has_aux` | [Autodiff](https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html), [Training and Metrics](https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html), [Parameters tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html) |
| `skills/brainstate/references/brainstate/transformation-vmap-expansion.md` | State axes, ensembles, sweeps, stochastic vmap, `in_states`, and `out_states` | [Vectorization](https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html), [Randomness](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html) |
| `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md` | Transform-safe loops, scans, branches, and checkpointed control flow | [Control Flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html) |
| `skills/brainstate/references/brainstate/brainstate-transformed-diagnostics.md` | Runtime checks, transformed debugging, NaN/Inf checks, callbacks, traced values, and recurring failure diagnosis | [Error Handling and Checks](https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html), [Debugging](https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html) |
| `skills/brainx-acceleration-audit/SKILL.md` | Performance, batching, sweeps, memory, GPU, and multi-device work | The acceleration skill plus the transform sources it conditionally opens |

##### Nested parameter

`skills/brainstate/references/brainstate/parameter-constraints-regularization.md` owns the operational workflow and alone selects this exhaustive API-family child:

| Nested child | Need | Crafting source |
|---|---|---|
| `skills/brainstate/references/brainstate/parameter-transforms-regularizers-catalog.md` | Exact transform signatures plus transform and regularizer selection by domain or modeling intent | [parameter-container API](https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html), [regularization API](https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html), parameters tutorial, constraint how-to |

#### Script References

`skills/brainstate/scripts/lif_neuron_model.py`

Source: [State and PyTrees](https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html)  
Role: State-role combination with explicit `.value` updates.  
Location: Direct skill script.

`skills/brainstate/scripts/modern_cnn.py`

Source: [Activations and normalization](https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html)  
Role: Full Module composition with convolution, normalization, pooling, dropout, and dense layers.  
Location: Direct skill script selected through the prebuilt-layer/activation branch.

`skills/brainstate/scripts/resnet.py`

Source: Not recorded in the current file; source must be established before treating it as canonical.  
Role: Residual Modules and dynamic child registration.  
Location: Direct skill script.



#### Boundaries and Common Failures

- Mutating raw Python attributes inside transformed code.
- State accessed without `.value`.
- Accidental State PyTree restructuring.
- All State types collected as trainable.
- `nn.Param`, `ParamState`, and `nn.Const` conflated.
- Child Modules not registered on attributes.
- Raw `jax.jit`/`grad`/`vmap` applied to State.
- Tiny operations compiled instead of stable whole steps.
- Blueprint references treated as exact API documentation.
- Dynamics or SNN tasks bypassing the dynamics parent.
- Advanced RNG opened before the normal seed-and-`brainstate.random` path.

---

### braincell

#### Purpose

- Boundary: own BrainCell single-compartment HH-style modeling and route morphology-based work through one multicompartment parent.
- Activate for channels, ions, current clamp, solver selection, FI curves, ablation, adaptation, rebound, or point-neuron populations.
- Primary path: choose unit convention → construct `SingleCompartment` → attach ions/channels → initialize → inject current → transform over time → validate.
- Advanced branches: area scaling, mixed-ion adaptation, channel/ion libraries, solver catalog, multicompartment morphology.

#### Essential Concepts

- `SingleCompartment` versus morphology-based `Cell`
- braincell.SingleCompartment collapses the morphology and discretization layers — there is exactly one compartment — and exposes ions and channels added imperatively in __init__.

```text
Direct model declaration
    ↓
Dynamical state and differential equations
    ↓
braincell.quad integration
    ↓
Updated states and spike detection
```

#### Multicompartment Modeling Lifecycle

```text
┌──────────────────────────────────────────────────────────────────┐
│ Declaration (what to model)                                      │
│   • Morphology          geometry: branches, radii, tree           │
│   • mech.*              channels, ions, clamps, synapses          │
│   • filter.*            regions and locsets (where)               │
└──────────────────────────────────────────────────────────────────┘
                                │
                         paint / place
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Discretization (_cv)                                             │
│   • CV                  one isopotential control volume           │
│   • CVPolicy            how many CVs each branch gets             │
└──────────────────────────────────────────────────────────────────┘
                                │
                              build
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Runtime (_compute)                                               │
│   • PointTree           execution graph over CVs                  │
│   • CellRuntimeState    frozen, JAX-friendly state                │
└──────────────────────────────────────────────────────────────────┘
                                │
                              step
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Integration (quad)                                               │
│   • DiffEqModule        defines f(t, y)                           │
│   • solver              advances y by dt                         │
└──────────────────────────────────────────────────────────────────┘
```

- Declaration is pure data. A braincell.mech Channel or Ion knows nothing about JAX, time, or state,
- Integration advances the state in time using a solver from solverlibrary.md
- `size` as independent batch/population dimension inside the super().__init__(size, solver=solver). solver names the integrator
- Must use BrainUnit quantities.
- HHTypedNeuron is the abstract base class
- Ion&channels, Match each channel's `root_type`
- Choose Fixed ions when reversal potential is constant
#### Simulation techniques
- must look at brainpy skill when dealing with network of cells
- must use brainstate.environ.context() to define the simulation environment
- must use brainstate.transform.for_loop(step, times) for timestamped steps
- Brainunit math functions that help with array: u.math.arange u.math.squeeze(), recommend to use
- When to use `MixIons` 
- Solver choice.

#### Canonical Workflow Scripts Included in the Skill

1. Minimal HH Cell
2. Run A Current-Clamp Simulation
3. Using Existing Ions And Channels
4. Vectorized FI Curve Pattern
5. Channel Ablation Pattern

#### Reference Routing

```text
braincell/
├── skills/brainevent/SKILL.md [shared skill]
├── area-scaled-hh-pattern.md
├── mixions-for-adaptation.md [shared]
├── ion-library.md
├── channel-library.md
├── solver-library-with-effects.md
├── array-creation.md
├── braincell-custom-ion-channel-authoring.md
└── multicompartment-cell-workflow.md
    ├── braincell-manual-morphology-construction.md
    ├── morphology-io-loading-validation.md
    ├── topology-building-and-visualization.md
    │   └── common-failures-index.md [shared diagnostic]
    ├── probe-reference.md
    │   └── common-failures-index.md [shared diagnostic]
    ├── filter-function-library.md
    └── cv-policy-reference.md
```

##### First-layer references

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` | morphology route: `Cell`, CVs, `paint`, `place`, clamps, probes, and geometry-dependent simulation | [Cell tutorial](https://brainx.chaobrain.com/braincell/tutorials/cell.html), then the nested sources below |
| `skills/braincell/references/area-scaled-hh-pattern.md` | Density-to-total conversion for capacitance, conductance, current, and cell area | The current skill's density-versus-total P0 rule and the existing extracted area-scaled pattern |
| `skills/braincell/references/mixions-for-adaptation.md` | First-layer adaptation, AHP/KCa, rebound, dynamic calcium, and `MixIons(k, ca)` composition | [Adaptation](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html), [T-current rebound](https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html), and [thalamic neurons](https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html) |
| `skills/braincell/references/libraries/ion-library.md` | Built-in ions, fixed/InitNernst/dynamic choices, concentration dynamics, and `MixIons` | [Ions and channels concept](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [ion tutorial](https://brainx.chaobrain.com/braincell/tutorials/ion.html), [ion API](https://brainx.chaobrain.com/braincell/apis/braincell.ion.html) |
| `skills/braincell/references/libraries/channel-library.md` | Built-in channel families, dependencies, selection, and the built-in-versus-custom boundary | [Ions and channels concept](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [channel tutorial](https://brainx.chaobrain.com/braincell/tutorials/channel.html), [channel API](https://brainx.chaobrain.com/braincell/apis/braincell.channel.html), [channel ablation](https://brainx.chaobrain.com/braincell/examples/channel_ablation.html), [adaptation example](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html) |
| `skills/braincell/references/libraries/solver-library-with-effects.md` | Integrator names, cable/composite solvers, speed/accuracy guidance, and numerical effects | [Integration concept](https://brainx.chaobrain.com/braincell/concepts/integration.html), [integration API](https://brainx.chaobrain.com/braincell/apis/integration.html), [solver guide](https://brainx.chaobrain.com/braincell/integration/solvers.html), [advanced integration](https://brainx.chaobrain.com/braincell/integration/advanced.html), [integration-methods example](https://brainx.chaobrain.com/braincell/examples/integration_methods.html) |
| `skills/braincell/references/braincell/braincell-custom-ion-channel-authoring.md` | Custom channel/ion extension after built-ins are exhausted | [Ions and channels concept](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [channel tutorial](https://brainx.chaobrain.com/braincell/tutorials/channel.html), [extending BrainCell](https://brainx.chaobrain.com/braincell/developer/extending.html) |

##### Nested reference tree under the multicompartment parent

| Exclusive nested child | Need | Crafting source |
|---|---|---|
| `skills/braincell/references/braincell/morphology-io-loading-validation.md` | SWC, ASC, NeuroML2, NeuroMorpho, validation, checkpoints, and post-load checks | [IO overview](https://brainx.chaobrain.com/braincell/file_formats/overview.html), [SWC](https://brainx.chaobrain.com/braincell/file_formats/swc.html), [ASC](https://brainx.chaobrain.com/braincell/file_formats/asc.html), [NeuroML2](https://brainx.chaobrain.com/braincell/file_formats/neuroml2.html), [NeuroMorpho](https://brainx.chaobrain.com/braincell/file_formats/neuromorpho.html), [checkpointing](https://brainx.chaobrain.com/braincell/file_formats/checkpointing.html), [morphology concept](https://brainx.chaobrain.com/braincell/concepts/morphology.html) |
| `skills/braincell/references/braincell/topology-building-and-visualization.md` | NodeTree, CV/branch/node views, placement verification, and visualization | [Visualization tutorial](https://brainx.chaobrain.com/braincell/tutorials/vis.html), [filter tutorial](https://brainx.chaobrain.com/braincell/tutorials/filter.html) |
| `skills/braincell/references/braincell/probe-reference.md` | State, mechanism, current, and trace probes plus missing-trace checks | [Mechanisms tutorial](https://brainx.chaobrain.com/braincell/tutorials/mech.html) |
| `skills/braincell/references/libraries/filter-function-library.md` | Region and locset selection for mechanisms, probes, and clamps | [Filter tutorial](https://brainx.chaobrain.com/braincell/tutorials/filter.html), [filter API](https://brainx.chaobrain.com/braincell/apis/filter.html), and the existing selector blueprint |
| `skills/braincell/references/libraries/cv-policy-reference.md` | CV policy selection, discretization effects, resolution, and cost | [Discretization concept](https://brainx.chaobrain.com/braincell/concepts/discretization.html), [Cell tutorial](https://brainx.chaobrain.com/braincell/tutorials/cell.html), and the existing CV blueprint |
| `skills/braincell/references/braincell/braincell-manual-morphology-construction.md` | Manual topology creation before `Cell` construction | Morphology concept, Cell tutorial, and existing blueprint |

`skills/braincell/references/diagnostics/common-failures-index.md` remains a second-level diagnostic child, selected only after the manual-morphology, topology, or probe child identifies a failure mode. It is not an eighth first-level child.

The multicompartment parent may reuse first-layer ion, channel, solver, and MixIons references; reuse does not make those files exclusive children.

#### Script References

- `references/scripts/hh_neuron_basics.py`; [source](https://brainx.chaobrain.com/braincell/examples/hh_neuron_basics.html); canonical point-neuron current clamp; skill-body support.
- `references/scripts/fi_curve.py`; [source](https://brainx.chaobrain.com/braincell/examples/fi_curve.html); vectorized current sweep; direct reference.
- `references/scripts/channel_ablation.py`; [source](https://brainx.chaobrain.com/braincell/examples/channel_ablation.html); current suppression comparison; channel-library branch.
- `references/scripts/calcium_channel_gating.py`; [source](https://brainx.chaobrain.com/braincell/examples/calcium_channel_gating.html); gating diagnostic; channel-library branch.
- `references/scripts/spike_frequency_adaptation.py`; [source](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html); dynamic calcium/KCa/AHP; `mixions-for-adaptation.md`.
- `references/scripts/t_current_rebound.py`; [source](https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html); post-inhibitory rebound; `mixions-for-adaptation.md`.
- `references/scripts/thalamic_neurons.py`; [source](https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html); advanced phenotype comparison; `mixions-for-adaptation.md`.
- `references/multicompartment/references/cell_multicompartment_reference.py`; [source](https://brainx.chaobrain.com/braincell/tutorials/cell.html); morphology-to-simulation workflow; multicompartment parent.

#### Boundaries and Common Failures

- `size=N` interpreted as compartments instead of independent cells.
- Density and total quantities mixed.
- Bare physical values passed to BrainCell.
- Channel installed on the wrong ion/root.
- Leak placed inside an ion container.
- `MixIons` order inconsistent with `root_type`.
- Custom channel authored before checking built-ins.
- Morphology leaf opened before the multicompartment parent.
- `paint` and `place` semantics reversed.
- CV policy, locset, probe key, or topology selected blindly.
- Network construction incorrectly kept in BrainCell instead of BrainPy.

---

### brainevent

#### Purpose

- Boundary: represent binary events and route them through dense, explicit sparse, generated, or fixed-degree connectivity.
- Activate for `BinaryArray`, CSR/CSC/COO decisions, JIT connectivity, fixed fan-in/out, event plasticity, or custom event operators.
- Primary path: classify event data → choose connectivity representation → construct → multiply → transform → validate shape/orientation.
- Advanced branches: sparse formats, connectivity variants, plasticity, custom operators.

#### Essential Concepts
- BrainEvent provides data structures and algorithms for event-driven computation on CPUs, GPUs, and TPUs. By processing only the active (non-zero) spikes in a network, it models brain dynamics far more efficiently than dense matrix operations — while integrating seamlessly with JAX’s autodiff, JIT, and vmap
- The brain computes with spikes — sparse, binary events. Wrap a spike vector in BinaryArray, and any matrix multiplication against it skips the zeros and processes only the neurons that fired.
#### dense arrays and any of brainevent’s sparse connectivity structures that A BinaryArray multiplies against
- `BinaryArray` and `spikes @ connectivity`.
- `coo2csr()`.
- `JITCScalarR` and seed stability.
- `FixedPostNumConn` versus `FixedPreNumConn`.
- Event-driven plasticity see synaptic-plasticity-modeling.md
- Custom CPU/GPU operator boundary.
#### Connectivity Decision Table we will include

| Format | Use when | Avoid when |
|---|---|---|
| Dense JAX/NumPy array | The matrix is small or genuinely dense (roughly more than 25% nonzero), or you need arbitrary per-entry weights with the simplest possible code. | The matrix is large and sparse, because storing and computing zeros wastes memory and compute. |
| CSR / CSC | You have an explicit, fixed sparse matrix and want fast row-oriented (CSR) or column-oriented (CSC) event-driven products. | Connectivity is generated randomly and the full matrix would not fit in memory. |
| JITC (`JITCScalarR`, `JITCNormalR`, `JITCUniformR`, …) | Connectivity is random with a fixed probability and should be regenerated on demand from a seed instead of materialized. | You need to inspect, mutate, or learn individual weights. |
| Fixed fan-in/out (`FixedPreNumConn`, `FixedPostNumConn`) | Each neuron has a fixed number of connections and you want that structure encoded directly. | Connection counts vary per neuron, or you need an explicit weight matrix. |

#### Canonical Workflow Scripts Included in the Skill

1. `BinaryArray`
2. Dense Connectivity
3. Explicit Sparse Connectivity: `CSR`
4. Generated Random Connectivity: `JITCScalarR`
5. Fixed Fan-Out Connectivity: `FixedPostNumConn`
6. JAX Transform Pattern

#### Reference Routing

```text
brainevent/
├── sparse-formats.md
├── JIT-connectivity-variants.md
├── Fixed-Connection-extension.md
├── synaptic-plasticity-modeling.md
├── custom-operators.md
└── scripts/
    ├── 102_EI_net_1996.py
    └── 204_joglekar_2018_propagation.py
```

All four required Markdown references are skill-local and already exist. Application-script selection and provenance live directly in the skill body.

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainevent/references/sparse-formats.md` | COO construction, CSR/CSC storage, conversion, and selection | [Sparse matrices tutorial](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/02_sparse_matrices.html), [sparse-data API](https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html), [utilities API](https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html) |
| `skills/brainevent/references/connectivity-variants.md` | JITC distributions/orientations, fixed fan-in/out, and format choice | [JIT connectivity](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/03_jit_connectivity.html), [fixed connections](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/04_fixed_connections.html), [format guide](https://brainx.chaobrain.com/brainevent/how-to/data-structures/choosing-a-sparse-format.html), [sparse-data API](https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html), [utilities API](https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html) |
| `skills/brainevent/references/synaptic-plasticity.md` | Pre/post event updates, CSR/dense routing, and STDP overlay | [Plasticity tutorial](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/05_synaptic_plasticity.html), [plasticity how-to](https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html), [operations API](https://brainx.chaobrain.com/brainevent/reference/apis/operations.html) |
| `skills/brainevent/references/custom-operators.md` | Route to Numba, Numba CUDA, Warp, C++, or CUDA extension paths | [Index](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/index.html), [Numba CPU](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html), [Numba CUDA](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/02_numba_cuda.html), [Warp](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/03_warp.html), [C++](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/04_cpp.html), [CUDA](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/05_cuda.html) |

#### Script References

- `references/scripts/102_EI_net_1996.py`; [source](https://raw.githubusercontent.com/chaobrain/brainpy.state/main/examples/brainpy_like/102_EI_net_1996.py); high-level E/I network; direct full-script reference.
- `references/scripts/204_joglekar_2018_propagation.py`; [source](https://raw.githubusercontent.com/chaobrain/brainpy.state/main/examples/brainpy_like/204_joglekar_2018_propagation.py); delayed spikes, JIT connectivity, and area mapping; connectivity-variants branch.

#### Boundaries and Common Failures

- Continuous analog values wrapped as binary events.
- Large random matrices materialized unnecessarily.
- JIT connectivity used where per-edge weights must be learned.
- COO mistaken for a finished matrix class.
- Fixed fan-in and fan-out reversed.
- Orientation inferred from class suffix without checking contraction.
- BrainEvent treated as a complete simulator.
- BrainMass coupling routed here without an explicit binary-event boundary.

---

### brainmass

#### Purpose

- Boundary: neural-mass models, simulation, stochastic ensembles, coupled networks, forward models, and parameter fitting.
- Activate for BrainMass model selection, `Simulator`, `Network`, BOLD/EEG/MEG, fitting, whole-brain workflows, or regime sweeps.
- Primary path: discover model → construct `*Step` → configure State/noise/units → simulate → observe → validate or fit.
- Advanced branches: model catalog, noise, coupling/delays, observations, fitting backends, datasets, analysis, task training, sweeps.

#### Essential Concepts

- BrainMass implements neural mass models with BrainState for differentiable, JAX-based whole-brain modeling.
- A `*Step` model defines regional dynamics; use `list_models()` for discovery and model selection.
- `Simulator` runs a model, `Network` couples regional nodes, forward models map hidden activity to BOLD/EEG/MEG signals, and `Fitter` tunes trainable parameters to data.
- Attach stochastic noise to the model and seed every reported stochastic run.
- Configure a `Network` with connectivity, distance, transmission speed, coupling, and delays only after establishing the simulation environment and global `dt`.
- Prefer gradient-based fitting when the workflow is differentiable; route to gradient-free fitting only when the objective or model requires it.
- Preserve BrainUnit quantities throughout simulation, coupling, observation, and fitting boundaries.
- Route reusable initialization, encoding, metrics, optimization, surrogate-gradient, and cognitive-task details to the matching Braintools references.

#### Simulation and Training Rules

- Use BrainState random APIs such as `rand`, `randn`, and `randint` for stochastic initialization, with explicit seed control when results are reported.
- Use `braintools.init` for reusable state and parameter initialization policies.
- Use Braintools encoders only when experimental or task inputs must be converted into spikes.
- Define simulation context with `brainstate.environ.context()`.
- Execute timestamped steps with `brainstate.transform.for_loop(step, times)`; use transform-safe `for_loop`/`scan` rollout and checkpointing for long runs.

#### Canonical Workflow Scripts Included in the Skill

1. Canonical Setup
2. Model Discovery
3. One Model Simulation
4. Noise + Random Seed Basics
5. Batching / Transform Basics
6. Small Network
7. Forward Models
8. Fitting With Gradients

#### Reference Routing

```text
brainmass/
├── modellibrary.md
├── noiseprocesses.md
├── coupling-network-api.md
├── forward-observation-api.md
├── fitting-with-objectives-api.md
│   ├── braintools/initializers.md [shared]
│   ├── braintools/metrics.md [shared]
│   ├── braintools/optimizers.md [shared]
│   └── braintools/surrogate-gradients.md [shared]
├── datasets-api.md
├── visualization-analysis-api.md
├── batch-transform-acceleration.md
├── horn-task-training.md
│   ├── braintools/cognitive-tasks.md [shared]
│   ├── braintools/encoders.md [shared]
│   ├── braintools/metrics.md [shared]
│   └── braintools/optimizers.md [shared]
└── parameter-sweeps-and-regime-analysis.md
```

The skill defines ten package references and routes to six shared Braintools references defined at the end of this plan.

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainmass/references/modellibrary.md` | Model inventory, categories, state variables, use cases, `list_models()`, and `ModelInfo` | [Models API](https://brainx.chaobrain.com/brainmass/reference/models.html), [utilities API](https://brainx.chaobrain.com/brainmass/reference/utilities.html) |
| `skills/brainmass/references/noiseprocesses.md` | Noise-family inventory, seeding, stochastic runs, and batched ensembles | [Noise API](https://brainx.chaobrain.com/brainmass/reference/noise.html), [noise tutorial](https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html) |
| `skills/brainmass/references/coupling-network-api.md` | Coupling mechanisms, delays, and network variants | [Coupling API](https://brainx.chaobrain.com/brainmass/reference/coupling.html), [network tutorial](https://brainx.chaobrain.com/brainmass/tutorials/04_building_a_network.html), [coupling/delays concept](https://brainx.chaobrain.com/brainmass/concepts/coupling_and_delays.html) |
| `skills/brainmass/references/forward-observation-api.md` | HRFBold, kernels, TemporalAverage, BOLDSignal, EEG/MEG, and lead fields | [Forward API](https://brainx.chaobrain.com/brainmass/reference/forward.html), [observation API](https://brainx.chaobrain.com/brainmass/reference/observation.html), [forward-model tutorial](https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html) |
| `skills/brainmass/references/fitting-with-objectives-api.md` | Simulator, Network, Fitter/FitResult, objective functions, and backend boundaries | [Orchestration API](https://brainx.chaobrain.com/brainmass/reference/orchestration.html), [gradient fitting](https://brainx.chaobrain.com/brainmass/tutorials/06_fitting_with_gradients.html), [gradient-free fitting](https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html), [custom objective](https://brainx.chaobrain.com/brainmass/howto/custom_objective.html) |
| `skills/brainmass/references/datasets-api.md` | Dataset registration/loading, Connectome, Signal, and task containers | [Datasets API](https://brainx.chaobrain.com/brainmass/reference/datasets.html) |
| `skills/brainmass/references/visualization-analysis-api.md` | Plotting, FC/FCD, and spectral analysis | [Visualization API](https://brainx.chaobrain.com/brainmass/reference/viz.html), [analysis how-to](https://brainx.chaobrain.com/brainmass/howto/analyze_results.html) |
| `skills/brainmass/references/batch-transform-acceleration.md` | JIT, transformed loops, `scan`, `vmap`, checkpointing, batched initial conditions, and sweeps | [BrainMass batch and accelerate](https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html), with [BrainTrace batching](https://brainx.chaobrain.com/braintrace/tutorials/batching.html) only for the vmap-per-sample comparison already used by the skill |
| `skills/brainmass/references/horn-task-training.md` | HORN components, task datasets, direct optimizer loops, and held-out metrics | [HORN API](https://brainx.chaobrain.com/brainmass/reference/horn.html), [task-training tutorial](https://brainx.chaobrain.com/brainmass/tutorials/08_training_on_tasks.html), [HORN case study](https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html) |
| `skills/brainmass/references/parameter-sweeps-and-regime-analysis.md` | Regime exploration and sensitivity analysis distinct from fitting | [Parameter-sweeps how-to](https://brainx.chaobrain.com/brainmass/howto/parameter_sweeps.html) |

#### Script References

- `references/scripts/gradient-free-fitting.py`; [source](https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html); Nevergrad and derivative-free SciPy; fitting parent.
- `resting-state-meg-whole-brain-pipeline.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/case_studies/resting_state_meg.html); network → MEG → FC; coupling → forward-observation.
- `eeg-fitting-with-gradients.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/case_studies/eeg_fitting.html); gradient fitting and EEG recovery; fitting parent.
- `seizure-epileptor-case-study.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/case_studies/seizure_epileptor.html); disease dynamics; model library.
- `wong-wang-decision-making.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/case_studies/decision_making.html); stochastic decision trials; model library.
- `horn-cognitive-task-training.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html); task training; HORN branch.
- `hopf-bifurcation-single-node.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/hopf.html); minimal oscillator; model library.
- `wilson-cowan-ei-dynamics.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wilson_cowan.html); E/I population rates; model library.
- `jansen-rit-eeg-proxy.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/jansen_rit.html); EEG proxy; model library → forward-observation.
- `kuramoto-synchronization.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/kuramoto.html); coupled oscillators; model library.
- `wong-wang-dmf-resting-state.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wong_wang_exc_inh.html); dynamic mean field; model library.
- `linear-baseline-node.py`; [source](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/linear.html); analytical sanity check; optional model-library script.

#### Boundaries and Common Failures

- BrainMass treated as a full general-purpose framework.
- Noise passed to `Simulator.run()` instead of the model.
- Delay-coupled `Network` built before global `dt`.
- Batched output axis misunderstood.
- Wrong monitor path for network node State.
- Raw oscillatory traces fitted with unsuitable pointwise loss.
- Gradient-free fitting chosen by default.
- BOLDSignal used when fast differentiable HRF convolution is intended.
- Units dropped at observation boundaries.
- Task training expanded into the canonical body.

---

### brainpy

#### Purpose

- Boundary: native BrainPy-State point-neuron, synapse, projection, plasticity, readout, SNN simulation, and surrogate-gradient training workflows.
- Activate for `brainpy.state` models, projections, E/I networks, delays, readouts, plasticity, SNN training, or NEST-compatible porting.
- Primary path: choose native versus NEST path → select components → initialize → run transformed rollout → monitor/train → validate.
- branches: component catalogs, custom models, training, Braintools, gallery scripts, NEST compatibility.

#### Essential Concepts

- brain simulation and brain-inspired computing are the same computation, expressed once. The neurons, synapses, and projections you assemble to simulate a biophysical network are the exact objects you train with gradients and scale with linear-memory online learning.
- Look at the brainstate skill for knowledge in Modules
- State initialization and BrainUnit quantities.
- Neuron and synapse anatomy.
- Projection roles: `comm`, `syn`, `out`, `post`.
- AlignPre versus AlignPost.
- Projection-before-post update order.
#### Simulation techniques
- Random Sampling rand, randn, randint, is useful for parameter intialization , basic random seed knowledge.
- use the braintool.init apis to initialize states and parameters for reusable initialization policy, use braintool encoders when need to converts experimental/data input into spikes
- must use brainstate.environ.context() to define the simulation environment
- must use brainstate.transform.for_loop(step, times) for timestamped steps, `for_loop`/`scan` rollout and checkpointing.
- Brainunit math functions that help with array: u.math.arange u.math.squeeze(), recommend to use

### Training
- Surrogate gradients and `ParamState` selection.

#### Canonical Workflow Scripts Included in the Skill

1. Classify native BrainPy versus NEST-compatible request.
2. Select neuron, synapse, output, projection, and optional readout.
3. Construct the Module/network with units.
4. Initialize all State.
5. Execute time through BrainState control flow.
6. Add loss/optimizer only for training.
7. Validate projection order, State reset, output shapes, units, spikes, and gradients.

Minimal inline scripts: single-neuron rollout and two-population synapse/projection workflow.

#### Reference Routing

```text
brainpy/
├── skills/brainevent/SKILL.md [shared skill]
├── brainpy-neuron-library.md
├── brainpy-synapse-library.md
├── brainpy-synaptic-outputs.md
├── brainpy-projection-library.md
├── brain-dynamics-delay-protocol.md
├── brain-event-driven-operators.md
├── array-creation.md
├── brainpy-plasticity.md
├── brainpy-custom-models.md
├── brainpy-training.md
│   ├── braintools/encoders.md [shared]
│   ├── braintools/metrics.md [shared]
│   ├── braintools/optimizers.md [shared]
│   └── braintools/surrogate-gradients.md [shared]
├── brainpy-readouts-and-inputs.md
├── NEST-compatible/
│   └── nest-workflow.md
│       ├── model-library.md
│       ├── synapse-and-connectivity.md
│       ├── devices.md
│       ├── network-building.md
│       ├── divergence-and-parity.md
│       ├── integration-categories.md
│       └── scripts/
│           ├── brunel_alpha.py
│           ├── brunel_delta.py
│           ├── brette_et_al_2007.py
│           ├── synapsecollection.py
│           ├── evaluate_tsodyks2_synapse.py
│           ├── clopath_synapse_spike_pairing.py
│           └── spatial_gaussex.py
└── braintools/initializers.md [shared]
```

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainpy/references/brainpy-neuron-library.md` | Neuron catalog and selection | [Neuron API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-neurons.html), [neuron-selection how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-choose-neuron.html) |
| `skills/brainpy/references/brainpy-synapse-library.md` | Synaptic dynamics and receptor filters | [Synapse API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synapses.html) |
| `skills/brainpy/references/brainpy-synaptic-outputs.md` | COBA/CUBA/MgBlock outputs and current-versus-conductance selection | [Synaptic-output API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synouts.html), [COBA/CUBA how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-coba-cuba-synapses.html) |
| `skills/brainpy/references/brainpy-projection-library.md` | Projection APIs, AlignPre/AlignPost, direct-current projections, gap junctions, and delays | [Projection API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-projections.html), [alignment concept](https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html), [delays how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html) |
| `skills/brainpy/references/brainpy-plasticity.md` | STP/STD state and projection integration | [Plasticity API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-plasticity.html), [short-term-plasticity how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-short-term-plasticity.html) |
| `skills/brainpy/references/brainpy-custom-models.md` | Custom Neuron/Synapse anatomy, ODE steps, and paper reproduction | [Paper-reproduction how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-reproduce-a-paper.html), [BrainPy gallery](https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html) |
| `skills/brainpy/references/brainpy-training.md` | Differentiability, surrogate gradients, ParamState, BPTT, and checkpointed rollouts | [Differentiability concept](https://brainx.chaobrain.com/brainpy-state/concepts/differentiability.html), [train-an-SNN tutorial](https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html), [surrogate-gradient how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-surrogate-gradients.html), [checkpointing how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html) |
| `skills/brainpy/references/brainpy-readouts-and-inputs.md` | Readout heads, spike/input generators, Poisson helpers, and encoders | [Readout API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-readouts.html), [readout how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-readouts.html), [input API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-inputs.html) |
| `skills/brainstate/references/brainstate-dynamics/brain-dynamics-delay-protocol.md` | Delay APIs and buffer behavior | [delay tutorial](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html) |
| `skills/brainstate/references/brainstate-dynamics/brain-dynamics-event-driven-operators.md` | Sparse event operators and connectivity | [event-driven tutorial](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/03_event_driven_operators.html) |


##### NEST-compatible nested branch

| Nested lookup area | Need | Crafting sources | Disposition |
|---|---|---|---|
| `Model Library.md` | Select NEST-compatible neurons and inspect neuron-model APIs | [Models](https://brainx.chaobrain.com/brainpy-state/nest-style/models.html), [neuron API](https://brainx.chaobrain.com/brainpy-state/apis/nest-neurons.html) | Keep as a compact area in `nest-workflow.md` |
| `Synapse And Connectivity.md` | Static/special synapses, plasticity, connection rules, synapse specs, and realized connectivity | [synapse API](https://brainx.chaobrain.com/brainpy-state/apis/nest-synapses.html), [plasticity API](https://brainx.chaobrain.com/brainpy-state/apis/nest-plasticity.html), [connectivity](https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html) | Keep as a compact area in `nest-workflow.md` |
| `Devices` | Generators, recorders, detectors, source semantics, direction, and result readback | [devices guide](https://brainx.chaobrain.com/brainpy-state/nest-style/devices.html), [device API](https://brainx.chaobrain.com/brainpy-state/apis/nest-devices.html) | Keep as a compact area in `nest-workflow.md` |
| `Network Building.md` | `Simulator`, `NodeView`, `SimulationResult`, `SynapseCollection`, projection/connection APIs, and spatial primitives | [network tutorial](https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/03-connect-network.html), [network API](https://brainx.chaobrain.com/brainpy-state/apis/nest-network.html), [spatial API](https://brainx.chaobrain.com/brainpy-state/apis/nest-spatial.html), [spatial guide](https://brainx.chaobrain.com/brainpy-state/nest-style/spatial.html) | Keep as a compact area in `nest-workflow.md` |
| `Divergence And Parity.md` | Porting differences, STDP parameter placement, recording/stochastic parity, validation, and NEST mismatches | [divergence index](https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/index.html), [validation status](https://brainx.chaobrain.com/brainpy-state/nest-style/validation-status.html), [STDP divergence](https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/stdp.html) | Keep as a compact area in `nest-workflow.md` |
| `Integration Categories.md` | Numerical and integration behavior by NEST-compatible model family | [integration categories](https://brainx.chaobrain.com/brainpy-state/nest-style/integration-categories.html) | Keep as a compact area in `nest-workflow.md` |

#### Script References

Native scripts:

- `103_COBA_2005.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/103_COBA_2005.py) — canonical E/I COBA network; projection branch.
- `106_COBA_HH_2007.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/106_COBA_HH_2007.py) — custom HH network; custom-model branch.
- `107_gamma_oscillation_1996.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/107_gamma_oscillation_1996.py) — custom neuron/synapse; custom-model branch.
- `109_fast_global_oscillation.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/109_fast_global_oscillation.py) — `DeltaProj` and delay; projection branch.
- `201_surrogate_grad_lif_fashion_mnist.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/201_surrogate_grad_lif_fashion_mnist.py) — real-data SNN training; training branch.
- `references/brainstate-dynamics/scripts/training-snn.py` — [source](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html) — compact SNN training; training branch.

NEST-compatible external scripts:

- `brunel_alpha.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_alpha.py) — alpha-synapse Brunel network.
- `brunel_delta.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_delta.py) — delta-synapse voltage-weight semantics.
- `brette_et_al_2007.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brette_et_al_2007.py) — comparative network workflow.
- `synapsecollection.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/synapsecollection.py) — synapse inspection/manipulation.
- `evaluate_tsodyks2_synapse.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/evaluate_tsodyks2_synapse.py) — short-term plasticity parity.
- `clopath_synapse_spike_pairing.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/clopath_synapse_spike_pairing.py) — plasticity protocol.
- `spatial_gaussex.py` — [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_gaussex.py) — spatial connectivity.

Location for all NEST scripts: `NEST-compatible/nest-workflow.md` full-script branch.

#### Boundaries and Common Failures

- Native and NEST APIs mixed.
- State not initialized or reset.
- Time advanced with a Python loop.
- Projection applied after postsynaptic update.
- `comm`, `syn`, `out`, and `post` treated as one role.
- AlignPost used for unsuitable nonlinear synapse dynamics.
- Unitless membrane or synaptic values.
- All State differentiated.
- BrainEvent treated as a separate simulator.
- Online-learning APIs expected from BrainPy instead of BrainTrace.
- Placeholder `scan` block left without an official script source.

---

### braintrace

#### Purpose

- Boundary: online learning with eligibility-trace propagation, BrainTrace layers/primitives, compiler graphs, and online-learning batching.
- Activate for D-RTRL, ES-D-RTRL, pp-prop, eligibility traces, `braintrace.compile`, hidden groups, ETP primitives, or excluded-weight debugging.
- Primary path: define recurrent model → compile once → inspect graph → run online learner → differentiate → validate traced weights.
- Advanced branches: primitives, algorithms, compiler diagnostics, hidden-state/batching modes.

#### Essential Concepts

- Online learning and eligibility traces.
- Use built-in`braintrace.nn` first.
- `braintrace.compile`.
- Decision Table: Algorithmn choice between `D_RTRL`, `ES_D_RTRL`, `pp_prop`.
- Hidden State, `HiddenGroupState`, `HiddenTreeState`.
- Inspecting `ETraceGraph` by `learner.report` and `show_graph()`.
- Vmap compilation and per-sample State.
- Compile-once shape stability.
- Supported-control-flow limitations.

#### Canonical Workflow Scripts Included in the Skill

1. Classify RNN/SNN and memory/accuracy requirements.
2. Prefer a `braintrace.nn` model.
3. Define hidden State.
4. Compile once with representative input and algorithm.
5. Inspect report and graph.
6. Run the learner through transformed time execution.
7. Differentiate selected `ParamState`.
8. Validate traced/excluded weights, batch State, gradients, and reuse.

Minimal inline script: GRU → `braintrace.compile(..., D_RTRL, ...)` → State-targeted gradient.

#### Reference Routing

```text
braintrace/
├── primitive-ops-and-transforms.md
├── algorithms-customization.md
├── compiler-graph-debugging.md
├── state-batching-workflows.md
├── braintools/metrics.md [shared]
└── braintools/optimizers.md [shared]
```

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/braintrace/references/primitive-ops-and-transforms.md` | ETP primitives, matmul/conv/sparse/LoRA/element-wise ops, transform hooks, and custom registration | [Concepts](https://brainx.chaobrain.com/braintrace/quickstart/concepts.html), [compiler internals](https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html), [ETP primitives](https://brainx.chaobrain.com/braintrace/tutorials/etp_primitives.html), [custom transforms](https://brainx.chaobrain.com/braintrace/tutorials/customizing_primitive_transforms.html), [primitives API](https://brainx.chaobrain.com/braintrace/apis/primitives.html) |
| `skills/braintrace/references/algorithms-and-customization.md` | Algorithm-by-algorithm selection and custom algorithm extension | [Algorithms API](https://brainx.chaobrain.com/braintrace/apis/algorithms.html), [custom algorithms](https://brainx.chaobrain.com/braintrace/advanced/custom_algorithms.html) |
| `skills/braintrace/references/compiler-graph-debugging.md` | `ETraceGraph`, hidden groups, relations, diagnostics, exclusions, limitations, and workarounds | [Compiler internals](https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html), [limitations](https://brainx.chaobrain.com/braintrace/advanced/limitations.html), [graph visualization](https://brainx.chaobrain.com/braintrace/tutorials/graph_visualization.html) |
| `skills/braintrace/references/state-batching-workflows.md` | Hidden-state variants, initialization/reset, single-sample mode, vmap batching, and multi-step input | [Hidden states](https://brainx.chaobrain.com/braintrace/tutorials/hidden_states.html), [batching](https://brainx.chaobrain.com/braintrace/tutorials/batching.html) |

#### Script References

Core workflows:

- `rnn-online-learning.py` — [source](https://brainx.chaobrain.com/braintrace/quickstart/rnn_online_learning.html) — GRU copying-memory workflow; skill-body support.
- `snn-online-learning.py` — [source](https://brainx.chaobrain.com/braintrace/quickstart/snn_online_learning.html) — recurrent SNN with ES-D-RTRL; algorithms branch.

Default bundle:

- `examples/drtrl/09-classification-mnist.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/09-classification-mnist.py) — D-RTRL classification; algorithms.
- `examples/pp_prop/12-classification-neuromorphic.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/12-classification-neuromorphic.py) — pp-prop SNN; algorithms.
- `examples/drtrl/02-batching-vmap.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/02-batching-vmap.py) — per-sample State batching; state batching.
- `examples/pp_prop/06-batching-batched.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/06-batching-batched.py) — directly batched primitive; state batching.
- `examples/pp_prop/14-knob-vjp-method-contrast.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/14-knob-vjp-method-contrast.py) — temporal-credit contrast; algorithms.
- `examples/drtrl/11-knob-fast-solve.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/11-knob-fast-solve.py) — speed/equivalence knob; algorithms.

Operator branches:

- `examples/drtrl/07-operator-lora.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/07-operator-lora.py) — LoRA primitive; primitive reference.
- `examples/pp_prop/09-operator-sparse.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/09-operator-sparse.py) — masked/sparse connectivity; primitive reference.
- `examples/pp_prop/11-operator-conv.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/11-operator-conv.py) — convolutional ETP; primitive reference.

Optional specialized scripts:

- `examples/003-snn-memory-and-speed-evaluation-all.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/003-snn-memory-and-speed-evaluation-all.py) — heavy benchmark; advanced algorithms/performance.
- `examples/pp_prop/04-neurons-coba-ei-rsnn.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/04-neurons-coba-ei-rsnn.py) — Dale-law E/I RSNN; advanced algorithms.
- `examples/pp_prop/_shared.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/_shared.py) — support dependency only.
- `examples/drtrl/_shared.py` — [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/_shared.py) — support dependency only.

#### Boundaries and Common Failures

- Temporal recurrent weight uses ordinary `x @ w`.
- Compiler invoked inside the training loop.
- Hidden State or batch semantics designed manually before trying `compile(..., vmap=True)`.
- Excluded readout weights assumed to be a compiler failure.
- Custom model trusted without graph inspection.
- Unsupported `lax` control flow or nested mapping inside `update()`.
- Raw parameter transformation detaches eligibility traces.
- BrainMass-derived rollout text copied without BrainTrace-specific validation.
- Placeholder `scan` example retained without an official source.

---

### brainx-acceleration-audit

#### Purpose

- Boundary: diagnose and plan/refactor performance improvements in BrainX/BrainState simulations.
- Activate for speed, GPU use, vectorization, batching, sweeps, memory, many trials, compile/runtime separation, throughput, or multi-device scaling.
- Primary path: hot-path inventory → pattern classification → prioritization → one-axis rewrite → semantic validation → warm benchmark.
- Advanced branches: exact control flow, JIT, vmap, grad, RNG, checkpointing, multi-device work.

#### Essential Concepts

- Optimization priority: correctness → State/RNG safety → shape stability → warm runtime → memory → device scale.
- Population arrays and transformed time loops.
- Batched and mapped State.
- Parameter gradients versus per-sample gradients.
- Stable JIT boundary.
- Single-device cleanup before multi-device work.

#### Canonical Workflow Scripts Included in the Skill

1. Inventory shapes, State, loops, transforms, RNG, host interaction, and connectivity.
2. Classify inefficiencies using the skill's pattern vocabulary.
3. Rank impact, semantic risk, and confidence.
4. Build a tiny deterministic baseline.
5. Rewrite one axis at a time.
6. Compare outputs, final State, RNG, gradients, shapes, warm runtime, and memory.
7. Report findings, proposed patch, validation, and residual risk.

Canonical compositions: `jit(scan(step))`, `jit(vmap(run_trial))`, batched-State time scans, mapped ensembles, compiled `grad(loss_over_batch)`, and checkpointed long sequences.

#### Reference Routing

```text
brainx-acceleration-audit/
├── brainstate/
│   ├── brainstate-control-flow-patterns.md
│   ├── transformation-jit-expansion.md
│   ├── transformation-vmap-expansion.md
│   └── transformation-grad-expansion.md
└── brainstate-randomness-reproducibility/
    └── randomness-and-reproducibility.md
        └── advanced-randomness.md
```

- `skills/brainstate/SKILL.md` is a required upstream route before nontrivial State-aware rewrites.

This skill owns duplicated local transform and randomness references for the exact semantics it audits.

##### First-layer routes

| Route | Open when | Crafting source |
|---|---|---|
| `skills/brainstate/SKILL.md` | Any nontrivial state-aware rewrite | Owning BrainState skill |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-jit-expansion.md` | JIT boundaries, static args, recompilation, or benchmarking | JIT and Transformation Essentials sources |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-vmap-expansion.md` | Batch, trial, ensemble, State-axis, or RNG mapping | Vectorization and Randomness sources |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-grad-expansion.md` | Finite-difference replacement, training gradients, or ParamState differentiation | Autodiff, Training, and Parameters sources |
| `skills/brainx-acceleration-audit/references/brainstate/brainstate-control-flow-patterns.md` | Time/recurrent loops, scan/for-loop/while-loop, or checkpointing | Control Flow source |
| `skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` | Seed/key restoration or independent mapped randomness | Randomness corpus |

##### Nested randomness

The acceleration skill and its transform references route only to the local randomness parent. That parent alone selects:

| Nested child | Open when | Crafting source |
|---|---|---|
| `skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/advanced-randomness.md` | Advanced stream, mapped-key, or restoration behavior | Same randomness corpus |

---

### brainx-general-guard

This skill owns local copies of the randomness references it conditionally opens and keeps installation as a semantic skill route.

#### Reference Routing

| Route | Open when | Crafting source |
|---|---|---|
| `skills/brainx-install/SKILL.md` | Setup, import, backend, device, version, or package mismatch | Owning installation skill |
| `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` | Stochastic behavior, seed control, random trials, dropout/noise, or reproducibility | Randomness tutorial and API |

The guard's source anchors are [Thinking in BrainState](https://brainx.chaobrain.com/brainstate/getting_started/thinking_in_brainstate.html), [BrainCell units](https://brainx.chaobrain.com/braincell/concepts/units.html), [BrainCell mechanisms](https://brainx.chaobrain.com/braincell/concepts/mechanisms.html), and [BrainState randomness](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html).

---

### brainx-install

#### Purpose

- Boundary: inspect, diagnose, plan, modify with approval, and verify a compatible BrainX environment.
- Activate for installation, upgrade, downgrade, repair, version mismatch, Python compatibility, CPU/CUDA/TPU selection, or import failures.
- Primary path: inspect → collect versions → match release tuple → assess Python/hardware → propose exact changes → obtain approval → modify → verify.

#### Essential Concepts

- BrainX release tuple.
- Subpackage versions need not numerically match.
- Meta-package extras: CPU, CUDA 12/13, TPU.
- Existing environment-manager preservation.
- Mandatory human approval for installation.
- Python and JAX version match

#### Canonical Workflow Scripts Included in the Skill

1. Inspect project manifests, lockfiles, interpreter, and manager.
2. Collect installed BrainX/JAX versions.
3. Match an official BrainX release tuple.
4. Check Python compatibility.
5. Inspect intended backend and devices.
6. Classify the environment.
7. Present exact commands and package changes.
8. Obtain explicit approval.
9. Apply the minimal approved change.
10. Verify tuple, imports, dependencies, hardware, and project behavior.

Minimal inline scripts: read-only shell/Python inspection commands and post-install verification commands.

#### Reference Routing

```text
brainx-install/
├── compatibility-and-release-matching.md
```

- Compatibility matching is the first decision after inspection.
- Hardware selection should not open unless CPU/GPU/TPU choice is involved.
- Repair opens only after diagnosis and before an approved mutation.
- The current `SKILL.md` frontmatter is malformed: its closing delimiter is a long hyphen line instead of `---`.

Create a skill-local `skills/brainx-install/references/` directory.

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainx-install/references/compatibility-and-release-matching.md` | Release tuples, exact/partial matching, release drift, historical/yanked releases, and compatibility evidence | [BrainX summary](https://brainx.chaobrain.com/summ/) plus the skill's compatibility-classification sections |

No nested Markdown layer is declared.

#### Script References

- No standalone scripts are bundled.
- Inline inspection/verification commands remain appropriate because environment managers differ.
- Release source: [BrainX releases](https://brainx.chaobrain.com/summ/).
- Backend source: [BrainX installation](https://brainx.chaobrain.com/summ/install.html).

#### Boundaries and Common Failures

- Latest subpackage versions installed independently.
- Environment manager guessed.
- CPU/CUDA/TPU silently selected.
- Mutation performed before approval.
- Working historical release upgraded unnecessarily.
- Import success treated as tuple compatibility.
- GPU visibility treated as consent.
- Lockfiles or manifests modified without approval.
- Broad eager dependency upgrade used.
- Malformed frontmatter prevents reliable skill discovery.

---

## Benchmark Documents

- [BrainCell benchmark](benchmark.md)
- [BrainPy benchmark](benchmarkbrainpy.md)
- [BrainMass benchmark](benchmark-brainmass.md)

## Braintools References

Use one shared reference file per Braintools capability. Consumer skills route to these files instead of maintaining package-local duplicates. `skills/braintools/` is a shared reference directory, not a standalone skill.

| Canonical reference | Consumers | Need | Crafting sources |
|---|---|---|---|
| `skills/braintools/references/cognitive-tasks.md` | BrainMass | Build and generate cognitive-task trials for task-training workflows | [Cognitive-task API](https://brainx.chaobrain.com/braintools/apis/cogtask.html) |
| `skills/braintools/references/encoders.md` | BrainMass, BrainPy | Convert experimental or task inputs with latency, rate, Poisson, population, Bernoulli, delta, step-current, spike-count, temporal, or rank-order encoders and related spike operations | [Encoder API](https://brainx.chaobrain.com/braintools/apis/braintools.html) |
| `skills/braintools/references/initializers.md` | BrainMass, BrainPy | Select and compose reusable state, parameter, weight, and distance-modulated connectivity initializers | [Initializer API](https://brainx.chaobrain.com/braintools/apis/init.html) |
| `skills/braintools/references/metrics.md` | BrainMass, BrainPy, BrainTrace | Select losses and evaluation metrics for fitting, task training, simulation analysis, and online learning, including classification, regression, ranking, spike-train, synchronization, LFP, and connectivity metrics | [Metric API](https://brainx.chaobrain.com/braintools/apis/metric.html) |
| `skills/braintools/references/optimizers.md` | BrainState, BrainMass, BrainPy, BrainTrace | Select optimizers, learning-rate schedules, Optax bridges, and SciPy or Nevergrad wrappers for training, fitting, and online parameter updates | [Optimization API](https://brainx.chaobrain.com/braintools/apis/optim.html), [optimization tutorials](https://brainx.chaobrain.com/braintools/optim/index.html) |
| `skills/braintools/references/surrogate-gradients.md` | BrainMass, BrainPy | Select functional or object-style surrogate gradients and tune their shape parameters for differentiable workflows containing non-differentiable spike functions | [Surrogate-gradient API](https://brainx.chaobrain.com/braintools/apis/surrogate.html) |
