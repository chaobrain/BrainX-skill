---
name: brainstate
description: Use for BrainState mutable State and `.value`, ParamState and other State roles, Module graphs, environment-scoped simulations, state initialization, operational randomness, size-aware neural-network composition, state-aware jit/grad/vmap, or a BrainState training step.
---

# Brainstate

## Purpose and boundary

Use this skill for BrainState's general stateful workflow: mutable `State`, registered `brainstate.nn.Module` graphs, environment-scoped simulation, state initialization and collection, randomness, state-aware transforms, and simulation or training.

Canonical path:

`classify State roles -> construct Modules -> register State and children -> initialize -> set the run environment -> transform the whole operation -> validate State and outputs`

Open the routed reference for specialized variants; route specialized neuronal or network dynamics to the matching BrainCell or BrainPy-State skill.

## Underlying principle of BrainState

BrainState reconciles mutable models with JAX by making mutation explicit: store every value that changes during transformed execution in `State`. Its State roles match distinct model lifecycles; for example, `HiddenState` holds voltages, firing rates, and other dynamics updated each simulation step.

Wrap the complete stateful operation in `brainstate.transform`. Unlike raw JAX transforms, its JAX-like `jit`, `grad`, and `vmap` discover State reads and writes, differentiate selected State collections, and map or share State during vectorization.

### API structure

Choose the namespace that owns the operation:

| API | Use |
|---|---|
| `brainstate` | Store values that mutate during transformed execution in `State`, and mark their lifecycle with semantic State subclasses. |
| `brainstate.graph` | Inspect, split, merge, or reconstruct Module and State graphs while preserving shared references and cycles. |
| `brainstate.nn` | Build registered `Module` graphs from parameters, layers, dynamics, delays, and metrics. |
| `brainstate.transform` | Apply State-aware JIT, differentiation, vectorization, parallelization, or control flow to a complete stateful operation. |
| `brainstate.interop` | Convert supported standard-layer models between `brainstate.nn`, Flax NNX, Flax Linen, and Equinox. |
| `brainstate.random` | Seed and sample reproducibly through stateful JAX keys with automatic splitting. |
| `brainstate.util` | Organize supporting mappings, filters, representations, dataclasses, and caches that do not require model-graph identity; use `graph` for Module or State structure. |
| `brainstate.typing` | Annotate arrays, shapes, dtypes, keys, filters, and PyTrees with JAX-, NumPy-, and BrainUnit-compatible types. |
| `brainstate.mixin` | Add reusable behavioral contracts, computation modes, or deferred `.desc()` construction to components. |
| `brainstate.environ` | Share run settings such as time, fitting mode, precision, and platform without storing them in model State or threading them through Module signatures. |

### 1. State is the mutation boundary

`State` is a typed, mutable container for an array or stable PyTree; read and replace it through `.value`, including inside BrainState transformations. Treat it as a fixed-structure slot: keep static configuration in ordinary attributes and preserve value type, shape, dtype, and PyTree structure across writes.

#### Create scalar, array, and pytree state

```python
import brainstate
import brainstate.nn as nn
import brainunit as u
import jax.numpy as jnp

counter = brainstate.State(jnp.array(0))
vector = brainstate.State(jnp.zeros(10))
neuron = brainstate.State({
    "V": jnp.zeros(5),
    "u": jnp.ones(5),
})

value = neuron.value
neuron.value = {
    "V": value["V"] + 0.1,
    "u": value["u"],
}

with brainstate.check_state_value_tree():
    neuron.value = {
        "V": jnp.zeros(5),
        "u": jnp.ones(5),
    }
```


#### Substate

State subclasses remain State containers but act as semantic markers for filtering and model organization. Use these non-parameter roles for hidden or runtime values:

| Role | Use |
|---|---|
| `HiddenState` | Internal activations or dynamical state retained between updates |
| `ShortTermState` | Transient runtime values such as current input or last spike time |
| `LongTermState` | Persistent non-parameter values such as running statistics |

```python
h = brainstate.HiddenState(jnp.zeros(5))
last_spike = brainstate.ShortTermState(jnp.full(5, -1e7))
running_mean = brainstate.LongTermState(jnp.zeros(5))
```

### 2. Modules form registered state graphs

Subclass `brainstate.nn.Module`, assign each `State` and child `Module` to an attribute, and implement computation in `update()`; assignment registers a traversable graph whose State leaves can be filtered by role:

```python
params = model.states(brainstate.ParamState)
```

#### Add state to a module

```python
class Counter(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = brainstate.ShortTermState(jnp.array(0))

    def update(self, x):
        self.count.value = self.count.value + 1
        return x * self.count.value


counter_module = Counter()
for _ in range(5):
    print(counter_module(jnp.array(10.0)), counter_module.count.value)
```

The output advances through `(10, 1)`, `(20, 2)`, ..., `(50, 5)` because the registered State is updated explicitly.

#### Use basic prebuilt layers

```python
brainstate.random.seed(42)

linear = nn.Linear(in_size=(10,), out_size=(5,))
relu = nn.ReLU()
conv = nn.Conv2d(
    in_size=(28, 28, 3),
    out_channels=32,
    kernel_size=3,
    padding="SAME",
)

x = brainstate.random.randn(8, 10)
y = relu(linear(x))
assert y.shape == (8, 5)
```

Open the layer and activation catalogs instead of guessing an uncommon class name or signature.

#### Parameter

`nn.Param` wraps an underlying `ParamState` with an optional constraint transform and regularizer; model code reads `.value()`, while optimizers update `.val`.

| Role | Use |
|---|---|
| `ParamState` | Trainable weights, biases, or other unconstrained values that need no parameter transform or regularizer |
| `nn.Param` | A trainable parameter that needs the richer transform or regularization contract |
| `nn.Const` | A fixed forward value kept in the Module graph but excluded from `ParamState` collection, gradients, and optimizer updates |

Use `nn.SoftplusT(lower=L)` for a value constrained to `(L, infinity)`; `.value()` stays in range while `.val` remains unconstrained. Do not confuse parameter transforms with execution transforms such as `brainstate.transform.jit`.

```python
w = brainstate.ParamState(brainstate.random.randn(10, 5) * 0.1)
b = brainstate.ParamState(jnp.zeros(5))

gain = nn.Param(jnp.array(1.0))
positive_tau = nn.Param(jnp.array(2.0), t=nn.SoftplusT(lower=0.1))
fixed_scale = nn.Const(jnp.array(10.0))

print(gain.value())          # IdentityT by default; usable and stored values coincide
print(positive_tau.value())  # constrained model value, always greater than 0.1
print(positive_tau.val)      # underlying unconstrained ParamState
print(fixed_scale.value())   # fixed value, excluded from ParamState collection
```

### 3. Size inference drives composition

Size-aware Modules carry feature-shape metadata without the batch dimension, so composition code can construct and validate each next layer before execution.

| API | Description |
|---|---|
| `Module.in_size` | Set the expected per-sample feature shape on the first or a standalone layer when it is known; the layer uses it to initialize shape-dependent values, validate inputs, and infer `out_size`, returning a size tuple or `None`. |
| `Module.out_size` | Read the inferred per-sample output shape after construction when wiring the next layer; it returns a size tuple or `None` and avoids duplicating shape calculations. |
| `nn.Sequential(first, *layers)` | Use for an ordered input-output pipeline; it feeds each runtime output to the next layer, propagates size metadata through the chain, and exposes the first `in_size` and final `out_size`. |
| `Layer.desc(**kwargs)` | Use after the first layer when the next layer's `in_size` should come from the preceding `out_size`; it stores the other constructor arguments, and `Sequential` replaces the descriptor with a concrete layer initialized with that inferred size. |

```python
model = nn.Sequential(
    nn.Linear(in_size=(10,), out_size=(8,)),
    nn.ReLU(),
    nn.Linear.desc(out_size=(2,)),
)

x = brainstate.random.randn(4, 10)
y = model(x)

assert model.layers[2].in_size == (8,)
assert model.out_size == (2,)
assert y.shape == (4, 2)
```

Open `references/size-inference-variations.md` to compose `ComplexNet` with `Sequential` and `.desc()`, or when convolution, pooling, and flattening make size propagation non-obvious.

### 4. Environment context drives simulations

Use `brainstate.environ` to share run settings across a computation, where scoped contexts override persistent defaults and model code reads the active value.

| API | Description |
|---|---|
| `brainstate.environ.context(**settings)` | Use for one simulation, training phase, evaluation phase, or step; it pushes temporary settings, inherits unspecified outer values, and restores the previous values on exit, including exceptional exit. |
| `brainstate.environ.get_dt()` | Use inside numerical dynamics that require the active integration step; it returns `dt` from the selected environment and raises `KeyError` when `dt` is unset. |
| `brainstate.nn.init_all_states(target, **kwargs)` | Use after constructing a stateful Module and before its first rollout; it calls `init_state()` across the graph in `@call_order` order and returns the initialized target. |
| `brainstate.transform.for_loop(step, *xs)` | Use for a State-aware time loop; it slices each input along its leading axis and returns the stacked per-step outputs. |

```python
with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
    brainstate.nn.init_all_states(net)
    times = u.math.arange(
        0.0 * u.ms,
        100.0 * u.ms,
        brainstate.environ.get_dt(),
    )
    step_indices = jnp.arange(times.shape[0])

    def step(t, i):
        with brainstate.environ.context(t=t, i=i):
            return net.update(input_current)

    outputs = brainstate.transform.for_loop(
        step,
        times,
        step_indices,
    )
```

Open `references/simulation-environment.md` for persistent defaults, generic setting access, nested or isolated environments, precision and platform controls, `exp_euler_step()`, and environment-specific failures.

### 5. State-aware transforms

Wrap the complete forward, simulation, or training step in `brainstate.transform`; its JAX-like `jit`, `grad`, and `vmap` track State reads and writes, while raw JAX transforms can lose State mutations.

#### Canonical transformation setup

```python
brainstate.random.seed(0)
model = nn.Linear(in_size=(3,), out_size=(1,))
x_train = brainstate.random.randn(64, 3)
y_train = brainstate.random.randn(64, 1)
params = model.states(brainstate.ParamState)
```

#### State-aware jit example

```python
forward = brainstate.transform.jit(model)
prediction = forward(x_train)
assert prediction.shape == (64, 1)
```

The first compatible call traces and compiles the complete forward pass; later compatible calls reuse it while BrainState handles State effects.

#### Gradient and parameter update example

`grad` differentiates with respect to a State collection and returns gradients keyed by the same State paths. `return_value=True` returns the loss from the same pass.

```python
def loss_fn():
    return jnp.mean((model(x_train) - y_train) ** 2)


grads, loss = brainstate.transform.grad(
    loss_fn,
    params,
    return_value=True,
)()

for key in params:
    params[key].value -= 0.1 * grads[key]
```

#### Composed training-step transform

```python
@brainstate.transform.jit
def train_step():
    grads, loss = brainstate.transform.grad(
        loss_fn,
        params,
        return_value=True,
    )()
    for key in params:
        params[key].value -= 0.1 * grads[key]
    return loss


loss = train_step()
```

Use `jit(grad(...))` as the default compiled training-step backbone. Open the optimizer reference when manual updates should become optimizer-managed updates.

#### State-aware `vmap`

```python
def predict_one(x_row):
    return model(x_row[None, :])[0]


predict_batch = brainstate.transform.vmap(predict_one)
batched_prediction = predict_batch(x_train)
assert batched_prediction.shape == (64, 1)
```

This maps a function written for one example over a batch. Open the `vmap` expansion for mapped State axes, ensembles, parameter sweeps, or the documented `state_in_axes` / `state_out_axes` controls. The routed tutorial does not define the rough draft's `in_states` / `out_states` names.

### 6. Randomness

Use the NumPy-like `brainstate.random` API through the global `DEFAULT` `RandomState`, which splits keys automatically; call `seed()` before any sequence that must be reproducible.

#### Seed when the sequence needs reproducibility

```python
brainstate.random.seed(42)
x1 = brainstate.random.rand(5)

brainstate.random.seed(42)
x2 = brainstate.random.rand(5)

assert jnp.allclose(x1, x2)
```

#### Generate random input spikes

The official SNN training pattern compares uniform `[0, 1)` samples with the dimensionless per-step spike probability `firing_rate * dt`. The result is a Boolean `(time, batch, input)` tensor:

```python
num_steps, batch_size, num_inputs = 100, 128, 100
firing_rate = 5.0 * u.Hz

with brainstate.environ.context(dt=1.0 * u.ms):
    input_spikes = (
        brainstate.random.rand(num_steps, batch_size, num_inputs)
        < firing_rate * brainstate.environ.get_dt()
    )

assert input_spikes.shape == (num_steps, batch_size, num_inputs)
```

#### Shuffle the training order

`brainstate.random.shuffle(x, axis=0)` returns a shuffled copy; JAX arrays are immutable, so it does not modify `x` in place. Generate one order and apply it to every aligned training array:

```python
order = brainstate.random.shuffle(jnp.arange(x_train.shape[0]))
x_epoch = x_train[order]
y_epoch = y_train[order]
```

Use independent `RandomState` instances, key save/restore, stochastic mapping, or checkpoint-aware randomness only through the randomness reference.

## Script references

- `scripts/integrator_rnn.py`: full stateful sequence-training workflow with a custom RNN cell, trainable initial state, optimization, compiled steps, and evaluation.
- `scripts/lif_neuron_model.py`: extended combination of `HiddenState`, `ShortTermState`, and `ParamState` with explicit `.value` updates.
- `scripts/modern_cnn.py`: full convolution, normalization, activation, pooling, dropout, and dense Module composition. Select it through the layer or activation branch.
- `scripts/resnet.py`: residual Modules and dynamically registered child blocks.

## Reference routing

Route by the outcome the task needs, then open only the smallest reference that owns that variant.

### State graphs, collections, and lifecycle

| Reference | Open when |
|---|---|
| `references/state-graph-operations.md` | Break a stateful `Module` into `GraphDef` plus State PyTrees so raw JAX can transform or checkpoint the values, then reconstruct the graph without losing sharing or cycles |
| `references/state_collections_and_utilities.md` | Structure, filter, freeze, flatten, and inspect the configs or PyTree mappings around a model without changing graph identity; use `DictManager`, `DotDict`, `FrozenDict`, and declarative filters |
| `references/collective_model_operations.md` | Initialize, reset, restore, or invoke one shared method across an entire Module graph without manual traversal; use `call_order`, `call_all_fns`, `init_all_states`, and their vmapped variants |

### Simulation environment

| Reference | Open when |
|---|---|
| `references/simulation-environment.md` | Choose persistent defaults, generic setting access, scoped or nested overrides, Module-bound contexts, or an isolated `EnvironmentState`; configure precision and platform, or advance environment-driven dynamics with `exp_euler_step()` |

### Model composition, extension, and interoperation

| Reference | Open when |
|---|---|
| `references/size-inference-variations.md` | Prevent shape mismatches when spatial operations make `out_size` non-obvious; resolve convolution, padding, pooling, and `Flatten` variants so size-aware `Sequential` / `.desc()` composition remains valid |
| `references/extension_mechanisms.md` | Extend BrainState without rewriting core Modules: compose reusable `Mixin` behavior, defer construction with `ParamDesc`, centralize runtime semantics with `Mode`, or observe and enforce State access through hooks |
| `references/model-interop-and-migration.md` | Reuse standard-layer architectures and weights across BrainState, Flax NNX/Linen, or Equinox, or port a PyTorch workflow into BrainState's explicit State and transform model |

### Parameters, optimization, and randomness

| Reference | Open when |
|---|---|
| `references/brainstate/parameter-constraints-regularization.md` | Encode valid parameter domains and modeling priors while optimization remains unconstrained; use `nn.Param` transforms, explicit regularization or prior penalties, and `nn.Const` for fixed graph values |
| `references/braintools-optimizer-reference.md` | Replace manual `ParamState` updates with the right optimization strategy inside the canonical training step; select a `braintools.optim` optimizer or scheduler, `OptaxOptimizer`, `ScipyOptimizer`, or `NevergradOptimizer` |
| `references/brainstate/randomness-and-reproducibility.md` | Control independence and exact replay of dropout, noise, and stochastic trials across transforms and checkpoints; use custom `RandomState` streams, direct key management, parallel key preparation, and checkpoint restoration |

The remaining nested reference has one inbound route:

- Only `parameter-constraints-regularization.md` may open `parameter-transforms-regularizers-catalog.md`.

### Layer libraries

| Reference | Open when |
|---|---|
| `references/libraries/prebuilt-layer-library.md` | Keep graph registration, parameter roles, size metadata, and fit behavior BrainState-native by selecting an existing layer instead of hand-rolling a standard linear, convolutional, normalization, pooling, padding, or dropout operator |
| `references/libraries/prebuilt-activation-library.md` | Decide whether nonlinear behavior belongs in Module composition or as a pure lowercase function inside `update()` or raw JAX code, then select the exact activation symbol |

### Stateful transformations

| Reference | Open when |
|---|---|
| `references/brainstate/transformation-jit-expansion.md` | Compile a whole stateful step while keeping State writes observable, or expose mutation explicitly to cross into raw `jax.jit`; use `JittedFunction` cache controls and static specialization around that boundary |
| `references/brainstate/transformation-grad-expansion.md` | Define exactly what a stateful computation differentiates with `argnums`, `grad_states`, or both, and how gradients, loss, and auxiliary values return; use `StateFinder`, Jacobian/Hessian transforms, or optimizer fitting overlays |
| `references/brainstate/transformation-vmap-expansion.md` | Lift one stateful computation into batches, ensembles, or sweeps while deciding per-instance versus shared State, write-back, and random-key behavior; use `vmap2` with `state_in_axes` / `state_out_axes` |
| `references/brainstate/brainstate-control-flow-patterns.md` | Express repeated or conditional State updates as compiled JAX control flow instead of Python loops and branches; choose `scan`, `for_loop`, checkpointed loops, bounded iteration, or lazy branches by carry, gradient, and memory needs |
| `references/brainstate/brainstate-transformed-diagnostics.md` | See what actually happens at runtime inside traced stateful code and enforce value-dependent invariants; use `checkify`, `jit_error_if`, `debug_nan`, runtime print/callback, or conditional breakpoints |

Do not route to dynamics or solver references from this skill; they are outside the architecture supplied for this BrainState skill.
