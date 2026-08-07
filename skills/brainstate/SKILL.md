---
name: brainstate
description: BrainState is the central stateful execution infrastructure for BrainX modeling and simulation. Use this skill for mutable State and `.value`, ParamState and other State roles, Module graphs, environment-scoped simulations, state initialization, operational randomness, size-aware neural-network composition, state-aware jit/grad/vmap, or a BrainState training step.
---

# BrainState

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

In BrainX modeling, seed BrainState's stateful generator before model construction or simulation so initialization, stochastic inputs, noise, dropout, and data order follow the same automatically split key sequence.

| API | Description |
|---|---|
| `brainstate.random.DEFAULT` | Global `RandomState` used by module-level random functions; it manages and splits JAX keys automatically. |
| `brainstate.random.seed(seed)` | Set the global seed for JAX and NumPy; call it before random work that must be reproducible. |
| `brainstate.random.rand(*shape)` / `random(size)` | Draw uniform samples in `[0, 1)` for stochastic thresholds such as spike or dropout probabilities. |
| `brainstate.random.randn(*shape)` / `normal(loc, scale, size)` | Draw standard or parameterized Gaussian samples for parameter initialization and model noise. |
| `brainstate.random.bernoulli(p, size)` | Draw Bernoulli samples for binary events and masks. |
| `brainstate.random.shuffle(x, axis=0)` / `permutation(x)` | Randomly reorder data, trials, or indices; `shuffle()` returns a shuffled copy along the selected axis. |

#### Generate reproducible stochastic input

```python
brainstate.random.seed(42)

num_steps, batch_size, num_inputs = 100, 128, 100
firing_rate = 5.0 * u.Hz

with brainstate.environ.context(dt=1.0 * u.ms):
    spike_probability = firing_rate * brainstate.environ.get_dt()
    input_spikes = (
        brainstate.random.rand(num_steps, batch_size, num_inputs)
        < spike_probability
    )

assert input_spikes.shape == (num_steps, batch_size, num_inputs)
```

Open `references/brainstate/randomness-and-reproducibility.md` for independent `RandomState` streams, direct key control, stochastic mapping, exact replay, or checkpointed RNG State.

## Reference routing

Route by the outcome the task needs, then open only the smallest reference that owns that variant.

### State graphs, collections, and lifecycle

| Reference | Open when |
|---|---|
| `references/state-graph-operations.md` | Splitting, transforming, checkpointing, or reconstructing Module graphs while preserving sharing and cycles |
| `references/state_collections_and_utilities.md` | Filtering, freezing, flattening, configuring, or inspecting supporting mappings without graph identity |
| `references/collective_model_operations.md` | Initializing, resetting, restoring, or invoking methods across a Module graph, including vmapped lifecycle operations |

### Simulation environment

| Reference | Open when |
|---|---|
| `references/simulation-environment.md` | Managing persistent, scoped, nested, Module-bound, or isolated environments, including precision, platform, and `exp_euler_step()` |

### Model composition, extension, and interoperation

| Reference | Open when |
|---|---|
| `references/size-inference-variations.md` | Resolving convolution, padding, pooling, or `Flatten` size propagation in `Sequential` / `.desc()` pipelines |
| `references/extension_mechanisms.md` | Adding reusable `Mixin` behavior, deferred `ParamDesc` construction, runtime `Mode` semantics, or State hooks |
| `references/model-interop-and-migration.md` | Converting models or weights among BrainState, Flax NNX/Linen, Equinox, and PyTorch |

### Parameters, optimization, and randomness

| Reference | Open when |
|---|---|
| `references/brainstate/parameter-constraints-regularization.md` | Using `nn.Param` transforms, regularizers, priors, penalties, or `nn.Const` |
| `references/braintools/optimizer.md` | Selecting a `braintools.optim`, Optax, SciPy, or Nevergrad optimizer or scheduler |
| `references/brainstate/randomness-and-reproducibility.md` | Creating independent random streams or exact stochastic replay across transforms and checkpoints |

The remaining nested reference has one inbound route:

- Only `references/brainstate/parameter-constraints-regularization.md` may open `references/brainstate/parameter-transforms-regularizers-catalog.md`.

### Layer libraries

| Reference | Open when |
|---|---|
| `references/libraries/prebuilt-layer-library.md` | Selecting a BrainState linear, convolutional, normalization, pooling, padding, or dropout layer |
| `references/libraries/prebuilt-activation-library.md` | Selecting a Module activation or pure lowercase activation function |

### Stateful transformations

| Reference | Open when |
|---|---|
| `references/brainstate/transformation-jit-expansion.md` | Controlling State write-back, raw `jax.jit` boundaries, caching, or static specialization |
| `references/brainstate/transformation-grad-expansion.md` | Controlling `argnums`, `grad_states`, gradient returns, higher-order transforms, or fitting overlays |
| `references/brainstate/transformation-vmap-expansion.md` | Mapping or sharing State across batches, ensembles, or sweeps, including State axes and randomness |
| `references/brainstate/brainstate-control-flow-patterns.md` | Choosing transform-safe loops, scans, branches, checkpointed control flow, or memory-efficient training through long rollouts |
| `references/brainstate/brainstate-transformed-diagnostics.md` | Debugging runtime values or enforcing invariants inside transformed stateful code |

Do not route to dynamics or solver references from this skill; they are outside the architecture supplied for this BrainState skill.

## Application script examples

Open one script only when its pattern matches the task; each is a complete, runnable program.

| Script | Open when |
|---|---|
| `scripts/integrator_rnn.py` | Building a full stateful sequence-training workflow with a custom RNN cell, trainable initial state, optimization, compiled steps, and evaluation. |
| `scripts/lif_neuron_model.py` | Combining `HiddenState`, `ShortTermState`, and `ParamState` in one Module with explicit `.value` updates. |
| `scripts/modern_cnn.py` | Composing convolution, normalization, activation, pooling, dropout, and dense Modules; select it through the layer or activation branch. |
| `scripts/resnet.py` | Building residual Modules and dynamically registered child blocks. |
