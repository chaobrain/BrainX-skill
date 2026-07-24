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

BrainState makes mutable programs compatible with JAX's functional model. A `State` is the explicit mutation boundary: values that change, such as neuron voltages, firing rates, parameters, hidden activations, or optimizer buffers, live in `.value`; ordinary attributes remain static.

This fits neuroscience simulation and training because brain models contain values with different lifecycles: membrane potentials and firing rates evolve during every simulation step, while weights are optimized across training steps. e.g `HiddenState` marks evolving neural dynamics and `ParamState` marks trainable values, so code can update simulation state separately from the parameters selected for gradients.

BrainState transforms mirror JAX's `jit`, `grad`, and `vmap`, but understand `State`: they thread state reads and writes, differentiate State collections, and share or map State during vectorization. Raw JAX transforms expect explicit, side-effect-free state flow, so `jax.jit` can discard ordinary mutation. Read and write `.value`, then wrap the complete stateful operation in `brainstate.transform`.

### Api structure

| API | Feature description |
|---|---|
| `brainstate` | Defines mutable `State` classes, semantic State roles, tracing utilities, hooks, and State-specific errors; use it to create and manage values that must participate in BrainState transformations. |
| `brainstate.graph` | Traverses, filters, splits, merges, and reconstructs object graphs while preserving shared references and cycles; use it for structural State and Module graph operations. |
| `brainstate.nn` | Provides the `Module` system, parameter containers and transforms, neural-network layers, dynamics, delays, and metrics; use it to assemble artificial or spiking models. |
| `brainstate.transform` | Extends JAX transformations with State handling for JIT compilation, automatic differentiation, vectorization, parallelization, and control flow; use it around complete stateful computations. |
| `brainstate.interop` | Converts supported layers and `Sequential` models between `brainstate.nn` and Flax NNX, Flax Linen, or Equinox while transferring weights; use it for framework migration or integration. |
| `brainstate.random` | Provides a stateful, NumPy-like interface to JAX random generation with automatic key splitting, seeding, and probability distributions; use it for reproducible initialization, sampling, and simulation noise. |
| `brainstate.util` | Supplies mapping, filtering, pretty-printing, dataclass, caching, and dictionary helpers; use it to manipulate BrainState collections and supporting data structures. |
| `brainstate.typing` | Exposes JAX-, NumPy-, and BrainUnit-compatible aliases and protocols for arrays, shapes, dtypes, keys, filters, and PyTrees; use it to annotate public APIs precisely. |
| `brainstate.mixin` | Defines reusable behavior mixins, deferred parameter descriptors, computation modes, and type utilities; use it to add shared contracts or `.desc()`-style construction to components. |
| `brainstate.environ` | Manages global defaults and scoped overrides for time, training mode, precision, platform, and related run settings; use it to share configuration across a computation. |

### 1. State is the mutation boundary

`State` encapsulates model values that change over time. It can wrap Python scalars, arrays, `jax.Array` values, dictionaries, lists, or another stable PyTree structure; its value remains mutable after compilation. Read and write it through `.value`.

State provides three operational guarantees:

- Its value can be updated inside JIT-compiled functions.
- State checks value type, shape, and, when requested, PyTree structure.
- BrainState transformations can discover and manage its reads and writes.

Only values inside `State` are mutable in transformed code. Keep ordinary Python attributes for static configuration. Preserve the original PyTree structure when assigning `.value`.

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

`brainstate.nn.Module` is the base class for BrainState modules. It provides automatic child registration, State collection, inspection, and integration with BrainState transformations. Assign each `State` and child `Module` to an attribute so the model becomes a nested Module graph with State objects at the leaves.

Modules keep related State and computation together, can be reused after construction, and compose into larger graphs. Collect only the semantic State role required by the operation:

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
pool = nn.MaxPool2d(
    in_size=conv.out_size,
    kernel_size=(2, 2),
    stride=(2, 2),
    channel_axis=-1,
)

x = brainstate.random.randn(8, 10)
y = relu(linear(x))
assert y.shape == (8, 5)
```

Open the layer and activation catalogs instead of guessing an uncommon class name or signature.

#### Parameter

A `ParamState` is a bare trainable array; that is all most layers need. `nn.Param` is a richer parameter container built around an underlying `ParamState`: it adds an optional bijective parameter transform and an optional regularizer. With `nn.Param`, model computation reads the usable value with `.value()`, while the optimizer updates the unconstrained `ParamState` in `.val`.

| Role | Use |
|---|---|
| `ParamState` | Trainable weights, biases, or other unconstrained values that need no parameter transform or regularizer |
| `nn.Param` | A trainable parameter that needs the richer transform or regularization contract |
| `nn.Const` | A fixed parameter-like value kept inside the Module graph |

`nn.Const` is an `nn.Param` with `fit=False`. It is excluded when collecting `ParamState` objects, so gradients and optimizers leave it unchanged.

`nn.SoftplusT(lower=L)` is the canonical positive-domain parameter transform. It maps an unconstrained optimizer value to `(L, infinity)`, so `.value()` stays strictly above `L` regardless of how `.val` changes. Parameter transforms constrain values; they are distinct from execution transforms such as `brainstate.transform.jit`.

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

Every size-aware `brainstate.nn.Module` exposes `in_size` and `out_size` as feature shapes without the batch dimension. When the input size is known, the Module computes its output size. `nn.Sequential` propagates one layer's `out_size` into the next layer, and `.desc()` creates a descriptor that is instantiated when that input size becomes available.
#### Compose `ComplexNet` with `Sequential` and `.desc()`

```python
class ComplexNet(brainstate.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_size,
                out_channels=16,
                kernel_size=3,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.Conv2d.desc(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.Conv2d.desc(
                out_channels=64,
                kernel_size=3,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.MaxPool2d.desc(
                kernel_size=(2, 2),
                stride=(2, 2),
                channel_axis=-1,
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(in_size=self.features.out_size),
            nn.Linear.desc(out_size=256),
            nn.ReLU(),
            nn.Linear.desc(out_size=10),
        )

    def update(self, x):
        return self.classifier(self.features(x))


brainstate.random.seed(42)
net = ComplexNet(in_size=(32, 32, 3))
x_image = brainstate.random.randn(2, 32, 32, 3)
y_image = net(x_image)

assert net.features.out_size == (8, 8, 64)
assert net.classifier.layers[0].out_size == (4096,)
assert net.classifier.out_size == (10,)
assert y_image.shape == (2, 10)
```

Open `references/size-inference-variations.md` for convolution formulas, padding/stride edge cases, pooling reduction, and flatten-size variants.

### 4. Environment context drives simulations

Many parts of a simulation need the same settings, such as the time step (`dt`), training mode (`fit`), and numerical precision. Instead of passing these values to every Module, store them in `brainstate.environ`. Use `set()` for a lasting default or `context()` to apply settings only inside a `with` block; code inside that scope reads them with `get()` or a typed accessor such as `get_dt()`.

`EnvironmentState` is the object that stores these settings separately for each thread. Most users do not create one directly because BrainState provides a default environment. Despite its name, it is not a model `State` and does not belong in a Module graph.

Usually, wrap the relevant simulation, training block, or step in `brainstate.environ.context()`.

#### Initialize and run a time-indexed simulation

`brainstate.nn.init_all_states(target, **kwargs)` walks the Module graph and calls `init_state()` on every node, respecting `@call_order` decorators. Call it after constructing a stateful model and before its first rollout; pass `batch_size=` when its hidden states need a batch axis.

Given a network with 11 neurons and `net.spike.value.shape == (11,)`, use this canonical rollout:

```python
num_neurons = 11
total = 5_000.0 * u.ms
input_current = 3.0 * u.mA

with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
    brainstate.nn.init_all_states(net)
    times = u.math.arange(
        0.0 * u.ms,
        total,
        brainstate.environ.get_dt(),
    )
    step_indices = jnp.arange(times.shape[0])

    def step(t, i):
        with brainstate.environ.context(t=t, i=i):
            net.update(input_current)
        return t, net.spike.value

    sampled_times, spikes = brainstate.transform.for_loop(
        step,
        times,
        step_indices,
    )

assert sampled_times.shape == (50_000,)
assert spikes.shape == (50_000, num_neurons)
```

`brainstate.transform.for_loop` slices inputs along their leading axis and stacks each per-step return. Therefore a per-step spike vector with shape `(11,)` becomes a trajectory with shape `(time, neuron)`, here `(50000, 11)`. `spikes[time_index, neuron_index]` is that neuron's event at that step; for a Boolean spike State, `True` is the spike event and becomes `1` only if converted to a numeric representation.

Use `brainstate.nn.EnvironContext(layer, **context)` only when one Module should automatically run with the same settings on every call. Its `update()` method also accepts a `context` dictionary when one call needs different settings.

```python
evaluation_model = brainstate.nn.EnvironContext(model, fit=False)
prediction = evaluation_model.update(inputs)
```

#### Advanced environment configuration

Construct a separate `EnvironmentState` only for an explicitly isolated configuration, and pass it consistently through the `env=` argument of `set()`, `context()`, and `get()` or `get_dt()`.

| Setting | Precise use |
|---|---|
| `dt` | Numerical integration step; set before initialization or rollout and read with `brainstate.environ.get_dt()` |
| `t` | Current simulation time; set for each step and read with `brainstate.environ.get("t")` |
| `i` | Current integer step index; set for each step and read with `brainstate.environ.get("i")` |
| `fit` | Set `True` for training and `False` for evaluation; layers such as dropout and batch normalization observe it consistently |
| `precision` | Default numerical precision (`8`, `16`, `32`, `64`, or `"bf16"`) for the scoped computation; inspect with `brainstate.environ.get_precision()` |

#### Use `exp_euler_step` for element-wise continuous dynamics

`brainstate.nn.exp_euler_step()` advances an ODE or SDE by one environment `dt`. It exactly integrates the diagonal linearized part and is well suited to per-neuron or per-synapse decay/growth dynamics:

```python
def decay(v, t, tau):
    return -v / tau


v = jnp.ones(num_neurons) * u.mV
with brainstate.environ.context(dt=0.1 * u.ms):
    v_next = brainstate.nn.exp_euler_step(
        decay,
        v,
        0.0 * u.ms,
        10.0 * u.ms,
    )
```

The state must use a supported floating dtype, and a state with units `[X]` requires drift units `[X] / [time]`. Only the diagonal of the drift Jacobian is integrated exponentially; off-diagonal coupling is treated explicitly like forward Euler, so do not treat this as an exact solver for strongly cross-coupled systems.

### 5. State-aware transforms

`brainstate.transform` mirrors JAX's `jit`, `grad`, and `vmap`, but tracks the `State` objects a model reads and writes. Raw `jax.jit` can discard State writes; wrap the complete stateful operation in `brainstate.transform` and prefer whole forward, simulation, or training steps over fragmented transforms.

#### Canonical transformation setup

The scripts below share one model and dataset so the `jit`, `grad`, composed training-step, and `vmap` decisions are not re-explained four times.

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

All `brainstate.random` functions use the global `brainstate.random.DEFAULT` `RandomState` unless a separate stream or key is supplied. Use the NumPy-like sampling API directly for stochastic inputs, data order, initialization, and simulation noise.

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
