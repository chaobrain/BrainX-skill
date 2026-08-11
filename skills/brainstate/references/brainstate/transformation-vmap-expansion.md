# BrainState vmap expansion

Use this reference after the minimal `brainstate.transform.vmap` example in
`skills/brainstate/SKILL.md` when argument axes, State axes, random-key policy,
nested mappings, or unexpected State writes require explicit control.

This reference covers only behavior demonstrated by these routed official tutorials:

- Vectorization: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html
- Random Number Generation: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

The Vectorization tutorial demonstrates filter-based `vmap2`; the current generated API pages define both mapping contracts. Use `vmap` when specific State instances must be declared with `in_states` and `out_states`. Use `vmap2` when State filters, automatic output-axis inference, or `unexpected_out_state_mapping` are required. Do not silently substitute one contract for the other.

## Selection map

| Need | Control | Key constraint |
|---|---|---|
| Map or broadcast function arguments | `in_axes` | Use an integer axis to map and `None` to share. |
| Place the mapped output axis | `out_axes` | Match the downstream layout explicitly. |
| Map with no mapped input argument | `axis_size` | Supply the size because it cannot be inferred. |
| Use collectives over the mapped axis | `axis_name` | The collective must use the same name. |
| Map explicitly named State instances | `vmap(..., in_states=..., out_states=...)` | Declare every mapped input State and written output State; undeclared batched writes raise `BatchAxisError`. |
| Map mutable State but share parameters | `vmap2(..., state_in_axes=..., state_out_axes=...)` with State filters | Define both input mapping and write-back for per-instance State. |
| Handle a written State omitted from `state_out_axes` | `vmap2(..., unexpected_out_state_mapping=...)` | The default `'auto'` infers and scatters the mapped axis; declare State axes when a coincidental leading-size match would be ambiguous. |
| Choose independent or shared random draws | BrainState automatic splitting or one broadcast JAX key | Shared keys intentionally correlate mapped results. |
| Build a custom mapping primitive | `StatefulMapping` | Use only when `vmap` or `vmap2` cannot express the required primitive. |

The Vectorization tutorial imports the detailed mapping API and State filter as follows:

```python
import jax
import jax.numpy as jnp
import brainstate
from brainstate.transform import vmap2
from brainstate.util.filter import OfType
```

## Map sweeps and broadcast fixed arguments

`in_axes` controls how batch dimensions are mapped over function arguments and works identically to `jax.vmap`. Use an axis number for a swept argument and `None` for a value shared by every mapped call.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#the-in-axes-parameter

```python
def weighted_sum(x, weight):
    """Compute weighted sum: x * weight"""
    return x * weight


vmap_weighted = vmap2(weighted_sum, in_axes=(0, None))
batch_x = jnp.array([1.0, 2.0, 3.0])
single_weight = 2.0
result = vmap_weighted(batch_x, single_weight)
```

Here `batch_x` supplies the sweep axis and `single_weight` is broadcast. To map corresponding candidates from two sweeps, use `in_axes=(0, 0)`, as in the tutorial's batched matrix/vector product.

`out_axes` controls where the mapped dimension appears in the result. Set it deliberately when downstream code expects the batch somewhere other than axis 0.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#the-out-axes-parameter

```python
def create_vector(scalar):
    return jnp.array([scalar, scalar * 2, scalar * 3])


vmap_axis1 = vmap2(create_vector, in_axes=0, out_axes=1)
result_axis1 = vmap_axis1(jnp.array([1.0, 2.0]))
assert result_axis1.shape == (3, 2)
```

When every input is static, the mapped size cannot be inferred from an argument; supply `axis_size` explicitly.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#the-axis-size-parameter

```python
def generate_sequence(unused=None):
    return jnp.arange(3)


vmap_generate = vmap2(generate_sequence, in_axes=None, axis_size=5)
result = vmap_generate()
assert result.shape == (5, 3)
```

Name the mapped axis only when a collective operation needs it. The tutorial uses `axis_name='batch'` with `jax.lax.pmean`:

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#the-axis-name-parameter

```python
def normalize_batch(x):
    batch_mean = jax.lax.pmean(x, axis_name='batch')
    return x - batch_mean


vmap_normalize = vmap2(normalize_batch, in_axes=0, axis_name='batch')
normalized = vmap_normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
```

## Map per-instance state and share parameters

`state_in_axes` and `state_out_axes` are BrainState-specific parameters that control how `State` objects are batched. The tutorial's State Axis Inference example maps `ShortTermState` along axis 0 while leaving `ParamState` shared; use that demonstrated separation for an ensemble with per-instance mutable State and shared parameters.

Sources:

- https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#state-aware-parameters-state-in-axes-and-state-out-axes
- https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#state-axis-inference

```python
class StatefulComputation(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = brainstate.ShortTermState(jnp.zeros(3))
        self.param = brainstate.ParamState(jnp.array(1.0))

    def __call__(self, x):
        self.temp.value = self.temp.value + x
        return self.temp.value * self.param.value


model = StatefulComputation()
vmap_model = vmap2(
    model,
    in_axes=0,
    out_axes=0,
    state_in_axes={0: OfType(brainstate.ShortTermState)},
    state_out_axes={0: OfType(brainstate.ShortTermState)},
)

inputs = jnp.array([1.0, 2.0, 3.0])
outputs = vmap_model(inputs)

assert jnp.allclose(model.temp.value, jnp.array([1.0, 2.0, 3.0]))
assert model.param.value == 1.0
```

This bundles the State-axis decision with the executable pattern: each mapped element owns one `temp` value, while every element reads the same scalar parameter. For an ordinary `nn.Module` with no State-axis filters, the tutorial says Module states are typically shared, or broadcast, across the batch by default.

A shared State remains unbatched and is omitted from the mapped State axis. Keep it read-only during the mapped call. Broadcasting identical values to shape `(batch, ...)` and then mapping axis 0 creates independent copies with batch-scaled storage; it does not share one State. Map a State that each lane may write, but keep an unbatched State shared only when concurrent lanes do not update it.

### Choose the State declaration contract

The generated APIs use distinct, non-alias parameter names:

| API | State input mapping | State output mapping | Undeclared batched write |
|---|---|---|---|
| `brainstate.transform.vmap` | `in_states` by State instance | `out_states` by State instance | Raises `BatchAxisError`. |
| `brainstate.transform.vmap2` | `state_in_axes` by State filter | `state_out_axes` by State filter | Controlled by `unexpected_out_state_mapping`; default `'auto'` infers and scatters. |

Sources: https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.transform.vmap.html and https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.transform.vmap2.html.

## Handle state writes outside `state_out_axes`

`unexpected_out_state_mapping` controls behavior when a State is written but is
not covered by `state_out_axes`. The current generated `vmap2` API defaults to
`'auto'`, which infers the output axis and scatters the write. The `'raise'`
policy raises `BatchAxisError`; `'warn'` and `'ignore'` scatter with or without
a warning. The tutorial demonstrates an explicit `'ignore'` policy with a
written `LongTermState` that is not listed in State-axis filters.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#the-unexpected-out-state-mapping-parameter

Exact policy source: https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.transform.vmap2.html

```python
temp_state = brainstate.ShortTermState(jnp.array(0.0))
write_state = brainstate.LongTermState(jnp.asarray(0.0))


def update_temp(x):
    temp_state.value = temp_state.value + x
    write_state.value = temp_state.value
    return temp_state.value


vmap_ignore = vmap2(
    update_temp,
    in_axes=0,
    unexpected_out_state_mapping='ignore',
)
result = vmap_ignore(jnp.array([1.0, 2.0, 3.0]))
```

Declare the State axes when an unexpected read-modify-write could be mistaken
for per-lane State because its leading size happens to match the mapped size.
Do not assume that mapping only the function arguments also specifies the
output behavior of every State the function mutates.

## Choose independent or shared randomness

`brainstate.transform.vmap` automatically splits keys for `brainstate.random.RandomState`, so each mapped element receives a unique random key. Seed before the mapped call when the whole experiment must be reproducible; do not manually add `RandomState` State-axis filters for this ordinary independent-sample path.

Sources:

- https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#automatic-key-splitting-for-randomstate
- https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html#setting-seeds

```python
brainstate.random.seed(42)


def sample_normal(scale):
    return brainstate.random.normal(0.0, scale)


vmap_sample = vmap2(sample_normal, in_axes=0)
scales = jnp.array([1.0, 2.0, 3.0, 4.0])
samples = vmap_sample(scales)
```

This is also the smallest stochastic parameter-sweep pattern: the scale is mapped, and BrainState supplies an independent key to each element.

If every mapped element must reuse the same random draw, the tutorial uses the JAX random API and broadcasts one key with `in_axes=None`.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#controlling-random-keys-using-jax-s-random-api

```python
def sample_with_jax_key(key, scale):
    return jax.random.normal(key, ()) * scale


vmap_shared_key = vmap2(
    sample_with_jax_key,
    in_axes=(None, 0),
)

shared_key = jax.random.PRNGKey(0)
scales = jnp.array([1.0, 2.0, 3.0, 4.0])
samples_shared = vmap_shared_key(shared_key, scales)
```

Stop here for ordinary stochastic mapping. For `get_key`, `set_key`, `split_key`, custom `RandomState` streams, checkpoint restoration, or parallel key assignment, open `randomness-and-reproducibility.md`; that reference owns direct key manipulation rather than duplicating it here.

Randomness boundary source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html#advanced-key-management

## Nest and compose mappings

Nest `vmap2` when the function must map over more than one batch dimension. Each layer owns one mapped axis.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#nested-vmap2

```python
def matrix_elem_product(x, y):
    return x * y


vmap_rows = vmap2(matrix_elem_product, in_axes=(0, 0))
vmap_matrix = vmap2(vmap_rows, in_axes=(0, 0))

matrix_a = jnp.ones((3, 4))
matrix_b = jnp.arange(12).reshape(3, 4)
result = vmap_matrix(matrix_a, matrix_b)
```

The tutorial also composes BrainState `grad`, `vmap2`, and `jit`. Map the per-example gradient first, then compile the mapped function.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#combining-with-other-transforms

```python
from brainstate.transform import grad, jit


def loss_fn(x, target):
    pred = x ** 2
    return jnp.sum((pred - target) ** 2)


batched_grad = vmap2(
    grad(loss_fn, argnums=0),
    in_axes=(0, 0),
)
batched_grad_jit = jit(batched_grad)

batch_x = jnp.array([1.0, 2.0, 3.0])
batch_targets = jnp.array([2.0, 4.0, 6.0])
gradients = batched_grad_jit(batch_x, batch_targets)
```

## Use `StatefulMapping` only for a custom primitive

The tutorial states that `vmap` is a thin wrapper around `StatefulMapping`, which performs State discovery, input/output axis mapping, intermediate-representation compilation, and State management. Advanced users can instantiate it directly for a custom mapping primitive.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html#direct-use-of-statefulmapping

```python
import functools
from brainstate.transform import StatefulMapping

counter_state = brainstate.ShortTermState(jnp.zeros(3))


def increment(delta):
    counter_state.value = counter_state.value + delta
    return counter_state.value


custom_mapping = StatefulMapping(
    increment,
    in_axes=0,
    out_axes=0,
    state_in_axes={0: OfType(brainstate.ShortTermState)},
    state_out_axes={0: OfType(brainstate.ShortTermState)},
    name="custom_increment",
    mapping_fn=functools.partial(jax.vmap, spmd_axis_name=None),
)

results = custom_mapping(jnp.array([1.0, 2.0, 3.0]))
```

Do not infer semantics for `vmap_new_states`, `vmap2_new_states`, `map`, `pmap2`, `pmap2_new_states`, `shard_map`, or `unvmap` from their names. The routed Vectorization tutorial lists those APIs in navigation but does not teach their contracts.

## High-impact checks

- Match every non-`None` `in_axes` entry to an actual mapped dimension of the corresponding argument.
- Use `vmap` for State-instance declarations and `vmap2` for filter-based State-axis policies; do not mix their parameter names.
- Prefer an owning package's native batch or `size` axis when it already represents the independent conditions.
- Decide whether mutable State is shared or mapped; use `state_in_axes` and `state_out_axes` together for per-instance State write-back.
- Set `axis_size` when all inputs are static.
- Declare State axes or set an explicit policy when automatic output-axis inference would be ambiguous.
- Expect automatic independent `RandomState` keys; use the documented broadcast-key pattern only when shared randomness is intentional.
- Route direct key manipulation through the randomness reference rather than duplicating it here.
