# BrainEvent connectivity variants

Use this reference after choosing generated or fixed-degree connectivity in the root skill. Select the exact API from the tables; each code block constructs only one representative variation.

## JITC distribution and orientation variants

JITC classes regenerate sparse connectivity from compact distribution parameters, probability, and seed; `R` is row-oriented and `C` is column-oriented.

| API | Description |
|---|---|
| `JITCScalarR(weight, prob=None, seed=None, *, shape, ...)` | Use for row-oriented connectivity when every realized edge has one shared weight. |
| `JITCScalarC(weight, prob=None, seed=None, *, shape, ...)` | Use for the column-oriented shared-weight variation. |
| `JITCNormalR(loc, scale=None, prob=None, seed=None, *, shape, ...)` | Use for row-oriented connectivity whose realized weights follow a normal distribution. |
| `JITCNormalC(loc, scale=None, prob=None, seed=None, *, shape, ...)` | Use for the column-oriented normal-distribution variation. |
| `JITCUniformR(low, high=None, prob=None, seed=None, *, shape, ...)` | Use for row-oriented connectivity whose realized weights stay within uniform bounds. |
| `JITCUniformC(low, high=None, prob=None, seed=None, *, shape, ...)` | Use for the column-oriented uniform-distribution variation. |

```python
import brainevent
import brainstate
import jax.numpy as jnp

n_pre = 1000
n_post = 500

# Shared weight, connection probability, and seed.
connectivity = brainevent.JITCScalarR(
    (0.1, 0.1, 12345),
    shape=(n_pre, n_post),
)
print("shape:", connectivity.shape)
print("weight:", connectivity.weight)
print("probability:", connectivity.prob)
print("seed:", connectivity.seed)

brainstate.random.seed(0)
spikes = brainevent.BinaryArray(
    brainstate.random.bernoulli(0.05, size=(n_pre,))
)
postsynaptic_input = spikes @ connectivity

print("active inputs:", jnp.sum(spikes.value))
print("output shape:", postsynaptic_input.shape)
print("nonzero outputs:", jnp.sum(postsynaptic_input > 0))

# A stable constructor seed reproduces the same generated graph.
first = brainevent.JITCScalarR(
    (0.1, 0.1, 999),
    shape=(100, 50),
)
replay = brainevent.JITCScalarR(
    (0.1, 0.1, 999),
    shape=(100, 50),
)
different = brainevent.JITCScalarR(
    (0.1, 0.1, 888),
    shape=(100, 50),
)
test_spikes = brainevent.BinaryArray(jnp.ones(100, dtype=bool))

first_result = test_spikes @ first
replay_result = test_spikes @ replay
different_result = test_spikes @ different

assert postsynaptic_input.shape == (n_post,)
assert jnp.allclose(first_result, replay_result)
print("different seed matches:", jnp.allclose(first_result, different_result))
```

This example mirrors the official homogeneous JITC construction, event multiplication, and seed-replay cells while constructing only `JITCScalarR`. Keep `seed` stable for reproducible connectivity. Do not use any JITC variation when individual edges must be inspected, persistently mutated, or learned.

## Application: ultra-large JITC network

The official practice network composes two generated layers whose storage remains a few distribution parameters even when their logical connection counts reach millions.

```python
import brainevent
import brainstate
import jax
import jax.numpy as jnp


class MassiveJITCNetwork:
    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        connection_probability=0.01,
        seed=0,
    ):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.w1 = brainevent.JITCNormalR(
            (0.0, 0.05, connection_probability, seed),
            shape=(n_input, n_hidden),
        )
        self.w2 = brainevent.JITCNormalR(
            (0.0, 0.1, connection_probability * 2, seed + 1),
            shape=(n_hidden, n_output),
        )

    def forward(self, input_spikes):
        hidden_input = input_spikes @ self.w1
        hidden_spikes = brainevent.BinaryArray(hidden_input > 0.2)
        output = hidden_spikes @ self.w2
        return output, hidden_spikes


network = MassiveJITCNetwork(
    n_input=100_000,
    n_hidden=50_000,
    n_output=1_000,
    connection_probability=0.01,
    seed=2024,
)

brainstate.random.seed(999)
input_spikes = brainevent.BinaryArray(
    brainstate.random.bernoulli(0.001, size=(100_000,))
)
output, hidden_spikes = network.forward(input_spikes)
output = jax.block_until_ready(output)

assert output.shape == (1_000,)
print("active inputs:", jnp.sum(input_spikes.value))
print("active hidden neurons:", jnp.sum(hidden_spikes.value))
print("active outputs:", jnp.sum(output > 0.5))
```

This adapts “Practice: Ultra-large-scale spiking neural network.” Use it only when generated weights are acceptable: neither layer exposes persistent individual edges for inspection or learning.

## Choose row or column orientation

Use the orientation that keeps the indexed contraction dimension contiguous; do not assume an `R` or `C` suffix alone determines the fastest implementation for every workload.

| API | Description |
|---|---|
| `*R` JITC classes | Use for the normal forward row-oriented event contraction, conceptually following CSR. |
| `*C` JITC classes | Use for column-oriented or transpose-centered contraction, conceptually following CSC. |
| `brainevent.benchmark_function()` | Use when orientation performance is uncertain; benchmark the complete representative product rather than construction alone. |

Keep the same shape, event pattern, distribution parameters, seed, backend, warmup, and trial count when comparing orientations.

## Fixed-degree variants

Fixed-degree structures encode whether a constant connection count belongs to each presynaptic or postsynaptic neuron.

| API | Description |
|---|---|
| `FixedNumPerPre(data, indices=None, *, shape, ...)` | Use for fixed fan-out: every presynaptic neuron has the same number of target indices, and `data` and `indices` use shape `(num_pre, connections_per_pre)`. |
| `FixedNumPerPost(data, indices=None, *, shape, ...)` | Use for fixed fan-in: every postsynaptic neuron has the same number of source indices, and `data` and `indices` use shape `(num_post, connections_per_post)`. |
| `FixedPostNumConn` | Recognize as the deprecated alias of `FixedNumPerPre`; replace it when maintaining older code. |
| `FixedPreNumConn` | Recognize as the deprecated alias of `FixedNumPerPost`; replace it when maintaining older code. |

```python
import brainevent
import brainstate
import jax.numpy as jnp

n_pre = 100
n_post = 50
connections_per_pre = 10

brainstate.random.seed(42)
indices = brainstate.random.randint(
    0,
    n_post,
    size=(n_pre, connections_per_pre),
)
weights = (
    brainstate.random.normal(size=(n_pre, connections_per_pre))
    * 0.1
)

connectivity = brainevent.FixedNumPerPre(
    (weights, indices),
    shape=(n_pre, n_post),
)

print("shape:", connectivity.shape)
print("weight shape:", connectivity.data.shape)
print("index shape:", connectivity.indices.shape)
print("neuron 0 targets:", connectivity.indices[0])

brainstate.random.seed(999)
spikes = brainevent.BinaryArray(
    brainstate.random.bernoulli(0.1, size=(n_pre,))
)
postsynaptic_input = spikes @ connectivity

print("active inputs:", jnp.sum(spikes.value))
print("output shape:", postsynaptic_input.shape)
print("nonzero outputs:", jnp.sum(postsynaptic_input != 0))

assert connectivity.data.shape == (n_pre, connections_per_pre)
assert postsynaptic_input.shape == (n_post,)
```

This example mirrors the official fixed fan-out construction and propagation workflow, replacing its deprecated `FixedPostNumConn` alias with current `FixedNumPerPre`. Reverse the storage orientation and index meaning when selecting `FixedNumPerPost`; do not reuse the fan-out array shape unchanged.

## Application: fixed-degree cortical E/I network

The official cortical practice composes fixed fan-out E-to-E, E-to-I, and I-to-E pathways, then repeatedly thresholds their event-driven inputs into new population spikes.

```python
import brainevent
import brainstate
import jax.numpy as jnp
import numpy as np


class CorticalNetwork:
    def __init__(
        self,
        n_exc=800,
        n_inh=200,
        exc_fanout=50,
        inh_fanout=30,
        seed=0,
    ):
        self.n_exc = n_exc
        self.n_inh = n_inh
        brainstate.random.seed(seed)

        ee_indices = brainstate.random.randint(
            0,
            n_exc,
            size=(n_exc, exc_fanout),
        )
        ee_weights = (
            brainstate.random.normal(size=(n_exc, exc_fanout))
            * 0.05
        )
        self.w_ee = brainevent.FixedNumPerPre(
            (ee_weights, ee_indices),
            shape=(n_exc, n_exc),
        )

        ei_indices = brainstate.random.randint(
            0,
            n_inh,
            size=(n_exc, exc_fanout // 2),
        )
        ei_weights = (
            brainstate.random.normal(size=(n_exc, exc_fanout // 2))
            * 0.08
        )
        self.w_ei = brainevent.FixedNumPerPre(
            (ei_weights, ei_indices),
            shape=(n_exc, n_inh),
        )

        ie_indices = brainstate.random.randint(
            0,
            n_exc,
            size=(n_inh, inh_fanout),
        )
        ie_weights = (
            -jnp.abs(
                brainstate.random.normal(size=(n_inh, inh_fanout))
            )
            * 0.15
        )
        self.w_ie = brainevent.FixedNumPerPre(
            (ie_weights, ie_indices),
            shape=(n_inh, n_exc),
        )

    def forward(self, exc_spikes, inh_spikes):
        exc_input = (
            exc_spikes @ self.w_ee
            + inh_spikes @ self.w_ie
        )
        inh_input = exc_spikes @ self.w_ei
        return exc_input, inh_input


network = CorticalNetwork(seed=2024)
brainstate.random.seed(0)
exc_spikes = brainevent.BinaryArray(
    brainstate.random.bernoulli(0.1, size=(800,))
)
inh_spikes = brainevent.BinaryArray(jnp.zeros(200, dtype=bool))

exc_counts = []
inh_counts = []
for _ in range(100):
    exc_input, inh_input = network.forward(exc_spikes, inh_spikes)
    external_input = brainstate.random.normal(size=(800,)) * 0.3
    exc_spikes = brainevent.BinaryArray(
        exc_input + external_input > 0.3
    )
    inh_spikes = brainevent.BinaryArray(inh_input > 0.25)
    exc_counts.append(int(jnp.sum(exc_spikes.value)))
    inh_counts.append(int(jnp.sum(inh_spikes.value)))

assert exc_spikes.value.shape == (800,)
assert inh_spikes.value.shape == (200,)
print("mean E/I spikes:", np.mean(exc_counts), np.mean(inh_counts))
```

This adapts “Practice: Biologically realistic cortical network,” replaces deprecated fixed-degree aliases, removes plotting, and makes inhibitory weights strictly non-positive. Treat the threshold loop as the tutorial’s connectivity demonstration, not as a complete biophysical neuron model.

## Sources

- Tutorial 3, JIT Connection Matrices: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/03_jit_connectivity.html
- Tutorial 4, Fixed Connection Count Structures: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/04_fixed_connections.html
- Choose a connectivity format: https://brainx.chaobrain.com/brainevent/how-to/data-structures/choosing-a-sparse-format.html
- Sparse Matrix Data Structures API: https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html
- Utility Functions API: https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html
