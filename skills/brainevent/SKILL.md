---
name: brainevent
description: Use when representing binary spike events with BrainEvent, multiplying BinaryArray values through dense, CSR, CSC, JIT-generated, or fixed-degree connectivity, choosing among BrainEvent connectivity formats, applying event-driven plasticity, or routing custom operator work.
---

## Purpose and boundary

Use BrainEvent to represent binary spikes and communicate them through dense, explicit sparse, generated, or fixed-degree connectivity. Route neuron, synapse, channel, state, and neural-mass dynamics to their owning BrainX packages; never use `BinaryArray` for analog activity or BrainEvent as a complete simulator.

## Underlying principle of BrainEvent

`BinaryArray` marks boolean or 0/1 data for event-driven `spikes @ connectivity`, which processes active events through the selected dense-value, explicit-edge, generated, or fixed-degree representation; require `spikes.shape[-1] == connectivity.shape[0]`.

## API structure overview

| API family | Responsibility |
|---|---|
| `BinaryArray` | Mark boolean or 0/1 vectors and matrices for event-driven multiplication. |
| Dense JAX/NumPy arrays | Store every connection weight for small or genuinely dense matrices. |
| COO triplets, `CSR`, `CSC` | Build and store explicit sparse connectivity. |
| `JITCScalar*`, `JITCNormal*`, `JITCUniform*` | Regenerate random connectivity from distribution parameters and a seed instead of materializing edges. |
| `FixedNumPerPre`, `FixedNumPerPost` | Encode a fixed fan-out or fan-in directly; recognize `FixedPostNumConn` and `FixedPreNumConn` as deprecated aliases. |
| `update_*_on_binary_pre`, `update_*_on_binary_post` | Update stored CSR or dense weights from spike-triggered plasticity rules. |
| Custom-operator APIs | Extend BrainEvent with Numba, Warp, C++, or CUDA kernels when built-ins are insufficient. |

## Choose a connectivity representation

Choose the representation before constructing it; every family supports the same `BinaryArray @ connectivity` call site but has a different storage and mutation contract.

| Representation | Use when | Avoid when |
|---|---|---|
| Dense JAX/NumPy array | The matrix is small or genuinely dense, roughly more than 25% nonzero, or arbitrary per-edge values require the simplest storage. | A large matrix is mostly zero. |
| COO triplets -> `CSR` or `CSC` | Edges are explicit, fixed, reusable, inspectable, or mutable. Use CSR for row-oriented work and CSC for column-oriented work. | Random connectivity is too large to materialize. |
| `JITC*` | Connectivity is random with fixed probability and must be regenerated from compact parameters and a stable seed. | Individual edges must be inspected, mutated, or learned. |
| `FixedNumPerPre` / `FixedNumPerPost` | Each neuron has a fixed number of outputs or inputs and that topology should be encoded directly. | Connection counts vary per neuron. |

Use explicit `CSR` or `CSC` for stored edges, `JITC*` for random and huge connectivity, and fixed-degree structures for constant fan-in or fan-out.

## Represent and multiply binary events

`BinaryArray` wraps boolean or 0/1 event data and preserves a uniform multiplication interface across connectivity representations.

| API | Description |
|---|---|
| `BinaryArray(value)` | Use at the binary event boundary; it wraps a boolean or 0/1 vector or matrix, dispatches `@` to event-driven operations, and returns an event representation compatible with JAX transformations. |
| `spikes @ connectivity` | Use after the spike dimension and connectivity input dimension agree; it selects the implementation for the connectivity representation and returns the weighted postsynaptic input. |

```python
import brainevent
import jax.numpy as jnp

spikes = brainevent.BinaryArray([1, 0, 1, 0])
connectivity = jnp.array([
    [0.5, 0.1],
    [0.2, 0.4],
    [0.3, 0.7],
    [0.6, 0.2],
])
postsynaptic_input = spikes @ connectivity

assert postsynaptic_input.shape == (2,)
```

Use a dense array only when its storage cost is acceptable. Open `references/sparse-formats.md` when explicit edges should be compressed instead.

## Build explicit sparse connectivity

COO describes edges for construction, while `CSR` and `CSC` store the finished explicit matrix for row- or column-oriented operations.

| API | Description |
|---|---|
| COO `row`, `col`, `data` triplets | Use while assembling sparse edges from coordinate records; BrainEvent has no standalone COO matrix class, so convert the indices before multiplication. |
| `coo2csr(row_ids, col_ids, *, shape)` | Use to convert COO indices to CSR; it returns `(indptr, indices, order)`, and `data[order]` aligns values with the compressed row order. |
| `coo_to_csc_index(pre_ids, indices, *, shape)` | Use to convert COO indices directly to CSC index arrays; use the returned permutation to align stored values. |
| `csr_to_csc_index(csr_indptr, csr_indices, *, shape, ...)` | Use when an existing CSR topology must become column-oriented; it returns CSC indices and, by default, the value permutation. |
| `CSR(data, indices=None, indptr=None, *, shape, ...)` | Use for explicit row-oriented sparse connectivity and the normal forward `BinaryArray @ connectivity` path; it stores nonzero values, column indices, and row pointers. |
| `CSC(data, indices=None, indptr=None, *, shape, ...)` | Use for explicit column-oriented sparse connectivity or transpose-centered work; it stores nonzero values, row indices, and column pointers. |

```python
import brainevent
import jax.numpy as jnp

connectivity = brainevent.CSR(
    (
        jnp.array([0.5, 0.2, 0.7, 0.4]),
        jnp.array([0, 1, 0, 1]),
        jnp.array([0, 1, 2, 3, 4]),
    ),
    shape=(4, 2),
)
spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ connectivity

assert postsynaptic_input.shape == (2,)
```

This example constructs only `CSR`; do not duplicate it for every format. Open `references/sparse-formats.md` when edges begin as COO, CSC orientation is required, or formats must be converted.

## Generate random connectivity

JITC matrices store a probability, weight-distribution parameters, and a seed, then regenerate the required connections during computation instead of storing individual edges.

| API | Description |
|---|---|
| `JITCScalarR(weight, prob=None, seed=None, *, shape, ...)` | Use for row-oriented connectivity with one shared nonzero weight; it regenerates a reproducible graph from `prob` and `seed`. |
| `JITCScalarC(weight, prob=None, seed=None, *, shape, ...)` | Use for the column-oriented form of shared-weight generated connectivity. |
| `JITCNormalR(loc, scale=None, prob=None, seed=None, *, shape, ...)` | Use for row-oriented generated weights drawn from a normal distribution. |
| `JITCNormalC(loc, scale=None, prob=None, seed=None, *, shape, ...)` | Use for the column-oriented normal-weight form. |
| `JITCUniformR(low, high=None, prob=None, seed=None, *, shape, ...)` | Use for row-oriented generated weights bounded by a uniform distribution. |
| `JITCUniformC(low, high=None, prob=None, seed=None, *, shape, ...)` | Use for the column-oriented uniform-weight form. |

```python
import brainevent
import jax.numpy as jnp

n_pre = 100_000
n_post = 100_000
connectivity = brainevent.JITCScalarR(
    (0.5, 0.01, 7),
    shape=(n_pre, n_post),
)
spikes = brainevent.BinaryArray(
    jnp.zeros(n_pre, dtype=bool).at[::1000].set(True)
)
postsynaptic_input = spikes @ connectivity

assert postsynaptic_input.shape == (n_post,)
```

This example constructs only `JITCScalarR`. Keep the seed stable when the realized graph must remain reproducible. Open `references/connectivity-variants.md` when choosing a weight distribution, row/column orientation, or benchmarking an uncertain contraction.

## Encode fixed-degree connectivity

Fixed-degree structures store one connection count per relevant neuron population, so choose the class by whether the invariant is fan-in or fan-out.

| API | Description |
|---|---|
| `FixedNumPerPre(data, indices=None, *, shape, ...)` | Use when every presynaptic neuron has the same number of outputs; it stores data and target indices with shape `(num_pre, connections_per_pre)`. |
| `FixedNumPerPost(data, indices=None, *, shape, ...)` | Use when every postsynaptic neuron receives the same number of inputs; it stores data and source indices with shape `(num_post, connections_per_post)`. |
| `FixedPostNumConn` | Recognize as the deprecated alias of `FixedNumPerPre`; migrate new code to the current name. |
| `FixedPreNumConn` | Recognize as the deprecated alias of `FixedNumPerPost`; migrate new code to the current name. |

```python
import brainevent
import jax.numpy as jnp

connectivity = brainevent.FixedNumPerPre(
    (
        jnp.array([
            [0.5, 0.2],
            [0.4, 0.1],
            [0.3, 0.6],
            [0.7, 0.2],
        ]),
        jnp.array([
            [0, 2],
            [1, 2],
            [0, 1],
            [1, 2],
        ]),
    ),
    shape=(4, 3),
)
spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ connectivity

assert postsynaptic_input.shape == (3,)
```

This example constructs only `FixedNumPerPre`. Open `references/connectivity-variants.md` when fixed fan-in is required, a deprecated alias appears in existing code, or the stored index shape must be checked.

## Transform and verify the product

Keep the complete event-driven product inside the JAX transform, then verify shape, orientation, and reproducibility rather than relying on a class suffix alone.

| API | Description |
|---|---|
| `jax.jit(function)` | Use to compile a complete event-driven product; it traces `BinaryArray` and connectivity PyTrees and returns a compiled callable for compatible shapes. |
| `jax.vmap(function, ...)` | Use to batch independent event-driven products; it maps the same communication rule over the selected array axis. |
| `brainevent.benchmark_function(function, ...)` | Use when row- versus column-oriented performance is uncertain; it benchmarks the actual workload and returns timing statistics. |

```python
import jax

@jax.jit
def communicate(spikes, connectivity):
    return spikes @ connectivity

postsynaptic_input = communicate(spikes, connectivity)
```

Verify `postsynaptic_input.shape == connectivity.shape[1:]` for vector input and confirm that repeated JITC runs with the same seed reproduce the intended graph.

## Reference routing

| Reference | Open when |
|---|---|
| `references/sparse-formats.md` | Open when explicit connectivity begins as COO, requires CSR/CSC conversion, or needs row/column storage decisions; it contains the exact construction and conversion APIs and storage invariants. |
| `references/connectivity-variants.md` | Open when choosing among all six JITC distribution/orientation variants or between fixed fan-in and fan-out; it contains constructor semantics, index shapes, deprecated alias mapping, seed rules, and benchmarking guidance. |
| `references/synaptic-plasticity.md` | Open when pre- or postsynaptic events must update stored CSR or dense weights; it contains all four public update variants and one CSR STDP pattern. |
| `references/custom-operators.md` | Open when built-in operations are insufficient; it routes Numba, Numba CUDA, Warp, C++, and CUDA extension paths to the exact official tutorial. |
| `references/scripts/102_EI_net_1996.py` | Open for a complete high-level E/I network already using `brainpy.state.AlignPostProj` and `brainstate.nn.EventFixedProb`; it preserves unit-aware weights, initialization, compiled time loops, and visualization. |
| `references/scripts/204_joglekar_2018_propagation.py` | Open for delayed spikes, area mapping, and vmapped `JITCScalarC` communication; it preserves delays, seeds, external-data assumptions, and BrainPy compatibility details. |

## Boundaries and common failures

- Wrap only boolean or 0/1 events in `BinaryArray`; keep continuous values as ordinary arrays or BrainUnit quantities.
- Match the spike vector length to the left connectivity dimension before compiling.
- Do not materialize a huge random matrix when a JITC generation rule is sufficient.
- Do not use JITC when individual connection weights must be inspected, mutated, or learned; use stored `CSR`, `CSC`, or dense weights.
- Treat COO as construction input, not as a BrainEvent matrix class.
- Apply the `order` or permutation returned by index-conversion utilities to the corresponding data values.
- Do not reverse fixed fan-in and fan-out: `FixedNumPerPre` fixes outputs per presynaptic neuron; `FixedNumPerPost` fixes inputs per postsynaptic neuron.
- Migrate deprecated `FixedPostNumConn` to `FixedNumPerPre` and deprecated `FixedPreNumConn` to `FixedNumPerPost`.
- Do not infer the fastest orientation from `R` or `C` alone; benchmark the actual contraction.
- Route spike-triggered weight mutation to `references/synaptic-plasticity.md` and custom kernels to `references/custom-operators.md`.
