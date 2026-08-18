# Legacy BrainPy connectivity

Use this reference to choose and materialize explicit legacy `brainpy.connect` connectivity, convert among dense/COO/CSR forms, or perform transmission with an implicit random matrix through `brainpy.math.jitconn`. Keep synapse dynamics and projection assembly in `synaptic projections.md` or `customize neuron and synpase.md`.

## Choose explicit or implicit connectivity

Explicit connectors define a reusable wiring object whose representations can be inspected; JIT connectivity generates a random matrix during each matrix-vector operation and does not store the connection graph.

| Need | Use | Constraint |
|---|---|---|
| Inspect edges, degrees, or a connection matrix | `brainpy.connect` | Materialize the required representation with `.require(...)`. |
| Reuse the exact graph across different operations | `brainpy.connect` | Keep the connector or extracted indices. |
| Apply sparse or event-driven transmission over stored CSR edges | `bm.sparse.csrmv` or `bm.event.csrmv` | Supply `indices`, `indptr`, and values that share one orientation. |
| Avoid storing a large random matrix | `bm.jitconn.*` | Supply shape, distribution parameters, probability, and an explicit integer seed. |

Do not use `jitconn` when downstream analysis must recover individual edges or modify a specific synapse.

## Build explicit connectivity

Call a connector with `(pre_size, post_size)` unless those sizes were supplied as `pre=` and `post=` constructor arguments.

### Construct from known edges

| API | Description |
|---|---|
| `bp.connect.MatConn(conn_mat)` | Use an existing dense connection matrix. |
| `bp.connect.IJConn(i, j)` | Use paired presynaptic and postsynaptic edge indices. |
| `bp.connect.CSRConn(indices, indptr)` | Use an existing CSR representation. |
| `bp.connect.SparseMatConn(sparse_mat)` | Use an existing supported sparse matrix. |

### Generate random graphs

| API | Description |
|---|---|
| `bp.connect.FixedProb(prob, pre_ratio=1.0, include_self=True, allow_multi_conn=False, seed=None)` | Connect possible pairs independently with fixed probability. |
| `bp.connect.FixedPreNum(num, ...)` | Give each postsynaptic neuron a fixed number of presynaptic inputs. |
| `bp.connect.FixedPostNum(num, ...)` | Give each presynaptic neuron a fixed number of postsynaptic targets. |
| `bp.connect.FixedTotalNum(num, ...)` | Generate a fixed total number of connections. |
| `bp.connect.GaussianProb(sigma, ...)` | Make connection probability decay with Gaussian distance inside one population. |
| `bp.connect.ProbDist(dist, prob, ...)` | Connect pairs within a maximum distance at a specified probability. |
| `bp.connect.SmallWorld(...)` | Build a Watts-Strogatz small-world graph. |
| `bp.connect.ScaleFreeBA(...)` | Build a Barabasi-Albert preferential-attachment graph. |
| `bp.connect.ScaleFreeBADual(...)` | Build the dual Barabasi-Albert variant. |
| `bp.connect.PowerLaw(...)` | Build the Holme-Kim power-law graph with clustering. |

### Generate regular graphs

| API | Description |
|---|---|
| `bp.connect.One2One()` | Connect equal-size groups by matching indices. |
| `bp.connect.All2All(include_self=True)` | Connect every presynaptic neuron to every postsynaptic neuron. |
| `bp.connect.GridFour(...)` | Connect four nearest grid neighbors. |
| `bp.connect.GridEight(...)` | Connect eight nearest grid neighbors. |
| `bp.connect.GridN(N, ...)` | Connect the neighborhood defined by radius `N`. |

```python
import brainpy as bp
import brainpy.math as bm

pre_num, post_num = 4, 3
connector = bp.connect.FixedProb(prob=0.5, seed=123)(pre_num, post_num)

conn_mat = connector.require('conn_mat')
indices, indptr = connector.require('pre2post')

assert conn_mat.shape == (pre_num, post_num)
assert indptr.shape == (pre_num + 1,)
assert indices.ndim == 1
```

**Invariant:** Preserve representation orientation. In `pre2post`, `indptr` partitions edges by presynaptic neuron and `indices` stores postsynaptic indices.

## Convert connection representations

Use conversion helpers when source data and the selected communication operator require different formats.

| API | Description |
|---|---|
| `bp.connect.mat2coo(mat)` | Convert a dense matrix to paired COO indices. |
| `bp.connect.mat2csr(mat)` | Convert a dense matrix to CSR `indices` and `indptr`. |
| `bp.connect.mat2csc(mat)` | Convert a dense matrix to CSC. |
| `bp.connect.coo2csr(pre_ids, post_ids, ...)` | Group COO edges into presynaptic-oriented CSR. |
| `bp.connect.coo2csc(pre_ids, post_ids, ...)` | Group COO edges into postsynaptic-oriented CSC. |
| `bp.connect.coo2mat(pre_ids, post_ids, ...)` | Materialize a dense matrix from COO edges. |
| `bp.connect.csr2coo(indices, indptr)` | Expand CSR into paired edge indices. |
| `bp.connect.csr2csc(indices, indptr, ...)` | Change sparse orientation. |
| `bp.connect.csr2mat(indices, indptr, ...)` | Materialize a dense matrix from CSR. |

Keep edge-aligned weights in the same order as their sparse indices. If a conversion returns a permutation, apply it to the weight vector before transmission.

## Transmit through stored sparse connectivity

Use sparse matrix-vector multiplication for graded presynaptic activity and event-driven multiplication when the presynaptic vector is boolean or spike-like.

| API | Description |
|---|---|
| `bm.sparse.csrmv(data, indices, indptr, vector, shape, transpose=...)` | Multiply a CSR matrix by a dense graded vector. |
| `bm.event.csrmv(data, indices, indptr, events, shape, transpose=...)` | Skip inactive presynaptic events during CSR transmission. |

```python
weights = bm.ones(indices.shape)
pre_activity = bm.asarray([1.0, 0.0, 2.0, 0.0])
pre_spikes = bm.asarray([True, False, True, False])

graded_post = bm.sparse.csrmv(
    weights,
    indices=indices,
    indptr=indptr,
    vector=pre_activity,
    shape=(pre_num, post_num),
    transpose=True,
)
event_post = bm.event.csrmv(
    weights,
    indices=indices,
    indptr=indptr,
    events=pre_spikes,
    shape=(pre_num, post_num),
    transpose=True,
)

assert graded_post.shape == event_post.shape == (post_num,)
```

Use the event operator only when zero/nonzero or boolean event semantics match the presynaptic signal. Do not convert continuous rates to booleans merely to select the event kernel.

## Generate connectivity during transmission

The six `jitconn` operators combine random connection generation with matrix-vector multiplication.

| API | Weight model and input |
|---|---|
| `bm.jitconn.mv_prob_homo(vector, weight, conn_prob, seed, *, shape, ...)` | Graded vector; one scalar weight on every generated edge. |
| `bm.jitconn.mv_prob_uniform(vector, w_low, w_high, conn_prob, seed, *, shape, ...)` | Graded vector; edge weights sampled uniformly. |
| `bm.jitconn.mv_prob_normal(vector, w_mu, w_sigma, conn_prob, seed, *, shape, ...)` | Graded vector; edge weights sampled normally. |
| `bm.jitconn.event_mv_prob_homo(events, weight, conn_prob, seed, *, shape, ...)` | Event vector; one scalar weight on every generated edge. |
| `bm.jitconn.event_mv_prob_uniform(events, w_low, w_high, conn_prob, seed, *, shape, ...)` | Event vector; edge weights sampled uniformly. |
| `bm.jitconn.event_mv_prob_normal(events, w_mu, w_sigma, conn_prob, seed, *, shape, ...)` | Event vector; edge weights sampled normally. |

```python
post = bm.jitconn.event_mv_prob_normal(
    pre_spikes,
    w_mu=0.5,
    w_sigma=0.1,
    conn_prob=0.2,
    seed=123,
    shape=(pre_num, post_num),
    transpose=True,
)

assert post.shape == (post_num,)
```

Always pass an explicit integer `seed`. With `seed=None`, the implementation draws a host NumPy seed on each eager call, but a JAX trace captures that seed as a Python constant and reuses it on subsequent jitted calls.

Do not assume separate calls with `transpose=False` and `transpose=True` sample an identical matrix/transposed-matrix pair. The API documents `outdim_parallel=True` as the setting for that equivalence, with a possible speed cost; validate equivalence for the exact operator before depending on it.

## Common failures

- Do not materialize `conn_mat` for a large sparse graph unless an operation truly needs the dense matrix.
- Do not interpret `pre2post` indices as postsynaptic-oriented CSC data.
- Do not separate edge weights from the index order that defines them.
- Do not use an event kernel for graded activity.
- Do not omit the `jitconn` seed inside `jit`, `vmap`, or repeated experiment code.
- Do not use implicit JIT connectivity when exact edge inspection, plasticity, or edge-specific modification is required.

## Routing

Open `route activity through connectivity.md` for communication-layer selection and weighted transmission. Open `synaptic projections.md` for projection anatomy and update order. Open `synpase properties.md` for synaptic state and output semantics.

## Sources mirrored

- https://brainpy.readthedocs.io/apis/connect.html
- https://brainpy.readthedocs.io/tutorial_math/Dedicated_Operators.html
- https://brainpy.readthedocs.io/apis/brainpy.math.jitconn.html
