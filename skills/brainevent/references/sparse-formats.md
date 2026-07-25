# BrainEvent sparse formats

Use this reference when explicit sparse connectivity must be assembled, converted, stored, or reoriented. Keep `CSR` as the single representative code path; choose other variations from the API table.

## Storage model

COO triplets are flexible construction records, while CSR and CSC compress the row or column dimension for event-driven products.

| API | Description |
|---|---|
| COO `row`, `col`, `data` triplets | Use when edges arrive as coordinate records; they describe nonzero locations and values but are not a standalone BrainEvent matrix class. |
| `coo2csr(row_ids, col_ids, *, shape)` | Use to convert COO indices to CSR; it returns `(indptr, indices, order)`, where `data[order]` produces row-compressed value order. |
| `coo_to_csc_index(pre_ids, indices, *, shape)` | Use to convert COO indices to CSC; it returns column pointers, row indices, and the value permutation required by the generated order. |
| `csr_to_coo_index(indptr, indices)` | Use to convert CSR indices back to COO; it returns `(row_ids, col_ids)`, with `col_ids` equal to the input column indices. |
| `csr_to_csc_index(csr_indptr, csr_indices, *, shape, include_perm=True, ...)` | Use to convert an existing CSR topology to CSC; it returns CSC structure and, by default, the permutation for the stored values. |
| `CSR(data, indices=None, indptr=None, *, shape, ...)` | Use for explicit row-oriented sparse connectivity and common forward `BinaryArray @ connectivity`; it stores values, column indices, and row pointers. |
| `CSC(data, indices=None, indptr=None, *, shape, ...)` | Use for explicit column-oriented connectivity, transpose-centered products, or column access; it stores values, row indices, and column pointers. |

```python
import brainevent
import jax.numpy as jnp

# Build a 4 x 5 sparse matrix from COO triplets:
# (0, 1) = 1.5, (1, 3) = 2.0, (2, 0) = 0.5, (3, 4) = 3.0.
shape = (4, 5)
row = jnp.array([0, 1, 2, 3])
col = jnp.array([1, 3, 0, 4])
data = jnp.array([1.5, 2.0, 0.5, 3.0])

indptr, indices, order = brainevent.coo2csr(row, col, shape=shape)
connectivity = brainevent.CSR(
    (data[order], indices, indptr),
    shape=shape,
)

print("shape:", connectivity.shape)
print("stored values:", connectivity.nse)
print("dense form:\n", connectivity.todense())

spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ connectivity

assert postsynaptic_input.shape == (5,)
assert jnp.allclose(
    postsynaptic_input,
    jnp.array([0.5, 1.5, 0.0, 0.0, 0.0]),
)
```

This example mirrors the official COO-to-CSR workflow and constructs only `CSR`. Keep `data[order]`: the conversion sorts the index structure and returns the value permutation separately. For CSC, use the matching conversion utility and apply its returned permutation.

## Selection rules

| Format | Use when | Critical invariant |
|---|---|---|
| COO triplets | Building, importing, or editing an edge list. | Convert before multiplication; keep values aligned with the returned permutation. |
| `CSR` | The contraction is row-oriented, especially presynaptic `BinaryArray @ connectivity` forward propagation. | `indptr` has one boundary per row plus the terminal boundary; `indices` are column indices. |
| `CSC` | The contraction or access is column-oriented or transpose-centered. | `indptr` has one boundary per column plus the terminal boundary; `indices` are row indices. |

Prefer CSR with `BinaryArray` for the common forward spiking-network path. Benchmark CSR and CSC when the actual contraction pattern makes orientation performance uncertain.

## Application: two-layer sparse spiking network

The official practice network constructs two reusable CSR layers, thresholds each weighted event product into the next `BinaryArray`, and evaluates firing statistics across multiple sparse inputs.

```python
import brainevent
import brainstate
import jax.numpy as jnp
import numpy as np


def dense_to_csr(weights):
    row, col = jnp.where(weights != 0)
    data = weights[row, col]
    indptr, indices, order = brainevent.coo2csr(
        row,
        col,
        shape=weights.shape,
    )
    return brainevent.CSR(
        (data[order], indices, indptr),
        shape=weights.shape,
    )


class SparseSpikingNetwork:
    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        connection_probability=0.1,
        seed=0,
    ):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        brainstate.random.seed(seed)
        mask1 = brainstate.random.bernoulli(
            connection_probability,
            size=(n_input, n_hidden),
        )
        weights1 = (
            brainstate.random.normal(size=(n_input, n_hidden))
            * 0.1
            * mask1
        )
        self.w1 = dense_to_csr(weights1)

        mask2 = brainstate.random.bernoulli(
            connection_probability,
            size=(n_hidden, n_output),
        )
        weights2 = (
            brainstate.random.normal(size=(n_hidden, n_output))
            * 0.1
            * mask2
        )
        self.w2 = dense_to_csr(weights2)

    def forward(
        self,
        input_spikes,
        hidden_threshold=0.3,
        output_threshold=0.5,
    ):
        hidden_input = input_spikes @ self.w1
        hidden_spikes = brainevent.BinaryArray(
            hidden_input > hidden_threshold
        )
        output_input = hidden_spikes @ self.w2
        output_spikes = brainevent.BinaryArray(
            output_input > output_threshold
        )
        return output_input, output_spikes, hidden_spikes


network = SparseSpikingNetwork(
    n_input=500,
    n_hidden=200,
    n_output=10,
    connection_probability=0.15,
    seed=42,
)

input_counts = []
hidden_counts = []
output_counts = []

brainstate.random.seed(999)
for _ in range(100):
    input_spikes = brainevent.BinaryArray(
        brainstate.random.bernoulli(0.1, size=(500,))
    )
    output, output_spikes, hidden_spikes = network.forward(input_spikes)

    input_counts.append(int(jnp.sum(input_spikes.value)))
    hidden_counts.append(int(jnp.sum(hidden_spikes.value)))
    output_counts.append(int(jnp.sum(output_spikes.value)))

assert output.shape == (10,)
print("layer connections:", network.w1.nse, network.w2.nse)
print(
    "average active neurons:",
    np.mean(input_counts),
    np.mean(hidden_counts),
    np.mean(output_counts),
)
```

This adapts section 6, “Practice: Building a sparse connection neural network.” Construct CSR weights once and reuse them across samples; only event vectors change during forward propagation.

## Sources

- Tutorial 2, Sparse Data Structures - CSR and CSC: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/02_sparse_matrices.html
- Sparse Matrix Data Structures API: https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html
- Utility Functions API: https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html
