# BrainEvent synaptic plasticity

Use this reference when binary pre- or postsynaptic events must modify stored synaptic weights. Plasticity is an overlay on the core `BinaryArray @ connectivity` workflow, not a prerequisite for ordinary event-driven communication.

BrainEvent's event-driven plasticity operators touch only weights connected to neurons that fired, following the same event-sparsity principle as its matrix products.

## Choose the update operator

| Weight storage | Presynaptic event triggers update | Postsynaptic event triggers update | Use when |
|---|---|---|---|
| CSR | `update_csr_on_binary_pre()` | `update_csr_on_binary_post()` | Connectivity is sparse and fixed; only stored synapses should be visited |
| Dense | `update_dense_on_binary_pre()` | `update_dense_on_binary_post()` | A small fully connected layer already uses a dense weight matrix |

Use the `*_on_binary_pre` direction when the source neuron firing triggers the rule. Use the `*_on_binary_post` direction when the target neuron firing triggers it.

## Canonical CSR STDP overlay

The official tutorial maintains exponentially decaying pre/post traces, applies a spike-triggered update to the CSR `data`, and rebuilds the CSR object while preserving its structural arrays.

```python
import brainevent
import brainstate
import jax.numpy as jnp
import numpy as np

n_pre = 100
n_post = 50
connection_probability = 0.1

# Create explicit sparse weights and convert their COO indices to CSR.
brainstate.random.seed(42)
mask = brainstate.random.bernoulli(
    connection_probability,
    size=(n_pre, n_post),
)
weights_dense = (
    brainstate.random.uniform(0.0, 0.5, size=(n_pre, n_post))
    * mask
)
row, col = jnp.where(weights_dense != 0)
data = weights_dense[row, col]
indptr, indices, order = brainevent.coo2csr(
    row,
    col,
    shape=(n_pre, n_post),
)
csr_weights = brainevent.CSR(
    (data[order], indices, indptr),
    shape=(n_pre, n_post),
)

# Initialize traces and the official presynaptic-triggered STDP parameters.
pre_trace = jnp.zeros(n_pre)
post_trace = jnp.zeros(n_post)
decay_pre = np.exp(-1.0 / 20.0)
decay_post = np.exp(-1.0 / 20.0)
A_plus = 0.005
initial_mean = csr_weights.data.mean()

brainstate.random.seed(100)
for _ in range(500):
    pre_spike = brainstate.random.bernoulli(0.05, size=(n_pre,))
    post_spike = brainstate.random.bernoulli(0.05, size=(n_post,))

    pre_trace = (
        pre_trace * decay_pre
        + pre_spike.astype(jnp.float32)
    )
    post_trace = (
        post_trace * decay_post
        + post_spike.astype(jnp.float32)
    )

    new_data = brainevent.update_csr_on_binary_pre(
        weight=csr_weights.data,
        indices=csr_weights.indices,
        indptr=csr_weights.indptr,
        pre_spike=pre_spike,
        post_trace=post_trace * A_plus,
        w_min=0.0,
        w_max=1.0,
        shape=csr_weights.shape,
    )
    csr_weights = brainevent.CSR(
        (new_data, csr_weights.indices, csr_weights.indptr),
        shape=csr_weights.shape,
    )

print("connections:", csr_weights.nse)
print("initial mean weight:", initial_mean)
print("final mean weight:", csr_weights.data.mean())
print(
    "final range:",
    csr_weights.data.min(),
    csr_weights.data.max(),
)
```

This mirrors the official sparse-network construction and learning loop while using only the presynaptic CSR update API. Preserve `indices`, `indptr`, and `shape`; the operator updates weight values, not the sparse topology.

For a bidirectional STDP rule, combine the corresponding pre- and post-triggered operators with the appropriate traces and signs from the learning rule. Do not silently substitute the pre-triggered operator for a post-triggered update.

## Application: adaptive two-layer spiking network

The official self-learning practice stores two CSR layers, propagates binary events through both, maintains decaying traces, and applies presynaptic-triggered updates after each sample.

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


class AdaptiveSpikingNetwork:
    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        connection_probability=0.15,
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
            brainstate.random.uniform(
                0.0,
                0.3,
                size=(n_input, n_hidden),
            )
            * mask1
        )
        self.w1 = dense_to_csr(weights1)

        mask2 = brainstate.random.bernoulli(
            connection_probability,
            size=(n_hidden, n_output),
        )
        weights2 = (
            brainstate.random.uniform(
                0.0,
                0.3,
                size=(n_hidden, n_output),
            )
            * mask2
        )
        self.w2 = dense_to_csr(weights2)

        self.input_trace = jnp.zeros(n_input)
        self.hidden_trace = jnp.zeros(n_hidden)
        self.output_trace = jnp.zeros(n_output)
        self.decay = np.exp(-1.0 / 20.0)
        self.learning_rate = 0.008

    def forward(self, input_spikes, learning=True):
        hidden_input = input_spikes @ self.w1
        hidden_spikes = brainevent.BinaryArray(hidden_input > 0.5)
        output_input = hidden_spikes @ self.w2
        output_spikes = brainevent.BinaryArray(output_input > 0.8)

        input_events = input_spikes.value
        hidden_events = hidden_spikes.value
        output_events = output_spikes.value
        self.input_trace = (
            self.input_trace * self.decay
            + input_events.astype(jnp.float32)
        )
        self.hidden_trace = (
            self.hidden_trace * self.decay
            + hidden_events.astype(jnp.float32)
        )
        self.output_trace = (
            self.output_trace * self.decay
            + output_events.astype(jnp.float32)
        )

        if learning:
            new_w1 = brainevent.update_csr_on_binary_pre(
                weight=self.w1.data,
                indices=self.w1.indices,
                indptr=self.w1.indptr,
                pre_spike=input_events,
                post_trace=self.hidden_trace * self.learning_rate,
                w_min=0.0,
                w_max=1.0,
                shape=self.w1.shape,
            )
            self.w1 = brainevent.CSR(
                (new_w1, self.w1.indices, self.w1.indptr),
                shape=self.w1.shape,
            )

            new_w2 = brainevent.update_csr_on_binary_pre(
                weight=self.w2.data,
                indices=self.w2.indices,
                indptr=self.w2.indptr,
                pre_spike=hidden_events,
                post_trace=self.output_trace * self.learning_rate,
                w_min=0.0,
                w_max=1.0,
                shape=self.w2.shape,
            )
            self.w2 = brainevent.CSR(
                (new_w2, self.w2.indices, self.w2.indptr),
                shape=self.w2.shape,
            )

        return output_input, hidden_spikes, output_spikes


network = AdaptiveSpikingNetwork(
    n_input=200,
    n_hidden=100,
    n_output=10,
    connection_probability=0.12,
    seed=2024,
)
initial_w1 = network.w1.data.mean()
initial_w2 = network.w2.data.mean()
activity = []

brainstate.random.seed(0)
for _ in range(50):
    epoch_output_count = 0
    for _ in range(20):
        input_spikes = brainevent.BinaryArray(
            brainstate.random.bernoulli(0.1, size=(200,))
        )
        output, _, output_spikes = network.forward(
            input_spikes,
            learning=True,
        )
        epoch_output_count += int(jnp.sum(output_spikes.value))
    activity.append(epoch_output_count / 20)

assert output.shape == (10,)
assert network.w1.data.mean() >= initial_w1
assert network.w2.data.mean() >= initial_w2
print("W1 mean:", initial_w1, "->", network.w1.data.mean())
print("W2 mean:", initial_w2, "->", network.w2.data.mean())
print("final output activity:", activity[-1])
```

This adapts “Practice: Building a Self-Learning Neural Network,” replaces manual CSR row assembly with `coo2csr()`, and removes visualization-only history. It demonstrates the tutorial’s potentiation path; add the post-triggered operator when the learning rule also requires depression.

## Storage boundary

- Prefer CSR plasticity for large sparse networks with fixed connectivity.
- Prefer dense plasticity for small fully connected layers already represented densely.
- Do not choose JITC when individual connection weights must be persistently updated; use stored CSR or dense weights.

## Exact API routing

Open the Matrix Operations API for exact signatures and associated primitives:

- CSR: `update_csr_on_binary_pre`, `update_csr_on_binary_post`, and their `*_p` primitives.
- Dense: `update_dense_on_binary_pre`, `update_dense_on_binary_post`, and their `*_p` primitives.

API: https://brainx.chaobrain.com/brainevent/reference/apis/operations.html

## Sources

- Tutorial 5: Synaptic Plasticity Modeling - Foundation of Learning and Memory: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/05_synaptic_plasticity.html
- Apply event-driven synaptic plasticity: https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html
