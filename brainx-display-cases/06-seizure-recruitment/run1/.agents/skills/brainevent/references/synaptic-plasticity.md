# BrainEvent synaptic plasticity

Use this reference when binary pre- or postsynaptic events must modify stored synaptic weights. Plasticity is an overlay on the core `BinaryArray @ connectivity` workflow, not a prerequisite for ordinary event-driven communication.

BrainEvent's event-driven plasticity operators touch only weights connected to neurons that fired. Keep persistent weights and temporal traces in BrainState State so the complete online update can run inside State-aware transforms.

## Choose the update operator

| Weight storage | Presynaptic event triggers update | Postsynaptic event triggers update | Use when |
|---|---|---|---|
| CSR | `update_csr_on_binary_pre()` | `update_csr_on_binary_post()` | Connectivity is sparse and fixed; only stored synapses should be visited. |
| Dense | `update_dense_on_binary_pre()` | `update_dense_on_binary_post()` | A small fully connected layer already uses a dense weight matrix. |

Use the `*_on_binary_pre` direction when the source neuron firing triggers the rule. Use the `*_on_binary_post` direction when the target neuron firing triggers it. For a bidirectional STDP rule, apply both directions with the traces and signs defined by that rule; do not substitute one trigger direction for the other.

## Run online plasticity through State

Each timestep reads event arrays, advances trace State, and writes persistent weight State; lower that complete ordered update through `for_loop` and compile it with BrainState `jit`.

| API | Description |
|---|---|
| `brainstate.LongTermState(weight)` | Use for learned weights that must persist across trial resets and later rollouts; assign the update result through `.value`. |
| `brainstate.ShortTermState(trace)` | Use for decaying pre- or postsynaptic traces that belong to one sequence; reset them at every independent sequence boundary. |
| `update_csr_on_binary_pre(...)` | Use when active presynaptic rows update stored CSR data from a postsynaptic trace; it returns new data with the same shape and optionally clips it to `w_min` and `w_max`. |
| `brainstate.transform.for_loop(step, *events)` | Use for the ordered event sequence; it slices the leading time axis and preserves State effects between steps. |
| `brainstate.transform.jit(run)` | Use around the complete online sequence so State discovery and write-back remain transform-aware. |

```python
import brainevent
import brainstate
import brainunit as u
import jax.numpy as jnp


class PlasticCSR(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (3, 2)
        self.indices = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
        self.indptr = jnp.array([0, 2, 4, 6], dtype=jnp.int32)
        self.weight = brainstate.LongTermState(
            jnp.full(6, 0.2, dtype=jnp.float32)
        )
        self.post_trace = brainstate.ShortTermState(
            jnp.zeros(2, dtype=jnp.float32)
        )

    def update(self, pre_spike, post_spike):
        decay = u.math.exp(-brainstate.environ.get_dt() / (20.0 * u.ms))
        self.post_trace.value = (
            self.post_trace.value * decay
            + post_spike.astype(jnp.float32)
        )
        self.weight.value = brainevent.update_csr_on_binary_pre(
            weight=self.weight.value,
            indices=self.indices,
            indptr=self.indptr,
            pre_spike=pre_spike,
            post_trace=self.post_trace.value * 0.01,
            w_min=0.0,
            w_max=1.0,
            shape=self.shape,
        )
        return self.weight.value


pre_events = jnp.array(
    [[False, False, False], [True, False, False], [False, True, False]]
)
post_events = jnp.array(
    [[True, False], [False, False], [False, True]]
)

with brainstate.environ.context(dt=1.0 * u.ms):
    rule = PlasticCSR()

    @brainstate.transform.jit
    def learn():
        return brainstate.transform.for_loop(
            rule.update,
            pre_events,
            post_events,
        )

    weight_history = learn()

learned = brainevent.CSR(
    (rule.weight.value, rule.indices, rule.indptr),
    shape=rule.shape,
)
drive = brainevent.BinaryArray(jnp.array([True, False, False])) @ learned

assert weight_history.shape == (3, 6)
assert drive.shape == (2,)
```

Preserve `indices`, `indptr`, and `shape`; the operator updates stored weight values, not sparse topology. When trials are independent except for learned weights, keep `weight` sequential but reset `post_trace`, neural State, delays, and other per-trial State at every trial boundary; a silent interval is not a reset. If mapped lanes learn independently, give each lane separate weight and trace State. Do not `vmap` sequential trials when trial N+1 must consume weights learned in trial N.

## Storage boundary

- Prefer CSR plasticity for large sparse networks with fixed connectivity.
- Prefer dense plasticity for small fully connected layers already represented densely.
- Do not choose JITC when individual connection weights must be persistently updated; use stored CSR or dense weights.
- Keep dimensionless efficacy weights dimensionless until a named synaptic or current boundary scales them into a physical quantity.

## Exact API routing

Open the Matrix Operations API for exact signatures and associated primitives:

- CSR: `update_csr_on_binary_pre`, `update_csr_on_binary_post`, and their `*_p` primitives.
- Dense: `update_dense_on_binary_pre`, `update_dense_on_binary_post`, and their `*_p` primitives.

API: https://brainx.chaobrain.com/brainevent/reference/apis/operations.html

Open the official Synaptic Plasticity tutorial when choosing trace equations, update signs, or a bidirectional learning rule. The BrainEvent operator applies the supplied update; it does not choose the scientific learning rule.

## Common failures

- Storing learned weights or traces in ordinary attributes while a transformed timestep mutates them.
- Running timesteps in a Python loop instead of one State-aware transformed sequence.
- Resetting persistent learned weights together with per-sequence traces.
- Updating CSR data while changing or discarding `indices`, `indptr`, or `shape`.
- Using a presynaptic-triggered operator for a postsynaptic-triggered term.
- Interpreting a potentiation-only example as a complete STDP rule.

## Sources

- Tutorial 5: Synaptic Plasticity Modeling - Foundation of Learning and Memory: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/05_synaptic_plasticity.html
- Apply event-driven synaptic plasticity: https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html
