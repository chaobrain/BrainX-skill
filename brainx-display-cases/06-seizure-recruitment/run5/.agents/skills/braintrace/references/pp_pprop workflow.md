# pp-prop workflow

Use this reference when lower trace memory makes input/output-factorized
`pp_prop` preferable to D-RTRL, especially for recurrent SNN training. Open
`algorithm selection.md` for other estimators and `batching.md` for batching
ownership beyond the explicit mapped workflow shown here.

## Choose pp-prop

`pp_prop` carries input-side and output-side eligibility factors with
exponential smoothing instead of a parameter-dimensional trace.

| API or option | Description |
|---|---|
| `braintrace.pp_prop` | Use for the current input/output-factorized estimator. Its trace storage is `O(B * (I + O))`, subject to the documented model and operator assumptions. |
| `braintrace.ES_D_RTRL` | Use only for compatibility with historical code; new code should use `braintrace.pp_prop`. |
| `decay_or_rank=<float>` | Use a value in `(0, 1)` to set the exponential decay directly. Larger values retain a longer history. |
| `decay_or_rank=<int>` | Use a positive integer to select `decay = (rank - 1) / (rank + 1)`. It does not allocate that many independently stored low-rank factors. |

The factorization discards correlations represented by an unfactorized trace.
Lower memory therefore does not imply BPTT-equivalent gradients, and the
selected `decay_or_rank` is part of the method definition that must be reported
with results.

## Compare with D-RTRL

Choose between the two estimators by trace representation and acceptable
approximation, not by treating either as a universal BPTT replacement.

| Aspect | `D_RTRL` | `pp_prop` |
|---|---|---|
| Eligibility trace | Parameter-shaped with a diagonal hidden-Jacobian approximation. | Input/output-factorized with exponential smoothing. |
| Trace memory | `O(B * |theta|)`. | `O(B * (I + O))`. |
| Main approximation | Diagonal hidden-to-hidden Jacobian structure. | Input/output factorization plus exponential smoothing. |
| Use when | Parameter-shaped trace memory is feasible. | Lower trace memory is required and factorization assumptions are acceptable. |

Validate either estimator against BPTT on a reduced version of the intended
model before relying on gradient fidelity.

## Build an ETP-aware recurrent SNN

Use ETP-aware projections for trainable parameter paths, a stateful spiking
neuron for recurrent dynamics, and an ETP-aware leaky readout for continuous
sequence outputs.

| API | Description |
|---|---|
| `braintrace.nn.Linear(...)` | Use for the input-plus-recurrent projection so its weights participate through ETP primitives. |
| `brainpy.state.LIF(...)` | Use for leaky integrate-and-fire dynamics; provide a differentiable surrogate spike function. |
| `braintrace.nn.LeakyRateReadout(...)` | Use to integrate recurrent spikes into a continuous classification output. |
| `brainstate.environ.context(dt=...)` | Use to scope the physical simulation time step around model construction and execution. |

```python
import jax.numpy as jnp
import brainstate
import braintools
import braintrace
import brainunit as u
import brainpy.state


class LIF_SNN(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.linear = braintrace.nn.Linear(
            n_in + n_rec,
            n_rec,
            w_init=braintools.init.KaimingNormal(scale=50., unit=u.mA),
            b_init=braintools.init.ZeroInit(unit=u.mA),
        )
        self.neuron = brainpy.state.LIF(
            n_rec,
            tau=20. * u.ms,
            R=1. * u.ohm,
            V_th=0.1 * u.mV,
            V_reset=0. * u.mV,
            V_rest=0. * u.mV,
            spk_fun=braintools.surrogate.ReluGrad(),
            spk_reset='soft',
        )
        self.readout = braintrace.nn.LeakyRateReadout(
            n_rec,
            n_out,
            tau=20. * u.ms,
            w_init=braintools.init.KaimingNormal(),
        )

    def update(self, spike_input):
        recurrent_spikes = self.neuron.get_spike()
        projection_input = jnp.concatenate(
            [spike_input, recurrent_spikes],
            axis=-1,
        )
        self.neuron(self.linear(projection_input))
        return self.readout(self.neuron())
```

The projection weights use current units so the LIF term `I * R` is compatible
with the voltage-valued threshold and membrane State.

## Train a mapped spike sequence

Create one mapped model, initialize its independent recurrent State, construct
`pp_prop` around it directly, and compile from one complete batched time step.

| API | Description |
|---|---|
| `brainstate.nn.Map(model, init_map_size=B)` | Use once to own batch mapping and independent recurrent State across lanes. |
| `braintrace.pp_prop(mapped_model, decay_or_rank=...)` | Use with an explicitly mapped model; do not pass that model to `compile(..., vmap=True)`. |
| `learner.compile_graph(inputs[0])` | Use once with a complete step shaped `(batch_size, n_in)`. |
| `learner.etrace_grad(inputs, step_fn=..., has_aux=True, ...)` | Use to scan time, accumulate online gradients, and retain per-step outputs needed for a sequence prediction. |

```python
n_steps, batch_size = 100, 40
n_in, n_rec, n_out = 50, 128, 10

labels = jnp.arange(batch_size, dtype=jnp.int32) % n_out
channel_class = jnp.arange(n_in) * n_out // n_in
active = labels[:, None] == channel_class[None, :]
firing_probability = jnp.where(active, 0.6, 0.0)

brainstate.random.seed(37)
inputs = brainstate.random.bernoulli(
    firing_probability,
    size=(n_steps, batch_size, n_in),
).astype(jnp.float32)

with brainstate.environ.context(dt=1. * u.ms):
    model = LIF_SNN(n_in, n_rec, n_out)
    mapped_model = brainstate.nn.Map(model, init_map_size=batch_size)
    mapped_model.init_all_states()

    learner = braintrace.pp_prop(mapped_model, decay_or_rank=0.5)
    learner.compile_graph(inputs[0])

    optimizer = braintools.optim.Adam(3e-3)
    optimizer.register_trainable_weights(learner.param_states)

    @brainstate.transform.jit
    def train_step(sequence, targets):
        brainstate.nn.reset_all_states(mapped_model)
        learner.reset_state()

        def step_loss(spikes):
            output = learner(spikes)
            loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
                output,
                targets,
            ).mean()
            return loss, output

        grads, step_losses, outputs = learner.etrace_grad(
            sequence,
            step_fn=step_loss,
            has_aux=True,
            reduction='sum',
            return_value=True,
        )
        grads = brainstate.nn.clip_grad_norm(grads, 1.0)
        optimizer.update(grads)

        predictions = jnp.argmax(jnp.sum(outputs, axis=0), axis=-1)
        accuracy = jnp.mean(predictions == targets)
        return step_losses.mean(), accuracy

    loss, accuracy = train_step(inputs, labels)
    assert loss.shape == ()
    assert accuracy.shape == ()
```

`step_loss` must call `learner` exactly once. Reset model State and eligibility
State together before each independent spike sequence. `reduction='sum'`
preserves the accumulated per-step scale used by this workflow; tune the
learning rate for that reduction.

## Interpret checks and exclusions

Use joint loss and accuracy evidence on a reproducible task as a teaching
check, not a convergence guarantee. Different trace information means pp-prop
need not reach the same loss as D-RTRL on a matched run.

A weight that reaches hidden State only through another trainable ETP weight is
excluded by the compiler's non-parametric-tail invariant. This prevents double
counting and does not indicate an optimizer failure.

Open `ETP operators.md` when a custom SNN projection does not participate. Open
`pre-built-braintrace-layer.md` when selecting recurrent or readout layers.

## Sources

- [SNN Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/snn_online_learning.html)
- [pp_prop: input/output-factorized online gradients](https://brainx.chaobrain.com/braintrace/tutorials/pp_prop.html)

The factorization rules, decay semantics, SNN workflow, and comparison above
are trimmed and reorganized from the official pages without changing their API
names or documented guarantees.
