# pp-prop workflow

Use this reference when input/output-factorized `pp_prop` is preferable to D-RTRL, especially for recurrent SNNs and silent-delay tasks. Open `batching.md` for batching ownership and `ETP operators.md` when a custom projection does not participate.

## Choose pp-prop

`pp_prop` carries input-side and output-side eligibility factors with exponential smoothing instead of a parameter-dimensional trace.

| API or option | Description |
|---|---|
| `braintrace.pp_prop` | Use when factorized `O(B * (I + O))` trace storage is preferable to parameter-shaped storage. |
| `braintrace.ES_D_RTRL` | Use only for compatibility with historical code; prefer `pp_prop`. |
| `decay_or_rank=<float>` | Use a value in `(0, 1)` to set exponential decay directly; larger values retain longer history. |
| `decay_or_rank=<int>` | Use a positive integer to select `decay = (rank - 1) / (rank + 1)`; it does not allocate that many stored factors. |

Factorization discards correlations present in an unfactorized trace. Report `decay_or_rank` with results and validate the estimator against BPTT on a reduced matched model when gradient fidelity matters.

## Build an ETP-aware recurrent SNN

Use ETP-aware projections for trainable parameter paths, stateful BrainPy-State neurons for dynamics, and an ETP-aware leaky readout for continuous outputs.

| API | Description |
|---|---|
| `braintrace.nn.Linear(...)` | Use for an input-plus-recurrent projection whose weights participate through ETP primitives. |
| `braintrace.sparse_matmul(x, weight_data, sparse_mat=...)` | Use for fixed sparse recurrent structure; pass raw numeric/bool spikes as `x` and a BrainEvent `DataRepresentation` as `sparse_mat`. |
| `brainpy.state.LIF(...)` | Use for leaky integrate-and-fire dynamics with a differentiable surrogate spike function. |
| `braintrace.nn.LeakyRateReadout(...)` | Use to integrate recurrent spikes into a continuous task output. |
| `brainstate.environ.context(dt=...)` | Use to scope the physical simulation step around construction and execution. |

```python
import brainpy.state
import brainstate
import braintrace
import braintools
import brainunit as u


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

    def update(self, x):
        previous = self.neuron.get_spike()
        current = self.linear(u.math.concatenate([x, previous], axis=-1))
        return self.readout(self.neuron(current))
```

The projection uses current units so `I * R` is voltage-compatible with membrane State and threshold.

## Train a batched spike sequence

Compile one-step behavior with per-sample mapped State, reset both State domains, accumulate masked step gradients through a scan, and update once per sequence.

| API | Description |
|---|---|
| `braintrace.compile(..., braintrace.pp_prop, batch_size=B, vmap=True, decay_or_rank=...)` | Use to create independent recurrent and trace State for each batch lane. |
| `brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)` | Use around one learner call to return the online gradient, masked loss, and output. |
| `brainstate.transform.scan(...)` | Use to advance every time step and carry the summed gradient. |
| `brainstate.nn.clip_grad_norm(...)` | Use after accumulation when gradient clipping is required. |

```python
import jax
import jax.numpy as jnp

n_steps, batch_size = 60, 16
n_in, n_rec, n_out = 2, 48, 2
inputs = jnp.zeros((n_steps, batch_size, n_in))
labels = jnp.arange(batch_size, dtype=jnp.int32) % n_out
loss_mask = jnp.concatenate([jnp.zeros(50), jnp.ones(10)])

with brainstate.environ.context(dt=1. * u.ms):
    model = LIF_SNN(n_in, n_rec, n_out)
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        inputs[0],
        batch_size=batch_size,
        vmap=True,
        decay_or_rank=0.9,
    )
    algorithm = learner.module
    algorithm.report.show(2)
    assert algorithm.report.counts['errors'] == 0

    weights = model.states(brainstate.ParamState)
    optimizer = braintools.optim.Adam(lr=2e-3)
    optimizer.register_trainable_weights(weights)

    mapped_states = learner.states('new')

    @brainstate.transform.vmap(in_states=mapped_states)
    def reset_sequence():
        brainstate.nn.reset_all_states(learner)

    def step_loss(cue, mask):
        output = learner(cue)
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
            output,
            labels,
        ).mean()
        return loss * mask, output

    step_grad = brainstate.transform.grad(
        step_loss,
        weights,
        has_aux=True,
        return_value=True,
    )

    def train_sequence():
        reset_sequence()
        initial_grads = jax.tree.map(
            jnp.zeros_like,
            {key: state.value for key, state in weights.items()},
        )

        def accumulate(grads, sample):
            current, loss, output = step_grad(*sample)
            grads = jax.tree.map(lambda x, y: x + y, grads, current)
            return grads, (loss, output)

        grads, (losses, outputs) = brainstate.transform.scan(
            accumulate,
            initial_grads,
            (inputs, loss_mask),
        )
        optimizer.update(brainstate.nn.clip_grad_norm(grads, 1.0))
        return losses.sum() / loss_mask.sum(), outputs
```

`step_loss` calls the learner on every cue, delay, and report step. The zero mask removes the loss signal before the report window without freezing hidden or eligibility State. Do not put `optimizer.update()` inside the scan unless the intended experiment explicitly uses per-timestep parameter changes and reports that schedule.

## Validate silent-delay memory

A lower training loss does not prove that the first cue survived the delay; use a control that changes the claimed memory variable while preserving the evaluation path.

| Check | Required evidence |
|---|---|
| Silent interval | Assert that cue input is exactly zero during the declared delay. |
| Frozen evaluation | Fix evaluation trials, seeds, masks, and metrics before viewing the trained result. |
| First-cue ablation | Replace or permute only the first cue and rerun the same mapped rollout; accuracy should fall materially relative to intact trials. |
| Robustness | Prefer held-out stochastic cue realizations or a nearby longer delay when the task permits them. |
| Schedule | Report sequences and optimizer updates separately; one scan followed by one update means one optimizer update per sequence. |

For a deterministic two-cue task with only four combinations, first-cue ablation is the minimum discriminating control. Perfect accuracy on the same four training combinations is insufficient by itself.

## Interpret compiler diagnostics

Use `algorithm.report.show(2)` and structured fields together. A parameter may have a rejected unbatched candidate and an included batched relation in the same report. Require zero errors and check that the intended recurrent path appears in `etrace_weights`; do not classify it from warning text alone.

Open `compiler_internal.md` only when structured report records cannot explain the relation. Open `ETP operators.md` for sparse input and structure contracts.

## Sources

- [BrainTrace `v0.2.4` pp-prop tutorial](https://github.com/chaobrain/braintrace/blob/v0.2.4/docs/tutorials/pp_prop.md)
- [BrainTrace `v0.2.4` delayed-match example](https://github.com/chaobrain/braintrace/blob/v0.2.4/examples/pp_prop/02-neurons-alif-dms.py)
- [BrainTrace `v0.2.4` working-memory example](https://github.com/chaobrain/braintrace/blob/v0.2.4/examples/pp_prop/03-neurons-gif-working-memory.py)
