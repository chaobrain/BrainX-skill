# D-RTRL

Use this reference when implementing or validating the parameter-dimensional D-RTRL workflow. Open `algorithm selection.md` when choosing another estimator and `batching.md` before adding a batch axis.

## Understand the approximation

`D_RTRL` carries parameter-dimensional eligibility traces forward while its default recurrence drops cross-hidden terms from the hidden-to-hidden Jacobian.

| API | Description |
|---|---|
| `braintrace.D_RTRL` | Use when parameter-shaped trace memory is feasible; it computes approximate online gradients without retaining the full sequence graph. |
| `braintrace.compile(model, braintrace.D_RTRL, example_step, ...)` | Use once to initialize State, construct the learner, and compile the ETP graph from a representative step. |
| `learner.etrace_grad(..., step_fn=..., reduction=...)` | Use to accumulate online gradients across one sequence. |

Trace memory scales as `O(B * |theta|)`, where `B` is batch size and `theta` is the participating parameter set. Do not describe D-RTRL as generally identical to BPTT; equality depends on the recurrence and mathematical regime being tested.

## Follow the sequence lifecycle

Reset recurrent and eligibility State together, accumulate gradients across the complete sequence, then update parameters once.

| Step | Action | Important result |
|---|---|---|
| 1. Build | Compose ETP-aware recurrent and readout layers. | Parameter-to-hidden paths remain visible to the compiler. |
| 2. Compile | Call `braintrace.compile(...)` once with one real per-step input. | The graph and State layout are reused. |
| 3. Inspect | Show report level 2 and inspect structured participation fields. | Intended recurrent weights are confirmed rather than inferred from warning text. |
| 4. Reset | Reset model State and learner State at every independent sequence. | Hidden State, eligibility traces, and running index share one boundary. |
| 5. Accumulate | Call `etrace_grad` with one learner call per step loss. | Every time step advances online State while gradients sum across time. |
| 6. Update | Clip if needed and call the optimizer once after the sequence. | The printed sequence count matches the optimizer schedule. |

```python
import jax.numpy as jnp
import brainstate
import braintrace
import braintools


class SequenceModel(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = braintrace.nn.MiniGRU(in_size=1, out_size=6)
        self.readout = braintrace.nn.Linear(6, 1)

    def update(self, x):
        return self.readout(self.rnn(x))


model = SequenceModel()
inputs = jnp.linspace(-1.0, 1.0, 12).reshape(12, 1, 1)
targets = 0.7 * inputs + 0.2
learner = braintrace.compile(
    model,
    braintrace.D_RTRL,
    inputs[0],
    batch_size=1,
)
learner.report.show(2)

weights = model.states(brainstate.ParamState)
optimizer = braintools.optim.SGD(lr=0.005)
optimizer.register_trainable_weights(weights)


def reset_sequence():
    brainstate.nn.reset_all_states(model, batch_size=1)
    learner.reset_state(batch_size=1)


def step_loss(x, target):
    return jnp.mean((learner(x) - target) ** 2)


def train_sequence():
    reset_sequence()
    grads, losses = learner.etrace_grad(
        inputs, targets, step_fn=step_loss, reduction='sum', return_value=True,
    )
    optimizer.update(grads)
    return losses.mean()


loss = train_sequence()
assert jnp.isfinite(loss)
```

Pass a time-major `mask` to `etrace_grad` when only selected steps contribute to the loss. The learner still advances both State systems at masked steps.

## Reset both State domains

| API | State reset |
|---|---|
| `brainstate.nn.reset_all_states(model, batch_size=B)` | Resets native batched recurrent State. |
| `learner.reset_state(batch_size=B)` | Resets native batched eligibility traces and the running index. |
| Mapped reset from `batching.md` | Resets a compile-owned vmap wrapper lane by lane without collapsing its State axis. |

Consecutive learner calls otherwise continue from installed State. Preserve continuation only when the calls belong to one trajectory.

## Interpret compiler participation

Use `learner.report.show(2)` and inspect `counts`, `etrace_weights`, `excluded_weights`, and diagnostics together. A readout that does not feed hidden State is correctly non-temporal. A warning for an unbatched candidate can coexist with an included batched relation for the same parameter; do not decide participation from warning strings alone.

Open `ETP operators.md` when a custom layer's intended recurrent parameter is absent. Open `compiler_internal.md` only when structured report fields cannot explain the decision.

## Keep the BPTT boundary explicit

| Property | D-RTRL | BPTT |
|---|---|---|
| Gradient timing | Updates eligibility information during the forward sequence. | Backpropagates after unrolling the sequence. |
| Sequence storage | Does not retain the full unrolled graph. | Stores or rematerializes sequence activations. |
| Gradient claim | Uses the configured recurrent approximation. | Provides the matched full-sequence baseline. |

Use a reduced matched task as a BPTT oracle only when the estimator's documented regime predicts equality. A descending loss verifies training mechanics, not gradient equivalence.

## Sources

- [D-RTRL tutorial](https://brainx.chaobrain.com/braintrace/tutorials/drtrl.html)
- [Algorithms API](https://brainx.chaobrain.com/braintrace/apis/algorithms.html)
