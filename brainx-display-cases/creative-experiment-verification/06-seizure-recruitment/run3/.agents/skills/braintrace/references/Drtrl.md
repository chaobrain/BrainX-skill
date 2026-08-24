# D-RTRL

Use this reference when implementing or validating the canonical
parameter-dimensional D-RTRL workflow, including sequence resets, online
gradient accumulation, and comparison with BPTT. Open `algorithm selection.md`
when deciding among estimators and `batching.md` before adding a batch axis.

## Understand the approximation

`D_RTRL` carries parameter-dimensional eligibility traces forward while its
default diagonal recurrence drops cross-hidden terms from the
hidden-to-hidden Jacobian.

| API | Description |
|---|---|
| `braintrace.D_RTRL` | Use when parameter-shaped trace memory is feasible. It computes approximate gradients during the forward sequence without retaining the full unrolled graph. |
| `braintrace.compile(model, braintrace.D_RTRL, example_input, ...)` | Use as the normal one-call setup. It constructs the learner and compiles its ETP graph from the representative step. |
| `learner.etrace_evolve(inputs, return_outputs=...)` | Use for evaluation or a loss-free prefix. It advances hidden and eligibility State without accumulating a loss gradient. |
| `learner.etrace_grad(inputs, *targets, step_fn=..., ...)` | Use for online sequence gradients. It owns the temporal loop, accumulation, masking, and reduction. |

Trace memory scales as `O(B * |theta|)`, where `B` is batch size and `theta` is
the participating parameter set. Wide recurrent layers can therefore be
memory-intensive even though memory does not grow with sequence length.

**Guarantee:** Do not describe D-RTRL as generally identical to BPTT. Equality
depends on the configured recurrence, VJP path, and mathematical regime being
tested.

## Follow the sequence lifecycle

Reset recurrent and eligibility State together at each independent sequence;
then let the learner drive the complete sequence and update parameters once.

| Step | Action | Important result |
|---|---|---|
| 1. Build | Compose ETP-aware recurrent and readout layers. | Participating parameter-to-hidden paths remain visible to the compiler. |
| 2. Compile | Call `braintrace.compile(...)` once with a representative input. | The learner reuses one compiled ETP graph across training epochs. |
| 3. Register | Give the optimizer the model's trainable `ParamState` collection. | Gradient keys and optimizer weights refer to the same State objects. |
| 4. Reset | Call `brainstate.nn.reset_all_states(...)` and `learner.reset_state(...)`. | Hidden State, eligibility traces, and the learner's running index start at the same boundary. |
| 5. Drive | Define a local loss that calls the learner exactly once, then call `etrace_grad()`. | The learner accumulates per-step online gradients. |
| 6. Update | Apply the returned gradients and verify descent on a deterministic task. | A lower loss checks training mechanics, not BPTT equivalence. |

## Train a deterministic MiniGRU

Use the same small model and affine sequence task as the official pp-prop
walkthrough when comparing estimators; changing only the algorithm isolates the
trace rule.

```python
import brainstate
import braintools
import braintrace
import jax.numpy as jnp

brainstate.random.seed(7)


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

weights = model.states(brainstate.ParamState)
optimizer = braintools.optim.SGD(lr=0.08)
optimizer.register_trainable_weights(weights)


def reset_sequence():
    brainstate.nn.reset_all_states(model, batch_size=1)
    learner.reset_state(batch_size=1)


def evaluate():
    reset_sequence()
    predictions = learner.etrace_evolve(inputs, return_outputs=True)
    return jnp.mean((predictions - targets) ** 2)


def local_loss(x, target):
    prediction = learner(x)
    return jnp.mean((prediction - target) ** 2)


def train_epoch(_):
    reset_sequence()
    grads, step_losses = learner.etrace_grad(
        inputs,
        targets,
        step_fn=local_loss,
        reduction='mean',
        return_value=True,
    )
    optimizer.update(grads)
    return step_losses.mean()


initial_loss = evaluate()
training_losses = brainstate.transform.for_loop(
    train_epoch,
    jnp.arange(25),
)
final_loss = evaluate()

assert final_loss < initial_loss
```

`reduction='mean'` divides by the total mask weight, which is the number of
steps when no mask is supplied. `local_loss` must call `learner` exactly once;
`etrace_grad()` owns every other part of the sequence loop.

## Reset both State domains

The model and learner carry different temporal information, so resetting only
one invalidates independent-sequence comparisons.

| API | State reset |
|---|---|
| `brainstate.nn.reset_all_states(model, batch_size=1)` | Resets the MiniGRU hidden State. |
| `learner.reset_state(batch_size=1)` | Resets eligibility traces and the learner's running index. |

Consecutive `etrace_evolve()` or `etrace_grad()` calls otherwise continue from
the currently installed State. Preserve that continuation only when the calls
belong to one trajectory.

## Interpret compiler participation

A parameter participates in temporal learning only when an ETP primitive marks
its path to hidden State.

| Situation | Interpretation |
|---|---|
| Recurrent weight reaches hidden State through an ETP primitive. | The compiler records a temporal relation and D-RTRL carries its eligibility trace. |
| Readout does not feed hidden State. | The parameter remains trainable but is correctly reported as non-temporal; its instantaneous gradient does not require a recurrent trace. |
| A weight reaches hidden State only through another trainable ETP weight. | The compiler excludes the upstream weight under the non-parametric-tail invariant to prevent double counting. |

Open `ETP operators.md` when a custom layer's parameter is unexpectedly absent.
Treat a documented non-temporal or excluded relation as a diagnostic, not an
optimizer failure.

## Keep the BPTT boundary explicit

| Property | D-RTRL | BPTT |
|---|---|---|
| Gradient timing | Updates eligibility information during the forward sequence. | Backpropagates after unrolling the sequence. |
| Sequence storage | Does not retain the full unrolled graph. | Stores intermediate activations, so memory grows with sequence length. |
| Gradient claim | Uses the configured diagonal recurrence and VJP approximations. | Provides the full-sequence baseline for a matched model and loss. |

Use a reduced matched task as a BPTT oracle only when the estimator's
mathematical regime predicts equality. A descending D-RTRL loss verifies the
workflow but does not establish gradient equality.

For explicit `brainstate.nn.Map` setup, compile directly with
`braintrace.D_RTRL(mapped_model).compile_graph(batched_step)` and follow
`batching.md`; never map an already mapped model again with `vmap=True`.

## Sources

- [RNN Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/rnn_online_learning.html)
- [D-RTRL: diagonal online gradient learning](https://brainx.chaobrain.com/braintrace/tutorials/drtrl.html)

The lifecycle, approximation boundary, examples, and diagnostics above are
trimmed and reorganized from the official pages without changing their API
names or guarantees.
