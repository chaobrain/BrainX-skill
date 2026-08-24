# Batching

Use this reference when adding a batch axis to BrainTrace `0.2.4`, selecting native batched execution or independent mapped lanes, resetting mapped State, or scanning time-major data.

## Choose one batching owner

| Path | Use when | Owner and constraint |
|---|---|---|
| Compile-owned vmap | Use when each example needs independent recurrent and eligibility State. | Pass the unwrapped model to `braintrace.compile(..., batch_size=B, vmap=True)` once. |
| Native batch | Use when the model and ETP operators accept a leading batch axis directly. | Pass `batch_size=B` and a batched example to `compile(..., vmap=False)`. |
| Explicit mapping | Use only when compile-owned vmap cannot express the State construction. | Follow the official `vmap_new_states` plus `brainstate.nn.Vmap` pattern and verify reset shapes. |
| Single stream | Use for one unbatched trajectory. | Omit `batch_size` and `vmap`. |

Use exactly one owner. Do not pass an already mapped model to `compile(..., vmap=True)`.

## Compile independent lanes

Compile-owned vmap creates per-lane State and returns a `brainstate.nn.Vmap`; its `.module` is the underlying BrainTrace learner.

| API | Description |
|---|---|
| `braintrace.compile(model, algorithm, example_step, batch_size=B, vmap=True, ...)` | Use with `example_step.shape == (B, features)`; compilation strips axis 0 to build one lane and creates mapped State across `B` lanes. |
| `mapped_learner.module.report` | Use for compilation diagnostics; the report belongs to the underlying learner. |
| `mapped_learner.states('new')` | Use to select the mapped State collection for a lane-preserving reset. |
| `brainstate.transform.vmap(in_states=mapped_states)` | Use to map reset over exactly the State created by compile-owned vmap. |

```python
import jax
import jax.numpy as jnp
import brainstate
import braintrace


class SimpleGRU(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
        self.out = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.out(self.rnn(x))


batch_size = 16
example_step = jnp.zeros((batch_size, 10))
model = SimpleGRU(10, 64, 5)

mapped_learner = braintrace.compile(
    model,
    braintrace.D_RTRL,
    example_step,
    batch_size=batch_size,
    vmap=True,
)
algorithm = mapped_learner.module
algorithm.report.show(2)

mapped_states = mapped_learner.states('new')


@brainstate.transform.vmap(in_states=mapped_states)
def reset_sequence():
    brainstate.nn.reset_all_states(mapped_learner)
```

Capture the shapes of mapped dynamical State before and after the first reset. If any leading lane axis changes, follow BrainState's collective-operations reference and restore an exact snapshot instead of continuing with the broken reset.

## Compile a native batch

Use native batching when the model's one-step call already consumes `(B, features)` and initializes State with `batch_size=B`.

```python
learner = braintrace.compile(
    SimpleGRU(10, 64, 5),
    braintrace.D_RTRL,
    jnp.zeros((16, 10)),
    batch_size=16,
)
```

Reset native batched execution with `brainstate.nn.reset_all_states(model, batch_size=16)` and `learner.reset_state(batch_size=16)`. Prefer the compile-owned vmap path when State constructors or custom cells do not implement native batch initialization consistently.

## Accumulate time-major gradients

Keep time on axis 0 and batch on axis 1. The scan owns time; compilation owns batch.

```python
weights = model.states(brainstate.ParamState)


def step_loss(inp, target, mask):
    prediction = mapped_learner(inp)
    loss = jnp.mean((prediction - target) ** 2)
    return loss * mask, prediction


step_grad = brainstate.transform.grad(
    step_loss,
    weights,
    has_aux=True,
    return_value=True,
)


def accumulate(grads, sample):
    current, loss, prediction = step_grad(*sample)
    grads = jax.tree.map(lambda total, value: total + value, grads, current)
    return grads, (loss, prediction)


initial_grads = jax.tree.map(
    jnp.zeros_like,
    {key: state.value for key, state in weights.items()},
)
grads, (losses, outputs) = brainstate.transform.scan(
    accumulate,
    initial_grads,
    (inputs, targets, loss_mask),
)
optimizer.update(grads)
```

Every step must call the learner exactly once, including masked cue and delay steps. A zero mask suppresses the learning signal; it does not suppress recurrent or eligibility-State evolution.

## Use sequence-data wrappers

| API | Description |
|---|---|
| `braintrace.SingleStepData(data)` | Use to mark `data` as one step; a plain array has the same interpretation. |
| `braintrace.MultiStepData(sequence)` | Use when one learner call should consume axis 0 as a multi-step window. |

Do not use `MultiStepData` as a substitute for the explicit scan when the workflow needs per-step masks, outputs, losses, or a gradient sum followed by one optimizer update.

Open `Drtrl.md` or `pp_pprop workflow.md` for estimator-specific training. Open `skills/package-skills/brainstate/references/collective_model_operations.md` when an explicit mapped lifecycle is unavoidable.

## Sources

- [BrainTrace `v0.2.4` compile modes](https://github.com/chaobrain/braintrace/blob/v0.2.4/examples/tests/test_compile_modes.py)
- [BrainTrace `v0.2.4` mapped batching example](https://github.com/chaobrain/braintrace/blob/v0.2.4/examples/pp_prop/05-batching-vmap.py)
- [BrainTrace `v0.2.4` native batching example](https://github.com/chaobrain/braintrace/blob/v0.2.4/examples/pp_prop/06-batching-batched.py)
