# Batching

Use this reference when adding a batch axis to BrainTrace online learning,
choosing which API owns mapping, compiling from a batched example, or wrapping
single-step and multi-step inputs. Keep model update logic single-sample and
assign batch mapping to exactly one API.

## Choose one mapping owner

Map-based batching keeps parameters shared while giving each batch lane an
independent copy of recurrent State.

| Path | Use when | Mapping owner |
|---|---|---|
| Explicit map | Use for the recommended path when mapping and State initialization must be visible. | Create one `brainstate.nn.Map`; construct the algorithm around it directly. |
| Compile-owned map | Use when the one-call compiler should create the wrapper. | Pass an unmapped model to `braintrace.compile(..., vmap=True)`. |
| Single stream | Use for one stream or step-by-step debugging. | Use an unmapped model and unbatched example; no API owns a batch transformation. |

**Mapping invariant:** Use exactly one mapping owner. Never pass an existing
`brainstate.nn.Map` to `braintrace.compile(..., vmap=True)` because that maps
the model twice.

## Map and compile a batch

Create and initialize one `Map`, construct the algorithm with that mapped
model, then compile from one complete batched time step.

| API | Description |
|---|---|
| `brainstate.nn.Map(model, init_map_size=B)` | Use to replicate mutable model State across `B` lanes while sharing parameter State. |
| `mapped_model.init_all_states()` | Use before algorithm construction to initialize the independent recurrent State copies. |
| `braintrace.D_RTRL(mapped_model)` | Use to construct the selected algorithm directly around an explicitly mapped model. |
| `mapped_algo.compile_graph(example_input)` | Use once with an example shaped `(batch_size, n_in)` so the batch axis remains inside the compiled ETP graph. |

```python
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


model = SimpleGRU(10, 64, 5)
batch_size = 16
example_input = jnp.zeros((batch_size, 10))

mapped_model = brainstate.nn.Map(model, init_map_size=batch_size)
mapped_model.init_all_states()

mapped_algo = braintrace.D_RTRL(mapped_model)
mapped_algo.compile_graph(example_input)
```

The mapped learner applies the wrapped single-sample update over axis 0. Its
parameters remain shared and its recurrent State remains independent across
lanes; an input shaped `(16, 10)` produces an output shaped `(16, 5)` for this
model.

## Train over time-major batches

Keep time on axis 0 and the batch on axis 1, then let the learner's sequence
driver own the temporal scan and gradient accumulation.

| API | Description |
|---|---|
| `mapped_algo.etrace_grad(inputs, ..., step_fn=..., reduction=...)` | Use for online gradients over time-major input. It calls `step_fn` across axis 0 and returns gradient keys matching `mapped_algo.param_states`. |
| `mapped_algo.etrace_evolve(inputs, ...)` | Use for loss-free sequence evolution. It advances every mapped lane without accumulating a loss gradient. |
| `reduction='sum'` | Use when accumulated per-step gradients must not be divided by the number of steps. |

```python
@brainstate.transform.jit
def train_step(inputs, targets):
    """inputs: (n_steps, batch_size, n_in); targets: (batch_size, n_out)."""
    def step_loss(inp):
        prediction = mapped_algo(inp)
        return jnp.mean((prediction - targets) ** 2)

    return mapped_algo.etrace_grad(
        inputs,
        step_fn=step_loss,
        reduction='sum',
    )


inputs = jnp.ones((20, 16, 10))
targets = jnp.zeros((16, 5))
grads = train_step(inputs, targets)
# Register mapped_algo.param_states with the optimizer that consumes grads.
```

`step_fn` must call `mapped_algo` exactly once. Call `etrace_grad()` and
`etrace_evolve()` on the mapped learner itself; accessing a wrapped `.module`
bypasses mapped lanes.

## Use single-sample mode

Use an unbatched model when processing one stream or diagnosing a single step;
do not create a `Map` or request `vmap`.

```python
single_model = SimpleGRU(10, 64, 5)
single_algo = braintrace.compile(
    single_model,
    braintrace.D_RTRL,
    jnp.zeros(10),
)

output = single_algo(jnp.ones(10))
assert output.shape == (5,)
```

Omitting `batch_size` keeps model State unbatched and the example input contains
only the feature axis.

## Wrap temporal data

Use data wrappers to distinguish one forward step from an input whose first
axis is time.

| API | Description |
|---|---|
| `braintrace.SingleStepData(data)` | Use to mark `data` as one forward step. A plain array has the same single-step interpretation. |
| `braintrace.MultiStepData(sequence)` | Use to mark axis 0 as time. The algorithm scans that axis internally instead of treating the complete sequence as one step. |

```python
one_step = braintrace.SingleStepData(jnp.ones(10))
many_steps = braintrace.MultiStepData(jnp.ones((20, 10)))
```

For mapped data, preserve the same axis meaning: a complete time-major sequence
has shape `(n_steps, batch_size, n_in)`, and each compiled step has shape
`(batch_size, n_in)`.

Open `algorithm selection.md` when batching cost changes the estimator choice.
Open `Drtrl.md` or `pp_pprop workflow.md` for estimator-specific State reset and
training behavior.

## Source

[Batching Strategies](https://brainx.chaobrain.com/braintrace/advanced/batching.html).
The ownership rules, workflows, shapes, and wrapper behavior above are
condensed from the official page's prose and code blocks.
