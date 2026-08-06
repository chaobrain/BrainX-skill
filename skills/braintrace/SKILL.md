---
name: braintrace
description: Use BrainTrace for eligibility-trace online learning in recurrent or spiking BrainX models, including ETP-aware construction, D-RTRL or pp-prop selection, compilation, graph inspection, sequence gradients, and eligibility-State lifecycle.
---

# BrainTrace

## Purpose and boundary

Use BrainTrace to train recurrent or spiking models by carrying temporal credit forward in eligibility State. Follow this path:

`build with braintrace.nn -> choose an algorithm -> compile once -> inspect -> reset both State systems -> train -> validate`

Route general `State`, `Module`, transformations, and optimizers to BrainState; neuron and synapse dynamics to BrainPy-State or BrainCell; and offline BPTT without eligibility traces away from BrainTrace.

## Underlying principle of BrainTrace

Hidden State represents recurrent memory. BrainTrace discovers State read and written by one model step and groups related values for online Jacobian computation.

ETP operations represent parameter paths that receive temporal credit. The operation selects participation: an ETP primitive marks a plain `brainstate.ParamState` input in JAX IR; the equivalent unmarked JAX operation does not.

Eligibility State represents accumulated gradient history. An `ETraceGraph` connects ETP-marked parameters to hidden groups so an algorithm can update this State during the forward sequence.

## Online learning versus BPTT

BrainTrace improves sequence-memory efficiency; it does not guarantee lower total memory or faster runtime for every model.

| Concern | BrainTrace | BPTT |
|---|---|---|
| Temporal credit | Accumulate eligibility traces during the forward drive. | Traverse the unrolled sequence in reverse. |
| Memory in sequence length $T$ | $O(1)$ per step; do not retain the trajectory. | $O(T)$ to store or rematerialize the trajectory. |
| Gradient | Depends on the chosen online estimator and approximation. | Differentiates the chosen unrolled graph. |

Trace State still scales with the model and algorithm. `D_RTRL` uses diagonal hidden recurrence with parameter-dimensional traces; use factorized `pp_prop` when that trace cost is unsuitable. Do not claim equality with BPTT without the algorithm's documented guarantee and a reduced comparison.

## API structure overview

| API | Responsibility |
|---|---|
| `braintrace.nn` | ETP-aware layers; use these first. |
| ETP operations | Custom layer construction when prebuilt layers are insufficient. |
| `braintrace.compile()` | Hidden-State discovery, graph compilation, and learner construction. |
| `D_RTRL`, `pp_prop` | Parameter-dimensional or factorized online estimators. |
| Compiled learner | Graph inspection, eligibility-State reset, and sequence execution. |

## Build and compile a learner

Compile one representative step to connect ETP-marked parameters to recurrent hidden groups before training.

| API | Description |
|---|---|
| `braintrace.nn.MiniGRU(...)` | Use for the canonical recurrent layer; it applies ETP-aware parameter operations and carries hidden State. |
| `braintrace.nn.Linear(...)` | Use for an ETP-aware projection; the compiler excludes it from temporal traces when it does not feed recurrent State. |
| `braintrace.compile(model, algorithm, example, batch_size=...)` | Use once with the real step shape; it initializes State, builds the `ETraceGraph`, and returns a learner. |
| `learner.report.show(1)` | Use after compilation to inspect included and excluded weights and their reasons. |
| `learner.show_graph()` | Use after the report to inspect hidden groups and parameter relations. |

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
    model, braintrace.D_RTRL, inputs[0], batch_size=1
)
learner.report.show(1)
learner.show_graph()
```

Open `references/pre-built-braintrace-layer.md` to select other layers, `references/ETP operators.md` to build a custom layer from built-in operations, or `references/custom ETP primitives.md` only when those operations cannot express the computation.

## Train and evaluate sequences

Let the learner own the time loop and eligibility updates; make the step loss call the learner exactly once.

| API | Description |
|---|---|
| `brainstate.nn.reset_all_states(model, batch_size=...)` | Reset recurrent model State at an independent sequence boundary. |
| `learner.reset_state(batch_size=...)` | Reset eligibility State at the same boundary. |
| `learner.etrace_grad(..., step_fn=..., reduction=...)` | Drive a supervised sequence and return reduced online gradients. |
| `learner.etrace_evolve(..., return_outputs=True)` | Drive a loss-free sequence and return stacked outputs. |
| `optimizer.update(grads)` | Apply the reduced gradients to registered parameters. |

```python
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

def step_loss(x, target):
    return jnp.mean((learner(x) - target) ** 2)

def train_epoch(_):
    reset_sequence()
    grads, losses = learner.etrace_grad(
        inputs, targets, step_fn=step_loss,
        reduction="mean", return_value=True,
    )
    optimizer.update(grads)
    return losses.mean()

initial_loss = evaluate()
losses = brainstate.transform.for_loop(train_epoch, jnp.arange(25))
final_loss = evaluate()
assert losses.shape == (25,)
assert final_loss < initial_loss
```

Open `references/Drtrl.md` for D-RTRL recurrence, memory, resets, and BPTT comparison; `references/pp_pprop workflow.md` for factorized traces or SNNs; and `references/batching.md` before adding manual or mapped batch axes.

## Reference routing

| Reference | Open when |
|---|---|
| `references/algorithm selection.md` | Choosing an estimator beyond the canonical D-RTRL and pp-prop boundary. |
| `references/customizing_primitive_transforms.md` | Applying a mask, constraint, normalization, LoRA, or bias transform to an ETP weight. |
| `references/custom algorithms.md` | Implementing an estimator that built-in algorithms cannot express. |
| `references/compiler_internal.md` | Public `learner.report` and `learner.graph` cannot explain custom graph behavior. |

## Boundaries and common failures

- Use prebuilt layers before built-in ETP operations, and built-in operations before custom primitives.
- Do not compile inside training or from an example with the wrong batch or feature shape.
- Do not treat a documented non-temporal exclusion as a failure; inspect its report reason.
- Reset both recurrent and eligibility State between independent sequences.
- Do not wrap `etrace_grad()` or `etrace_evolve()` in another time scan.
- Do not use the historical `ES_D_RTRL` name in new code; use `pp_prop`.
- Do not infer constant total memory, universal speed, or BPTT-equivalent gradients from $O(1)$ memory in sequence length.
