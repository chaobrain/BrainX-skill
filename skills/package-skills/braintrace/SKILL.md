---
name: braintrace
description: BrainTrace relieves training-memory pressure in recurrent and spiking models by replacing BPTT's sequence-length-dependent graph with eligibility traces. Use this skill for memory-efficient temporal training, long sequences, BPTT out-of-memory problems, activation-memory reduction, and BrainTrace estimator selection or compilation. Do not use for general speed optimization, offline BPTT, or model dynamics.
---

# BrainTrace

## Purpose and boundary

Use BrainTrace when recurrent or spiking training is limited by BPTT memory. It carries eligibility traces forward instead of retaining the full sequence graph, so memory is constant in sequence length but still scales with batch size, participating parameters, hidden dimensions, and the selected estimator.

Follow this path:

`build with braintrace.nn -> choose a trace representation -> compile once -> inspect -> reset both State systems -> accumulate sequence gradients -> update once -> verify`

This skill targets BrainTrace `0.2.4`, bundled by BrainX `v2026.7.9`. Do not use `etrace_grad()`, `etrace_evolve()`, `SequenceDriverMixin`, or `ETraceVmap`; those APIs start in BrainTrace `0.2.5`.

Route general `State`, `Module`, transformations, mapped lifecycle operations, and optimizers to BrainState. Route neuron and synapse dynamics to BrainPy-State or BrainCell. Use ordinary BrainState training when offline BPTT is acceptable.

## Underlying principle of BrainTrace

Hidden State represents recurrent memory. It carries neural activity between time steps.

ETP operations represent parameterized paths that need temporal credit. A parameter consumed by an ETP operation can receive an eligibility trace; a parameter consumed only by an ordinary JAX operation cannot.

`ETraceGraph` represents compiled relationships between ETP parameters and hidden-State groups. It records which parameter influences which recurrent state.

Eligibility State represents compressed gradient history. The selected algorithm updates it during each learner call.

## API structure overview

| API family | Responsibility |
|---|---|
| `braintrace.nn` | Prebuilt ETP-aware recurrent, linear, convolutional, sparse, and readout layers. Start here. |
| ETP operations | Mark parameterized operations in a custom layer for eligibility-trace participation. |
| `braintrace.compile()` | Initialize State, discover hidden and parameter paths, build `ETraceGraph`, and return a learner. |
| `D_RTRL`, `pp_prop`, SNN algorithms | Define how eligibility State is represented and updated. |
| Compiled learner | Advance one step, update eligibility State, expose compiler reports, and participate in BrainState differentiation. |

## Choose the memory strategy and compile

Compile one representative time step once; the selected algorithm determines eligibility-memory shape and approximation.

| API | Description |
|---|---|
| `braintrace.nn.MiniGRU(...)` | Use for the canonical recurrent model; it carries hidden State and applies ETP-aware parameter operations. |
| `braintrace.nn.Linear(...)` | Use for an ETP-aware projection; a readout that does not return to hidden State remains trainable but needs no temporal trace. |
| `braintrace.D_RTRL` | Use when parameter-shaped trace memory fits; it applies a diagonal hidden-recurrence approximation. |
| `braintrace.pp_prop` | Use when parameter-shaped traces are too costly, especially for recurrent SNNs; it stores input/output factors. |
| `braintrace.compile(model, algorithm, example_step, batch_size=..., vmap=...)` | Use once with the real per-step shape; it initializes State, compiles relations, rejects a graph with no online-trainable path, and returns a learner. |
| `learner.report.show(2)` | Use after compilation; level 2 includes diagnostics needed to distinguish included batched relations from rejected duplicate paths. |
| `learner.report.counts`, `etrace_weights`, `excluded_weights` | Use for programmatic participation checks; require zero errors and verify every intended recurrent weight explicitly. |

```python
import brainstate
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
learner = braintrace.compile(
    model,
    braintrace.D_RTRL,
    inputs[0],
    batch_size=1,
)
learner.report.show(2)
assert learner.report.counts['errors'] == 0
assert learner.report.etrace_weights
```

Open `references/pre-built-braintrace-layer.md` when selecting another ETP-aware layer. Open `references/Drtrl.md` for D-RTRL memory and validation. Open `references/pp_pprop workflow.md` when parameter-shaped traces are too large or the model is a recurrent SNN. Open `references/algorithm selection.md` only when neither canonical choice fits.

## Train and evaluate a sequence

Differentiate exactly one learner call per time step, carry the accumulated gradient through `brainstate.transform.scan`, and update parameters once after the sequence.

| API | Description |
|---|---|
| `brainstate.nn.reset_all_states(model, batch_size=...)` | Use at an independent sequence boundary to reset recurrent State. |
| `learner.reset_state(batch_size=...)` | Use at the same boundary to reset eligibility State and the running index. |
| `brainstate.transform.grad(step_loss, weights, ...)` | Use around one learner call; BrainTrace supplies the online parameter gradient from current eligibility State. |
| `brainstate.transform.scan(body, initial_grads, sequence)` | Use to advance time and sum per-step gradients in an explicit carry. |
| `optimizer.update(grads)` | Use once after the complete sequence gradient has been accumulated. |

```python
import jax
import braintools

targets = 0.7 * inputs + 0.2
weights = model.states(brainstate.ParamState)
optimizer = braintools.optim.SGD(lr=0.005)
optimizer.register_trainable_weights(weights)


def reset_sequence():
    brainstate.nn.reset_all_states(model, batch_size=1)
    learner.reset_state(batch_size=1)


def step_loss(x, target):
    return jnp.mean((learner(x) - target) ** 2)


loss_grad = brainstate.transform.grad(
    step_loss,
    weights,
    return_value=True,
)


def train_epoch(_):
    reset_sequence()
    initial_grads = jax.tree.map(
        jnp.zeros_like,
        {key: state.value for key, state in weights.items()},
    )

    def accumulate(grads, sample):
        step_grads, loss = loss_grad(*sample)
        grads = jax.tree.map(
            lambda total, value: total + value,
            grads,
            step_grads,
        )
        return grads, loss

    grads, losses = brainstate.transform.scan(
        accumulate,
        initial_grads,
        (inputs, targets),
    )
    optimizer.update(grads)
    return losses.mean()


losses = brainstate.transform.for_loop(train_epoch, jnp.arange(5))
reset_sequence()
predictions = brainstate.transform.for_loop(learner, inputs)

assert losses[-1] < losses[0]
assert predictions.shape == targets.shape
```

Open `references/batching.md` before adding a batch axis. For compile-owned vmap, reset the returned wrapper through its mapped State collection and inspect the underlying report through `.module`; do not apply a second mapping wrapper. Open the BrainState collective-operations reference when a manual mapped lifecycle is unavoidable.

## Reference routing

Escalate model construction from prebuilt layers to built-in ETP operations, then to a custom ETP primitive only when the previous level cannot express the computation.

| Reference | Open when | Contains |
|---|---|---|
| `references/pre-built-braintrace-layer.md` | Selecting another ETP-aware layer or composing the default model. | Layer families, parameterized-operation choices, and BrainState-owned supporting layers. |
| `references/Drtrl.md` | Using or validating parameter-shaped D-RTRL traces. | Memory scaling, recurrence, sequence resets, scan accumulation, and the BPTT boundary. |
| `references/pp_pprop workflow.md` | Parameter-shaped traces are too costly or a recurrent SNN needs factorized traces. | Input/output factorization, SNN integration, report-window training, and validation. |
| `references/algorithm selection.md` | Neither canonical memory strategy fits the estimator requirements. | BrainTrace `0.2.4` estimators, guarantees, and configuration options. |
| `references/batching.md` | Adding a batch axis, mapped model, or multi-step wrapper. | Mapping ownership, batched compilation, mapped resets, and sequence scan shape. |
| `references/ETP operators.md` | A prebuilt layer cannot express the required parameterized operation. | Built-in ETP operations, participation rules, sparse inputs, units, and transform behavior. |
| `references/custom ETP primitives.md` | No built-in ETP operation can express the computation. | Primitive registration, ETP rules, compiler integration, and validation. |
| `references/customizing_primitive_transforms.md` | An ETP weight needs masking, constraints, normalization, LoRA, or a bias transform. | Transform hooks and raw-parameter gradient attachment. |
| `references/custom algorithms.md` | No built-in BrainTrace `0.2.4` algorithm expresses the research method. | Estimator bases, trace lifecycle, solve hooks, and inspection. |
| `references/compiler_internal.md` | Structured report fields cannot explain custom graph behavior. | Discovery, graph execution, control-flow policy, and low-level diagnostics. |

## Boundaries and common failures

- Do not describe BrainTrace as constant-total-memory training. Memory is constant only in sequence length.
- Do not claim that an online estimator equals BPTT without its documented regime and a reduced gradient-oracle comparison.
- Do not compile inside training or compile from a full sequence; pass one representative time step with the real batch and feature shape.
- Do not treat every warning or excluded candidate as a failed parameter. Inspect `etrace_weights`, `excluded_weights`, and level-2 diagnostics together.
- Reset both recurrent State and eligibility State between independent sequences and verify mapped shapes survive a mapped reset.
- Use exactly one batching owner. Prefer `braintrace.compile(..., vmap=True)` for independent per-sample State or native batched compilation when the model supports it.
- Do not update the optimizer inside the temporal scan when the intended schedule is one update per sequence.
- Do not use BrainTrace `0.2.5` sequence-driver APIs with the bundled `0.2.4` release.
- Use `pp_prop` in new code; `ES_D_RTRL` is its compatibility alias.
