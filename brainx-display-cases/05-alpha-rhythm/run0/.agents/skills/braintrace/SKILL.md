---
name: braintrace
description: BrainTrace relieves training-memory pressure in recurrent and spiking models by replacing BPTT's sequence-length-dependent graph with eligibility traces. Use this skill for memory-efficient temporal training, long sequences, BPTT out-of-memory problems, activation-memory reduction, and BrainTrace estimator selection or compilation. Do not use for general speed optimization, offline BPTT, or model dynamics.
---

# BrainTrace

## Purpose and boundary

Use BrainTrace when recurrent or spiking training is limited by BPTT memory. Its broad purpose is to relieve sequence-length memory pressure: it accumulates eligibility traces during the forward sequence instead of retaining the full trajectory for reverse-mode differentiation.

Online learning is the mechanism for this memory efficiency, not the routing goal. Memory becomes constant in sequence length, not constant in total: eligibility storage still depends on the batch size, participating parameters, hidden dimensions, and selected algorithm.

Follow this path:

`build with braintrace.nn -> choose a trace representation -> compile once -> inspect -> reset both State systems -> drive the sequence -> update -> verify`

Route general `State`, `Module`, transformations, and optimizer behavior to BrainState. Route neuron and synapse dynamics to BrainPy-State or BrainCell. Use ordinary BrainState training when offline BPTT is acceptable and eligibility traces are not required.

## Underlying principle of BrainTrace

Hidden State represents the model's recurrent memory. It carries neural activity from one time step to the next.

ETP operations represent parameterized paths that need temporal credit. The operation marks participation: a parameter consumed by a BrainTrace ETP operation can receive an eligibility trace; the same parameter consumed only by an ordinary JAX operation cannot.

`ETraceGraph` represents the compiled relationship between ETP parameters and hidden-State groups. It tells the learner which parameter influences which recurrent state.

Eligibility State represents compressed gradient history. The selected algorithm updates it during each forward step, so training does not retain the unrolled sequence graph.

## API structure overview

| API family | Responsibility |
|---|---|
| `braintrace.nn` | Prebuilt ETP-aware recurrent, linear, convolutional, sparse, and readout layers. Start here. |
| ETP operations | Mark parameterized operations in a custom layer for eligibility-trace participation. |
| `braintrace.compile()` | Discover hidden State and ETP parameter paths, build `ETraceGraph`, and return a learner. |
| `D_RTRL`, `pp_prop`, other algorithms | Define how eligibility State is represented and updated. |
| Compiled learner | Inspect compilation, reset eligibility State, execute sequences, and return online gradients. |

## Choose the memory strategy and compile

Compile one representative time step once; the chosen algorithm determines the eligibility-memory shape and approximation.

| API | Description |
|---|---|
| `braintrace.nn.MiniGRU(...)` | Use for the canonical recurrent model; it carries hidden State and applies ETP-aware parameter operations. |
| `braintrace.nn.Linear(...)` | Use for an ETP-aware projection; the compiler treats it as non-temporal when its output does not return to hidden State. |
| `braintrace.D_RTRL` | Use as the default recurrent estimator when a parameter-shaped trace fits memory; it avoids BPTT activation storage but uses a diagonal hidden-recurrence approximation. |
| `braintrace.pp_prop` | Use when the parameter-shaped trace is too costly, especially for recurrent SNNs; it stores input/output factors and introduces a stronger factorization approximation. |
| `braintrace.compile(model, algorithm, example_step, batch_size=...)` | Use once with the real per-step shape; it initializes State, compiles parameter-to-hidden relations, and returns the learner. |
| `learner.report.show(1)` | Use immediately after compilation; it shows hidden groups, traced weights, excluded weights, and exclusion reasons. |
| `learner.show_graph()` | Use when the report needs structural confirmation; it displays the compiled parameter-to-hidden relationships. |

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
targets = 0.7 * inputs + 0.2

learner = braintrace.compile(
    model, braintrace.D_RTRL, inputs[0], batch_size=1
)
learner.report.show(1)
learner.show_graph()
```

Open `references/pre-built-braintrace-layer.md` when selecting another ETP-aware layer. Open `references/Drtrl.md` for D-RTRL memory, recurrence, and validation. Open `references/pp_pprop workflow.md` when parameter-shaped traces are too large or the model is a recurrent SNN. Open `references/algorithm selection.md` only when neither canonical choice fits.

## Train and evaluate a sequence

Let the learner own the time loop and eligibility updates; reset recurrent State and eligibility State together at every independent sequence boundary.

| API | Description |
|---|---|
| `brainstate.nn.reset_all_states(model, batch_size=...)` | Use at an independent sequence boundary; it resets the model's recurrent State. |
| `learner.reset_state(batch_size=...)` | Use at the same boundary; it resets eligibility State and the learner's time index. |
| `learner.etrace_evolve(inputs, return_outputs=True)` | Use for loss-free execution or evaluation; it advances hidden and eligibility State and returns stacked outputs when requested. |
| `learner.etrace_grad(..., step_fn=..., reduction=...)` | Use for supervised online learning; it drives the sequence, accumulates per-step gradients, applies masking and reduction, and returns gradients. |
| `optimizer.update(grads)` | Use after one sequence gradient has been reduced; it updates the registered parameters. |

```python
import braintools

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
    grads, step_losses = learner.etrace_grad(
        inputs,
        targets,
        step_fn=step_loss,
        reduction="mean",
        return_value=True,
    )
    optimizer.update(grads)
    return step_losses.mean()

initial_loss = evaluate()
losses = brainstate.transform.for_loop(train_epoch, jnp.arange(25))
final_loss = evaluate()

assert losses.shape == (25,)
assert final_loss < initial_loss
```

Open `references/batching.md` before adding a batch axis or mapping wrapper; it identifies which API must own mapping. Do not wrap `etrace_grad()` or `etrace_evolve()` in another time scan, and make `step_fn` call the learner exactly once.

## Reference routing

Escalate model construction from prebuilt layers to built-in ETP operations, then to a custom ETP primitive only when the previous level cannot express the computation.

| Reference | Open when | Contains |
|---|---|---|
| `references/pre-built-braintrace-layer.md` | Selecting another ETP-aware layer or composing the default model. | Layer families, parameterized-operation choices, and BrainState-owned supporting layers. |
| `references/Drtrl.md` | Using or validating parameter-shaped D-RTRL traces. | Memory scaling, recurrence, sequence resets, and the BPTT comparison boundary. |
| `references/pp_pprop workflow.md` | Parameter-shaped traces are too costly or a recurrent SNN needs factorized traces. | Input/output factorization, decay selection, SNN integration, and validation. |
| `references/algorithm selection.md` | Neither canonical memory strategy fits the estimator requirements. | Other built-in estimators, guarantees, configuration axes, and sequence-driver options. |
| `references/batching.md` | Adding a batch axis, mapped model, or multi-step wrapper. | Mapping ownership, batched compilation, State layout, and sequence-data wrappers. |
| `references/ETP operators.md` | A prebuilt layer cannot express the required parameterized operation. | Built-in ETP operations, participation rules, units, and transform behavior. |
| `references/custom ETP primitives.md` | No built-in ETP operation can express the computation. | Primitive registration, ETP rules, compiler integration, and validation. |
| `references/customizing_primitive_transforms.md` | An ETP weight needs masking, constraints, normalization, LoRA, or a bias transform. | Transform hooks and raw-parameter gradient attachment. |
| `references/custom algorithms.md` | No built-in algorithm or `ETraceConfig` expresses the research method. | Estimator bases, trace lifecycle, solve hooks, and inspection. |
| `references/compiler_internal.md` | `learner.report` and `learner.graph` cannot explain custom graph behavior. | Discovery, graph execution, control-flow policy, and low-level diagnostics. |

## Boundaries and common failures

- Do not describe BrainTrace as constant-total-memory training. Its memory is constant only in sequence length; trace storage still scales with the selected estimator and model.
- Do not claim that an online estimator equals BPTT unless its documented mathematical regime and a reduced gradient-oracle comparison establish that result.
- Do not compile inside training or compile from a full sequence; pass one representative time step with the real batch and feature shape.
- Do not treat every excluded weight as an error. A readout that does not feed hidden State is correctly reported as non-temporal and still receives instantaneous gradients.
- Reset both recurrent State and eligibility State between independent sequences.
- Use exactly one batching owner; do not map an already mapped model again in `braintrace.compile()`.
- Use `pp_prop` in new code; `ES_D_RTRL` is its compatibility alias.
