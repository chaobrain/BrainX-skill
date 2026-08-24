---
name: brainx-acceleration
description: Audits and refactors BrainX/BrainState simulation code for performance using state-aware BrainState transform patterns. Use for speed, acceleration, GPU performance, batching, vectorization, parameter sweeps, many neurons/trials, compile/runtime separation, memory reduction, throughput, multi-device execution, or performance-audit requests.
---

# BrainX Acceleration

Use when auditing or refactoring BrainX/BrainState simulation code for performance. Systematically finds NumPy/Python-loop inefficiencies and rewrites them into state-aware BrainState transform patterns: `jit`, scan/control flow, `vmap`, `grad`, `checkpointed_scan`, `pmap2`/`shard_map`, `State`/`ParamState`/`RandomState`, PyTrees, batching, RNG, and shape-stable compiled simulation logic.

Usually reached from `skills/brainstate/SKILL.md` when speed, acceleration, GPU performance, batching, vectorization, parameter sweeps, many neurons/trials, compile/runtime separation, memory reduction, throughput, multi-device execution, or a performance audit becomes the user request.

Use this skill to turn slow BrainX/BrainState simulation code into transform-friendly, state-aware array programs. The goal is not “sprinkle `jit` everywhere.” The goal is to expose the correct simulation axes to BrainState transformations while preserving state semantics, RNG behavior, shapes, gradients, and numerical results.

Optimize in this order: **correctness -> state/RNG safety -> shape stability -> warm runtime -> memory -> multi-device scale**.

## Reference usage

For any nontrivial State-aware rewrite, route through `skills/brainstate/SKILL.md` first. Load this skill's local references only when the specific rewrite needs exact transform or randomness semantics.

Do not invent BrainState arguments. If the exact API is uncertain, inspect local usage/tests or reference markdowns and state the assumption in the patch notes.

## Core mental model

Convert this:

```text
Python controls loops and mutable objects; arrays are small operations inside the loop.
```

into this:

```text
Arrays/states represent populations; BrainState transforms control repeated execution.
```

Axis map:

| Axis | Meaning | Slow NumPy/Python pattern | BrainState rewrite target |
|---|---|---|---|
| `N` | neurons, synapses, features, compartments | loop over scalar cells/synapses | population-shaped arrays + `jnp` ops inside `jit` |
| `T` | time/recurrent steps | Python loop calling `step` | `scan`, `for_loop`, `while_loop`; usually wrapped by `jit` |
| `B` | batch, trials, stimuli, subjects | loop over examples/trials | `vmap` or explicitly batched state |
| `E` | ensemble, parameter sweep, initial conditions, seeds | loop over model copies/configs | `vmap` with mapped states/parameters/RNG |
| `P` | trainable parameters/sensitivities | finite differences/manual perturbation loops | `grad`, `value_and_grad`, `jacrev`, `jacfwd` over args or `ParamState` |
| `D` | devices | multiprocessing/manual sharding | `pmap2`, `shard_map`, explicit sharding after single-device cleanup |

## Non-negotiable rules

- Use `brainstate.transform.*` for stateful BrainState code. Use raw `jax.*` transforms only for pure functions with explicit array inputs/outputs and no BrainState state writes.
- Use `jax.numpy as jnp` in transformed numerical paths. Keep `numpy as np` only for host preprocessing, file I/O, offline analysis, or static setup outside transforms.
- Mutable simulation quantities belong in explicit `State`, `ParamState`, or BrainState random-stream objects. Do not hide changing transformed values in ordinary Python attributes.
- Hot-path shapes, dtypes, and PyTree structure must be stable. Use masks, padding, bucketing, or static config instead of runtime shape creation.
- Do not automatically densify sparse/event/structured connectivity. First choose the right mathematical representation, then transform it.
- Do not claim acceleration from the first call. Separate compile time from warm execution time.
- Prefer small reviewable diffs with tiny deterministic equivalence tests.

## Investigation protocol

### 1. Build a hot-path inventory

For each candidate function (`step`, `update`, `forward`, `run`, `simulate`, `loss`, `train_step`, data/RNG helper), record:

```text
function/call site:
inputs: shapes, dtypes, axis meanings
states read: State/ParamState/RandomState, shapes, shared or per-trial
states written: State/ParamState/RandomState, shapes, shared or per-trial
loops: axis, trip count, nested axes
current transforms: none / raw jax / brainstate.transform
outputs: full history vs summary metrics
RNG: source, key/stream split policy, per-step/per-trial independence
shape changes: yes/no, cause
host interaction: print/log/float/item/np.asarray/callbacks
connectivity: dense/sparse/event/structured
```

Use searches as a first pass, then read the code before editing:

```bash
rg -n "import numpy|np\." .
rg -n "for .*range|while " .
rg -n "\.append\(|np\.array\(|np\.stack\(|jnp\.stack\(|concatenate\(" .
rg -n "jax\.jit|jax\.grad|jax\.vmap|brainstate\.transform" .
rg -n "\.value\s*=|\[[^\]]+\]\s*=" .
rg -n "random|np\.random|jax\.random|RandomState|split_key|seed" .
rg -n "finite|epsilon|eps|perturb|theta_plus|theta_minus|jacobian|hessian" .
rg -n "\.item\(|float\(|int\(|bool\(|np\.asarray|print\(|logging\." .
rg -n "jit\(|grad\(|vmap\(|scan\(" .
```

### 2. Classify every inefficient pattern

Use these labels in audit notes and patch comments when useful.

| Label | Evidence | Why slow/risky | Rewrite target |
|---|---|---|---|
| `loop-N-scalar` | `for i in range(N)` updates neuron/synapse scalars | Python overhead; no fused population kernel | array-shaped state + `jnp.where`/broadcast/scatter/sparse ops under `jit` |
| `object-neurons` | list/dict of neuron/synapse objects updated one by one | compiler sees Python objects, not arrays | one population/module with state arrays shaped `[N,...]` |
| `loop-T-python` | Python loop calls `step(x[t])` | dispatch per step; no compiled recurrent loop | `scan`, `for_loop`, or `while_loop` inside `jit` |
| `dynamic-list-history` | append outputs/states then stack | dynamic Python accumulation; memory waste | `scan` outputs or summary carry |
| `memory-T-full-history` | stores all `V[t]`/state when only rates/loss needed | unnecessary memory; worse autodiff storage | carry summary; record sparse snapshots; `checkpointed_scan` for differentiable long runs |
| `loop-B-python` | loop over trials/batch/stimuli | serial Python batch | `vmap` or batched state |
| `loop-E-sweep` | loop over seeds/configs/model copies | serial ensemble sweep | `vmap` with leading `[E,...]` state/param/RNG axes |
| `loop-P-finite-diff` | `eps`, `theta_plus`, `theta_minus`, perturb each param | O(P) or O(2P) simulations; noisy gradients | `grad`/`value_and_grad` over args or `ParamState` |
| `raw-jax-stateful` | `jax.jit/grad/vmap` wraps BrainState module/state writes | state updates may be lost or mishandled | `brainstate.transform.*` or explicit split/pass/return/merge |
| `np-in-transform` | `np.*` in code used by `jit/grad/vmap/scan` | host execution, tracer conversion, frozen constants | replace with `jnp.*` or move outside transform |
| `python-control-array` | Python `if/while/bool/int/float` depends on array value | tracer errors or host sync | `jnp.where`, `cond`, `while_loop`, `for_loop`, masks |
| `wrong-jit-boundary` | jitting tiny helpers, or making jitted functions inside loops | compile overhead; poor fusion | jit largest stable unit: `run`, `loss`, `train_step` |
| `transform-inside-loop` | `jit(...)`, `vmap(...)`, `grad(...)` constructed repeatedly | repeated tracing/compilation | define transforms once at module/init scope |
| `dynamic-shape` | shape depends on data/spikes/runtime length | recompilation or unsupported traced shape | static config, padding, masks, bucketing |
| `hidden-mutable-attr` | ordinary attrs mutated in transformed method | untracked state semantics | `State`/`ParamState`/explicit return value |
| `state-vmap-ambiguous` | `vmap` over stateful model without explicit state-axis decision | shared vs independent state bugs | set/verify `state_in_axes`, `state_out_axes`, unexpected state policy |
| `rng-reuse` | same key/seed reused across trials/time; `np.random` in hot path | correlated trials; unreproducible behavior | BrainState random streams or explicit split; map RNG axis under `vmap` |
| `host-sync` | `print`, `.item()`, `float(x)`, `np.asarray(x)` in hot loop | device-host synchronization | aggregate metrics; log outside compiled loop; debug-only callbacks |
| `accidental-densify` | converts sparse/event connectivity to dense `[N,N]` | memory/bandwidth blow-up | preserve sparse/event/structured representation |
| `already-fused` | large `jnp` matmul/conv/reduction already dominates | transforms may not help much | benchmark first; avoid churn |

### 3. Rank findings before rewriting

Score each finding:

```text
impact = loop trip count × work per iteration × call frequency
risk = state/RNG/gradient/memory semantic risk
confidence = evidence quality + shape clarity + API certainty
```

Rewrite high-impact, high-confidence items first. Do not refactor low-impact style issues unless they block a larger transform.

## Rewrite playbook

### A. Population axis `N`: scalar/object loop -> array state

Bad:

```python
for i in range(num_neurons):
    v[i] += dt * (-v[i] + i_ext[i]) / tau[i]
    if v[i] > threshold[i]:
        spike[i] = 1
        v[i] = reset[i]
```

Good shape pattern:

```python
v_new = V.value + dt * (-V.value + i_ext) / tau
spike = v_new > threshold
V.value = jnp.where(spike, reset, v_new)
```

Use `jnp.where`/masks for elementwise conditional reset. Use `.at[]` scatter or segment/sparse/event operations when updates are indexed. Avoid Python object-per-neuron design in hot paths.

### B. Time axis `T`: Python simulation loop -> transformed control flow

Bad:

```python
ys = []
for t in range(T):
    ys.append(step(xs[t]))
y = jnp.stack(ys)
```

Good full-history pattern:

```python
def body(carry, x_t):
    y_t = net(x_t)          # may read/write BrainState State
    return carry, y_t

@brainstate.transform.jit
def run(xs_T):
    _, ys_T = brainstate.transform.scan(body, None, xs_T)
    return ys_T
```

Good summary-only pattern:

```python
def body(summary, x_t):
    y_t = net(x_t)
    summary = summary + metric(y_t)
    return summary, None
```

Use `checkpointed_scan` when differentiating through long `T` and activation/state storage dominates memory. Before applying exact control-flow APIs, read `references/brainstate/brainstate-control-flow-patterns.md` if present.

### C. Batch/trial axis `B`: Python batch loop -> `vmap` or batched state

Bad:

```python
outs = []
for b in range(B):
    outs.append(run_one(xs_B[b]))
outs = jnp.stack(outs)
```

Good pattern:

```python
run_batch = brainstate.transform.jit(
    brainstate.transform.vmap(run_one, in_axes=0, out_axes=0)
)
outs = run_batch(xs_B)
```

Before rewriting, decide for every state:

```text
shared across B?    parameters, shared read-only config, global constants
independent per B?  membrane voltage, refractory state, synaptic traces, trial RNG, optimizer-free simulation state
written back?       final per-trial state, shared aggregate, or discarded temporary?
```

If independent state is required, use BrainState `vmap` state-axis controls such as `state_in_axes`/`state_out_axes` according to the local API. Read `references/brainstate/transformation-vmap-expansion.md` before editing exact arguments.

### D. Ensemble/sweep axis `E`: serial model copies -> mapped states/params

Bad:

```python
for e in range(E):
    set_params(params[e])
    results.append(run(inputs, seed=e))
```

Good shape intent:

```text
params/state/RNG: [E, ...]
inputs:          [E, T, ...] or shared [T, ...]
outputs:         [E, T, ...] or [E, summary]
```

Good transform intent:

```python
run_ensemble = brainstate.transform.jit(
    brainstate.transform.vmap(
        run_one_model,
        in_axes=..., out_axes=...,
        state_in_axes=..., state_out_axes=...,
    )
)
```

Use this for parameter sweeps, random initial conditions, seeds, lesions, perturbations, and independent circuit replicas. Verify memory: mapping `E` full histories can be huge.

### E. Parameter axis `P`: manual gradients -> `grad`

Bad finite differences:

```python
for i in range(num_params):
    theta_p = theta.at[i].add(eps)
    theta_m = theta.at[i].add(-eps)
    g = (loss(theta_p) - loss(theta_m)) / (2 * eps)
```

Good training intent:

```python
params = model.states(brainstate.ParamState)

def loss_fn(x, target):
    pred = run(x)
    return jnp.mean((pred - target) ** 2)

grad_fn = brainstate.transform.grad(loss_fn, grad_states=params, return_value=True)

@brainstate.transform.jit
def train_step(x, target):
    grads, loss = grad_fn(x, target)
    optimizer.update(grads)
    return loss
```

Use `grad(loss_over_batch)` for normal batch training. Use `vmap(grad(single_loss))` only for per-sample gradients, influence functions, uncertainty, or diagnostics; it can be memory-heavy. Read `references/brainstate/transformation-grad-expansion.md` before exact state/argument gradient edits.

### F. JIT boundary: compile the largest stable unit

Prefer:

```python
@brainstate.transform.jit
def train_step(batch):
    loss = ...
    grads = ...
    optimizer.update(grads)
    return loss, metrics
```

Avoid:

```python
for t in range(T):
    y = brainstate.transform.jit(tiny_op)(x[t])
```

Common good boundaries:

```text
step        if called many times and shape-stable
run         when it contains scan/control flow
loss_fn     when repeatedly evaluated
train_step  best boundary for training: forward + loss + grad + optimizer update
```

Read `references/brainstate/transformation-jit-expansion.md` before changing static args, diagnosing recompilation, or replacing raw `jax.jit` around stateful code.

### G. Composition patterns

Use these as intent patterns, then adapt to local BrainState API:

```text
single trial over time:              jit(scan(step))
batch of independent trials:         jit(vmap(run_one_trial))
time loop with batched state:        jit(scan(vmapped_step_or_batched_step))
ensemble of models:                  jit(vmap(run_one_model, mapped states))
normal training:                     jit(train_step using grad(loss_over_batch))
per-sample gradients:                jit(vmap(grad(single_sample_loss)))
long differentiable sequence:        jit(checkpointed_scan(step))
multi-device batch/ensemble/shard:   pmap2/shard_map after single-device cleanup
```

Axis order matters. Choose the order that preserves state semantics and minimizes memory. For example, `vmap(scan(step))` creates independent trials with independent recurrent state; `scan(vmap(step))` uses a time loop whose step is already batched.

### H. Randomness

Tiny RNG block: use one reproducible root seed or stream, then split/fold/map independent randomness for trials, parameter sweeps, stochastic dynamics, dropout/noise, and transformed execution. Route deeper seed/key restoration details to `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`.

Bad:

```python
for b in range(B):
    y = run_one(xs[b], key)   # same key reused
```

Good intent:

```text
one reproducible root seed/stream
split/fold/map independent keys for B/E/trials
split or advance per time step when stochastic dynamics require it
store RNG in BrainState-compatible state/stream if transformed code mutates it
```

Under `vmap`, verify each mapped element receives independent randomness. RNG correctness is part of validation, not a style preference.

### I. Multi-device `D`

Use `pmap2`, `shard_map`, or explicit sharding only after the single-device program is already transform-friendly. Good candidates: large independent trials, large ensembles, or explicit population/connectivity shards. Bad candidates: tiny Python-bound loops, logging-heavy code, shape-changing workloads, or dense memory blow-ups.

## Implementation sequence for Codex-style agents

1. **Do not rewrite immediately.** First produce an inventory and pattern table.
2. **Pick the highest-impact axis.** Usually `T`, then `N`, then `B/E`, then `P`, then `D`.
3. **Read reference markdowns only for transforms you will actually use.** Do not load all references by default.
4. **Write or identify a tiny deterministic test before refactoring.** Include final state checks, not only outputs.
5. **Refactor one semantic unit at a time.** Keep public APIs stable unless the user asks for deeper redesign.
6. **Preserve exact state ownership.** When changing `vmap`, explicitly document which states are shared, mapped, written back, or discarded.
7. **Preserve RNG semantics or intentionally improve them.** Document seed/key splitting.
8. **Avoid cosmetic churn.** Do not convert low-impact host setup code from `np` to `jnp` unless it enters transforms.
9. **Benchmark warm execution only after correctness.** Include compile time separately if measured.
10. **Leave API assumptions visible.** If BrainState syntax is uncertain, mark it and point to the needed reference markdown/test.

## Validation gates

A rewrite is not done until these are addressed:

| Gate | Required check |
|---|---|
| Outputs | old vs new on tiny deterministic input, with tolerance stated |
| Final state | compare final `State`/`ParamState` values where semantics require it |
| Shape stability | same shapes/dtypes/PyTree across repeated calls; no accidental recompiles |
| RNG | independent mapped trials/seeds; reproducible root seed behavior |
| Gradients | finite gradients; intended `ParamState`s receive grads; tiny training step moves loss or parameter in expected direction |
| Timing | cold compile separated from warm runtime; synchronize before timing when needed |
| Memory | full-history vs summary scan choice justified; no accidental dense `[N,N]` blow-up |
| State under `vmap` | shared vs independent state documented and tested |
| Host sync | no print/item/float/np.asarray in hot path except debug-only code |

## When not to transform

Be realistic. A transform may not help when:

- the workload is tiny and compile overhead dominates;
- the hot path is already a large fused `jnp`/XLA operation;
- runtime shapes genuinely change each call and cannot be bucketed/masked;
- performance is limited by memory bandwidth or sparse representation choices;
- I/O, logging, plotting, or Python data loading dominates;
- the proposed rewrite would densify a sparse/event model;
- state/RNG semantics are not understood well enough to preserve correctness.

In these cases, report the bottleneck and propose a safer benchmark or representation change instead of forcing `jit`/`vmap`.

## Required response format when using this skill

```markdown
## BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| ... | `loop-T-python` | `for t ... step(...)` | T | High | Medium | `scan` inside `jit` | High |

## Patch / rewrite plan

[Small concrete diff or code blocks. State which reference markdown was used for exact transform semantics. State API assumptions if any.]

## Validation plan

[Old/new tiny test, final-state check, RNG check, shape/recompile check, gradient check if training, warm timing method.]

## Remaining risks

[State sharing, RNG independence, sparse/dense memory, dynamic shapes, numerical differences, unresolved API questions.]
```

If the user asks for an actual code rewrite, include the patch. If the user asks only for an audit, do not rewrite yet; provide the table and prioritized plan.


## Reference routing

Do not guess exact BrainState transform syntax when changing transform behavior. `skills/brainstate/SKILL.md` is the required upstream route before any nontrivial State-aware rewrite.

This skill owns local transform and randomness references for the exact semantics it audits:

```text
brainx-acceleration-audit/
├── brainstate/
│   ├── brainstate-control-flow-patterns.md
│   ├── transformation-jit-expansion.md
│   ├── transformation-vmap-expansion.md
│   └── transformation-grad-expansion.md
└── brainstate-randomness-reproducibility/
    └── randomness-and-reproducibility.md
        └── advanced-randomness.md
```

### First-layer routes

| Route | Open when |
|---|---|
| `skills/brainstate/SKILL.md` | Any nontrivial State-aware rewrite |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-jit-expansion.md` | JIT boundaries, static arguments, recompilation, or benchmarking |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-vmap-expansion.md` | Batch, trial, ensemble, State-axis, or RNG mapping |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-grad-expansion.md` | Finite-difference replacement, training gradients, or `ParamState` differentiation |
| `skills/brainx-acceleration-audit/references/brainstate/brainstate-control-flow-patterns.md` | Time or recurrent loops, `scan`, `for_loop`, `while_loop`, or checkpointing |
| `skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` | Seed/key restoration or independent mapped randomness |

### Nested randomness

The acceleration skill and its transform references route only to the local randomness parent. That parent alone may select `skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/advanced-randomness.md` for advanced stream, mapped-key, or restoration behavior. Do not route directly to the advanced child.

## Script references

No fixed or standalone script is bundled with this skill. Build validation cases from the code under audit.
