# Audit and rewrite patterns

Use this reference to classify a hot path or compare execution-axis patterns after reading the acceleration skill.

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

Use these labels in audit notes when useful.

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

Use `checkpointed_scan` when differentiating through long `T` and activation/state storage dominates memory. Before applying exact control-flow APIs, read `brainstate` → `references/brainstate/brainstate-control-flow-patterns.md`.

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

If independent state is required, use `vmap(..., in_states=..., out_states=...)` for explicit State instances or `vmap2(..., state_in_axes=..., state_out_axes=...)` for State filters. Read `brainstate` → `references/brainstate/transformation-vmap-expansion.md` before editing exact arguments.

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
    brainstate.transform.vmap2(
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

Use `grad(loss_over_batch)` for normal batch training. Use `vmap(grad(single_loss))` only for per-sample gradients, influence functions, uncertainty, or diagnostics; it can be memory-heavy. Read `brainstate` → `references/brainstate/transformation-grad-expansion.md` before exact state/argument gradient edits.

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

Read `brainstate` → `references/brainstate/transformation-jit-expansion.md` before changing static args, diagnosing recompilation, or replacing raw `jax.jit` around stateful code.

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

Tiny RNG block: use one reproducible root seed or stream, then split/fold/map independent randomness for trials, parameter sweeps, stochastic dynamics, dropout/noise, and transformed execution. Route deeper seed/key restoration details to `brainstate` → `references/brainstate/randomness-and-reproducibility.md`.

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

## Audit response template

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
