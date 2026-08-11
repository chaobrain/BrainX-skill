# Batch and transform acceleration

Use this reference when `Simulator` is insufficient, a custom rollout needs transformed control flow, parameters or trials must be batched, a JAX benchmark is misleading, or reverse-mode differentiation through a long run exhausts memory.

## Choose the execution primitive

Use `Simulator` first. It already composes State initialization, `jit`, `for_loop`, environment time, monitors, transients, and sampling.

| API | Description |
|---|---|
| `brainmass.Simulator.run(...)` | Use for the standard model or network rollout; it compiles and stacks the complete run without a Python time loop. |
| `brainstate.transform.jit(fn)` | Use for a stable one-step or one-shot stateful computation; later compatible calls reuse the compiled program. |
| `brainstate.transform.for_loop(step, *xs)` | Use for many State-carrying steps that return fixed-structure outputs; it slices leading input axes and stacks per-step results. |
| `brainstate.transform.scan(body, init, xs)` | Use when an explicit carry must flow alongside model State through `body(carry, x) -> (carry, y)`. |
| `brainstate.transform.vmap(fn)(*batched_args)` | Use to map one computation over trial, input, or parameter axes without a Python batch loop. |
| `brainstate.transform.checkpointed_for_loop(...)` | Use for a long differentiated State-carrying rollout when rematerialization is required to reduce backward memory. |
| `brainstate.transform.checkpointed_scan(...)` | Use for the same memory tradeoff when the loop also has an explicit carry. |

Do not wrap a jitted step in a Python time loop. That still dispatches from Python once per step and prevents whole-rollout fusion.

## Write a custom State-carrying rollout

Use `for_loop` when model `State` already carries the dynamics and only monitored outputs need stacking.

```python
import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp

dt = 0.1 * u.ms
node = brainmass.HopfStep(in_size=64, a=0.25, w=0.3)

with brainstate.environ.context(dt=dt):
    node.init_all_states()

    def step(index):
        with brainstate.environ.context(
            i=index,
            t=index * dt,
        ):
            node.update()
        return node.x.value

    trajectory = brainstate.transform.for_loop(
        step,
        jnp.arange(300),
    )

assert trajectory.shape == (300, 64)
```

Return monitored values from the body. Do not append traced arrays to Python lists.

## Use an explicit carry

Use `scan` only when a value outside the model's registered State must pass between steps.

```python
node = brainmass.HopfStep(in_size=8, a=0.25, w=0.3)
node.init_all_states()
drive = 0.05 * jnp.sin(
    2.0 * jnp.pi * jnp.arange(300) / 300
)[:, None]

def body(running_sum, input_value):
    with brainstate.environ.context(t=0.0 * u.ms):
        node.update(input_value)
    output = node.x.value
    return running_sum + output, output

total, trajectory = brainstate.transform.scan(
    body,
    jnp.zeros(8),
    drive,
)

assert total.shape == (8,)
assert trajectory.shape == (300, 8)
```

Use the carry for custom running statistics, external controller State, curriculum State, or a loss accumulator. Do not duplicate State already owned by the model.

## Batch trials and parameters

For independent initial conditions or stochastic trials, use `Simulator.run(batch_size=B)`.

```python
brainstate.random.seed(0)
node = brainmass.HopfStep(
    in_size=4,
    a=0.1,
    w=0.3,
    noise_x=brainmass.OUProcess(
        4,
        sigma=0.1,
        tau=10.0 * u.ms,
    ),
)
result = brainmass.Simulator(node, dt=0.1 * u.ms).run(
    200.0 * u.ms,
    monitors=["x"],
    batch_size=16,
)
assert result["x"].shape == (2000, 16, 4)
```

For parameter batching, build the model inside the function passed to `vmap`. Open `parameter-sweeps-and-regime-analysis.md` for the complete grid pattern.

## Checkpoint long gradients

Replace `for_loop` or `scan` with its checkpointed counterpart only when reverse-mode memory is the limiting resource. Rematerialization lowers peak memory by recomputing forward values during the backward pass; it can increase runtime.

Keep the same body signature and result structure when switching. Validate both loss and gradients before and after the change.

## Benchmark correctly

Define the step once, define the full rollout once, and transform that rollout once outside every timed call. The first call measures compilation plus execution; later compatible calls to the same callable measure steady execution.

```python
import time

import brainmass
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

dt = 0.1 * u.ms
node = brainmass.HopfStep(in_size=64, a=0.25, w=0.3)
brainstate.nn.init_all_states(node)
steps = jnp.arange(300)

def step(index):
    with brainstate.environ.context(i=index, t=index * dt):
        node.update()
    return node.x.value

def rollout():
    with brainstate.environ.context(dt=dt):
        return brainstate.transform.for_loop(step, steps)

run_once = brainstate.transform.jit(rollout)

start = time.perf_counter()
first_result = run_once()
jax.block_until_ready(first_result)
first_call_seconds = time.perf_counter() - start

steady_seconds = []
for _ in range(5):
    start = time.perf_counter()
    result = run_once()
    jax.block_until_ready(result)
    steady_seconds.append(time.perf_counter() - start)
```

- Warm up the exact callable and shape once before reporting steady timing; report the first call separately when compilation cost matters.
- Block every timed result before stopping the timer because JAX dispatch is asynchronous.
- Restore the same initial State outside the timed region, or include the same reset inside every measured callable, when comparisons require identical starting conditions.
- Batch enough work to occupy an accelerator; one small neural-mass node may run faster on CPU.
- Keep `dt`, duration, shape, dtype, and monitor set identical across comparisons.

## Common failures

- A Python `for` or `while` loop around repeated State updates.
- Raw `jax.jit`, `jax.vmap`, or `jax.lax.scan` applied directly to State-aware code.
- `scan` used when no explicit carry exists.
- A model constructed outside a parameter-mapped function and mutated across mapped values.
- Variable-shape or Python-object results returned from transformed loop bodies.
- A step or rollout function reconstructed inside each timed repetition.
- JAX timing measured without warmup or `block_until_ready`.
- Checkpointing enabled before memory pressure is established.

## Official source

- `https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html`
