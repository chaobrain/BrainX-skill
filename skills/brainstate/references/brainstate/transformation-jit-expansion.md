# BrainState JIT Expansion

Use this reference after the main BrainState skill's minimal state-aware JIT workflow. It covers supplemental write-back behavior, `JittedFunction` cache controls, static specialization, compilation boundaries, and the boundary where raw `jax.jit` remains appropriate. It does not repeat the basic forward-pass example.

Official sources:

- https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

The examples share these imports:

```python
import jax
import jax.numpy as jnp

import brainstate
```

## Verify Write-Back Across Multiple States

Use this pattern when a compiled call updates more than one piece of hidden state and the caller must observe every update after each invocation. BrainState records and replays the state updates around the compiled executable, so the hidden states remain in sync.

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html

```python
class RunningMean(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.sum = brainstate.HiddenState(jnp.array(0.0))
        self.count = brainstate.HiddenState(jnp.array(0))

    def __call__(self, batch: jax.Array) -> jax.Array:
        self.sum.value += jnp.sum(batch)
        self.count.value += batch.size
        return self.sum.value / self.count.value


tracker = RunningMean()


@brainstate.transform.jit
def update_running_mean(batch: jax.Array) -> jax.Array:
    return tracker(batch)


for step in range(3):
    data = jnp.arange(4.0) + step
    print(f"step {step}: mean={float(update_running_mean(data)):.2f}")

assert float(tracker.sum.value) == 30.0
assert int(tracker.count.value) == 12
```

The three calls print means `1.50`, `2.00`, and `2.50`. Validate both the returned value and the final `State` values when write-back is part of the function's contract.

## Precompile Or Clear Cached Traces

`brainstate.transform.jit` returns a `JittedFunction`. The tutorial names `compile`, `clear_cache`, and `origin_fun` as helper surfaces and demonstrates the first two: `compile()` precompiles an executable for example inputs, while `clear_cache()` explicitly drops cached traces. Subsequent compatible calls otherwise reuse the compiled executable.

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html

```python
@brainstate.transform.jit
def softplus(x: jax.Array) -> jax.Array:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


example = jnp.ones((4,))
softplus.compile(example)
result = softplus(example)

softplus.clear_cache()
result_after_clear = softplus(jnp.linspace(-1.0, 1.0, 5))
```

The routed tutorial identifies `origin_fun` but does not demonstrate how to call it. Do not infer a usage contract from its name alone.

When JIT is disabled globally, BrainState falls back to the original Python implementation automatically:

```python
with jax.disable_jit():
    result_without_jit = softplus(example * 2.0)
```

## Specialize Python Control With Static Arguments

Static-argument handling mirrors `jax.jit`. Use `static_argnums` when a positional value controls Python structure; the tutorial specializes the compiled polynomial program by `degree`.

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html

```python
@brainstate.transform.jit(static_argnums=1)
def polynomial_series(x: jax.Array, degree: int) -> jax.Array:
    powers = [x ** (i + 1) for i in range(degree)]
    coeffs = jnp.arange(1, degree + 1, dtype=x.dtype)
    return jnp.tensordot(
        coeffs,
        jnp.stack(powers, axis=0),
        axes=1,
    )


x = jnp.array([1.0, 2.0])
p1 = polynomial_series(x, 3)
p2 = polynomial_series(x, 3)
p3 = polynomial_series(x, 4)

assert jnp.allclose(p1, jnp.array([6.0, 34.0]))
assert jnp.allclose(p2, p1)
assert jnp.allclose(p3, jnp.array([10.0, 98.0]))
```

Here `degree=3` and `degree=4` select different static specializations. The sources do not enumerate other cache-key or recompilation triggers, so do not add shape-, dtype-, or Python-object rules without a separate official source.

## Keep The Compilation Boundary At A Whole Step

`jit` traces a function on its first call, compiles it with XLA, and reuses the compiled version afterwards. Apply it to a whole forward pass or training step, not to tiny operations. For training, preserve the main skill's `jit(grad(...))` composition as one compiled step instead of wrapping its internal arithmetic separately.

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

The two routed tutorials demonstrate first-call compilation, later executable reuse, explicit precompilation, and cache clearing. They do not demonstrate a lowering API, a synchronized timing protocol, or a benchmark. Do not invent `lower()` or benchmarking instructions in this reference; route performance measurement to a source that documents it.

## Cross Into Raw `jax.jit` Only With A Pure Interface

Use raw `jax.jit` for a pure stateless function when it is the leanest choice. If the computation owns mutable values, pull each value out, pass it through the call signature, and return the updated value for the next call.

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html

```python
def running_mean_pure(carry, batch):
    total, count = carry
    total = total + jnp.sum(batch)
    count = count + batch.size
    return (total, count), total / count


jax_jitted = jax.jit(running_mean_pure)
carry = (jnp.array(0.0), jnp.array(0))

for step in range(3):
    batch = jnp.arange(4.0) + step
    carry, mean = jax_jitted(carry, batch)

total, count = carry
assert float(total) == 30.0
assert int(count) == 12
```

This boundary is explicit: the raw JAX function receives and returns `carry`; no BrainState mutation is hidden inside it. When a `GraphDef` or explicit module reconstruction is required, route to `../state-graph-operations.md` rather than expanding graph splitting here.

## Selection Checklist

- Use the main skill for the canonical state-aware JIT rule and minimal forward script.
- Use this reference for multi-State write-back, `compile()`, `clear_cache()`, static specialization, whole-step boundaries, or a pure-JAX boundary.
- Use `brainstate.transform.jit` for module-centric stateful code.
- Use raw `jax.jit` only when the computation is pure or all mutable values are threaded explicitly.
- Do not claim lowering, benchmark methodology, or undocumented recompilation triggers from these two sources.
