# BrainState transformed diagnostics

Use this reference after `skills/package-skills/brainstate/SKILL.md` when runtime values, State
updates, or failures must be inspected from transformed BrainState code. The
skill owns the core transformation model; this reference only adds diagnostics
that execute after tracing or turn value-dependent failures into data or
runtime exceptions.

## Choose the diagnostic outcome

Inside a compiled `jit` function, ordinary Python `assert` and `if` statements that depend on array values do not work because those values are abstract tracers at trace time. A plain `print(x)` also runs only at trace time and shows a tracer rather than the runtime value. Select the tool by what the caller needs:

| Need | Tool |
|---|---|
| Return failures for inspection | `brainstate.transform.checkify` with `check` |
| Raise on a bad runtime condition | `brainstate.transform.jit_error_if` |
| Find the primitive that first produces NaN or Inf | `brainstate.transform.debug_nan` or `debug_nan_if` |
| Print concrete runtime values | `jax.debug.print` |
| Send runtime values to Python for richer inspection | `jax.debug.callback` |
| Pause only when a runtime predicate is true | `brainstate.transform.breakpoint_if` |

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html

## Return inspectable errors with `checkify`

`checkify` transforms a function so that it returns an `(error, result)` pair instead of raising. The error object is inert until inspected: `error.get()` returns `None` if every check passed or the failure message otherwise. Bundle an explicit value condition with its checked function:

```python
import jax.numpy as jnp
import brainstate.transform as T


def safe_log(x):
    T.check(jnp.all(x > 0), 'x must be positive, got {}', x)
    return jnp.log(x)


checked = T.checkify(safe_log, errors=T.user_checks)

err, out = checked(jnp.array([1.0, 2.0]))
print(err.get())  # None

err, out = checked(jnp.array([-1.0, 2.0]))
print(err.get())  # x must be positive ...
```

Call `T.check_error(err)` at an outer boundary to turn a captured error back into an exception; it raises when the error is set and otherwise does nothing.

For failures that do not need an explicit `T.check(...)`, choose a predefined `errors` set:

| Set | Detects |
|---|---|
| `T.user_checks` | Explicit `check(...)` assertions |
| `T.nan_checks` | NaN values produced by any primitive |
| `T.float_checks` | NaN and Inf from floating-point operations |
| `T.div_checks` | Division by zero |
| `T.index_checks` | Out-of-bounds array indexing |
| `T.all_checks` | Every category above |

```python
nan_checked = T.checkify(lambda x: jnp.log(x), errors=T.nan_checks)
err, _ = nan_checked(jnp.array([-1.0]))
print(err.get())

index_checked = T.checkify(lambda arr, i: arr[i], errors=T.index_checks)
err, _ = index_checked(jnp.arange(3), 10)
print(err.get())
```

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html

## Raise from a compiled step with `jit_error_if`

Use `jit_error_if(pred, msg)` when a bad runtime condition should fail loudly rather than be threaded outward as an error object. It raises when `pred` is true inside `brainstate.transform.jit`, making it the documented guard for a training or simulation step precondition.

```python
import brainstate


@brainstate.transform.jit
def reciprocal(x):
    T.jit_error_if(
        jnp.any(x == 0.0),
        'reciprocal received a zero entry',
    )
    return 1.0 / x
```

If any entry is zero at runtime, the call raises with the supplied message instead of returning `inf`. The tutorial states that the check compiles to a cheap conditional and is safe to leave in production steps.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html

## Locate the first nan or inf with `debug_nan`

`debug_nan(fn, *args)` runs `fn` with NaN/Inf detection enabled and raises as soon as a non-finite value appears, naming the offending primitive. Keep the tested computation and the detector together:

```python
def unstable(x):
    y = x * 1e20
    return jnp.exp(y)


T.debug_nan(unstable, jnp.array([10.0]))
```

Use `T.debug_nan_if(has_nan, fn, *args)` when an upstream flag already suspects trouble and the somewhat costly detection should run only conditionally.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html

## Inspect runtime values and `State` mutations

`jax.debug.print` defers printing until runtime, so each execution prints concrete values rather than trace-time tracers. It can observe real `State` values on both sides of a mutation:

```python
import jax
import brainstate


class Accumulator(brainstate.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.total = brainstate.ShortTermState(jnp.zeros(size))

    def __call__(self, x):
        jax.debug.print('before: {s}', s=self.total.value)
        self.total.value = self.total.value + x
        jax.debug.print('after : {s}', s=self.total.value)
        return self.total.value


acc = Accumulator(3)
step = brainstate.transform.jit(acc)
_ = step(jnp.array([1.0, 2.0, 3.0]))
```

The same runtime print is documented inside `grad` and `vmap`: in a loss function it fires during the forward pass, and under `vmap` it runs once per batch element. Use `{name}` placeholders with keyword arguments as shown above.

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html

## Send runtime values to a python callback

`jax.debug.callback` passes runtime values to an arbitrary Python function for inspection such as shapes and summary statistics. The callback must not return a value used by the computation.

```python
def summarize(name, value):
    print(
        f'[{name}] shape={value.shape} '
        f'min={float(jnp.min(value)):.3f} '
        f'max={float(jnp.max(value)):.3f} '
        f'mean={float(jnp.mean(value)):.3f}'
    )


@brainstate.transform.jit
def forward(x):
    jax.debug.callback(summarize, 'activations', x)
    return jnp.tanh(x)
```

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html

## Pause only on a bad runtime condition

`breakpoint_if(pred)` enters JAX's interactive debugger only when `pred` is true at runtime. Use it to halt on a rare condition without stopping every iteration:

```python
@brainstate.transform.jit
def guarded(x):
    T.breakpoint_if(jnp.any(~jnp.isfinite(x)))
    return x * 2.0
```

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html

## Official sources

- https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html
- https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html
