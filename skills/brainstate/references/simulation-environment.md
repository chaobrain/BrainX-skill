# Simulation environment

Use this reference when a task needs more than the canonical `brainstate.environ.context(dt=..., fit=...)` rollout in `SKILL.md`: nested overrides, persistent defaults, isolated environments, precision or platform control, one-step exponential-Euler integration, or diagnosis of leaked and missing settings.

Sources:

- [Time and Environment](https://brainx.chaobrain.com/brainstate/concepts/time_and_environment.html)
- [`brainstate.environ` API](https://brainx.chaobrain.com/brainstate/apis/environ.html)
- [`brainstate.nn.exp_euler_step` API](https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.exp_euler_step.html)

## Treat the environment as run context

Put values that describe the run, rather than one Module, in `brainstate.environ`. The environment is a stack of settings:

- `set()` installs a persistent default that remains active until changed or reset.
- `context()` pushes temporary values and restores the previous values on exit, including exceptional exit.
- `get(key, default=...)` searches the active context before persistent settings; it returns the value or explicit default and otherwise raises `KeyError`.
- Typed accessors such as `get_dt()`, `get_precision()`, and `get_platform()` enforce the corresponding setting contract.
- A nested context inherits its outer context and overrides only the keys it supplies.

Do not treat `EnvironmentState` as model `State`. It stores thread-local environment configuration and does not belong in a Module graph or State collection.

Choose the narrowest lifetime:

| Need | Use |
|---|---|
| One simulation, training phase, evaluation phase, or step | `brainstate.environ.context(...)` |
| A persistent default in the selected environment | `brainstate.environ.set(...)` |
| The same context on every call to one Module | Use `brainstate.nn.EnvironContext(layer, **context)`; the wrapper executes the Module with the stored settings, merges optional call-specific settings, and returns the Module's result |
| Configuration isolated from the default environment | Create `brainstate.environ.EnvironmentState()` and pass it through every `env=` argument; operations then use its independent thread-local setting stack instead of the default environment |

Prefer `context()` for normal simulation and training. Use `set()` only when later code should intentionally inherit the setting.

## Advanced environment configuration

Construct a separate `EnvironmentState` only for an explicitly isolated configuration, and pass it consistently through the `env=` argument of `set()`, `context()`, and `get()` or `get_dt()`.

| Setting | Precise use |
|---|---|
| `dt` | Numerical integration step; set before initialization or rollout and read with `brainstate.environ.get_dt()` |
| `t` | Current simulation time; set for each step and read with `brainstate.environ.get("t")` |
| `i` | Current integer step index; set for each step and read with `brainstate.environ.get("i")` |
| `fit` | Set `True` for training and `False` for evaluation; dropout and batch normalization observe it consistently |
| `precision` | Default numerical precision (`8`, `16`, `32`, `64`, or `"bf16"`); inspect with `brainstate.environ.get_precision()` |
| `platform` | Computing platform (`"cpu"`, `"gpu"`, or `"tpu"`); configure globally with `set()` or `set_platform()` |
| `host_device_count` | Number of host devices; configure globally before device-dependent work |

Keep `dt` unit-aware in physical simulations. A rate with units `[X] / [time]` multiplied by a unit-aware `dt` produces an increment with units `[X]`; a unit mismatch then fails instead of silently changing the model.

Do not set `platform` or `host_device_count` through `context()`. The API requires these process-level settings to be set globally.

## Nest training and evaluation scopes

Let an inner context override only the setting that changes:

```python
with brainstate.environ.context(dt=0.1 * u.ms, fit=True):
    train_step()

    with brainstate.environ.context(fit=False):
        validation = evaluate()

    train_step()
```

The inner evaluation observes the outer `dt` and its own `fit=False`. On exit, `fit=True` is restored without mutating the persistent defaults.

An unset required key raises instead of supplying a guessed value. Set `dt` before calling code that uses `get_dt()`; use an explicit `default=` with `get()` only when that fallback is part of the intended behavior.

## Isolate a configuration

Use the isolated branch only when two configurations must not share defaults:

```python
isolated_env = brainstate.environ.EnvironmentState()
brainstate.environ.set(
    dt=0.05 * u.ms,
    precision=64,
    env=isolated_env,
)

with brainstate.environ.context(fit=False, env=isolated_env):
    isolated_dt = brainstate.environ.get_dt(env=isolated_env)
    isolated_fit = brainstate.environ.get("fit", env=isolated_env)
```

Omitting `env=isolated_env` from any operation silently switches that operation back to the default environment, so keep the argument consistent across the complete workflow.

## Advance one continuous-dynamics step

Use `brainstate.nn.exp_euler_step()` for one ODE or SDE step with the active environment `dt`; it returns the next state after integrating the diagonal linearized drift exponentially.

```python
def decay(v, t, tau):
    return -v / tau


v = jnp.ones(num_neurons) * u.mV
with brainstate.environ.context(dt=0.1 * u.ms):
    v_next = brainstate.nn.exp_euler_step(
        decay,
        v,
        0.0 * u.ms,
        10.0 * u.ms,
    )
```

Use a supported floating dtype. A state with units `[X]` requires drift units `[X] / [time]`; an SDE diffusion term requires `[X] / sqrt([time])`. Only the diagonal of the drift Jacobian is integrated exponentially, while off-diagonal coupling uses plain Euler, so do not use this step as an exact solver for strongly coupled systems.

## Common failures

- Use `context()` instead of `set()` when a setting must not leak beyond one run.
- Set `dt` before initialization when initialization depends on the integration step.
- Set `t` and `i` inside the per-step scope when model code reads current time or step index.
- Keep `fit` explicit around both training and evaluation.
- Do not mix reads from the default environment with writes to an isolated `EnvironmentState`.
- Do not assume `precision` changes every explicitly typed array; it governs BrainState's default dtype helpers.
