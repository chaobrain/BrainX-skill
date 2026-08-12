# Simulator input and monitor API

Use this reference when a BrainMass run needs driven inputs, custom observables, sampling, independent trials, or explicit initialization and JIT controls; use `batch-transform-acceleration.md` only when this runner cannot express the workflow.

## Use the standard runner

`Simulator` owns the normal initialization, environment, transformed time loop, monitoring, transient, and sampling lifecycle.

| API | Description |
|---|---|
| `brainmass.Simulator(model, dt=...)` | Use to bind a model or `Network` to an integration step for the standard run path. |
| `Simulator.run(duration, *, inputs=None, monitors=None, transient=None, sample_every=None, batch_size=None, init_states=True, jit=True)` | Use to execute a duration and return a dict of time-major monitored values plus unit-aware `ts`. It initializes State and JIT-compiles by default. |

Do not manually derive the number of steps, initialize State, scope `dt`, build a transformed loop, downsample, or reconstruct timestamps when these controls express the run.

## Supply model inputs

Choose the input form from how `model.update()` receives its arguments.

| `inputs` form | Behavior |
|---|---|
| `None` | Call `model.update()` without arguments. |
| Array shaped `(n_steps, ...)` | Slice the leading time axis and pass each row as one update argument. |
| Callable `(i, t)` returning one value | Evaluate it each step and pass the result as one update argument. |
| Callable `(i, t)` returning a tuple | Evaluate it each step and splat the tuple into positional update arguments. |

Use a callable for a constant vector drive. Passing that vector directly makes `Simulator` interpret its leading axis as time.

## Select monitors

Monitors record values after each model update.

| `monitors` form | Result |
|---|---|
| `None` | Store the return from `model.update()` under `"output"`. |
| List of State names | Store each named State under its own key. |
| Callable accepting the model | Store the returned observable under `"output"`. |
| Dict from output names to State names or callables | Store each selected State or derived observable under its dict key. |

`transient` discards a leading duration or step count. `sample_every=k` keeps every `k`th post-update sample and applies the same selection to `ts`.

## Run driven conditions as independent trials

Keep the scientific condition axis in `in_size` and the stochastic realization axis in `batch_size`.

```python
import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp

brainstate.random.seed(7)
evidence = jnp.asarray([-0.2, 0.0, 0.2])
model = brainmass.WongWangStep(
    in_size=evidence.size,
    noise_s1=brainmass.GaussianNoise(evidence.size, sigma=0.02 * u.nA),
    noise_s2=brainmass.GaussianNoise(evidence.size, sigma=0.02 * u.nA),
)
result = brainmass.Simulator(model, dt=0.1 * u.ms).run(
    2.0 * u.ms,
    inputs=lambda _i, _t: evidence,
    monitors={"decision_variable": lambda m: m.S1.value - m.S2.value},
    sample_every=5,
    batch_size=4,
)

assert result["decision_variable"].shape == (4, 4, 3)
assert model.S1.value.shape == (4, 3)
assert u.math.allclose(
    result["ts"],
    jnp.asarray([0.1, 0.6, 1.1, 1.6]) * u.ms,
)
```

The result layout is `(sampled_time, trial, evidence)`. Transpose or remove units only at an explicit analysis or serialization boundary.

## Override lifecycle controls only deliberately

| Control | Use when |
|---|---|
| `init_states=False` | Continuing intentionally from existing model State; otherwise keep the default reset. |
| `jit=False` | Debugging transformed execution or isolating compilation behavior; otherwise keep JIT enabled. |
| `batch_size=N` | Initializing and running `N` independent model and noise State realizations. |

Use a custom BrainState rollout only when required inputs, monitors, State effects, explicit carry, checkpointing, or a stable benchmark boundary cannot be represented here.

## Common failures

- Passing a constant condition vector as an array and unintentionally treating it as time-major input.
- Flattening conditions and trials into `in_size` even though `batch_size` represents independent realizations.
- Reconstructing time from `dt` instead of using the sampled, unit-aware `ts` result.
- Disabling initialization or JIT without a continuation, debugging, or benchmarking requirement.
- Rebuilding the runner with `environ.context()` and `for_loop` for a workflow already covered by `Simulator.run()`.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/orchestration.html`
- `https://brainx.chaobrain.com/brainmass/reference/generated/brainmass.Simulator.html`
