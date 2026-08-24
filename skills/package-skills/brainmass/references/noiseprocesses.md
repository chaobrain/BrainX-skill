# BrainMass noise processes

Use this reference when choosing a stochastic spectrum or correlation structure, attaching noise to a model, generating noise directly, or validating stochastic units, State, batches, and reproducibility.

## Choose a process

Use the simplest spectrum that expresses the scientific assumption.

| API | Description |
|---|---|
| `brainmass.GaussianNoise(in_size, mean=..., sigma=...)` | Use for independent Gaussian samples at every step; it is stateless and has a flat spectrum. |
| `brainmass.WhiteNoise(...)` | Use as the semantic alias of `GaussianNoise` when the white-noise assumption should be explicit. |
| `brainmass.OUProcess(in_size, mean=..., sigma=..., tau=...)` | Use for stationary mean-reverting drive with exponential autocorrelation; larger `tau` produces slower, smoother fluctuations. |
| `brainmass.BrownianNoise(in_size, sigma=...)` | Use for a stateful random walk with a `1/f^2` spectrum and variance that grows over time. |
| `brainmass.ColoredNoise(in_size, sigma=..., beta=...)` | Use when a custom `1/f^beta` spectrum is required. |
| `brainmass.PinkNoise(...)` | Use for the `beta=1` convenience form with a `1/f` spectrum. |
| `brainmass.BlueNoise(...)` | Use for the `beta=-1` convenience form with high-frequency emphasis. |
| `brainmass.VioletNoise(...)` | Use for the `beta=-2` convenience form with stronger high-frequency emphasis. |

`Noise` is the abstract common contract; instantiate a concrete process. Set `sigma` in the same unit as the model variable receiving the noise.

## Attach noise to a model

Noise is sampled inside `update()`, so attach it to the model's documented noise parameter and leave the `Simulator` call unchanged.

| API | Description |
|---|---|
| `brainstate.random.seed(seed)` | Reset the BrainState random stream before any stochastic construction or run that must be reproducible. |
| `<Model>Step(..., noise_<var>=process)` | Attach a process to the documented State component during construction. |
| `Simulator.run(..., batch_size=N)` | Initialize `N` independent model and noise realizations and return a time-major trajectory with a batch axis. |

```python
import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp

def noisy_run(seed):
    brainstate.random.seed(seed)
    node = brainmass.HopfStep(
        in_size=1,
        a=-0.05,
        w=0.3,
        noise_x=brainmass.OUProcess(
            in_size=1,
            sigma=0.1,
            tau=20.0 * u.ms,
        ),
    )
    return brainmass.Simulator(node, dt=0.1 * u.ms).run(
        100.0 * u.ms,
        monitors=["x"],
    )["x"]

first = noisy_run(7)
replay = noisy_run(7)
different = noisy_run(8)

assert jnp.allclose(first, replay)
assert not jnp.allclose(first, different)
```

Seed before each run whose exact stream matters. Seeding once does not make later calls reuse the same draws.

## Run an ensemble

Use `batch_size` when uncertainty across realizations is part of the result. The output ordering is time, then batch, then the model's region shape.

```python
brainstate.random.seed(2)
node = brainmass.HopfStep(
    in_size=4,
    a=-0.05,
    w=0.3,
    noise_x=brainmass.OUProcess(4, sigma=0.12, tau=20.0 * u.ms),
)
ensemble = brainmass.Simulator(node, dt=0.1 * u.ms).run(
    200.0 * u.ms,
    monitors=["x"],
    batch_size=12,
)

assert ensemble["x"].shape == (2000, 12, 4)
```

Use distributions across the batch dimension; do not report one stochastic trace as an ensemble statistic.

## Generate noise directly

Use a noise process as the `Simulator` model when a workflow needs a standalone drive or a spectral check. With `monitors=None`, `Simulator` stores the return from `update()` under `"output"`.

```python
process = brainmass.OUProcess(
    in_size=3,
    sigma=0.5 * u.Hz,
    tau=50.0 * u.ms,
)
samples = brainmass.Simulator(process, dt=0.1 * u.ms).run(
    100.0 * u.ms,
    monitors=None,
)

assert samples["output"].shape == (1000, 3)
assert u.get_unit(samples["output"]) == u.Hz
```

Stateless Gaussian noise needs no persistent process State. OU, Brownian, and colored processes do.

## Common failures

- Passing noise to `Simulator.run()` instead of the model constructor.
- Guessing a generic `noise=` attribute when the model documents `noise_x`, `noise_E`, or another component-specific parameter.
- Using unitless `sigma` for a unit-bearing State without confirming that the model permits it.
- Treating Brownian noise as stationary.
- Comparing noisy runs without controlling the random stream.
- Reading `(time, batch, regions)` as batch-major output.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/noise.html`
- `https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html`
