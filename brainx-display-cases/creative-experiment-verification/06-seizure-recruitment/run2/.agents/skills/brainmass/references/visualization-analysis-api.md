# Visualization and analysis

Use this reference when plotting BrainMass results or computing time-series, functional-connectivity, functional-connectivity-dynamics, and spectral summaries. Keep scientific metric computation in Braintools and use `brainmass.viz` as the thin optional plotting layer.

## Plot results

`brainmass.viz` imports matplotlib lazily. Install the BrainMass visualization extra before using it; core `brainmass` imports do not require matplotlib.

| API | Description |
|---|---|
| `brainmass.viz.plot_timeseries(signal, ts=None, labels=None, ax=None, ...)` | Plot one or more time-major regional or sensor trajectories; it accepts unit-aware signal and time inputs. |
| `brainmass.viz.plot_phase_portrait(x, y, ax=None, ...)` | Plot one State variable against another for oscillator or attractor inspection. |
| `brainmass.viz.plot_connectivity(matrix, labels=None, ax=None, ...)` | Plot any square structural, functional, FCD, or regime matrix as a heatmap. |
| `brainmass.viz.plot_functional_connectivity(data, is_matrix=False, labels=None, ax=None, ...)` | Compute FC from a trajectory or plot an existing matrix when `is_matrix=True`. |
| `brainmass.viz.plot_power_spectrum(signal, dt, ax=None, loglog=False, ...)` | Compute and plot the PSD of one one-dimensional signal with sampling interval `dt`. |

Every helper accepts `ax=` and returns the `matplotlib.axes.Axes` it drew on. Pass axes explicitly when composing figures.

## Compute standard summaries

Strip units explicitly before a metric that operates on scale-free raw arrays. Keep the sampling interval and labels separately.

| API | Description |
|---|---|
| `braintools.metric.functional_connectivity(signal)` | Compute pairwise correlation across regions from a `(time, regions)` trajectory. |
| `braintools.metric.functional_connectivity_dynamics(signal, window_size, step_size)` | Slide an FC window and correlate windowed FC matrices to return an `(n_window, n_window)` FCD matrix. |
| `braintools.metric.power_spectral_density(signal, dt, nperseg=None, noverlap=None, freq_range=None)` | Use for a one-sided Welch PSD of a `(time,)` or `(time, channels)` signal; pass `dt` as a time `Quantity` or a float in seconds, and it returns frequencies in Hz plus PSD shaped `(frequency,)` or `(frequency, channels)`. |
| `brainmass.objectives.fc_corr()` | Score correlation between prediction and target FC without manually recomputing the comparison logic. |
| `brainmass.objectives.fcd_distribution(fcd_matrix, ...)` | Estimate the off-diagonal FCD value distribution used by distributional FCD objectives. |

## Canonical analysis workflow

```python
import brainmass
import brainstate
import brainunit as u
import braintools
import numpy as np

dt = 0.1 * u.ms
brainstate.environ.set(dt=dt)
brainstate.random.seed(0)

connectome = brainmass.datasets.load_dataset("example_connectome")
n_region = connectome.weights.shape[0]
node = brainmass.HopfStep(
    in_size=n_region,
    a=0.1,
    w=0.3,
    noise_x=brainmass.OUProcess(
        n_region,
        sigma=0.1,
        tau=10.0 * u.ms,
    ),
)
network = brainmass.Network(
    node,
    conn=connectome.weights,
    distance=connectome.distances,
    speed=10.0 * u.mm / u.ms,
    coupling="diffusive",
    coupled_var="x",
    k=0.5,
)
result = brainmass.Simulator(network, dt=dt).run(
    4000.0 * u.ms,
    monitors=lambda model: model.node.x.value,
    transient=400.0 * u.ms,
    sample_every=10,
)

signal = u.get_magnitude(result["output"])
fc = braintools.metric.functional_connectivity(signal)
fcd = braintools.metric.functional_connectivity_dynamics(
    signal,
    window_size=100,
    step_size=20,
)

assert fc.shape == (n_region, n_region)
assert fcd.ndim == 2 and fcd.shape[0] == fcd.shape[1]
```

`sample_every=10` records every 1 ms when `dt=0.1 ms`; use the recorded `ts` difference as the analysis sampling interval rather than the integration `dt`.

## Compare with data

Use the same preprocessing, region order, duration policy, and sampling rate before comparing summaries.

```python
target = brainmass.datasets.load_dataset("example_signal")
score = brainmass.objectives.fc_corr()(
    signal,
    target.signal,
)
assert np.isfinite(float(score))
```

Use `as_loss=True` only when the score is passed to a minimizer. A high FC score does not validate spectra, FCD, amplitudes, or parameter plausibility.

## Analyze spectra

Use one channel at a time unless the metric explicitly accepts multichannel input.

```python
record_dt = result["ts"][1] - result["ts"][0]
frequencies, power = braintools.metric.power_spectral_density(
    signal[:, 0],
    record_dt,
)
assert frequencies.shape == power.shape
```

The returned frequencies are in hertz. A plain numeric `dt` is always interpreted in seconds; do not pass a value expressed in milliseconds without its unit. Use the difference between recorded timestamps because `sample_every` can make it differ from the integration step.

Freeze a signal-amplitude or in-band-power floor before inspecting condition outcomes. Report the dominant frequency as undefined below that floor: `argmax` always selects a bin even when the trace contains no scientifically identifiable oscillation. Choose `nperseg` deliberately when the default resolution cannot distinguish the claimed frequency band.

## Common failures

- Computing FC over the region axis instead of the time axis.
- Comparing matrices whose region labels are in different orders.
- Using integration `dt` after `sample_every` changed the recorded sampling interval.
- Treating FCD as static FC or fitting the full matrix when the intended target is its off-diagonal distribution.
- Plotting a unit-bearing `Quantity` by implicit NumPy conversion instead of using `brainmass.viz` or `u.get_magnitude`.
- Claiming scientific validation from a single summary.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/viz.html`
- `https://brainx.chaobrain.com/brainmass/howto/analyze_results.html`
