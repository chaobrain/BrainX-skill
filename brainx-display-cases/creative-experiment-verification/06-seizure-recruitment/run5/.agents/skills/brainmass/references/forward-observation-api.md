# Forward and observation models

Use this reference when mapping simulated regional activity to BOLD, EEG, or MEG; choosing an HRF kernel or hemodynamic path; downsampling a trajectory; or validating lead-field shapes and units.

## Choose the measurement path

| API | Need | Important boundary |
|---|---|---|
| `HRFBold` | Fast differentiable BOLD for fitting | Linear HRF convolution over a completed trajectory; prefer this canonical BOLD path for optimization. |
| `BOLDSignal` | Biophysical hemodynamic State | Four-State Balloon-Windkessel ODE; choose for flow, volume, and deoxyhemoglobin dynamics. |
| `TemporalAverage` | Anti-aliased temporal downsampling | Block-average non-overlapping windows and drop a trailing partial window. |
| `EEGLeadFieldModel` | Physically calibrated EEG | Preserve lead-field, dipole, and voltage units. |
| `MEGLeadFieldModel` | Physically calibrated MEG | Preserve lead-field, dipole, and magnetic-field units. |
| `LeadFieldModel` | Generic calibrated sensor projection | Configure the source scale, dipole unit, and sensor unit explicitly. |
| `LeadfieldReadout` | Learnable unitless sensor head | Learn the projection matrix end to end; do not present it as a calibrated physical lead field. |

Choose the observation before defining the objective. Hidden neural State and empirical sensor data are not interchangeable targets.

## Convolution BOLD

`HRFBold` temporally averages the neural drive, convolves it with a closed-form HRF, and decimates the result to `period`.

| API | Description |
|---|---|
| `brainmass.HRFBold(period, downsample_period, kernel)` | Use for a trajectory-level neural-to-BOLD operator; pass the simulation `dt` at call time and receive time-major BOLD. |
| `brainmass.FirstOrderVolterraHRFKernel(...)` | Use for the TVB first-order Volterra underdamped kernel. |
| `brainmass.GammaHRFKernel(...)` | Use for a peak-normalized gamma HRF. |
| `brainmass.DoubleExponentialHRFKernel(...)` | Use for the documented difference of damped oscillations. |
| `brainmass.MixtureOfGammasHRFKernel(...)` | Use for the SPM-style difference-of-gammas HRF. |
| `brainmass.HRFKernel` | Subclass only when the required impulse response is absent; a kernel call returns dimensionless `h(t)`. |

```python
import brainmass
import brainunit as u
import jax.numpy as jnp

time = jnp.arange(2000.0)
neural = 1.0 + 0.5 * jnp.sin(
    2.0 * jnp.pi * time[:, None] / 800.0
)
observer = brainmass.HRFBold(
    period=200.0 * u.ms,
    downsample_period=4.0 * u.ms,
    kernel=brainmass.FirstOrderVolterraHRFKernel(
        duration=400.0 * u.ms,
    ),
)
bold = observer(neural, dt=1.0 * u.ms)

assert bold.ndim == 2
assert bold.shape[1] == 1
```

Use a raw magnitude only where `HRFBold` expects a unitless neural trajectory. Preserve the original `dt` and region ordering as explicit metadata.

## Balloon-Windkessel BOLD

`BOLDSignal` evolves vasodilatory signal, blood flow, blood volume, and deoxyhemoglobin State before computing dimensionless BOLD.

| API | Description |
|---|---|
| `brainmass.BOLDSignal(in_size, **hemodynamic_params)` | Use for the nonlinear biophysical path; size it to the source regions and initialize all State before stepping. |
| `BOLDSignal.update(z)` | Advance the hemodynamic State from one neural-drive sample. |
| `BOLDSignal.bold()` | Read the BOLD observation derived from the current hemodynamic State. |

The current documented tutorial drives this ODE with a dimensionless environment `dt` and restores the unit-aware neural `dt` afterward. Do not pass a time `Quantity` into that path without checking the installed release, because its RK2 update mixes numeric hemodynamic time with `dt`.

Discard a physiologically justified hemodynamic transient before comparison. Do not compare HRF-convolution and Balloon-Windkessel traces at exactly zero lag and treat the mismatch as an implementation error.

## Temporal averaging

| API | Description |
|---|---|
| `brainmass.TemporalAverage(period)` | Use to average non-overlapping windows of `round(period / dt)` samples; it preserves units and drops the trailing incomplete window. |
| `Simulator.run(..., sample_every=k)` | Use for point decimation when averaging is not required; it records every `k`th post-update value. |

```python
signal = jnp.arange(20.0).reshape(20, 1)
averaged = brainmass.TemporalAverage(period=5.0 * u.ms)(
    signal,
    dt=1.0 * u.ms,
)
assert averaged.shape == (4, 1)
```

Use temporal averaging when aliasing matters; do not describe point decimation as an average.

## EEG and MEG lead fields

A physical lead-field operator applies a matrix from regional dipole sources to sensor measurements while enforcing dimensional compatibility.

| API | Description |
|---|---|
| `brainmass.LeadFieldModel(in_size, out_size, L, sensor_unit, dipole_unit, scale=...)` | Use for an explicitly configured physical projection; `scale` converts the source observable to dipole moment when needed. |
| `brainmass.EEGLeadFieldModel(in_size, out_size, L, sensor_unit=..., ...)` | Use for the EEG specialization, normally producing volts from a lead field in voltage per dipole moment. |
| `brainmass.MEGLeadFieldModel(...)` | Use for the MEG specialization when its installed signature matches the required magnetic units. |
| `brainmass.LeadfieldReadout(...)` | Use for a unitless trainable matrix with optional normalization; fit it as a model parameter rather than claiming physical calibration. |

Keep region and sensor axes consistent with the installed constructor. In the current tutorial, the lead field is shaped `(regions, sensors)` and a time-major source `(time, regions)` produces `(time, sensors)`.

```python
import numpy as np

n_region, n_sensor = 4, 6
rng = np.random.RandomState(1)
lead_field = jnp.asarray(rng.rand(n_region, n_sensor)) * (
    u.volt / (u.nA * u.meter)
)
eeg_model = brainmass.EEGLeadFieldModel(
    in_size=(n_region,),
    out_size=(n_sensor,),
    L=lead_field,
    sensor_unit=u.volt,
)
```

Use a source observable that represents the intended modality, such as `JansenRitStep.eeg()` for a cortical EEG proxy. A lead field cannot correct an inappropriate neural source model.

## Validate an observation

- Confirm source shape `(time, regions)` and output shape `(time, sensors)` or `(time, regions)` for BOLD.
- Confirm sampling interval and transient separately for neural and observation stages.
- Confirm region labels remain aligned with connectome and lead-field rows.
- Confirm units at every physical projection and strip them only at documented array boundaries.
- Compare simulated and empirical data in the same modality, reference, sampling rate, and preprocessing space.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/forward.html`
- `https://brainx.chaobrain.com/brainmass/reference/observation.html`
- `https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html`
