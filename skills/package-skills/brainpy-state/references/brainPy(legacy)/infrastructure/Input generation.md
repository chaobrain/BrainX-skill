# Legacy BrainPy input generation

Use this reference to construct time-major stimulus arrays with
`brainpy.inputs` for a legacy BrainPy simulation. Use the resulting arrays as
iterable `DSRunner` inputs; do not substitute Braintools input objects or
`brainpy.state` APIs in legacy code.

## Choose an input family

All input constructors use `dt` to discretize physical duration. Choose by the
shape of the intended stimulus.

### Piecewise and transient inputs

| API | Description |
|---|---|
| `bp.inputs.section_input(values, durations, dt=None, return_length=False)` | Use for consecutive sections with independently specified values and durations; set `return_length=True` to also return total duration. |
| `bp.inputs.constant_input(I_and_duration, dt=None)` | Use for the older `(value, duration)` piecewise-constant form. Prefer `section_input()` in new legacy-compatible code. |
| `bp.inputs.spike_input(sp_times, sp_lens, sp_sizes, duration, dt=None)` | Use for short current pulses at explicit times; lengths and sizes may be scalars or one value per spike. |
| `bp.inputs.ramp_input(c_start, c_end, duration, t_start=0, t_end=None, dt=None)` | Use for a linear ramp between `t_start` and `t_end` within a longer stimulus. |

### Stochastic inputs

| API | Description |
|---|---|
| `bp.inputs.wiener_process(duration, dt=None, n=1, t_start=0.0, t_end=None, seed=None)` | Use for Wiener increments sampled from a zero-mean normal distribution with scale `sqrt(dt)`. |
| `bp.inputs.ou_process(mean, sigma, tau, duration, dt=None, n=1, t_start=0.0, t_end=None, seed=None)` | Use for a mean-reverting Ornstein-Uhlenbeck process. |

Set `seed=` whenever a stochastic stimulus must be reproducible. Set `n=` to
generate multiple independent traces.

### Periodic inputs

| API | Description |
|---|---|
| `bp.inputs.sinusoidal_input(amplitude, frequency, duration, dt=None, t_start=0.0, t_end=None, bias=False)` | Use for a sinusoidal waveform active over the selected time interval. |
| `bp.inputs.square_input(amplitude, frequency, duration, dt=None, bias=False, t_start=0.0, t_end=None)` | Use for an oscillatory square waveform active over the selected time interval. |

## Construct an injection-ready stimulus

The leading axis is simulation time. Make the constructor `dt` equal to the
runner `dt`, then mark the array as iterable when injecting it through
`DSRunner`.

```python
import brainpy as bp

dt = 0.1
current, duration = bp.inputs.section_input(
    values=[0.0, 1.0, 0.0],
    durations=[100.0, 300.0, 100.0],
    dt=dt,
    return_length=True,
)

assert duration == 500.0
assert current.shape == (5000,)

runner = bp.DSRunner(
    target=model,
    inputs=[('input', current, 'iter')],
    monitors=['V'],
    dt=dt,
)
runner.run(duration)
```

If `dt` is omitted, the constructor uses BrainPy's global default. An array
created under one `dt` and consumed under another silently changes its physical
duration, so pass the same explicit value at both boundaries.

## Compose values and shapes

Input constructors broadcast heterogeneous section values to their maximum
compatible shape. For example, scalar, vector, and matrix sections become one
time-major array with the broadcast data shape after the leading time axis.

```python
import brainpy as bp
import brainpy.math as bm

current = bp.inputs.section_input(
    values=[0.0, bm.ones(10), bm.ones((3, 10))],
    durations=[100.0, 300.0, 100.0],
    dt=0.1,
)

assert current.shape == (5000, 3, 10)
```

Build a more complex stimulus by combining compatible generated arrays with
ordinary BrainPy math operations. Align `duration`, `dt`, and broadcast shape
before addition or multiplication; the constructors do not align two already
generated time grids for you.

## Source-backed failures

- Treat axis 0 as time. Do not pass a batch-major array as an unbatched iterable
  `DSRunner` input.
- Use `return_length=True` only with `section_input()`; `constant_input()`
  returns its legacy current-and-duration result directly.
- Give `sp_times`, `sp_lens`, and `sp_sizes` compatible lengths when the latter
  two are not scalars.
- Keep `rate`, probability, and event encoders out of this reference; these
  functions construct currents and stochastic processes, not Braintools data
  preprocessing pipelines.

## Official sources

- `https://brainpy.readthedocs.io/apis/inputs.html`
- `https://brainpy.readthedocs.io/tutorial_toolbox/inputs.html`
