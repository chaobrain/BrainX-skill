# Braintools input currents

Use this reference to generate unit-aware stimulation currents, waveforms,
pulses, and stochastic processes. Prefer the composable class API when a
protocol combines, transforms, repeats, or sequences several signals; use the
functional API for one direct array.

## Choose an interface

| Interface | Use when | Result |
|---|---|---|
| Composable classes | The protocol needs algebra, clipping, smoothing, time shifts, repetition, overlays, or concatenation | Returns an `Input` object; call it to generate the array. |
| Functional helpers | One signal can be generated in a single call | Returns the generated array immediately. |

Names ending in `Input` or `_input` are deprecated aliases. Use the class and
function names documented below.

## Time-step and composition model

An `Input` stores a duration and reads its time step from
`brainstate.environ`. Calling it generates a cached array; pass
`recompute=True` when the stochastic or environmental inputs must be evaluated
again.

| API | Description |
|---|---|
| `Input(duration)` | Base class for composable currents. It exposes `duration`, `dt`, `n_steps`, and `shape`. |
| `input_obj(recompute=False)` | Generate and return the current array. |
| `input_obj.scale(factor)` | Multiply all values by a factor. |
| `input_obj.shift(time_shift)` | Delay or advance the signal in time. |
| `input_obj.clip(min_val, max_val)` | Bound the signal values. |
| `input_obj.smooth(tau)` | Apply exponential smoothing. |
| `input_obj.repeat(n_times)` | Repeat the pattern. |
| `input_obj.apply(func)` | Apply a custom transformation to the generated values. |
| `a + b`, `a - b`, `a * b`, `a / b`, `-a` | Combine compatible signals pointwise. |
| `a & b` | Concatenate signals sequentially in time. |
| `a \| b` | Overlay signals by taking their pointwise maximum. |

```python
import brainstate
import brainunit as u
from braintools.input import Constant, Ramp, Sinusoidal

with brainstate.environ.context(dt=0.1 * u.ms):
    baseline = Constant([(0.2 * u.nA, 500 * u.ms)])
    ramp = Ramp(0 * u.nA, 1 * u.nA, 500 * u.ms)
    oscillation = Sinusoidal(
        amplitude=0.5 * u.nA,
        frequency=10 * u.Hz,
        duration=500 * u.ms,
    )
    protocol = (baseline + ramp + 0.5 * oscillation).clip(
        0 * u.nA,
        2 * u.nA,
    )
    current = protocol()

assert current.shape[0] == protocol.n_steps
```

**Invariant:** set or scope `brainstate.environ.dt` before constructing and
generating the protocol. A different `dt` changes the number of samples.

## Basic currents

| Composable API | Functional API | Use |
|---|---|---|
| `Section(values, durations)` | `section(values, durations, return_length=False)` | Build consecutive constant sections from parallel value and duration sequences. |
| `Constant(I_and_duration)` | `constant(I_and_duration)` | Build piecewise-constant sections from `(value, duration)` pairs. |
| `Step(amplitudes, step_times, duration=None)` | `step(amplitudes, step_times, duration=None)` | Change among discrete levels at specified times. |
| `Ramp(c_start, c_end, duration, t_start=None, t_end=None)` | `ramp(c_start, c_end, duration, t_start=None, t_end=None)` | Interpolate linearly, optionally only within a subinterval. |

`Constant` takes a sequence of pairs:

```python
from braintools.input import Constant

protocol = Constant([
    (0.0 * u.nA, 100 * u.ms),
    (1.0 * u.nA, 200 * u.ms),
    (0.0 * u.nA, 100 * u.ms),
])
```

Use `Section(values=[...], durations=[...])` when separate arrays are more
convenient.

## Pulse generators

| Composable API | Functional API | Use |
|---|---|---|
| `Spike(sp_times, duration, sp_lens=..., sp_sizes=...)` | `spike(sp_times, sp_lens, sp_sizes, duration)` | Place short current spikes at explicit times. |
| `GaussianPulse(amplitude, center, sigma, duration)` | `gaussian_pulse(amplitude, center, sigma, duration, ...)` | Generate a Gaussian-shaped pulse. |
| `ExponentialDecay(amplitude, tau, duration, ...)` | `exponential_decay(amplitude, tau, duration, ...)` | Generate a one-sided exponential decay. |
| `DoubleExponential(amplitude, tau_rise, tau_decay, ...)` | `double_exponential(amplitude, tau_rise, tau_decay, ...)` | Generate a rise-and-decay synaptic-current shape. |
| `Burst(...)` | `burst(...)` | Generate repeated pulses within one or more bursts. |

## Waveforms

| Composable API | Functional API | Use |
|---|---|---|
| `Sinusoidal(amplitude, frequency, duration, ...)` | `sinusoidal(amplitude, frequency, duration, ...)` | Generate a fixed-frequency sine wave. |
| `Square(amplitude, frequency, duration, ...)` | `square(amplitude, frequency, duration, ...)` | Generate a square wave; configure its duty cycle deliberately. |
| `Triangular(amplitude, frequency, duration, ...)` | `triangular(amplitude, frequency, duration, ...)` | Generate a triangular periodic wave. |
| `Sawtooth(amplitude, frequency, duration, ...)` | `sawtooth(amplitude, frequency, duration, ...)` | Generate a sawtooth periodic wave. |
| `Chirp(amplitude, f_start, f_end, duration, ...)` | `chirp(amplitude, f_start, f_end, duration, ...)` | Sweep frequency over time. |
| `NoisySinusoidal(amplitude, frequency, ..., noise_amplitude=...)` | `noisy_sinusoidal(amplitude, frequency, ..., noise_amplitude=...)` | Add stochastic noise to a sinusoid. |

## Stochastic processes

| Composable API | Functional API | Use |
|---|---|---|
| `WienerProcess(duration, n=1, t_start=None, t_end=None, sigma=1.0, seed=None)` | `wiener_process(duration, sigma=1.0, n=1, ...)` | Model Brownian fluctuations with independent Gaussian increments. |
| `OUProcess(mean, sigma, tau, duration, n=1, t_start=None, t_end=None, seed=None)` | `ou_process(mean, sigma, tau, duration, n=1, ...)` | Model mean-reverting fluctuations around `mean`. |
| `Poisson(rate, duration, n=1, t_start=None, t_end=None, amplitude=1.0, seed=None)` | `poisson(rate, duration, amplitude=1.0, n=1, ...)` | Generate independent Poisson spike-current trains. |

Use `seed=` for reproducibility and `n=` for multiple independent processes.
For Poisson input, ensure `rate * dt` is a valid per-bin spike probability.
Call with `recompute=True` only when a new stochastic realization is intended.

## Composition wrappers

These classes are normally created through operators or methods rather than
instantiated directly.

| API | Description |
|---|---|
| `Composite(input1, input2, operator)` | Pointwise combination of two inputs. |
| `ConstantValue(value, duration)` | Promote a scalar to a duration-matched input. |
| `Sequential(*inputs)` | Time concatenation corresponding to `&`. |
| `TimeShifted(input_obj, time_shift)` | Result of `.shift()`. |
| `Clipped(input_obj, min_val=None, max_val=None)` | Result of `.clip()`. |
| `Smoothed(input_obj, tau)` | Result of `.smooth()`. |
| `Repeated(input_obj, n_times)` | Result of `.repeat()`. |
| `Transformed(input_obj, func)` | Result of `.apply()`. |

Pointwise operands must have compatible duration, shape, and units. Use `&`
when durations differ and should occur one after another.

## Official source

- `https://brainx.chaobrain.com/braintools/apis/input.html`
