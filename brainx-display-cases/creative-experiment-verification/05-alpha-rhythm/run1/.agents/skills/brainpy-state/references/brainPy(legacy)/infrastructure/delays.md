# Legacy BrainPy delay buffers

Use this reference when a legacy `brainpy.math` model needs an explicit history
buffer indexed by simulation time or by integer simulation steps. Do not replace
these buffers with `brainpy.state` or BrainState delay APIs when maintaining a
legacy BrainPy model.

## Choose the delay coordinate

Choose the buffer from the coordinate used by the model to request history.

| API | Use when | Critical behavior |
|---|---|---|
| `bm.TimeDelay(delay_target, delay_len, before_t0=None, t0=0.0, dt=None, name=None, interp_method='linear_interp')` | The model requests an earlier value by absolute simulation time. | Stores `ceil(delay_len / dt) + 1` samples and interpolates off-grid requests by default. |
| `bm.LengthDelay(delay_target, delay_len, initial_delay_data=None, name=None, batch_axis=None, update_method='rotation')` | The model requests an earlier value by an integer number of steps. | Stores `delay_len + 1` samples; `retrieve(0)` is the latest value after an update. |
| `bm.NeuTimeDelay(...)` | Compatibility code names the neutral time-delay alias. | It is an alias of `TimeDelay`; do not choose it as a distinct mechanism. |
| `bm.NeuLenDelay(...)` | Compatibility code names the neutral length-delay alias. | It is an alias of `LengthDelay`; do not choose it as a distinct mechanism. |

Use `TimeDelay` when physical time and interpolation are part of the model. Use
`LengthDelay` when the delay has already been discretized into steps.

## Retrieve time-indexed history

`TimeDelay` tracks its own current time; call it with the absolute past time to
retrieve and call `update()` once whenever a new current sample enters the
history.

| API | Description |
|---|---|
| `delay(time, indices=None)` | Retrieve the value at absolute `time`; optionally select `indices` from the result. |
| `delay.update(value)` | Append the new current value and advance the buffer's current time by `dt`. |
| `delay.reset(delay_target, delay_len, t0=0.0, before_t0=None)` | Rebuild the history and restart its time coordinate. |
| `interp_method='linear_interp'` | Use for continuous off-grid retrieval; interpolate between adjacent stored samples. |
| `interp_method='round'` | Use when retrieval must snap to the nearest stored time step. |

```python
import brainpy.math as bm

delay = bm.TimeDelay(
    bm.zeros(3),
    delay_len=1.0,
    dt=0.1,
    before_t0=lambda t: t,
)

# Before t0, the callable supplies a value broadcast to the target shape.
past = delay(-0.2)
assert past.shape == (3,)
assert bm.allclose(past, bm.array([-0.2, -0.2, -0.2]))
```

The requested time must not exceed the buffer's current time and must remain
within the retained window. Supply `before_t0` when prehistory is not the
default initialized data. A callable receives the requested time; an array is
laid out as `(num_delay, ...)`, with the longest delay first.

## Retrieve step-indexed history

`LengthDelay` makes discrete delay semantics explicit: update the buffer with
the newest sample, then retrieve an integer number of steps into the past.

| API | Description |
|---|---|
| `delay.retrieve(delay_len, *indices)` | Retrieve an integer step offset and optionally index the stored value. |
| `delay.update(value=None)` | Insert `value`; when constructed from a `bm.Variable`, omit `value` to insert that variable's current value. |
| `delay.reset(delay_target, delay_len=None, initial_delay_data=None, batch_axis=None)` | Reinitialize the target, retained length, prehistory, and optional batch axis. |
| `bm.ROTATE_UPDATE` | Pass this `'rotation'` constant as `update_method`; use the default circular-buffer update. |
| `bm.CONCAT_UPDATE` | Pass this `'concat'` constant as `update_method`; use explicit concatenation when that storage behavior is required. |

```python
import brainpy.math as bm

signal = bm.Variable(bm.array([0.0]))
delay = bm.LengthDelay(signal, delay_len=2)

signal.value = bm.array([1.0])
delay.update()
signal.value = bm.array([2.0])
delay.update()

assert bm.allclose(delay.retrieve(0), bm.array([2.0]))
assert bm.allclose(delay.retrieve(1), bm.array([1.0]))
assert bm.allclose(delay.retrieve(2), bm.array([0.0]))
```

`delay_len` passed to `retrieve()` must be integer-valued and smaller than the
stored sample count. If `initial_delay_data` is an array, arrange its leading
axis as delays `1, 2, ..., delay_len`; this ordering changed in BrainPy 2.2.3.2,
so inspect old checkpoints before reusing their raw buffer arrays.

## Source-backed failures

- Construct either buffer from a BrainPy or JAX array, not an arbitrary Python
  container.
- Establish `dt` before constructing `TimeDelay`; omitting `dt` captures the
  current global BrainPy time step.
- Do not request fractional steps from `LengthDelay`; use `TimeDelay` when
  interpolation is required.
- Do not update a history more than once per logical simulation step.
- Preserve the delay-buffer State when wrapping the containing legacy object in
  `bm.jit`, `bm.for_loop`, or another object-oriented transformation.

## Official sources

- `https://brainpy.readthedocs.io/apis/brainpy.math.delayvars.html`
- Generated `TimeDelay`, `LengthDelay`, `NeuTimeDelay`, and `NeuLenDelay` pages
  linked from that API index.
