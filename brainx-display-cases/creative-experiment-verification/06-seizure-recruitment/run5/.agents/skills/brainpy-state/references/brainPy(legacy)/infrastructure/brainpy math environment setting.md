# Legacy BrainPy math environment settings

Use this reference to set or scope legacy `brainpy.math` integration step, computation mode, precision, default dtypes, platform, host-device count, and device-memory behavior. Set environment values before constructing models whose shapes or variables depend on them.

## Choose persistent or scoped settings

`bm.set(...)` changes process-wide defaults, while `bm.environment(...)` temporarily overrides selected settings and restores the previous environment on exit.

| API | Description |
|---|---|
| `bm.set(mode=None, dt=None, x64=None, float_=None, int_=None, bool_=None, complex_=None, ...)` | Persist one or more defaults for subsequent legacy BrainPy work. |
| `bm.set_environment(...)` | Alias for setting the default computation environment. |
| `bm.environment(mode=None, dt=None, x64=None, float_=None, int_=None, bool_=None, complex_=None, ...)` | Scope selected settings with a context manager and restore prior values afterward. |
| `bm.batching_environment(dt=None, batch_size=1, ...)` | Scope `BatchingMode` and its batch size for batched prediction or readout fitting. |
| `bm.training_environment(dt=None, batch_size=1, ...)` | Scope `TrainingMode` and its batch size while constructing trainable models. |

```python
import brainpy.math as bm

original_dt = bm.get_dt()

with bm.environment(dt=0.05, x64=True, float_=bm.float64):
    assert bm.get_dt() == 0.05
    assert bm.get_float() == bm.float64
    # Construct and run the precision-sensitive model here.

assert bm.get_dt() == original_dt
```

Use `bm.set(dt=...)` for a deliberate application default. Use a context when one experiment, training phase, or analysis needs a local override.

## Set integration step and mode

The integration step controls the default `dt` consumed by legacy integrators and `DSRunner`; the mode controls whether models allocate unbatched, batched, or trainable state.

| API | Description |
|---|---|
| `bm.set_dt(dt)` | Persist the default numerical integration step. |
| `bm.get_dt()` | Return the active integration step. |
| `bm.set_mode(mode)` | Persist a `NonBatchingMode`, `BatchingMode`, or `TrainingMode`. |
| `bm.get_mode()` | Return the active mode object. |
| `bm.nonbatching_mode` | Use for ordinary single-instance simulation. |
| `bm.batching_mode` | Use for batched execution without trainable-mode allocation. |
| `bm.training_mode` | Use when model construction must create trainable parameters. |

```python
import brainpy as bp
import brainpy.math as bm

with bm.training_environment(dt=0.1, batch_size=32):
    model = bp.dnn.Dense(20, 2)
    assert isinstance(bm.get_mode(), bm.TrainingMode)

assert isinstance(bm.get_mode(), bm.NonBatchingMode)
```

**Invariant:** Enter batching or training mode before constructing mode-dependent layers. Changing mode after construction does not retroactively change parameter roles or state shapes.

## Set precision and default dtypes

Enable x64 before constructing precision-sensitive arrays or analyzers; setting only a float default does not itself enable JAX 64-bit support.

| API | Description |
|---|---|
| `bm.enable_x64()` | Enable JAX 64-bit values for subsequent computation. |
| `bm.disable_x64()` | Disable JAX 64-bit values. |
| `bm.set_float(dtype)` | Set the default floating dtype. |
| `bm.get_float()` | Return the default floating dtype. |
| `bm.set_int(dtype)` | Set the default integer dtype. |
| `bm.get_int()` | Return the default integer dtype. |
| `bm.set_bool(dtype)` | Set the default boolean dtype. |
| `bm.get_bool()` | Return the default boolean dtype. |
| `bm.set_complex(dtype)` | Set the default complex dtype. |
| `bm.get_complex()` | Return the default complex dtype. |
| `bm.dftype()` | Return the active default float dtype. |
| `bm.ditype()` | Return the active default integer dtype. |

Use 64-bit computation for numerical dynamics analysis when root finding or eigenvalue classification is unstable at 32-bit precision. Keep 32-bit computation for normal simulation or training unless accuracy tests justify the added memory and compute cost.

## Select platform and host devices

Platform selection affects JAX backend initialization and must occur at program startup.

| API | Description |
|---|---|
| `bm.set_platform(platform)` | Select `'cpu'`, `'gpu'`, or `'tpu'`; call before arrays or compiled functions initialize the backend. |
| `bm.get_platform()` | Return the active platform name. |
| `bm.set_host_device_count(n)` | Expose `n` logical CPU devices so CPU `pmap` workflows can run. Configure before backend initialization. |

```python
import brainpy.math as bm

# Put this at the start of the process, before model or array construction.
bm.set_platform('cpu')
assert bm.get_platform() == 'cpu'
```

Do not call `set_platform()` in the middle of a process and assume existing arrays or compiled functions migrate to the new backend.

## Control device memory

Use memory controls only at explicit process boundaries or between large independent model runs.

| API | Description |
|---|---|
| `bm.clear_buffer_memory(platform=None, array=True, transform=True, compilation=False, object_name=False)` | Clear selected BrainPy/JAX buffers and caches, useful between models executed in a Python loop. |
| `bm.enable_gpu_memory_preallocation()` | Enable the configured JAX GPU preallocation behavior. |
| `bm.disable_gpu_memory_preallocation(release_memory=True)` | Allocate GPU memory on demand; this can reduce reserved memory but may increase fragmentation risk. |

Do not clear buffers inside a simulation step or training step. Cache clearing invalidates reuse and can dominate runtime.

## Common failures

- Do not set `dt` after constructing data or delays whose lengths were computed from the previous step.
- Do not create a model before entering `training_environment()` when its layers depend on training mode.
- Do not request `float64` without enabling x64 and then assume the request was honored.
- Do not select CPU/GPU/TPU after JAX has initialized its backend.
- Do not use persistent global settings for one local experiment when a restoring context is sufficient.
- Do not reuse another package's environment context; legacy `brainpy.math` keeps separate settings.

## Sources mirrored

- https://brainpy.readthedocs.io/apis/brainpy.math.environment.html
