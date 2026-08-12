# Parameter initializers

Use this reference to choose and apply a legacy `brainpy.initialize` initializer for state arrays, trainable matrices, convolution kernels, or distance-dependent within-population weights. Keep these APIs separate from `brainpy.state` and from newer initializer packages.

## Selection map

| Need | Use | Key constraint |
|---|---|---|
| Zeros, ones, a constant, or an identity matrix | Fixed and direct initializers | Check whether the consumer needs a scalar-filled array or a matrix with identity structure. |
| Values from a named distribution | Random initializers | Set the distribution parameters and pass `seed` when the random sequence must be reproducible. |
| Fan-scaled neural-network weights | Variance-scaling presets | Preserve the initializer's `in_axis` and `out_axis` interpretation for the actual weight layout. |
| Orthogonal dense or convolutional weights | Orthogonal initializers | `DeltaOrthogonal` accepts only 3-D, 4-D, or 5-D shapes. |
| Spatially decaying recurrent weights | Decay initializers | Define encoded ranges and periodic boundaries from the modeled space, not from array shape alone. |

## Initializer contract

An initializer stores an initialization policy; calling it with a shape creates the concrete BrainPy array.

| API | Description |
|---|---|
| `Initializer.__call__(shape, dtype=None)` | Implement this contract for a custom reusable initializer; it receives the requested shape and returns initialized values. |

Pass an initializer object to a BrainPy API that accepts an initialization policy. Call it directly only when the concrete array is needed immediately.

## Fixed and direct initializers

Use these when initialization is deterministic and does not depend on fan statistics.

| API | Description |
|---|---|
| `ZeroInit()` | Fill the requested shape with zeros. |
| `Constant(value=1.0)` | Fill the requested shape with `value`. |
| `OneInit(value=1.0)` | Initialize with ones, optionally scaled by `value`. |
| `Identity(value=1.0)` | Return an identity matrix scaled by `value`; use for matrix-shaped recurrent or linear weights. |

## Random initializers

Use a direct distribution when its location, scale, or bounds are part of the model specification.

| API | Description |
|---|---|
| `Normal(mean=0.0, scale=1.0, seed=None)` | Draw normally distributed values with the requested mean and scale. |
| `Uniform(min_val=0.0, max_val=1.0, seed=None)` | Draw values uniformly from the requested interval. |
| `TruncatedNormal(loc=0.0, scale=1.0, lower=-2.0, upper=2.0, seed=None)` | Draw from a normal distribution restricted to the supplied bounds. |

Do not substitute a truncated normal for an ordinary normal silently; truncation changes both the support and the realized distribution.

## Variance-scaled initializers

Use these for dense or convolutional weights whose scale must depend on fan-in, fan-out, or their average.

| API | Description |
|---|---|
| `VarianceScaling(scale, mode, distribution, in_axis=-2, out_axis=-1, seed=None)` | Configure the scale, fan mode, distribution, and weight axes explicitly. |
| `KaimingUniform(scale=2.0, mode='fan_in', distribution='uniform', in_axis=-2, out_axis=-1, seed=None)` | Use the Kaiming uniform preset. |
| `KaimingNormal(scale=2.0, mode='fan_in', distribution='truncated_normal', in_axis=-2, out_axis=-1, seed=None)` | Use the Kaiming truncated-normal preset. |
| `XavierUniform(scale=1.0, mode='fan_avg', distribution='uniform', in_axis=-2, out_axis=-1, seed=None)` | Use the Xavier uniform preset. |
| `XavierNormal(scale=1.0, mode='fan_avg', distribution='truncated_normal', in_axis=-2, out_axis=-1, seed=None)` | Use the Xavier truncated-normal preset. |
| `LecunUniform(scale=1.0, mode='fan_in', distribution='uniform', in_axis=-2, out_axis=-1, seed=None)` | Use the LeCun uniform preset. |
| `LecunNormal(scale=1.0, mode='fan_in', distribution='truncated_normal', in_axis=-2, out_axis=-1, seed=None)` | Use the LeCun truncated-normal preset. |

**Invariant:** `in_axis` and `out_axis` determine the fan calculation. Do not copy their defaults to a weight layout whose input and output axes differ.

## Orthogonal initializers

Use orthogonal structure when the model or training method requires norm-preserving rows or columns.

| API | Description |
|---|---|
| `Orthogonal(scale=1.0, axis=-1, seed=None)` | Construct a uniformly distributed orthogonal matrix; a non-square shape receives orthonormal rows or columns according to its smaller side. |
| `DeltaOrthogonal(scale=1.0, axis=-1)` | Construct a delta-orthogonal convolution kernel; the requested shape must be 3-D, 4-D, or 5-D. |

## Distance-decay initializers

Use these to build within-population weight patterns from encoded positions.

| API | Description |
|---|---|
| `GaussianDecay(sigma, max_w, min_w=None, encoding_values=None, periodic_boundary=False, include_self=True, normalize=False)` | Build Gaussian-decaying weights; use `min_w` to suppress small weights and configure self-connections, normalization, encoded ranges, and periodic boundaries explicitly. |
| `DOGDecay(sigmas, max_ws, min_w=None, encoding_values=None, periodic_boundary=False, normalize=True, include_self=True)` | Build a difference-of-Gaussians pattern from the positive and negative widths and amplitudes. |

`encoding_values` defines the value range represented along each population axis. Set `periodic_boundary=True` only for a genuinely periodic represented variable such as an angle.

## Canonical workflow

Create reusable policies first, then let each policy materialize the shape required by the consuming model:

```python
import brainpy.initialize as init

voltage_initializer = init.Normal(mean=-60.0, scale=2.0, seed=7)
weight_initializer = init.XavierNormal(seed=11)

initial_voltage = voltage_initializer((4,))
weights = weight_initializer((4, 3))
recurrent_identity = init.Identity(value=0.5)((3, 3))

assert initial_voltage.shape == (4,)
assert weights.shape == (4, 3)
assert recurrent_identity.shape == (3, 3)
```

Keep the initializer object when several components should share one policy. Construct separate seeded initializer objects when independent reproducible random sequences are required.

## Sources

- https://brainpy.readthedocs.io/apis/initialize.html
