# Array Mechanics

Use this reference for array metadata, indexing, functional updates, shape and axis transformations, joins and splits, repetition, named-axis transformations, and backend conversion of an existing `Quantity`.

## Contents

- [Metadata and identity](#metadata-and-identity)
- [Indexing and functional updates](#indexing-and-functional-updates)
- [Quantity methods](#quantity-methods)
- [Functional structural API](#functional-structural-api)
- [Named-axis transformations](#named-axis-transformations)
- [Backend conversion](#backend-conversion)
- [Source-backed gotchas](#source-backed-gotchas)

## Metadata And Identity

| Property | Official meaning |
|---|---|
| `q.shape` | Shape of the mantissa array |
| `q.dtype` | Data type of the mantissa |
| `q.backend` | One of `'numpy'`, `'jax'`, `'cupy'`, `'torch'`, `'dask'`, or `'ndonnx'` |
| `q.itemsize` | Bytes in one mantissa element |
| `q.nbytes` | Total bytes consumed by the mantissa array |
| `q.strides` | Byte steps in each dimension, mirroring `numpy.ndarray.strides` |
| `q.flat` | 1-D iterator over mantissa elements with units preserved |
| `q.mT` | Transpose of the final two dimensions; requires at least 2-D |

Use `u.math.ndim(x)`, `u.math.shape(x)`, and `u.math.size(x)` when a functional query is preferable to a property lookup.

## Indexing And Functional Updates

Ordinary indexing preserves the unit. `q.at` provides a functionally pure update interface:

```python
import brainunit as u
import jax.numpy as jnp

x = jnp.arange(5.0) * u.mV
y = x.at[2].add(10 * u.mV)
# Expected: [0, 1, 12, 3, 4] mV; x remains unchanged.
selected = x.at[2].get()
# Expected: 2 mV.
```

The documented update mappings are:

| Functional form | In-place equivalent |
|---|---|
| `x = x.at[idx].set(y)` | `x[idx] = y` |
| `x = x.at[idx].add(y)` | `x[idx] += y` |
| `x = x.at[idx].multiply(y)` | `x[idx] *= y` |
| `x = x.at[idx].divide(y)` | `x[idx] /= y` |
| `x = x.at[idx].power(y)` | `x[idx] **= y` |
| `x = x.at[idx].min(y)` | `x[idx] = minimum(x[idx], y)` |
| `x = x.at[idx].max(y)` | `x[idx] = maximum(x[idx], y)` |
| `x = x.at[idx].apply(ufunc)` | `ufunc.at(x, idx)` |
| `x.at[idx].get()` | `x[idx]` |

None of these expressions modifies the original object; each returns a modified copy. Inside `jit()`, assignments such as `x = x.at[idx].set(y)` are documented as being applied in place.

## Quantity Methods

| Exact signature | One-line description | Example & result |
|---|---|---|
| `reshape(shape, order='C')` | Return a quantity with the same data but a new shape. | `q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV); q.reshape((3, 1)).shape`<br>`# (3, 1)` |
| `flatten()` | Return a 1-D copy of this quantity. | `q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV); q.flatten()`<br>`# Quantity([1. 2. 3. 4.], "mV")` |
| `squeeze(axis=None)` | Remove length-one axes from the array. | `q = u.Quantity(jnp.array([[[1.0]]]), unit=u.mV); q.squeeze().shape`<br>`# ()` |
| `expand_dims(axis)` | Insert new axes at the given positions. | `q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV); q.expand_dims(0).shape`<br>`# (1, 2)` |
| `unsqueeze(axis)` | Insert a length-one axis (PyTorch-style alias for `expand_dims()`). | `q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV); q.unsqueeze(0).shape`<br>`# (1, 2)` |
| `transpose(*axes)` | Return the array with axes transposed. | `q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV); q.transpose().shape`<br>`# (2, 2)` |
| `swapaxes(axis1, axis2)` | Interchange two axes of the array. | `q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV); q.swapaxes(0, 1).shape`<br>`# (2, 2)` |
| `repeat(repeats, axis=None)` | Repeat elements of the array. | `q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV); q.repeat(2)`<br>`# Quantity([1. 1. 2. 2.], "mV")` |
| `tile(reps)` | Construct an array by repeating this quantity. | `q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV); q.tile(2)`<br>`# Quantity([1. 2. 1. 2.], "mV")` |
| `split(indices_or_sections, axis=0)` | Split the array into multiple sub-arrays. | `q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV); parts = q.split(3); len(parts)`<br>`# 3` |
| `take(indices, axis=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None)` | Select elements from the array at the given indices. | `q = u.Quantity(jnp.array([10.0, 20.0, 30.0]), unit=u.mV); q.take(jnp.array([0, 2]))`<br>`# Quantity([10. 30.], "mV")` |
| `sort(axis=-1, stable=True, order=None)` | Sort the array in-place along the given axis. | `q = u.Quantity(jnp.array([3.0, 1.0, 2.0]), unit=u.mV); q.sort()`<br>`# Quantity([1. 2. 3.], "mV")` |
| `diagonal(offset=0, axis1=0, axis2=1)` | Return specified diagonals, preserving units. | `q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV); q.diagonal()`<br>`# Quantity([1. 4.], "mV")` |
| `trace(offset=0, axis1=0, axis2=1)` | Sum along diagonals of the array, preserving units. | `q = u.Quantity(jnp.eye(3), unit=u.mV); q.trace()`<br>`# Quantity(3., "mV")` |
| `astype(dtype)` | Return a copy of this quantity with the mantissa cast to `dtype`. | `q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV); q.astype(jnp.float64).dtype`<br>`# float64` |

## Functional Structural API

Use these `brainunit.math` functions when composing transformations without binding the operation to a method:

| Family | Exact APIs |
|---|---|
| Join | `concatenate(arrays, axis=0, dtype=None, **kwargs)`, `stack(arrays, axis=0, dtype=None, **kwargs)`, `block(arrays, **kwargs)`, `append(arr, values, axis=None, **kwargs)` |
| Stack variants | `row_stack`, `vstack`, `hstack`, `dstack`, `column_stack` |
| Split | `split`, `array_split`, `dsplit`, `hsplit`, `vsplit` |
| Minimum rank | `atleast_1d`, `atleast_2d`, `atleast_3d` |
| Broadcast | `broadcast_arrays(*args, **kwargs)`, `broadcast_to(array, shape, **kwargs)` |
| Shape | `reshape`, `moveaxis`, `transpose`, `swapaxes`, `expand_dims`, `squeeze`, `flatten`, `unflatten` |
| Repetition | `tile(A, reps, **kwargs)`, `repeat(a, repeats, axis=None, total_repeat_length=None, **kwargs)` |
| Selection | `take`, `gather`, `choose`, `compress`, `extract` |

The module API categorizes these as functions that keep units. Open the [complete functional structural API reference](array-mechanics/functional-structural-api.md) for every exact signature, official one-line description, and example with its result.

```python
a = [1, 2] * u.second
b = [3, 4] * u.second
joined = u.math.concatenate([a, b])
# Expected: [1, 2, 3, 4] s.
rows = u.math.stack([a, b])
# Expected: [[1, 2], [3, 4]] s with shape (2, 2).
parts = u.math.array_split(joined, 2)
# Expected: two Quantity arrays, [1, 2] s and [3, 4] s.
```

## Named-Axis Transformations

`einrearrange(x, pattern, **axes_lengths)` covers transpose, reshape, squeeze, unsqueeze, stack, and concatenate. Composition uses C-order enumeration.

```python
channels_last = u.math.einrearrange(x, 'b c h w -> b h w c')
# Expected: for shape-compatible x, axes change from (b, c, h, w) to (b, h, w, c); unit preserved.
flat = u.math.einrearrange(x, 'b h w c -> (b h w c)')
# Expected: for shape-compatible x, a one-dimensional Quantity with all elements; unit preserved.
split_batch = u.math.einrearrange(
    x,
    '(b1 b2) h w c -> b1 b2 h w c',
    b1=2,
)
# Expected: for shape-compatible x, the batch axis splits into axes 2 and b/2; unit preserved.
```

`einrepeat(x, pattern, **axes_lengths)` covers repeat, tile, and broadcast-like patterns:

```python
expanded = u.math.einrepeat(x, 'h w c -> h new_axis w c', new_axis=5)
# Expected: for shape-compatible x, a new length-5 axis is inserted after h; unit preserved.
tiled = u.math.einrepeat(x, 'h w c -> (2 h) (2 w) c')
# Expected: for shape-compatible x, h and w each double by repetition; unit preserved.
```

Use `einshape(x, pattern)` to map named axes to lengths. Use `einreduce()` and `einsum()` through `math-function-library.md` when axes are reduced or contracted.

## Backend Conversion

All `Quantity` backend methods retain the wrapper and attached unit:

| Exact signature | Target mantissa |
|---|---|
| `to_jax()` | JAX `Array` |
| `to_numpy()` | `numpy.ndarray` |
| `to_cupy(*, device=None)` | `cupy.ndarray` |
| `to_torch(*, device=None, dtype=None)` | `torch.Tensor` |
| `to_dask(*, chunks='auto')` | `dask.array.Array` |
| `to_ndonnx()` | Symbolic `ndonnx.Array` |

`u.math.as_numpy(x)` is different: for a `Quantity`, it returns the underlying mantissa in the current unit scale as a NumPy array.

## Source-Backed Gotchas

- `q.at` supports NumPy, JAX, CuPy, Torch, and Dask mantissas. The ndonnx backend raises `BackendError`; the API recommends `.to_numpy()` first.
- Repeated-index update semantics differ by backend. JAX, NumPy, and CuPy accumulate documented operations; some Torch and Dask operations use last-write-wins behavior.
- Dask functional updates support slice, scalar integer, 1-D integer, and boolean-mask indices; multidimensional fancy integer indexing raises `NotImplementedError`.
- `mT` transposes only the last two dimensions and requires at least two dimensions.
- `u.math.as_numpy(q)` drops the `Quantity` wrapper; `q.to_numpy()` keeps it.
- In Einstein patterns, axis order inside parentheses changes element order. A literal `1` or `()` creates a length-one axis, and either can be removed in the inverse pattern.

## Sources Mirrored

- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.Quantity.html
- https://brainunit.readthedocs.io/apis/brainunit.math.html
- https://brainunit.readthedocs.io/unit_operations/einstein_operations.html
