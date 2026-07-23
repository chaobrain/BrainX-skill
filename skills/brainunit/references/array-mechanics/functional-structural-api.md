# Functional Structural API

Open this second-level reference from `../array-mechanics.md` when selecting a `brainunit.math` function for joining, stacking, splitting, broadcasting, reshaping, repeating, or selecting values. The module API categorizes these as functions that keep units.

```python
import brainunit as u
import jax.numpy as jnp
```

Examples use concise `Quantity(..., "unit")` results; the exact displayed representation can vary by backend and BrainUnit version.

## Contents

- [Joining and stacking](#joining-and-stacking)
- [Splitting](#splitting)
- [Minimum rank and broadcasting](#minimum-rank-and-broadcasting)
- [Shape and axis operations](#shape-and-axis-operations)
- [Repetition](#repetition)
- [Selection](#selection)

## Joining And Stacking

| Exact signature | One-line description | Example & result |
|---|---|---|
| `concatenate(arrays, axis=0, dtype=None, **kwargs)` | Join a sequence of quantities or arrays along an existing axis. | `a = [1, 2] * u.second; b = [3, 4] * u.second; u.math.concatenate([a, b])`<br>`# Quantity([1 2 3 4], "s")` |
| `stack(arrays, axis=0, dtype=None, **kwargs)` | Join a sequence of quantities or arrays along a new axis. | `a = [1, 2, 3] * u.second; b = [4, 5, 6] * u.second; u.math.stack([a, b])`<br>`# Quantity([[1 2 3] [4 5 6]], "s")` |
| `block(arrays, **kwargs)` | Assemble a quantity or an array from nested lists of blocks. | `a = [[1, 2], [3, 4]] * u.second; u.math.block(a)`<br>`# Quantity([[1 2] [3 4]], "s")` |
| `append(arr, values, axis=None, **kwargs)` | Append values to the end of a quantity or an array. | `a = [1, 2, 3] * u.second; u.math.append(a, 4 * u.second)`<br>`# Quantity([1 2 3 4], "s")` |
| `row_stack(tup, dtype=None, **kwargs)` | Stack quantities or arrays in sequence vertically (row wise). | `a = [1, 2, 3] * u.meter; b = [4, 5, 6] * u.meter; u.math.row_stack([a, b])`<br>`# Quantity([[1 2 3] [4 5 6]], "m")` |
| `vstack(tup, dtype=None, **kwargs)` | Stack quantities or arrays in sequence vertically (row wise). | `a = [1, 2, 3] * u.meter; b = [4, 5, 6] * u.meter; u.math.vstack([a, b])`<br>`# Quantity([[1 2 3] [4 5 6]], "m")` |
| `hstack(arrays, dtype=None, **kwargs)` | Stack quantities or arrays in sequence horizontally (column wise). | `a = [1, 2, 3] * u.meter; b = [4, 5, 6] * u.meter; u.math.hstack([a, b])`<br>`# Quantity([1 2 3 4 5 6], "m")` |
| `dstack(arrays, dtype=None, **kwargs)` | Stack quantities or arrays in sequence depth wise (along the third axis). | `a = [[1], [2], [3]] * u.meter; b = [[4], [5], [6]] * u.meter; u.math.dstack([a, b])`<br>`# Quantity([[[1 4]] [[2 5]] [[3 6]]], "m")` |
| `column_stack(tup, **kwargs)` | Stack 1-D arrays as columns into a 2-D array. | `a = [1, 2, 3] * u.second; b = [4, 5, 6] * u.second; u.math.column_stack([a, b])`<br>`# Quantity([[1 4] [2 5] [3 6]], "s")` |

`row_stack` is documented as an alias of `vstack`; its callable signature therefore mirrors `vstack`.

## Splitting

| Exact signature | One-line description | Example & result |
|---|---|---|
| `split(a, indices_or_sections, axis=0, **kwargs)` | Split a quantity or array into a list of multiple sub-arrays. | `a = jnp.arange(9.0) * u.second; u.math.split(a, 3)`<br>`# [Quantity([0. 1. 2.], "s"), Quantity([3. 4. 5.], "s"), Quantity([6. 7. 8.], "s")]` |
| `array_split(ary, indices_or_sections, axis=0, **kwargs)` | Split an array into multiple sub-arrays. | `a = jnp.arange(9.0) * u.second; parts = u.math.array_split(a, 3); [x.shape for x in parts]`<br>`# [(3,), (3,), (3,)]` |
| `dsplit(a, indices_or_sections, **kwargs)` | Split a quantity or an array into multiple sub-arrays along the third axis (depth). | `a = jnp.arange(16.0).reshape(2, 2, 4) * u.meter; parts = u.math.dsplit(a, 2); [x.shape for x in parts]`<br>`# [(2, 2, 2), (2, 2, 2)]` |
| `hsplit(a, indices_or_sections, **kwargs)` | Split a quantity or an array into multiple sub-arrays horizontally (column-wise). | `a = jnp.arange(16.0).reshape(4, 4) * u.meter; parts = u.math.hsplit(a, 2); [x.shape for x in parts]`<br>`# [(4, 2), (4, 2)]` |
| `vsplit(a, indices_or_sections, **kwargs)` | Split a quantity or an array into multiple sub-arrays vertically (row-wise). | `a = jnp.arange(16.0).reshape(4, 4) * u.meter; parts = u.math.vsplit(a, 2); [x.shape for x in parts]`<br>`# [(2, 4), (2, 4)]` |

## Minimum Rank And Broadcasting

| Exact signature | One-line description | Example & result |
|---|---|---|
| `atleast_1d(*arys, **kwargs)` | View inputs as quantities or arrays with at least one dimension. | `u.math.atleast_1d(0 * u.second)`<br>`# Quantity([0], "s")` |
| `atleast_2d(*arys, **kwargs)` | View inputs as quantities or arrays with at least two dimensions. | `a = [1, 2, 3] * u.second; u.math.atleast_2d(a)`<br>`# Quantity([[1 2 3]], "s")` |
| `atleast_3d(*arys, **kwargs)` | View inputs as quantities or arrays with at least three dimensions. | `a = [[1, 2], [3, 4]] * u.meter; u.math.atleast_3d(a).shape`<br>`# (2, 2, 1)` |
| `broadcast_arrays(*args, **kwargs)` | Broadcast any number of arrays against each other. | `a = [1, 2, 3] * u.second; b = [[4], [5]] * u.second; result = u.math.broadcast_arrays(a, b); [x.shape for x in result]`<br>`# [(2, 3), (2, 3)]` |
| `broadcast_to(array, shape, **kwargs)` | Broadcast an array to a new shape. | `a = [1, 2, 3] * u.meter; u.math.broadcast_to(a, (2, 3))`<br>`# Quantity([[1 2 3] [1 2 3]], "m")` |

## Shape And Axis Operations

| Exact signature | One-line description | Example & result |
|---|---|---|
| `reshape(a, shape, order='C', **kwargs)` | Gives a new shape to a quantity or an array without changing its data. | `a = [1, 2, 3, 4] * u.second; u.math.reshape(a, (2, 2))`<br>`# Quantity([[1 2] [3 4]], "s")` |
| `moveaxis(a, source, destination, **kwargs)` | Moves axes of a quantity or an array to new positions. | `a = jnp.zeros((3, 4, 5)) * u.meter; u.math.moveaxis(a, 0, -1).shape`<br>`# (4, 5, 3)` |
| `transpose(a, axes=None, **kwargs)` | Permute the dimensions of a quantity or an array. | `a = jnp.ones((2, 3)) * u.second; u.math.transpose(a).shape`<br>`# (3, 2)` |
| `swapaxes(a, axis1, axis2, **kwargs)` | Interchange two axes of a quantity or an array. | `a = jnp.zeros((3, 4, 5)) * u.meter; u.math.swapaxes(a, 0, 2).shape`<br>`# (5, 4, 3)` |
| `expand_dims(a, axis, **kwargs)` | Expand the shape of a quantity or an array. | `a = [1, 2, 3] * u.meter; u.math.expand_dims(a, axis=0).shape`<br>`# (1, 3)` |
| `squeeze(a, axis=None, **kwargs)` | Remove single-dimensional entries from the shape of a quantity or an array. | `a = [[[1], [2], [3]]] * u.second; u.math.squeeze(a).shape`<br>`# (3,)` |
| `flatten(x, start_axis=None, end_axis=None, **kwargs)` | Flattens input by reshaping it into a one-dimensional tensor. | `a = [[1, 2], [3, 4]] * u.second; u.math.flatten(a)`<br>`# Quantity([1 2 3 4], "s")` |
| `unflatten(x, axis, sizes, **kwargs)` | Expands a dimension of the input tensor over multiple dimensions. | `a = [1, 2, 3, 4, 5, 6] * u.meter; u.math.unflatten(a, 0, (2, 3))`<br>`# Quantity([[1 2 3] [4 5 6]], "m")` |

## Repetition

| Exact signature | One-line description | Example & result |
|---|---|---|
| `tile(A, reps, **kwargs)` | Construct a quantity or an array by repeating `A` the number of times given by `reps`. | `a = [1, 2, 3] * u.meter; u.math.tile(a, 2)`<br>`# Quantity([1 2 3 1 2 3], "m")` |
| `repeat(a, repeats, axis=None, total_repeat_length=None, **kwargs)` | Repeat elements of a quantity or an array. | `a = [1, 2, 3] * u.second; u.math.repeat(a, 2)`<br>`# Quantity([1 1 2 2 3 3], "s")` |

## Selection

| Exact signature | One-line description | Example & result |
|---|---|---|
| `take(a, indices, axis=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None, **kwargs)` | Take elements from an array along an axis. | `a = [4, 3, 5, 7, 6, 8] * u.second; u.math.take(a, jnp.array([0, 1, 4]))`<br>`# Quantity([4 3 6], "s")` |
| `gather(input, dim, index, **kwargs)` | Gather values along an axis specified by `dim`, according to `index`. | `a = jnp.array([[1, 2], [3, 4]]) * u.mV; index = jnp.array([[0, 0], [1, 0]]); u.math.gather(a, 1, index)`<br>`# Quantity([[1 1] [4 3]], "mV")` |
| `choose(a, choices, mode='raise', **kwargs)` | Construct a quantity or an array from an index array and a set of arrays to choose from. | `choices = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]; u.math.choose(jnp.array([0, 1, 0]), choices)`<br>`# Array([1, 5, 3], dtype=int32)` |
| `compress(condition, a, axis=None, *, size=None, fill_value=None, **kwargs)` | Return selected slices of a quantity or an array along given axis. | `a = [1, 2, 3, 4] * u.meter; u.math.compress(jnp.array([0, 1, 1, 0]), a)`<br>`# Quantity([2 3], "m")` |
| `extract(condition, arr, *, size=None, fill_value=None, **kwargs)` | Return the elements of an array that satisfy some condition. | `a = jnp.array([1, 2, 3]) * u.meter; u.math.extract(a.mantissa > 1, a)`<br>`# Quantity([2 3], "m")` |

## Sources Mirrored

- https://brainunit.readthedocs.io/apis/brainunit.math.html
- Generated function pages linked from the module index: `https://brainunit.readthedocs.io/apis/generated/brainunit.math.<function>.html`
