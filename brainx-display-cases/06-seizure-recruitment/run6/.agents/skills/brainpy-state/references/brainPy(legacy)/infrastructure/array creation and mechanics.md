# Legacy BrainPy array creation and mechanics

Use this reference for legacy `brainpy.math` array construction, mutable `Array` mechanics, NumPy-style transformations, backend conversion, random sampling, and named-axis operations. Use `../brainpy legacy workflow.md` for dynamical `Variable` lifecycle and model execution.

## Choose the array representation

Legacy `brainpy.math` follows NumPy syntax on a JAX backend while adding mutable wrappers needed by BrainPy's object-oriented transformations.

| API | Use when |
|---|---|
| `bm.Array(value, dtype=None)` | Wrap array-like data in BrainPy's mutable multidimensional array type. |
| `bm.array(a, dtype=None, copy=True, order='K', ndmin=0)` | Create a BrainPy `Array` from values, with copy and minimum-rank controls. |
| `bm.asarray(a, dtype=None, order=None)` | Convert input to a BrainPy `Array` without requesting an unnecessary copy. |
| `bm.Variable(value_or_size, ...)` | Mark model state that changes inside object-oriented transforms. A tuple or list can be interpreted as a shape, so wrap literal values with `bm.asarray(...)`. |
| `bm.TrainVar(value, ...)` | Mark a mutable value that optimizers should retrieve as trainable. |
| `bm.as_jax(tensor, dtype=None)` | Cross a library boundary with a JAX array. The returned value is not the mutable BrainPy wrapper. |
| `bm.as_numpy(tensor, dtype=None)` | Materialize a host NumPy array for plotting, serialization, or non-JAX code. |
| `bm.as_variable(tensor, dtype=None)` | Convert an existing array to `Variable`; pass a BrainPy or JAX array when the input is data rather than a shape. |

```python
import brainpy.math as bm

x = bm.array([[1.0, 2.0], [3.0, 4.0]])
x[0, 1] = 10.0

state = bm.as_variable(bm.asarray([1.0, 2.0]))
state.value = bm.asarray([3.0, 4.0])

assert x.shape == (2, 2)
assert state.shape == (2,)
```

**Invariant:** Use `Variable` for values mutated inside `bm.jit`, `bm.cls_jit`, `bm.grad`, or another BrainPy object transform. An ordinary array attribute is captured as static data and later Python-side replacement does not update already compiled code.

## Create arrays

Use the NumPy-equivalent constructor name unless BrainPy-specific mutability or conversion is the actual decision.

| API | Description |
|---|---|
| `bm.zeros(shape, dtype=None)` | Create an array filled with zero. |
| `bm.ones(shape, dtype=None)` | Create an array filled with one. |
| `bm.full(shape, fill_value, dtype=None)` | Create an array filled with one explicit value. |
| `bm.empty(shape, dtype=None)` | Allocate an array whose entries must be overwritten before use. |
| `bm.zeros_like(a, ...)` | Create zeros with an existing array's shape and default dtype. |
| `bm.ones_like(a, ...)` | Create ones with an existing array's shape and default dtype. |
| `bm.full_like(a, fill_value, ...)` | Fill an existing array's shape with one explicit value. |
| `bm.arange(start, stop=None, step=None, dtype=None)` | Create half-open regularly spaced values. |
| `bm.linspace(start, stop, num=50, endpoint=True, ...)` | Create a fixed number of evenly spaced values. |
| `bm.logspace(start, stop, num=50, endpoint=True, base=10.0, ...)` | Create values with evenly spaced exponents. |
| `bm.eye(N, M=None, k=0, dtype=None)` | Create a matrix with ones on diagonal `k`. |
| `bm.identity(n, dtype=None)` | Create a square identity matrix. |
| `bm.diag(v, k=0)` | Construct a diagonal matrix or extract a diagonal according to input rank. |
| `bm.meshgrid(*xi, copy=True, sparse=False, indexing='xy')` | Expand coordinate vectors into coordinate grids. |

```python
times = bm.arange(0.0, 100.0, bm.get_dt())
currents = bm.full(times.shape, 2.5)
grid_x, grid_y = bm.meshgrid(
    bm.linspace(-1.0, 1.0, 5),
    bm.linspace(0.0, 2.0, 3),
    indexing='ij',
)

assert times.ndim == 1
assert currents.shape == times.shape
assert grid_x.shape == grid_y.shape == (5, 3)
```

Open `Input generation.md` when the values represent simulation stimuli with sections, ramps, oscillations, spike trains, or Poisson input rather than general-purpose arrays.

## Index, update, and inspect

`Array` permits NumPy-style in-place assignment even though its underlying JAX value is immutable; assignments update the wrapper's stored value.

| Operation | Result |
|---|---|
| `x[index]` | Select values with standard integer, slice, boolean, or advanced indexing. |
| `x[index] = value` | Replace selected values through the mutable BrainPy wrapper. |
| `x += value` and other augmented assignments | Update the wrapper in place. |
| `x.value` | Access the underlying JAX value of an `Array` or `Variable`. |
| `x.shape`, `x.ndim`, `x.size`, `x.dtype` | Inspect rank, extent, element count, and element type. |
| `bm.shape(x)`, `bm.ndim(x)`, `bm.size(x)` | Use functional metadata lookup in composed code. |

Assignment mutates aliases of the same wrapper:

```python
x = bm.arange(5)
alias = x
x[2] = 20

assert alias[2] == 20
```

Do not assume this behavior after `bm.as_jax(x)`; the returned JAX array uses functional `.at[...]` updates.

## Reshape and combine arrays

Use methods for local transformations and `bm` functions when composing several arrays.

| API | Description |
|---|---|
| `bm.reshape(a, shape, ...)` | Change shape without changing element order. |
| `bm.ravel(a, ...)` | Return a contiguous flattened view when possible. |
| `bm.flatten(a, start_dim=None, end_dim=None)` | Flatten all dimensions by default or only the selected dimension range. |
| `bm.squeeze(a, axis=None)` | Remove length-one axes. |
| `bm.expand_dims(a, axis)` | Insert a length-one axis. |
| `bm.transpose(a, axes=None)` | Permute axes. |
| `bm.swapaxes(a, axis1, axis2)` | Exchange two axes. |
| `bm.moveaxis(a, source, destination)` | Move selected axes while retaining the others' order. |
| `bm.concatenate(arrays, axis=0)` | Join arrays along an existing axis. |
| `bm.stack(arrays, axis=0)` | Join arrays along a new axis. |
| `bm.hstack(arrays)` | Stack according to NumPy's horizontal convention. |
| `bm.vstack(arrays)` | Stack according to NumPy's vertical convention. |
| `bm.dstack(arrays)` | Stack along a third axis. |
| `bm.column_stack(arrays)` | Stack one-dimensional arrays as columns. |
| `bm.block(arrays)` | Assemble an array from nested blocks. |
| `bm.split(a, indices_or_sections, axis=0)` | Split only when equal division or exact cut indices are known. |
| `bm.array_split(a, indices_or_sections, axis=0)` | Permit unequal sub-array lengths. |
| `bm.hsplit(a, indices_or_sections)` | Split using the horizontal convention. |
| `bm.vsplit(a, indices_or_sections)` | Split using the vertical convention. |
| `bm.dsplit(a, indices_or_sections)` | Split along the third axis. |
| `bm.broadcast_arrays(*args)` | Broadcast several arrays to one compatible shape. |
| `bm.broadcast_to(a, shape)` | Broadcast one array to an explicit shape. |
| `bm.repeat(a, repeats, axis=None)` | Repeat individual elements. |
| `bm.tile(a, reps)` | Repeat the whole array pattern. |
| `bm.take(a, indices, axis=None)` | Select values by integer indices. |
| `bm.choose(a, choices, ...)` | Select among arrays using an index array. |
| `bm.compress(condition, a, axis=None)` | Select entries along an axis using a boolean condition. |
| `bm.where(condition, x, y)` | Select elementwise between two values. |
| `bm.sort(a, axis=-1)` | Return values ordered along an axis. |
| `bm.argsort(a, axis=-1)` | Return indices that order values along an axis. |
| `bm.lexsort(keys, axis=-1)` | Return indices for an indirect stable sort over multiple keys. |
| `bm.flip(a, axis=None)` | Reverse element order along selected axes. |
| `bm.roll(a, shift, axis=None)` | Circularly shift elements. |

```python
a = bm.arange(6).reshape(2, 3)
b = bm.ones((2, 3))

rows = bm.concatenate([a, b], axis=0)
channels = bm.stack([a, b], axis=-1)
left, right = bm.split(a, [1], axis=1)

assert rows.shape == (4, 3)
assert channels.shape == (2, 3, 2)
assert left.shape == (2, 1) and right.shape == (2, 2)
```

## Reduce and contract arrays

Reduction axes determine which dimensions survive and must be chosen explicitly for batched or time-major model data.

| API | Description |
|---|---|
| `bm.sum(a, axis=None, ...)` | Sum values over selected axes. |
| `bm.mean(a, axis=None, ...)` | Compute the arithmetic mean over selected axes. |
| `bm.prod(a, axis=None, ...)` | Multiply values over selected axes. |
| `bm.min(a, axis=None, ...)` | Return minimum values. |
| `bm.max(a, axis=None, ...)` | Return maximum values. |
| `bm.std(a, axis=None, ...)` | Compute standard deviation. |
| `bm.var(a, axis=None, ...)` | Compute variance. |
| `bm.all(a, axis=None, ...)` | Require all boolean values along selected axes. |
| `bm.any(a, axis=None, ...)` | Require at least one true value along selected axes. |
| `bm.argmin(a, axis=None, ...)` | Return indices of minimum values. |
| `bm.argmax(a, axis=None, ...)` | Return indices of maximum values. |
| `bm.matmul(x1, x2, ...)` | Apply matrix multiplication with NumPy/JAX broadcasting rules. |
| `bm.dot(a, b, ...)` | Apply vector dot product or the documented higher-rank contraction. |
| `bm.tensordot(a, b, axes=2, ...)` | Contract explicitly selected axes. |
| `bm.einsum(subscripts, *operands, ...)` | Express a named index contraction. |

Use `axis=0` only after confirming that axis 0 is time or batch for the current workflow. Legacy trainers can accept different axis conventions through their own configuration.

## Use named-axis transformations

Use the `ein_*` functions when an axis pattern communicates the transformation more reliably than several positional reshape and transpose calls.

| API | Description |
|---|---|
| `bm.ein_rearrange(x, pattern, **axes_lengths)` | Reorder, compose, split, add, or remove axes without reduction. |
| `bm.ein_reduce(x, pattern, reduction, **axes_lengths)` | Rearrange and reduce with `sum`, `mean`, `max`, `min`, or another supported reduction. |
| `bm.ein_repeat(x, pattern, **axes_lengths)` | Rearrange while repeating or broadcasting elements. |

```python
x = bm.arange(2 * 3 * 4).astype(bm.float32).reshape(2, 3, 4)

flat = bm.ein_rearrange(x, 'b h w -> b (h w)')
pooled = bm.ein_reduce(x, 'b h w -> b', 'mean')
repeated = bm.ein_repeat(x, 'b h w -> b copies h w', copies=2)

assert flat.shape == (2, 12)
assert pooled.shape == (2,)
assert repeated.shape == (2, 2, 3, 4)
```

Axis order inside parentheses changes element order. A literal `1` or `()` represents a length-one axis and can be added or removed only where the pattern remains shape-compatible.

## Random-number behavior

`bm.random` manages a default `RandomState`, so ordinary calls do not require passing a JAX key. Seed the stream explicitly when results must be reproducible.

```python
bm.random.seed(123)
sample = bm.random.uniform(size=(2, 3))
normal = bm.random.normal(size=(2, 3))
```

Unlike direct `jax.random` calls, successive `bm.random` calls advance the stored key. Use an explicit `bm.random.RandomState(seed)` when independent reproducible streams are required.

## Common failures

- Do not pass a Python list of values directly to `bm.Variable` or `bm.as_variable` without checking whether it is interpreted as a shape; convert values with `bm.asarray` first.
- Do not mutate an ordinary JAX array with `x[index] = value`; retain the BrainPy `Array` wrapper or use JAX's functional update syntax.
- Do not convert to NumPy inside JIT-compiled model code.
- Do not use `bm.Array` where model state must be discovered and written back by an object-oriented transform; use `bm.Variable`.
- Do not reshape away the time or batch axis until the trainer or monitor convention has been verified.

## Sources mirrored

- https://brainpy.readthedocs.io/apis/brainpy.math.html
- https://brainpy.readthedocs.io/tutorial_math/Numpy_like_Operations.html
- https://brainpy.readthedocs.io/tutorial_math/einops_in_brainpy.html
