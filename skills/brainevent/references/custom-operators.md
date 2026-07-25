# BrainEvent custom operators

Use this reference when the required event-driven computation is not covered by BrainEvent's built-in operations and a custom CPU or GPU kernel must be written.

BrainEvent's custom-operator tutorials extend the package from high-level Numba/Warp decorators down to hand-written C++ and CUDA. These are extension paths, not part of the canonical `BinaryArray @ connectivity` workflow.

## Tutorial routing

| Implementation target | Open this official tutorial |
|---|---|
| Custom CPU operator with Numba | [Custom CPU Operators with Numba](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html) |
| Custom GPU operator with Numba CUDA | [Custom GPU Operators with Numba CUDA](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/02_numba_cuda.html) |
| Custom GPU operator with NVIDIA Warp | [Custom GPU Operators with Warp](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/03_warp.html) |
| Hand-written C++ CPU kernel | [Custom C++ (CPU) kernels](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/04_cpp.html) |
| Hand-written CUDA GPU kernel | [Custom CUDA (GPU) kernels](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/05_cuda.html) |

Choose the tutorial matching the implementation technology already required by the task. Read that tutorial before selecting decorators, signatures, launch configuration, compilation, or registration details; the index alone does not define those APIs.

## Canonical Numba CPU workflow

`numba_kernel()` wraps a compiled Numba function with an explicit JAX output specification, making the custom CPU operation callable inside JAX transformations.

| API | Description |
|---|---|
| `@numba.njit` | Use to compile the numerical CPU kernel; write outputs into the provided output array instead of allocating or returning an array from the kernel. |
| `numba_kernel(kernel, outs=...)` | Use to wrap the compiled kernel for JAX; declare every output with `jax.ShapeDtypeStruct`, and unwrap the returned tuple for a single output when required. |
| `jax.jit(function)` | Use around the complete pipeline after constructing the wrapper outside the transformed function; it compiles the custom call with surrounding JAX operations. |
| `XLACustomKernel(name)` | Use when the task needs named backend registration or multiple kernel implementations; open the official tutorial before defining generators and backends. |

```python
import brainevent
import jax
import jax.numpy as jnp
import numba

@numba.njit
def add_kernel(x, y, out):
    for i in range(out.size):
        out[i] = x[i] + y[i]


size = 512
add = brainevent.numba_kernel(
    add_kernel,
    outs=jax.ShapeDtypeStruct((size,), jnp.float32),
)

a = jnp.arange(size, dtype=jnp.float32)
b = jnp.ones(size, dtype=jnp.float32) * 3.0

result = add(a, b)
result = result[0] if isinstance(result, tuple) else result
assert jnp.allclose(result, a + b)

@jax.jit
def compiled_pipeline(x, y):
    value = add(x, y)
    value = value[0] if isinstance(value, tuple) else value
    return jnp.sin(value) * jnp.sqrt(jnp.abs(value) + 1.0)


output = compiled_pipeline(a, b)
assert output.shape == (size,)
```

This example mirrors the official `add_kernel` wrapper and JIT pipeline while keeping one custom kernel. Define output shapes and dtypes statically; construct reusable wrappers outside `jax.jit`.

## Application: CSR synaptic accumulation kernel

The official neuroscience application registers a row-parallel CSR matrix-vector kernel, then verifies the sparse synaptic accumulation against the equivalent dense product.

```python
import brainevent
import jax
import jax.numpy as jnp
import numba
import numpy as np
import scipy.sparse as sp


@numba.njit(parallel=True)
def csr_matvec_kernel(data, indices, indptr, vector, out):
    n_rows = indptr.size - 1
    for row in numba.prange(n_rows):
        total = out.dtype.type(0)
        for offset in range(indptr[row], indptr[row + 1]):
            total += data[offset] * vector[indices[offset]]
        out[row] = total


n_pre = 2_000
n_post = 1_000
connection_probability = 0.05
rng = np.random.default_rng(42)

dense = (
    rng.random((n_post, n_pre)) < connection_probability
).astype(np.float32)
dense *= rng.uniform(0.01, 0.5, dense.shape).astype(np.float32)
csr = sp.csr_matrix(dense)

data = jnp.asarray(csr.data, dtype=jnp.float32)
indices = jnp.asarray(csr.indices, dtype=jnp.int32)
indptr = jnp.asarray(csr.indptr, dtype=jnp.int32)
presynaptic_values = jnp.asarray(
    rng.random(n_pre),
    dtype=jnp.float32,
)
out_spec = jax.ShapeDtypeStruct((n_post,), jnp.float32)

csr_matvec = brainevent.numba_kernel(
    csr_matvec_kernel,
    outs=out_spec,
)

@jax.jit
def synaptic_accumulation(values):
    result = csr_matvec(data, indices, indptr, values)
    return result[0] if isinstance(result, tuple) else result


postsynaptic_values = synaptic_accumulation(presynaptic_values)
expected = jnp.asarray(dense) @ presynaptic_values

assert postsynaptic_values.shape == (n_post,)
assert jnp.allclose(postsynaptic_values, expected, atol=1e-5)
print("stored connections:", csr.nnz)
print(
    "maximum dense-reference error:",
    jnp.max(jnp.abs(postsynaptic_values - expected)),
)
```

This adapts “Neuroscience Example: Sparse CSR x Float-Vector Multiplication” to the current direct Numba wrapper and omits its hardware-dependent benchmark. Construct the wrapper once, keep its output specification static, and validate numerical equivalence before timing.

## Source

- Custom operators tutorial index: https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/index.html
- Custom CPU Operators with Numba: https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html
