# BrainEvent custom CPU operators

Use this reference when BrainEvent's built-in operations cannot express a CPU computation and the operation should run through Numba or compiled C++ as a JAX-compatible custom call. Open `custom-operators-gpu.md` instead for Numba CUDA, Warp, raw CUDA, Pallas, or Triton.

Define output shapes and dtypes statically, construct wrappers outside transformed functions, and validate every custom result against a trusted JAX implementation before measuring performance.

## Choose a CPU implementation

| Path | Use when | Main constraint |
|---|---|---|
| Numba CPU | The algorithm has irregular sparse access, sequential loops, or modest problem sizes and should remain Python-authored. | The kernel takes inputs followed by output buffers and writes results in place. |
| Raw C++ | Existing C++ should be reused or the operation needs compiler-level CPU control. | Declare inputs as `const BE::Tensor` and outputs as non-`const` `BE::Tensor`. |

Prefer `numba_kernel()` for one CPU implementation. Use `XLACustomKernel` when the operation also needs named backend dispatch, explicit batching or autodiff rules, or BrainEvent benchmarking.

## Prepare the CPU toolchain

Install `brainevent[cpu]`, which includes the Numba CPU backend. Raw C++ additionally requires a host compiler such as `g++` or `clang++`.

Run `brainevent.print_diagnostics()` before debugging kernel source when Numba or the compiler cannot be found. Treat `ModuleNotFoundError: numba` as a missing CPU extra, not as a kernel implementation failure.

## Follow the CPU kernel contract

| Contract | Required behavior |
|---|---|
| Inputs | Pass runtime arrays before output buffers. Do not mutate an input unless `input_output_aliases` declares the alias. |
| Outputs | Declare every output with `jax.ShapeDtypeStruct`; write into its output buffer. An `XLACustomKernel` callable must return one result per flattened output specification, as a tuple even for one output. |
| Wrapper lifetime | Construct and cache the wrapper at module definition or kernel-generator time, never inside `jax.jit`. |
| JIT | Call an already-created wrapper inside `jax.jit`; keep output shape and dtype trace-time static. |
| `vmap` | Set `vmap_method` or define an `XLACustomKernel` batching rule when the default behavior is incorrect or inefficient. |
| Autodiff | Register JVP and transpose rules when differentiation is required; a compiled forward kernel does not define derivatives automatically. |
| Verification | Compare shape, dtype, values, and mutation semantics with a JAX reference before benchmarking. |

## Write a Numba CPU kernel

Use Numba CPU for custom loops and sparse or irregular memory access that does not map cleanly to a massively parallel backend.

| API | Description |
|---|---|
| `@numba.njit` | Use to compile `kernel(inputs..., outputs...)`; it writes all results into output arrays and returns nothing. |
| `@numba.njit(parallel=True)` | Use when the outer loop has independent iterations; it enables CPU threading. |
| `numba.prange(...)` | Use for the independent outer loop; keep dependent inner loops sequential. |
| `numba_kernel(kernel, outs, *, vmap_method=None, input_output_aliases=None)` | Use to expose the Numba dispatcher to JAX; `outs` is one shape/dtype struct or a sequence for multiple outputs. |

```python
import brainevent
import jax
import jax.numpy as jnp
import numba


@numba.njit(parallel=True)
def add_kernel(x, y, out):
    for i in numba.prange(out.size):
        out[i] = x[i] + y[i]


n = 512
add = brainevent.numba_kernel(
    add_kernel,
    outs=jax.ShapeDtypeStruct((n,), jnp.float32),
)


@jax.jit
def add_then_scale(x, y):
    result = add(x, y)
    result = result[0] if isinstance(result, tuple) else result
    return result * 2.0


x = jnp.arange(n, dtype=jnp.float32)
y = jnp.ones(n, dtype=jnp.float32)
actual = add_then_scale(x, y)
assert jnp.allclose(actual, (x + y) * 2.0)
```

For multiple outputs, append all output buffers to the Numba signature and pass matching `ShapeDtypeStruct` values in the same order.

## Register a CPU operator

`XLACustomKernel` owns one logical primitive and selects a registered implementation for the current platform.

| API | Description |
|---|---|
| `XLACustomKernel(name, doc=None)` | Use to create a uniquely named custom primitive. |
| `def_kernel(backend, platform, generator, asdefault=False)` | Use to register a general generator for a named backend and platform; the first backend for a platform becomes its default. |
| `def_numba_kernel(generator, asdefault=False)` | Use to register a Numba generator for CPU. |
| `available_backends("cpu")` | Use to list registered CPU backends. |
| `set_default("cpu", backend)` | Use to select an already-registered CPU backend. |
| `get_default("cpu")` / `defaults` | Use to inspect the CPU default or all platform defaults. |

A generator reads trace-time metadata from keyword arguments and returns the runtime callable:

```python
import brainevent
import jax
import jax.numpy as jnp
import numba


@numba.njit(parallel=True)
def scale_kernel(x, scale, out):
    for i in numba.prange(out.size):
        out[i] = x[i] * scale[0]


def scale_numba_generator(**kwargs):
    out_spec = kwargs["outs"][0]

    def run(x, scale):
        result = brainevent.numba_kernel(
            scale_kernel,
            outs=out_spec,
        )(x, scale)
        return result if isinstance(result, tuple) else (result,)

    return run


scale = brainevent.XLACustomKernel("reference_cpu_scale")
scale.def_numba_kernel(scale_numba_generator)

x = jnp.arange(128, dtype=jnp.float32)
factor = jnp.array([3.0], dtype=jnp.float32)
out_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
actual = scale(x, factor, outs=[out_spec])[0]
assert jnp.allclose(actual, x * factor[0])
```

Do not read runtime array values while constructing the generator. Return a tuple matching flattened `outs`; a direct one-output wrapper may return a bare array, but `XLACustomKernel` lowering uses multiple-result semantics.

### Define transformations and benchmarks

| API | Description |
|---|---|
| `def_batching_rule(rule)` | Use to replace batching with a rule that returns batched outputs and output batch axes. |
| `register_general_batching()` | Use to restore the default `jax.lax.scan` batching rule; it is general but may be slower than native batching. |
| `def_jvp_rule(rule)` | Use to define one forward-mode derivative rule. |
| `def_jvp_rule2(*rules)` | Use to define one tangent contribution per input; pass `None` for zero contributions. |
| `def_transpose_rule(rule)` | Use to define reverse-mode cotangent propagation. |
| `def_call(function)` / `call(...)` | Use to register and invoke a high-level call function. |
| `def_benchmark_data(function)` | Use to register benchmark configurations for a platform. |
| `benchmark(...)` | Use after registering call and data functions; it compares CPU backends and can validate outputs with `rtol` and `atol`. |
| `def_tags(*tags)` | Use to attach searchable categories such as `csr`, `binary`, or `mv`. |

Do not claim differentiability until JVP and transpose behavior has been tested.

## Compile a raw C++ CPU kernel

Use raw C++ when existing C++ should be reused or compiler-level control matters more than Python authoring.

| API | Description |
|---|---|
| `load_cpp_inline(name, cpp_sources, functions=None, **options)` | Use for inline C++; it compiles with the host compiler and returns a `CompiledModule`. Pass function names for auto-detection, a dict for explicit `arg_spec`, or `None` for `// @BE` discovery. |
| `load_cpp_file(filepath, functions=None, *, name=None, **options)` | Use for one `.cpp` or `.cc` file; it forwards options to `load_cpp_inline()`. |
| `jax.ffi.ffi_call(target, out_spec, ...)` | Use after registration to call `"<target_prefix>.<function>"` from eager or jitted JAX code. |

```python
import brainevent
import jax
import jax.numpy as jnp

cpp_source = r"""
#include "brainevent/common.h"

void add_one(const BE::Tensor x, BE::Tensor out) {
    const float* input = x.data_ptr<float>();
    float* output = out.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        output[i] = input[i] + 1.0f;
    }
}
"""

brainevent.load_cpp_inline(
    name="reference_cpu_ops",
    cpp_sources=cpp_source,
    functions=["add_one"],
)

add_one = jax.ffi.ffi_call(
    "reference_cpu_ops.add_one",
    jax.ShapeDtypeStruct((3,), jnp.float32),
    vmap_method="broadcast_all",
)
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
assert jnp.allclose(add_one(x), x + 1.0)
```

## Declare C++ arguments

Every compiled function needs an `arg_spec` so BrainEvent can generate the XLA FFI wrapper.

| Token | C++ parameter | Use |
|---|---|---|
| `"arg"` | `const BE::Tensor` | Mark an input for wrapper generation. |
| `"ret"` | `BE::Tensor` | Mark a preallocated output written by the function. |
| `"attr.<name>"` | Inferred scalar type | Pass a scalar keyword attribute whose type is parsed from the C++ signature. |
| `"attr.<name>:<type>"` | Explicit scalar type | Pass a scalar keyword attribute with an explicit supported type. |

Use parameter order `inputs -> outputs -> scalar attributes`. The aliases `"args"`, `"rets"`, and `"attrs.<name>"` normalize to the corresponding tokens.

Supported scalar attribute types are:

| Token type | C++ type | Python call value |
|---|---|---|
| `bool` | `bool` | `True` or `False` |
| `int8` / `uint8` | `int8_t` / `uint8_t` | `numpy.int8` / `numpy.uint8` |
| `int16` / `uint16` | `int16_t` / `uint16_t` | `numpy.int16` / `numpy.uint16` |
| `int32` / `uint32` | `int32_t` / `uint32_t` | `numpy.int32` / `numpy.uint32` |
| `int64` / `uint64` | `int64_t` / `uint64_t` | `numpy.int64` / `numpy.uint64` |
| `float32` / `float64` | `float` / `double` | `numpy.float32` / `numpy.float64` |
| `complex64` / `complex128` | `std::complex<float>` / `std::complex<double>` | `numpy.complex64` / `numpy.complex128` |
| `float16` / `bfloat16` | `uint16_t` raw bits | Pass a `uint16` bit view and reinterpret it inside C++. |

Pass scalar attributes to the callable returned by `ffi_call`, not to `ffi_call` itself. Use the explicit token form for pointer types, `__half`, `__nv_bfloat16`, or non-standard type spellings.

`const BE::Tensor` marks an input only in wrapper metadata; it does not make the underlying memory read-only. Remove `const` from every tensor the function writes. If every tensor is `const`, auto-detection cannot find an output.

## Use the C++ tensor API

Include `"brainevent/common.h"` for `BE::Tensor` and checking macros. Do not include BrainEvent's internal FFI headers.

| API | Description |
|---|---|
| `tensor.data_ptr()` / `tensor.data_ptr<T>()` | Return the untyped or typed contiguous data pointer. |
| `tensor.ndim()` | Return the number of dimensions. |
| `tensor.size(i)` / `tensor.shape(i)` | Return one dimension size. |
| `tensor.stride(i)` | Return one C-contiguous stride. |
| `tensor.shape_ptr()` / `tensor.strides_ptr()` | Return pointers to all shape or stride entries. |
| `tensor.dtype()` | Return the `BE::DType` enum. |
| `tensor.element_size()` | Return bytes per element. |
| `tensor.numel()` / `tensor.nbytes()` | Return element count or total byte count. |
| `tensor.is_contiguous()` | Report whether storage is contiguous. |
| `dtype_size(dtype)` / `dtype_name(dtype)` | Return a dtype's byte width or readable name. |
| `BE_CHECK(condition) << message` | Fail with a descriptive runtime message when an invariant is false. |
| `BE_DISPATCH_FLOATING(...)` | Dispatch over float32 and float64. |
| `BE_DISPATCH_INTEGRAL(...)` | Dispatch over signed and unsigned integer dtypes. |
| `BE_DISPATCH_ALL_TYPES(...)` | Dispatch over all numeric floating and integral dtypes. |

`BE::DType` represents float16, float32, float64, bfloat16, signed and unsigned integers from 8 through 64 bits, bool, complex64, and complex128.

## Configure CPU compilation and caching

| Option or API | Description |
|---|---|
| `extra_cflags=[...]` | Pass additional host compiler flags. |
| `extra_ldflags=[...]` | Pass additional linker flags. |
| `extra_include_paths=[...]` | Add user include directories. |
| `build_directory=...` | Override the build directory for one load call. |
| `verbose=True` | Print the compiler command and detailed output. |
| `force_rebuild=True` | Ignore the cache and compile again. |
| `auto_register=False` | Compile without automatically registering FFI targets. |
| `target_prefix=...` | Override the prefix used by automatically registered targets. |
| `set_cache_dir(path)` / `get_cache_dir()` | Change or inspect the process cache directory. |
| `clear_cache(name=None)` | Remove all cache entries or only entries for one module; it returns the number removed. |

The default cache directory is `~/.cache/brainevent/`; `BRAINEVENT_CACHE_DIR` changes it globally. Cache keys include source, compiler flags, CPU architecture marker, compiler version, and BrainEvent version.

Repeated loading of the same cached shared library is safe. Registering a different module under an existing target name raises `KernelRegistrationError`.

## Inspect and diagnose CPU kernels

| API | Description |
|---|---|
| `CompiledModule.path` | Return the loaded shared-library path. |
| `CompiledModule.function_names` | Return the compiled user function names. |
| `CompiledModule.get_handler(name)` | Return the `ctypes` FFI handler without the generated `be_` prefix. |
| `register_ffi_target(target_name, module, func_name, *, platform="cpu")` | Register a compiled CPU handler manually. |
| `list_registered_targets()` | Return registered FFI target names in sorted order. |
| `print_diagnostics()` | Print the BrainEvent compilation environment. |

| Exception | Meaning |
|---|---|
| `KernelToolchainError` | The host compiler is missing or incompatible. |
| `CompilationError` / `KernelCompilationError` | The toolchain exists but source, types, shapes, or backend constraints prevented compilation. |
| `KernelRegistrationError` | An FFI target conflicts with a live registration or could not be registered. |
| `KernelNotAvailableError` | The requested CPU backend is not installed or version-compatible. |
| `KernelFallbackExhaustedError` | No registered CPU backend can handle the operation. |
| `KernelExecutionError` | Compilation succeeded but runtime execution failed. |

For compilation failures, rerun the load with `verbose=True`, then validate buffer shapes, dtypes, and bounds.

## Verify a CPU operator

1. Run the kernel eagerly on the smallest meaningful input.
2. Compare every output with a pure JAX reference, including shape and dtype.
3. Run the already-created wrapper inside `jax.jit`.
4. Test `vmap`, JVP, and reverse-mode gradients only when the operation promises them.
5. Exercise every registered CPU backend with the same fixtures.
6. Benchmark only after warmup and output blocking.

Do not write a custom CPU kernel when a built-in BrainEvent operation already provides the required semantics and maintained backend coverage.

## Official sources

- [Installation](https://brainx.chaobrain.com/brainevent/getting-started/installation.html)
- [Custom CPU operators with Numba](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html)
- [Custom C++ CPU kernels](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/04_cpp.html)
- [Custom-kernel framework API](https://brainx.chaobrain.com/brainevent/reference/apis/operator.html)
- [`arg_spec` system](https://brainx.chaobrain.com/brainevent/reference/kernels/arg-spec.html)
- [C++ API](https://brainx.chaobrain.com/brainevent/reference/kernels/cpp-api.html)
- [Caching](https://brainx.chaobrain.com/brainevent/reference/kernels/caching.html)
