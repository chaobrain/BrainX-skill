# BrainEvent custom GPU operators

Use this reference when BrainEvent's built-in operations cannot express a GPU computation and the operation should run through Numba CUDA, Warp, raw CUDA, Pallas, or Triton as a JAX-compatible custom call. Open `custom-operators-cpu.md` instead for Numba CPU or raw C++.

Define output shapes and dtypes statically, construct wrappers outside transformed functions, launch work on the XLA-managed stream, and validate every result against a trusted JAX implementation before measuring performance.

## Choose a GPU implementation

| Path | Use when | Main constraint |
|---|---|---|
| Numba CUDA single kernel | One launch needs CUDA thread, block, shared-memory, or synchronization control from Python. | Specify either `grid` plus `block`, or `launch_dims`. |
| Numba CUDA callable | The operation requires multiple launches, temporary device buffers, or GPU orchestration from Python. | The callable receives all inputs, all outputs, then the XLA stream; zero-dimensional inputs are unsupported. |
| Warp | GPU authoring should remain Python-like and benefit from Warp types or atomic operations. | Use an in/out buffer and atomics for scatter accumulation. |
| Raw CUDA | Existing CUDA C++ should be reused or the operation needs full `nvcc`, launch, and compiler control. | Include `<cuda_runtime.h>`, accept the XLA stream, and launch on that stream. |
| Pallas or Triton registration | An existing Pallas or Triton kernel should participate in BrainEvent dispatch. | BrainEvent provides registration hooks, not package-specific authoring tutorials for these languages. |

Prefer a direct wrapper for one GPU implementation. Use `XLACustomKernel` when backend selection, CPU fallback, transformation rules, or cross-backend benchmarking belongs to the operation's contract.

## Prepare the GPU toolchain

| Path | Requirement |
|---|---|
| Numba CUDA | Install `brainevent[cuda12]` or `brainevent[cuda13]` plus `numba`, and use an NVIDIA GPU with a working driver. |
| Warp | Install `brainevent[cuda12]` or `brainevent[cuda13]` plus `warp-lang`, and use an NVIDIA GPU with a working driver. |
| Raw CUDA | Install `brainevent[cuda12]` or `brainevent[cuda13]`, then provide an NVIDIA driver and a compatible host C++ compiler; the JAX CUDA packages supply `nvcc`, CUDA headers, and runtime libraries. |

Run `brainevent.print_diagnostics()` before debugging source when a compiler or backend cannot be found. Treat `ModuleNotFoundError` for Numba or Warp as a missing optional dependency, not as a kernel implementation failure.

## Follow the GPU kernel contract

| Contract | Required behavior |
|---|---|
| Inputs | Pass runtime arrays before output buffers. Do not mutate an input unless `input_output_aliases` declares the alias. |
| Outputs | Declare every output with `jax.ShapeDtypeStruct`; write into its output buffer. An `XLACustomKernel` callable must return one result per flattened output specification, as a tuple even for one output. |
| Stream | Launch every Numba CUDA or raw CUDA operation on the XLA-managed stream; do not create an unrelated stream. |
| Wrapper lifetime | Construct and cache the wrapper at module definition or kernel-generator time, never inside `jax.jit`. |
| JIT | Call an already-created wrapper inside `jax.jit`; keep output shape and dtype trace-time static. |
| `vmap` | Set `vmap_method` or define an `XLACustomKernel` batching rule when the default behavior is incorrect or inefficient. |
| Autodiff | Register JVP and transpose rules when differentiation is required; a compiled forward kernel does not define derivatives automatically. |
| Verification | Compare shape, dtype, values, and mutation semantics with a JAX reference before benchmarking. |

## Write a Numba CUDA kernel

Use `numba_cuda_kernel()` for one launch and `numba_cuda_callable()` for a multi-launch pipeline or temporary device storage.

| API | Description |
|---|---|
| `@cuda.jit` | Use to compile a GPU kernel; obtain thread indices with `cuda.grid(ndim)`, bounds-check them, and write outputs in place. |
| `numba_cuda_kernel(kernel, outs, *, grid=None, block=None, launch_dims=None, threads_per_block=256, shared_mem=0, vmap_method=None, input_output_aliases=None)` | Use for one CUDA launch. Supply `grid` with `block`, or use `launch_dims` for automatic decomposition; the forms are mutually exclusive. |
| `numba_cuda_callable(func, outs, *, vmap_method=None, input_output_aliases=None)` | Use for a Python callable that launches multiple kernels or allocates temporary device memory; its final argument is the XLA-managed Numba CUDA stream. |

```python
import brainevent
import jax
import jax.numpy as jnp
from numba import cuda


@cuda.jit
def threshold_kernel(values, threshold, spikes):
    i = cuda.grid(1)
    if i < values.size:
        spikes[i] = values[i] >= threshold[0]


n = 1024
detect_spikes = brainevent.numba_cuda_kernel(
    threshold_kernel,
    outs=jax.ShapeDtypeStruct((n,), jnp.bool_),
    launch_dims=n,
    threads_per_block=256,
)

values = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float32)
threshold = jnp.array([0.25], dtype=jnp.float32)
spikes = detect_spikes(values, threshold)
assert jnp.array_equal(spikes, values >= threshold[0])
```

Use explicit `grid` and `block` for fine launch control. Use `launch_dims=n` or `launch_dims=(m, n)` for automatic one- or multi-dimensional decomposition. Prefer block sizes that are multiples of 32.

### Launch multiple Numba CUDA kernels

The callable receives all input buffers, all output buffers, then the XLA stream:

```python
import brainevent
import jax
import jax.numpy as jnp
from numba import cuda


@cuda.jit
def square_kernel(x, temp):
    i = cuda.grid(1)
    if i < x.size:
        temp[i] = x[i] * x[i]


@cuda.jit
def sqrt_kernel(temp, out):
    import math

    i = cuda.grid(1)
    if i < out.size:
        out[i] = math.sqrt(temp[i])


def absolute_value(x, out, stream):
    n = x.shape[0]
    threads = 256
    blocks = (n + threads - 1) // threads
    temp = cuda.device_array(n, dtype=x.dtype)
    square_kernel[blocks, threads, stream](x, temp)
    sqrt_kernel[blocks, threads, stream](temp, out)


n = 512
absolute = brainevent.numba_cuda_callable(
    absolute_value,
    outs=jax.ShapeDtypeStruct((n,), jnp.float32),
)
x = jnp.linspace(-5.0, 5.0, n, dtype=jnp.float32)
assert jnp.allclose(absolute(x), jnp.abs(x))
```

Do not pass zero-dimensional arrays to `numba_cuda_callable()`; use a length-one array for scalar runtime values.

## Write a Warp GPU kernel

Use Warp when GPU acceleration should remain Python-authored and the kernel benefits from Warp's array types, launch model, or atomic operations.

| API | Description |
|---|---|
| `@warp.kernel` | Use to define a GPU kernel with annotated Warp array arguments; the body runs once per `warp.tid()`. |
| `warp.jax_experimental.jax_kernel(...)` | Use to expose the Warp kernel to JAX. Select `output_dims` for allocated outputs or `in_out_argnames` for caller-provided accumulation buffers. |
| `jaxinfo_to_warpinfo(info)` | Use to convert `jax.ShapeDtypeStruct` into a Warp array annotation inside a generator. |
| `jaxtype_to_warptype(dtype)` | Use to convert a JAX or NumPy dtype into a Warp scalar type. |

```python
import jax.numpy as jnp
import warp
from warp.jax_experimental import jax_kernel


@warp.kernel
def relu_kernel(
    x: warp.array(dtype=warp.float32, ndim=1),
    out: warp.array(dtype=warp.float32, ndim=1),
):
    i = warp.tid()
    out[i] = warp.max(x[i], warp.float32(0.0))


n = 1024
relu = jax_kernel(
    relu_kernel,
    launch_dims=[n],
    num_outputs=1,
    output_dims={"out": (n,)},
)

x = jnp.linspace(-2.0, 2.0, n, dtype=jnp.float32)
(actual,) = relu(x)
assert jnp.allclose(actual, jnp.maximum(x, 0.0))
```

For scatter-add synaptic accumulation, mark the accumulator as in/out and use an atomic update:

```python
@warp.kernel
def scatter_add_kernel(
    values: warp.array(dtype=warp.float32, ndim=1),
    targets: warp.array(dtype=warp.int32, ndim=1),
    postsynaptic_current: warp.array(dtype=warp.float32, ndim=1),
):
    edge = warp.tid()
    warp.atomic_add(
        postsynaptic_current,
        targets[edge],
        values[edge],
    )


values = jnp.array([0.5, 0.2, 0.7], dtype=jnp.float32)
targets = jnp.array([0, 1, 0], dtype=jnp.int32)
scatter = jax_kernel(
    scatter_add_kernel,
    launch_dims=[values.size],
    num_outputs=1,
    in_out_argnames=["postsynaptic_current"],
)
(current,) = scatter(
    values,
    targets,
    jnp.zeros(2, dtype=values.dtype),
)
assert jnp.allclose(current, jnp.array([1.2, 0.2]))
```

A plain write is incorrect when several threads may target the same output element.

## Register GPU and multi-backend operators

`XLACustomKernel` dispatches one logical primitive to a registered GPU backend and can also use the CPU implementation defined in `custom-operators-cpu.md` as a fallback.

| API | Description |
|---|---|
| `XLACustomKernel(name, doc=None)` | Use to create a uniquely named custom primitive. |
| `def_kernel(backend, platform, generator, asdefault=False)` | Use to register a general generator for a named backend and platform. |
| `def_numba_cuda_kernel(generator, asdefault=False)` | Use to register Numba CUDA for GPU. |
| `def_warp_kernel(generator, asdefault=False)` | Use to register Warp for GPU. |
| `def_cuda_raw_kernel(generator, asdefault=False)` | Use to register a generator that loads raw CUDA and returns its JAX FFI caller. |
| `def_pallas_kernel("gpu", generator, asdefault=False)` | Use to register an existing Pallas GPU kernel; the JAX version requirement is checked lazily. |
| `def_triton_kernel(generator, asdefault=False)` | Use to register an existing Triton GPU kernel. |
| `def_numba_kernel(generator, asdefault=False)` | Use to add the Numba CPU fallback defined in `custom-operators-cpu.md`. |
| `available_backends("gpu")` | Use to list registered GPU backends. |
| `set_default("gpu", backend)` | Use to select an already-registered GPU backend. |
| `get_default("gpu")` / `defaults` | Use to inspect the GPU default or all platform defaults. |

A Warp generator derives its types from trace-time output metadata and returns a tuple-producing runtime callable:

```python
import brainevent
import jax
import jax.numpy as jnp
import warp
from warp.jax_experimental import jax_kernel


def relu_warp_generator(**kwargs):
    out_spec = kwargs["outs"][0]
    n = out_spec.shape[0]
    array_type = brainevent.jaxinfo_to_warpinfo(out_spec)

    @warp.kernel
    def relu_kernel(x: array_type, out: array_type):
        i = warp.tid()
        out[i] = warp.max(x[i], array_type.dtype(0.0))

    def run(x):
        kernel = jax_kernel(
            relu_kernel,
            launch_dims=[n],
            num_outputs=1,
            output_dims={"out": (n,)},
        )
        return kernel(x)

    return run


relu = brainevent.XLACustomKernel("reference_gpu_relu")
relu.def_warp_kernel(relu_warp_generator)

x = jnp.linspace(-2.0, 2.0, 256, dtype=jnp.float32)
out_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
actual = relu(x, outs=[out_spec])[0]
assert jnp.allclose(actual, jnp.maximum(x, 0.0))
```

Do not read runtime array values while constructing a generator. Return a tuple matching flattened `outs`; Warp's `jax_kernel()` already returns a tuple.

### Define transformations and benchmarks

| API | Description |
|---|---|
| `def_batching_rule(rule)` | Use to replace batching with a rule that returns batched outputs and output batch axes. |
| `register_general_batching()` | Use to restore the default `jax.lax.scan` batching rule; it is general but may be slower than native GPU batching. |
| `def_jvp_rule(rule)` | Use to define one forward-mode derivative rule. |
| `def_jvp_rule2(*rules)` | Use to define one tangent contribution per input; pass `None` for zero contributions. |
| `def_transpose_rule(rule)` | Use to define reverse-mode cotangent propagation. |
| `def_call(function)` / `call(...)` | Use to register and invoke a high-level call function. |
| `def_benchmark_data(function)` | Use to register benchmark configurations for a platform. |
| `benchmark(...)` | Use after registering call and data functions; it compares GPU backends and can validate outputs with `rtol` and `atol`. |
| `def_tags(*tags)` | Use to attach searchable categories such as `csr`, `binary`, or `mv`. |

Do not claim differentiability until JVP and transpose behavior has been tested. Benchmark equivalent GPU backends with output comparison enabled before choosing a default.

## Compile a raw CUDA kernel

Use raw CUDA when the operation needs full CUDA C++ features, compiler flags, launch control, or an existing `.cu` implementation.

| API | Description |
|---|---|
| `load_cuda_inline(name, cuda_sources, functions=None, **options)` | Use for inline CUDA; it compiles with `nvcc`, caches the shared library, and registers discovered or explicitly listed functions. |
| `load_cuda_file(filepath, functions=None, *, name=None, **options)` | Use for one `.cu` file; it forwards options to `load_cuda_inline()`. |
| `load_cuda_dir(directory, functions=None, *, name=None, file_patterns=None, **options)` | Use to compile matching `.cu` sources from one directory as a module. |
| `jax.ffi.ffi_call(target, out_spec, ...)` | Use after registration to invoke the compiled target from eager or jitted JAX code. |

```python
import brainevent
import jax
import jax.numpy as jnp

cuda_source = r"""
#include <cuda_runtime.h>
#include "brainevent/common.h"

__global__ void add_kernel(
    const float* x,
    const float* y,
    float* out,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

// @BE vector_add arg arg ret stream
void vector_add(
    const BE::Tensor x,
    const BE::Tensor y,
    BE::Tensor out,
    int64_t stream
) {
    int64_t n = x.numel();
    add_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    BE_CHECK_KERNEL_LAUNCH();
}
"""

brainevent.load_cuda_inline(
    name="reference_cuda_ops",
    cuda_sources=cuda_source,
)

n = 1024
add = jax.ffi.ffi_call(
    "reference_cuda_ops.vector_add",
    jax.ShapeDtypeStruct((n,), jnp.float32),
)
x = jnp.ones(n, dtype=jnp.float32)
y = jnp.full(n, 2.0, dtype=jnp.float32)
assert jnp.allclose(add(x, y), x + y)
```

Always include `<cuda_runtime.h>`. Put the `stream` token last, accept it as `int64_t`, cast it to `cudaStream_t`, and launch every kernel on that stream.

## Declare CUDA arguments

Every raw CUDA function needs an `arg_spec` so BrainEvent can generate the XLA FFI wrapper.

| Token | C++ parameter | Use |
|---|---|---|
| `"arg"` | `const BE::Tensor` | Mark an input for wrapper generation. |
| `"ret"` | `BE::Tensor` | Mark a preallocated output written by the function. |
| `"attr.<name>"` | Inferred scalar type | Pass a scalar keyword attribute whose type is parsed from the C++ signature. |
| `"attr.<name>:<type>"` | Explicit scalar type | Pass a scalar keyword attribute with an explicit supported type. |
| `"stream"` | `int64_t` | Pass the CUDA stream handle; place it last. |

Use parameter order `inputs -> outputs -> scalar attributes -> stream`. The aliases `"args"`, `"rets"`, `"attrs.<name>"`, and `"ctx.stream"` normalize to the corresponding tokens.

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

`const BE::Tensor` marks an input only in wrapper metadata; it does not make the underlying memory read-only. Remove `const` from every tensor the function writes.

## Use the CUDA C++ API

Include `<cuda_runtime.h>` and `"brainevent/common.h"`. Do not include BrainEvent's internal FFI headers.

| API | Description |
|---|---|
| `tensor.data_ptr()` / `tensor.data_ptr<T>()` | Return the untyped or typed contiguous device pointer. |
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
| `BE_CUDA_CHECK(call)` | Check a CUDA runtime return code. |
| `BE_CHECK_KERNEL_LAUNCH()` | Check `cudaGetLastError()` after a launch. |
| `BE_DISPATCH_FLOATING(...)` | Dispatch over float32 and float64. |
| `BE_DISPATCH_INTEGRAL(...)` | Dispatch over signed and unsigned integer dtypes. |
| `BE_DISPATCH_ALL_TYPES(...)` | Dispatch over all numeric floating and integral dtypes. |

`BE::DType` represents float16, float32, float64, bfloat16, signed and unsigned integers from 8 through 64 bits, bool, complex64, and complex128.

## Configure GPU compilation and caching

| Option or API | Description |
|---|---|
| `optimization_level=0..3` | Select `-O<n>` for CUDA host and device compilation; the default is 3. Use 0 for source-level debugging. |
| `use_fast_math=True` | Enable flush-to-zero, approximate division and square root, and fused multiply-add; validate numerical changes before production use. |
| `extra_cuda_cflags=[...]` | Pass extra `nvcc` flags such as line information, register limits, or PTX assembler diagnostics. |
| `extra_ldflags=[...]` | Pass additional linker flags. |
| `extra_include_paths=[...]` | Add user include directories. |
| `compute_capability=...` | Override automatic GPU architecture detection. |
| `allow_cuda_graph=True` | Mark targets command-buffer compatible; disable it for host-side replay effects such as dynamic allocation or callbacks. |
| `build_directory=...` | Override the build directory for one load call. |
| `verbose=True` | Print the compiler command and detailed output. |
| `force_rebuild=True` | Ignore the cache and compile again. |
| `auto_register=False` | Compile without automatically registering FFI targets. |
| `target_prefix=...` | Override the prefix used by automatically registered targets. |
| `set_cache_dir(path)` / `get_cache_dir()` | Change or inspect the process cache directory. |
| `clear_cache(name=None)` | Remove all cache entries or only entries for one module; it returns the number removed. |

The default cache directory is `~/.cache/brainevent/`; `BRAINEVENT_CACHE_DIR` changes it globally. Cache keys include source, compiler flags, GPU architecture, compiler version, and BrainEvent version. Optimization and fast-math changes recompile; `allow_cuda_graph` changes registration only.

Repeated loading of the same cached shared library is safe. Registering a different module under an existing target name raises `KernelRegistrationError`.

## Inspect and diagnose GPU kernels

| API | Description |
|---|---|
| `CompiledModule.path` | Return the loaded shared-library path. |
| `CompiledModule.function_names` | Return the compiled user function names. |
| `CompiledModule.get_handler(name)` | Return the `ctypes` FFI handler without the generated `be_` prefix. |
| `register_ffi_target(target_name, module, func_name, *, platform="CUDA")` | Register a compiled CUDA handler manually. |
| `list_registered_targets()` | Return registered FFI target names in sorted order. |
| `print_diagnostics()` | Print the BrainEvent compilation environment. |

| Exception | Meaning |
|---|---|
| `KernelToolchainError` / `CUDANotInstalledError` | The compiler, CUDA installation, driver, or compatible backend is unavailable. |
| `CompilationError` / `KernelCompilationError` | The toolchain exists but source, types, shapes, or backend constraints prevented compilation. |
| `KernelRegistrationError` | An FFI target conflicts with a live registration or could not be registered. |
| `KernelNotAvailableError` | Warp, Pallas, Triton, or another requested GPU backend is unavailable. |
| `KernelFallbackExhaustedError` | No registered GPU backend can handle the operation. |
| `KernelExecutionError` | Compilation succeeded but runtime execution failed, such as an invalid memory access or device assertion. |

For compilation failures, rerun raw CUDA with `verbose=True` and `optimization_level=0`. For runtime failures, add `BE_CHECK_KERNEL_LAUNCH()`, validate buffer shapes and dtypes, and compare against a CPU or JAX reference.

## Verify a GPU operator

1. Run the kernel on the smallest meaningful input.
2. Compare every output with a pure JAX reference, including shape and dtype.
3. Run the already-created wrapper inside `jax.jit`.
4. Test `vmap`, JVP, and reverse-mode gradients only when the operation promises them.
5. Exercise every registered GPU backend with the same fixtures.
6. Benchmark after warmup and explicit output blocking.
7. Enable fast math or CUDA graphs only after correctness and side-effect constraints pass.

Do not write a custom GPU kernel when a built-in BrainEvent operation already provides the required semantics and maintained backend coverage.

## Official sources

- [Installation](https://brainx.chaobrain.com/brainevent/getting-started/installation.html)
- [Custom GPU operators with Numba CUDA](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/02_numba_cuda.html)
- [Custom GPU operators with Warp](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/03_warp.html)
- [Custom CUDA GPU kernels](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/05_cuda.html)
- [Custom-kernel framework API](https://brainx.chaobrain.com/brainevent/reference/apis/operator.html)
- [`arg_spec` system](https://brainx.chaobrain.com/brainevent/reference/kernels/arg-spec.html)
- [C++ API](https://brainx.chaobrain.com/brainevent/reference/kernels/cpp-api.html)
- [Compiler options](https://brainx.chaobrain.com/brainevent/reference/kernels/compiler-options.html)
- [Caching](https://brainx.chaobrain.com/brainevent/reference/kernels/caching.html)
