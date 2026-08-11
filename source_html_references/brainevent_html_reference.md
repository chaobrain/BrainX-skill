# BrainEvent HTML Reference

Purpose: choose direct BrainEvent tutorial, how-to, API-category, and custom-kernel HTML pages without routing through broad index pages.

## Table of Contents

- [Getting Started](#getting-started)
- [Tutorials — Data structures & operators](#tutorials--data-structures--operators)
- [Tutorials — Custom operators](#tutorials--custom-operators)
- [How-to — Working with data structures](#how-to--working-with-data-structures)
- [How-to — Building & extending](#how-to--building--extending)
- [Reference — Python API category subpages](#reference--python-api-category-subpages)
- [Reference — Custom kernel reference pages](#reference--custom-kernel-reference-pages)

## Getting Started

* `https://brainx.chaobrain.com/brainevent/getting-started/installation.html`
  Installation page for CPU/GPU/TPU installs, JAX device verification, and GPU custom-kernel compiler dependencies. ([brainx.chaobrain.com][1])

* `https://brainx.chaobrain.com/brainevent/getting-started/quickstart.html`
  Minimal first workflow: wrap spikes in `BinaryArray`, multiply against dense/sparse/JIT/fixed connectivity, and run inside JAX transforms. ([brainx.chaobrain.com][2])

## Tutorials — Data structures & operators

* `https://brainx.chaobrain.com/brainevent/tutorials/data-structures/01_eventarray_basics.html`
  Introduces `BinaryArray`, binary spike/event representation, event-driven matrix multiplication, and simple feedforward/event examples. ([brainx.chaobrain.com][3])

* `https://brainx.chaobrain.com/brainevent/tutorials/data-structures/02_sparse_matrices.html`
  Explains CSR/CSC sparse matrices, COO triplets, sparse connectivity construction, and use with `BinaryArray`. ([brainx.chaobrain.com][4])

* `https://brainx.chaobrain.com/brainevent/tutorials/data-structures/03_jit_connectivity.html`
  Teaches JIT-generated random connectivity: storing a generator instead of materializing huge connection matrices. ([brainx.chaobrain.com][5])

* `https://brainx.chaobrain.com/brainevent/tutorials/data-structures/04_fixed_connections.html`
  Covers fixed fan-in/fan-out connectivity structures such as `FixedPreNumConn` and `FixedPostNumConn`. ([brainx.chaobrain.com][6])

* `https://brainx.chaobrain.com/brainevent/tutorials/data-structures/05_synaptic_plasticity.html`
  Covers event-driven synaptic plasticity, STDP-style updates, CSR/dense plasticity operators, and adaptive network examples. ([brainx.chaobrain.com][7])

## Tutorials — Custom operators

* `https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html`
  Custom CPU operators with Numba; wraps `@numba.njit` kernels into JAX-compatible BrainEvent operations. ([brainx.chaobrain.com][8])

* `https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/02_numba_cuda.html`
  Custom GPU operators with Numba CUDA; covers `numba_cuda_kernel`, `numba_cuda_callable`, launch configs, and GPU performance tips. ([brainx.chaobrain.com][9])

* `https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/03_warp.html`
  Custom GPU operators with NVIDIA Warp; integrates Warp kernels into BrainEvent/JAX custom-kernel workflow. ([brainx.chaobrain.com][10])

* `https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/04_cpp.html`
  Plain C++ CPU kernels compiled and registered as JAX FFI targets through BrainEvent. ([brainx.chaobrain.com][11])

* `https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/05_cuda.html`
  Raw CUDA GPU kernels plus C++ launchers compiled by BrainEvent and called from JAX. ([brainx.chaobrain.com][12])

## How-to — Working with data structures

* `https://brainx.chaobrain.com/brainevent/how-to/data-structures/choosing-a-sparse-format.html`
  Decision guide for Dense vs CSR/CSC vs JITC vs fixed fan-in/fan-out connectivity. ([brainx.chaobrain.com][13])

* `https://brainx.chaobrain.com/brainevent/how-to/data-structures/jit-connectivity-large-networks.html`
  Practical guide for using JIT connectivity to scale large random networks without storing full matrices. ([brainx.chaobrain.com][14])

* `https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html`
  Task guide for choosing pre/post-synaptic plasticity operators and CSR vs dense plasticity updates. ([brainx.chaobrain.com][15])

* `https://brainx.chaobrain.com/brainevent/how-to/data-structures/unit-aware-computation.html`
  Shows how BrainEvent works with BrainUnit physical units for synaptic weights/currents. ([brainx.chaobrain.com][16])

## How-to — Building & extending

* `https://brainx.chaobrain.com/brainevent/how-to/building-extending/build-coba-cuba-network.html`
  Core pattern for building event-driven excitatory/inhibitory networks, pointing to CUBA/COBA balanced-network examples. ([brainx.chaobrain.com][17])

* `https://brainx.chaobrain.com/brainevent/how-to/building-extending/compile-raw-cuda-cpp.html`
  Recipe for compiling raw CUDA/C++ kernels and calling them from JAX with BrainEvent. ([brainx.chaobrain.com][18])

## Reference — Python API category subpages

* `https://brainx.chaobrain.com/brainevent/reference/apis/events.html`
  Event array types: `EventRepresentation` and `BinaryArray`. ([brainx.chaobrain.com][19])

* `https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html`
  Sparse/connectivity data structures: CSR, CSC, JITC variants, and fixed-connection structures. ([brainx.chaobrain.com][20])

* `https://brainx.chaobrain.com/brainevent/reference/apis/operations.html`
  Matrix operations: CSR, dense, JITC, fixed-connectivity, and plasticity operators. ([brainx.chaobrain.com][21])

* `https://brainx.chaobrain.com/brainevent/reference/apis/operator.html`
  Custom-kernel framework APIs: `XLACustomKernel`, Numba wrappers, CUDA/C++ loaders, cache utilities, diagnostics. ([brainx.chaobrain.com][22])

* `https://brainx.chaobrain.com/brainevent/reference/apis/errors.html`
  Error classes for math errors, kernel availability, compilation, fallback, execution, and CUDA install issues. ([brainx.chaobrain.com][23])

* `https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html`
  Utility functions for sparse index conversion, LFSR RNGs, benchmarking, JVP/batching helpers, and type conversion helpers. ([brainx.chaobrain.com][24])

* `https://brainx.chaobrain.com/brainevent/reference/apis/config.html`
  Runtime configuration API for Numba parallelism, LFSR algorithm choice, and backend selection. ([brainx.chaobrain.com][25])

## Reference — Custom kernel reference pages

These are also under the Reference navigation, separate from `/reference/apis/`.

* `https://brainx.chaobrain.com/brainevent/reference/kernels/index.html`
  Landing page for raw C++/CUDA custom-kernel toolchain reference. ([brainx.chaobrain.com][26])

* `https://brainx.chaobrain.com/brainevent/reference/kernels/arg-spec.html`
  Explains the `arg_spec` token system: `"arg"`, `"ret"`, `"stream"`, and scalar attributes. ([brainx.chaobrain.com][27])

* `https://brainx.chaobrain.com/brainevent/reference/kernels/cpp-api.html`
  C++ API reference for `BE::Tensor`, dtype handling, checking macros, and dispatch macros. ([brainx.chaobrain.com][28])

* `https://brainx.chaobrain.com/brainevent/reference/kernels/compiler-options.html`
  Compiler options for raw CUDA/C++ kernels, including optimization level, fast math, extra flags, and CUDA graph support. ([brainx.chaobrain.com][29])

* `https://brainx.chaobrain.com/brainevent/reference/kernels/caching.html`
  Explains compiled-kernel caching, cache keys, cache directory, force rebuild, and safe re-import behavior. ([brainx.chaobrain.com][30])
