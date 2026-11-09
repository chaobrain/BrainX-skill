# BrainEvent Central Documentation Sitemap

Purpose: choose direct BrainEvent tutorial, how-to, API-category, and custom-kernel HTML pages without routing through broad index pages.

## Table of contents

- [Root](#root)
- [Getting Started](#getting-started)
- [Tutorials — Data structures & operators](#tutorials--data-structures--operators)
- [Tutorials — Custom operators](#tutorials--custom-operators)
- [How-to — Working with data structures](#how-to--working-with-data-structures)
- [How-to — Building & extending](#how-to--building--extending)
- [Explanation](#explanation)
- [Reference — Python API category subpages](#reference--python-api-category-subpages)
- [Reference — Custom kernel reference pages](#reference--custom-kernel-reference-pages)

## Root

- [BrainEvent documentation](https://brainx.chaobrain.com/brainevent/) — Main landing page for event-driven computation with `BinaryArray`, sparse and fixed connectivity, and custom kernels.
- [Release notes](https://brainx.chaobrain.com/brainevent/reference/changelog.html) — Check operator renames, removed representations, and backend changes for the installed BrainEvent version.

## Getting Started

- [Installation](https://brainx.chaobrain.com/brainevent/getting-started/installation.html) — Installation page for CPU/GPU/TPU installs, JAX device verification, and GPU custom-kernel compiler dependencies.
- [Quickstart](https://brainx.chaobrain.com/brainevent/getting-started/quickstart.html) — Minimal first workflow: wrap spikes in `BinaryArray`, multiply against dense/sparse/JIT/fixed connectivity, and run inside JAX transforms.

## Tutorials — Data structures & operators

- [Binary events](https://brainx.chaobrain.com/brainevent/tutorials/events/binary-events.html) — Introduces `BinaryArray`, binary spike/event representation, event-driven matrix multiplication, and simple feedforward/event examples.
- [Sparse matrices](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/02_sparse_matrices.html) — Explains CSR/CSC sparse matrices, COO triplets, sparse connectivity construction, and use with `BinaryArray`.
- [JIT connectivity](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/03_jit_connectivity.html) — Teaches JIT-generated random connectivity: storing a generator instead of materializing huge connection matrices.
- [Fixed connections](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/04_fixed_connections.html) — Covers fixed fan-in/fan-out connectivity structures such as `FixedPreNumConn` and `FixedPostNumConn`.
- [Synaptic plasticity](https://brainx.chaobrain.com/brainevent/tutorials/events/synaptic-plasticity.html) — Covers event-driven synaptic plasticity, STDP-style updates, CSR/dense plasticity operators, and adaptive network examples.

## Tutorials — Custom operators

- [Numba CPU operators](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html) — Custom CPU operators with Numba; wraps `@numba.njit` kernels into JAX-compatible BrainEvent operations.
- [Numba CUDA operators](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/02_numba_cuda.html) — Custom GPU operators with Numba CUDA; covers `numba_cuda_kernel`, `numba_cuda_callable`, launch configs, and GPU performance tips.
- [Warp operators](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/03_warp.html) — Custom GPU operators with NVIDIA Warp; integrates Warp kernels into BrainEvent/JAX custom-kernel workflow.
- [C++ operators](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/04_cpp.html) — Plain C++ CPU kernels compiled and registered as JAX FFI targets through BrainEvent.
- [CUDA operators](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/05_cuda.html) — Raw CUDA GPU kernels plus C++ launchers compiled by BrainEvent and called from JAX.

## How-to — Working with data structures

- [Choosing a sparse format](https://brainx.chaobrain.com/brainevent/how-to/data-structures/choosing-a-sparse-format.html) — Decision guide for Dense vs CSR/CSC vs JITC vs fixed fan-in/fan-out connectivity.
- [JIT connectivity for large networks](https://brainx.chaobrain.com/brainevent/how-to/data-structures/jit-connectivity-large-networks.html) — Practical guide for using JIT connectivity to scale large random networks without storing full matrices.
- [Synaptic plasticity how-to](https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html) — Task guide for choosing pre/post-synaptic plasticity operators and CSR vs dense plasticity updates.
- [Unit-aware computation](https://brainx.chaobrain.com/brainevent/how-to/data-structures/unit-aware-computation.html) — Shows how BrainEvent works with BrainUnit physical units for synaptic weights/currents.

## How-to — Building & extending

- [Build a COBA/CUBA network](https://brainx.chaobrain.com/brainevent/how-to/building-extending/build-coba-cuba-network.html) — Core pattern for building event-driven excitatory/inhibitory networks, pointing to CUBA/COBA balanced-network examples.
- [Compile raw CUDA/C++](https://brainx.chaobrain.com/brainevent/how-to/building-extending/compile-raw-cuda-cpp.html) — Recipe for compiling raw CUDA/C++ kernels and calling them from JAX with BrainEvent.

## Explanation

Conceptual pages for choosing formats and understanding the event-driven / custom-kernel design. Prefer these when the agent needs the *why* before a how-to or tutorial.

- [What is event-driven computation?](https://brainx.chaobrain.com/brainevent/explanation/event-driven-computation.html) — Explains why discrete spike events are the primary compute unit and how event-driven ops differ from dense matrix work.
- [Sparse format trade-offs](https://brainx.chaobrain.com/brainevent/explanation/sparse-formats.html) — Compares Dense, CSR/CSC, JITC, and fixed fan-in/fan-out trade-offs for connectivity choice.
- [Just-in-time connectivity](https://brainx.chaobrain.com/brainevent/explanation/jit-connectivity.html) — Explains just-in-time connectivity: multiplying spikes by random connectivity too large to store.
- [Custom-kernel architecture](https://brainx.chaobrain.com/brainevent/explanation/custom-kernel-architecture.html) — Explains how plain C++/CUDA kernels are compiled and called from JAX through BrainEvent.
- [FAQ](https://brainx.chaobrain.com/brainevent/explanation/faq.html) — FAQ for GPU requirements, format choice, and common BrainEvent setup questions.

## Reference — Python API category subpages

- [Events](https://brainx.chaobrain.com/brainevent/reference/apis/events.html) — Event array types: `EventRepresentation` and `BinaryArray`.
- [Sparse data](https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html) — Sparse/connectivity data structures: CSR, CSC, JITC variants, and fixed-connection structures.
- [Operations](https://brainx.chaobrain.com/brainevent/reference/apis/operations.html) — Matrix operations: CSR, dense, JITC, fixed-connectivity, and plasticity operators.
- [Operator / custom kernels](https://brainx.chaobrain.com/brainevent/reference/apis/operator.html) — Custom-kernel framework APIs: `XLACustomKernel`, Numba wrappers, CUDA/C++ loaders, cache utilities, diagnostics.
- [Errors](https://brainx.chaobrain.com/brainevent/reference/apis/errors.html) — Error classes for math errors, kernel availability, compilation, fallback, execution, and CUDA install issues.
- [Utilities](https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html) — Utility functions for sparse index conversion, LFSR RNGs, benchmarking, JVP/batching helpers, and type conversion helpers.
- [Config](https://brainx.chaobrain.com/brainevent/reference/apis/config.html) — Runtime configuration API for Numba parallelism, LFSR algorithm choice, and backend selection.

## Reference — Custom kernel reference pages

These are also under the Reference navigation, separate from `/reference/apis/`.

- [Custom kernel index](https://brainx.chaobrain.com/brainevent/reference/kernels/index.html) — Landing page for raw C++/CUDA custom-kernel toolchain reference.
- [Argument specification](https://brainx.chaobrain.com/brainevent/reference/kernels/arg-spec.html) — Explains the `arg_spec` token system: `"arg"`, `"ret"`, `"stream"`, and scalar attributes.
- [C++ API](https://brainx.chaobrain.com/brainevent/reference/kernels/cpp-api.html) — C++ API reference for `BE::Tensor`, dtype handling, checking macros, and dispatch macros.
- [Compiler options](https://brainx.chaobrain.com/brainevent/reference/kernels/compiler-options.html) — Compiler options for raw CUDA/C++ kernels, including optimization level, fast math, extra flags, and CUDA graph support.
- [Caching](https://brainx.chaobrain.com/brainevent/reference/kernels/caching.html) — Explains compiled-kernel caching, cache keys, cache directory, force rebuild, and safe re-import behavior.
