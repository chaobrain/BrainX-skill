# BrainTrace Central Documentation Sitemap

Purpose: choose the official BrainTrace HTML page that directly supports an online-learning concept, model-building workflow, algorithm decision, compiler diagnostic, batching strategy, example, or API question.

## Supported release boundary

This repository targets BrainX `v2026.7.9`, which bundles BrainTrace `0.2.4`. Use the tagged `v0.2.4` sources below for exact APIs and executable composition. The live documentation routes later in this file describe BrainTrace `0.2.5` and must not be used to introduce `etrace_grad`, `etrace_evolve`, `SequenceDriverMixin`, `ETraceVmap`, `ETraceConfig`, or other post-`0.2.4` APIs into the supported skills.

- [BrainTrace `v0.2.4` public API](https://github.com/chaobrain/braintrace/blob/v0.2.4/braintrace/__init__.py) - Public algorithms, compiler types, ETP operators, and input wrappers available in the bundled release.
- [BrainTrace `v0.2.4` one-call compiler](https://github.com/chaobrain/braintrace/blob/v0.2.4/braintrace/_compile.py) - Exact `compile(...)` contract, native batching, compile-owned vmap, report behavior, and options.
- [BrainTrace `v0.2.4` pp-prop examples](https://github.com/chaobrain/braintrace/tree/v0.2.4/examples/pp_prop) - SNN composition, working-memory and delayed-match tasks, scan gradient accumulation, and batching paths.
- [BrainTrace `v0.2.4` D-RTRL examples](https://github.com/chaobrain/braintrace/tree/v0.2.4/examples/drtrl) - Parameter-dimensional online gradients, mapped and native batching, and VJP modes.
- [BrainTrace `v0.2.4` algorithms API source](https://github.com/chaobrain/braintrace/blob/v0.2.4/docs/apis/algorithms.rst) - Estimator classes and guarantees in the bundled release.
- [BrainTrace `v0.2.4` compiler API source](https://github.com/chaobrain/braintrace/blob/v0.2.4/docs/apis/compiler.rst) - Compilation reports, structured diagnostics, graph types, and executor contracts.
- [BrainTrace release notes](https://brainx.chaobrain.com/braintrace/changelog.html) - Use only to identify release boundaries; sequence drivers begin in `0.2.5`.

## Root

- [BrainTrace Documentation](https://brainx.chaobrain.com/braintrace/) — Main BrainTrace landing page for eligibility-trace online learning in stateful recurrent and spiking neural networks.

## Get Started

- [Installation](https://brainx.chaobrain.com/braintrace/quickstart/installation.html) — Installs BrainTrace for CPU, CUDA, or TPU and verifies the package, JAX backend, and available devices.
- [Quickstart](https://brainx.chaobrain.com/braintrace/quickstart/quickstart.html) — Builds and trains a small `braintrace.nn.MiniGRU` with `braintrace.compile()`, sequence-level eligibility-trace gradients, optimizer updates, and state reset.
- [Core Concepts](https://brainx.chaobrain.com/braintrace/quickstart/concepts.html) — Explains online learning, eligibility traces, operation-based parameter participation, the compiler-executor-algorithm architecture, and the main algorithm families.

## Online Training Tutorials

- [Online Training](https://brainx.chaobrain.com/braintrace/tutorials/online_training.html) — Routes between the RNN and SNN training workflows and highlights their shared initialization, compilation, sequence-driving, reset, and validation steps.
- [RNN Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/rnn_online_learning.html) — Trains a GRU on the copying task with D-RTRL, explicit batched state, `etrace_evolve()`, and `etrace_grad()`, then compares the demonstrated workflow with BPTT.
- [SNN Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/snn_online_learning.html) — Builds a recurrent LIF network from ETP-aware layers and trains it with input/output-factorized pp-prop eligibility traces.

## Algorithm Tutorials

- [Algorithm Tutorials](https://brainx.chaobrain.com/braintrace/tutorials/algorithm_tutorials.html) — Introduces the algorithm-learning path and compares parameter-dimensional and input/output-factorized trace estimators on a shared task.
- [D-RTRL: Diagonal Online Gradient Learning](https://brainx.chaobrain.com/braintrace/tutorials/drtrl.html) — Develops the D-RTRL estimator, sequence workflow, trace reset behavior, memory scaling, and relationship to BPTT.
- [pp-prop: Input/Output-Factorized Online Gradients](https://brainx.chaobrain.com/braintrace/tutorials/pp_prop.html) — Explains pp-prop factorization, `decay_or_rank`, its sequence workflow, and its memory-versus-approximation tradeoff.

## Model-Building Tutorials

- [Building Neural Networks for Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/foundations.html) — Organizes the dependency path from ETP operators through ETP-aware layers to recurrent hidden states.
- [Operators for Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/five_primitive_functions.html) — Introduces the core ETP operator families and their unit-aware and JAX-transformation contracts.
- [Neural Network Layers for Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/neural_network_layers.html) — Maps operation-based parameter selection into reusable ETP-aware linear, recurrent, convolutional, sparse, LoRA, and readout layers.
- [Hidden States for Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/hidden_states.html) — Covers hidden-state containers, compiler discovery and grouping, initialization, reset, batching, and relation inspection.

## Compiler and Runtime Tutorials

- [Compiler & Runtime](https://brainx.chaobrain.com/braintrace/tutorials/compiler_runtime.html) — Routes the workflow for compilation, hidden-group inspection, traced and excluded weights, relation diagnostics, and structural validation.
- [Graph Compilation](https://brainx.chaobrain.com/braintrace/tutorials/graph_compilation.html) — Shows the high-level `braintrace.compile()` path, lower-level graph compilation, and construction of the `ETraceGraph` relation graph.
- [Visualization](https://brainx.chaobrain.com/braintrace/tutorials/visualization.html) — Uses learner reports, graphs, and `show_graph()` to inspect hidden groups, traced parameters, exclusions, and diagnostic reasons.

## Advanced Guides

- [Batching Strategies](https://brainx.chaobrain.com/braintrace/advanced/batching.html) — Compares explicit `brainstate.nn.Map`, single-sample, and helper-owned vectorization patterns for independent recurrent state and eligibility traces.
- [Creating Custom ETP Primitives](https://brainx.chaobrain.com/braintrace/advanced/etp_primitives.html) — Defines primitive registration, trainable-input metadata, ETP rule registries, JAX rules, compiler integration, and soundness constraints.
- [Customizing Parameter Transforms for ETP Operators](https://brainx.chaobrain.com/braintrace/advanced/customizing_primitive_transforms.html) — Applies masks, constraints, normalization, LoRA transforms, and bias transforms while preserving gradients with respect to raw parameters.
- [Compiler Internals](https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html) — Walks through module extraction, hidden-group and relation discovery, hidden perturbations, primitive identification, and compilation diagnostics.
- [Developing Custom Algorithms](https://brainx.chaobrain.com/braintrace/advanced/custom_algorithms.html) — Extends the eligibility-trace algorithm hierarchy through trace storage, update, solve, reset, and learner-execution hooks.
- [Limitations & Workarounds](https://brainx.chaobrain.com/braintrace/advanced/limitations.html) — Documents control-flow handling, shape compatibility, relation constraints, compilation reuse, diagnostics, and performance tradeoffs.

## Examples

- [Spiking Neural Network Examples](https://brainx.chaobrain.com/braintrace/examples/snn_examples.html) — Routes runnable LIF, GIF, excitatory-inhibitory, convolutional, and memory-and-speed SNN examples.
- [Rate-Based RNN Examples](https://brainx.chaobrain.com/braintrace/examples/rnn_examples.html) — Routes GRU copying-task and MiniGRU integrator examples comparing online learning with BPTT.
- [pp-prop Examples](https://brainx.chaobrain.com/braintrace/examples/pp_prop_examples.html) — Indexes input/output-factorized examples for neuron models, batching, VJP modes, operators, classification, and decay-or-rank behavior.
- [D-RTRL Examples](https://brainx.chaobrain.com/braintrace/examples/drtrl_examples.html) — Indexes parameter-dimensional examples for batching, VJP modes, LoRA, convolution, classification, language modeling, and fast solving.

## Central API Reference Pages

- [Release Notes](https://brainx.chaobrain.com/braintrace/changelog.html) — Records BrainTrace releases, additions, fixes, and compatibility changes.
- [ETP Operators & Core Types](https://brainx.chaobrain.com/braintrace/apis/concepts.html) — Documents public ETP operators, sequence-data wrappers, eligibility-trace state, gradient helpers, and compiler error types.
- [Compiler, Executor & Diagnostics](https://brainx.chaobrain.com/braintrace/apis/compiler.html) — Documents graph compilation, hidden groups, parameter-operation relations, perturbations, executors, diagnostics, and compilation reports.
- [Online-Learning Algorithms](https://brainx.chaobrain.com/braintrace/apis/algorithms.html) — Documents `braintrace.compile()`, sequence drivers, algorithm configuration, estimator base classes, and the built-in online-learning algorithms.
- [Neural-Network Layers](https://brainx.chaobrain.com/braintrace/apis/nn.html) — Documents ETP-aware linear, embedding, convolutional, recurrent, sparse, LoRA, grouped, and readout layers.
- [Custom ETP Primitives API](https://brainx.chaobrain.com/braintrace/apis/primitives.html) — Documents primitive registration, `ETPPrimitive`, trainable-input metadata, ETP propagation rules, and automatically derived JAX rules.
