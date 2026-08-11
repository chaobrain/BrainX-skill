# BrainState Central Documentation Sitemap

Purpose: choose the official BrainState HTML page that directly supports a state, module, transform, neural-network, dynamics, interoperability, or API task.

Generated class pages are grouped separately from narrative documentation. Start from a narrative page when learning a workflow; open a generated class page when an exact signature, attribute, or method must be checked.

## Root

- [BrainState Documentation](https://brainx.chaobrain.com/brainstate/) — Main landing page for State-based programming, transformations, and neural-network modules.
- [Release Notes](https://brainx.chaobrain.com/brainstate/changelog.html) — Version history and public API changes.

## Getting Started

- [Installation](https://brainx.chaobrain.com/brainstate/getting_started/installation.html) — Installation paths for CPU, CUDA, TPU, and development environments.
- [Quickstart](https://brainx.chaobrain.com/brainstate/getting_started/quickstart.html) — First State, Module, transform, and training workflow.
- [Thinking in BrainState](https://brainx.chaobrain.com/brainstate/getting_started/thinking_in_brainstate.html) — Introduces the State, Module, and Transform mental model.

## Core Tutorials

- [Core Tutorial Index](https://brainx.chaobrain.com/brainstate/tutorials/core/index.html) — Sequential hub for state management, modules, layers, transforms, training, and randomness.
- [State Management](https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html) — Explains State objects, PyTrees, tracing, splitting, and value updates.
- [Module System Protocol](https://brainx.chaobrain.com/brainstate/tutorials/core/02_modules_and_graph.html) — Introduces Modules, graph nodes, state traversal, and composition.
- [Basic Neural Network Layers](https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html) — Builds networks with linear, convolution, pooling, and container layers.
- [Activation Functions and Normalization](https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html) — Covers activation functions, normalization, and feature standardization.
- [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html) — Explains parameter constraints, transformations, and regularization.
- [Transformations, the Essentials](https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html) — Teaches state-aware JIT, gradients, mapping, and control flow.
- [Training and Metrics](https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html) — Builds training loops with optimizers, losses, metrics, and evaluation.
- [Random Number Generation](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html) — Explains reproducible keys, streams, seeding, and stochastic modules.

## Transformation Tutorials

- [Transformation Tutorial Index](https://brainx.chaobrain.com/brainstate/tutorials/transformations/index.html) — Hub for compilation, differentiation, vectorization, parallelism, control flow, debugging, and IR.
- [JIT and Compilation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html) — Teaches stateful JIT and the BrainState-versus-JAX transform boundary.
- [Automatic Differentiation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html) — Covers argument and State gradients, Jacobians, Hessians, and training loops.
- [Vectorization](https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html) — Teaches mapped axes, batched States, and vectorized module execution.
- [Advanced Batching and Parallelism](https://brainx.chaobrain.com/brainstate/tutorials/transformations/04_advanced_batching.html) — Covers nested batching, `pmap`, sharding, and device execution.
- [Control Flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html) — Covers compiled branches, loops, scans, and checkpointed scans.
- [Error Handling and Runtime Checks](https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html) — Covers transformed assertions, runtime checks, and NaN diagnostics.
- [Debugging Transformed Code](https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html) — Uses breakpoints, callbacks, and inspection tools in transformed code.
- [IR Optimization and Code Generation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/08_ir_optimization_and_codegen.html) — Introduces lowering, IR optimization, code generation, and compilation inspection.

## Brain Dynamics Tutorials

- [Brain Dynamics Tutorial Index](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/index.html) — Hub for dynamics, delays, event operators, SNN construction, and SNN training.
- [Dynamics and Integration](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html) — Defines Dynamics modules, update hooks, integration loops, sizes, and delays.
- [Delay Protocol](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html) — Explains delay registration, buffers, interpolation, access, and timing.
- [Event-Driven Operators](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/03_event_driven_operators.html) — Introduces sparse event-driven communication and efficient updates.
- [Building a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html) — Composes populations, synapses, monitors, and simulation loops.
- [Training a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html) — Trains SNNs with surrogate gradients, losses, optimizers, and metrics.

## How-To Guides

- [Save and Load Checkpoints](https://brainx.chaobrain.com/brainstate/how_to/checkpoint_and_restore.html) — Saves, restores, and manages BrainState model checkpoints.
- [Graph and Node System](https://brainx.chaobrain.com/brainstate/how_to/inspect_and_edit_state_graph.html) — Inspects, traverses, splits, merges, edits, and analyzes State graphs.
- [Utility Toolkit](https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html) — Filters, organizes, freezes, copies, and formats State collections.
- [Collective Operations](https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html) — Applies collective operations across modules, nodes, and State collections.
- [Mixin System](https://brainx.chaobrain.com/brainstate/how_to/custom_states_and_mixins.html) — Defines custom State types, parameter descriptors, and reusable mixins.
- [Observe and Intercept State Access with Hooks](https://brainx.chaobrain.com/brainstate/how_to/state_hooks.html) — Registers read, write, restore, and initialization hooks around State access.
- [Constrain and Regularize Parameters](https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html) — Applies parameter transforms and regularization objects to learnable values.
- [Interoperate with Flax and Equinox](https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html) — Converts supported layers and structures between BrainState, Flax, and Equinox.
- [Migrating Concepts from PyTorch to BrainState](https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html) — Maps PyTorch modules, parameters, mutation, training mode, and transforms to BrainState concepts.

## Concepts

- [Why State-Based?](https://brainx.chaobrain.com/brainstate/concepts/why_state_based.html) — Explains why explicit State enables composable transformations over mutable programs.
- [The State Model](https://brainx.chaobrain.com/brainstate/concepts/the_state_model.html) — Defines State identity, value mutation, tracing, and lifecycle semantics.
- [The Parameter Model](https://brainx.chaobrain.com/brainstate/concepts/the_parameter_model.html) — Explains parameter States, constraints, and training semantics.
- [The Graph Model](https://brainx.chaobrain.com/brainstate/concepts/the_graph_model.html) — Explains graph nodes, shared references, splitting, merging, and traversal.
- [Transformation Semantics](https://brainx.chaobrain.com/brainstate/concepts/transformation_semantics.html) — Explains how BrainState transformations discover and thread State effects.
- [Time and Environment](https://brainx.chaobrain.com/brainstate/concepts/time_and_environment.html) — Explains shared environment values such as time step, fitting mode, precision, and platform.
- [The Typing System](https://brainx.chaobrain.com/brainstate/concepts/the_typing_system.html) — Documents BrainState type aliases and typing conventions.

## Examples

- [Deep Learning Examples](https://brainx.chaobrain.com/brainstate/examples/deep_learning/index.html) — Hub for CNN, VAE, and recurrent-network training examples.
- [Image Classification with CNNs](https://brainx.chaobrain.com/brainstate/examples/deep_learning/image_classification.html) — Builds and trains a CNN image classifier.
- [Variational Autoencoder on MNIST](https://brainx.chaobrain.com/brainstate/examples/deep_learning/variational_autoencoder.html) — Builds an MNIST VAE with encoder, decoder, loss, and latent visualization.
- [Training Recurrent Neural Networks](https://brainx.chaobrain.com/brainstate/examples/deep_learning/integrator_rnn.html) — Trains a recurrent network on a temporal integration task.
- [Brain Dynamics Examples](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/index.html) — Hub for complete neuron and spiking-network applications.
- [Hodgkin-Huxley Neuron Model](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/hodgkin_huxley_neuron.html) — Implements and simulates a unit-aware Hodgkin-Huxley Dynamics module.
- [Simulating Spiking Neural Networks](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/snn_simulation.html) — Simulates LIF populations, connections, monitors, and network dynamics.
- [Training Spiking Neural Networks](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/snn_training.html) — Trains a spiking classifier with surrogate gradients.

## Central API Reference Pages

- [Core `brainstate` Module](https://brainx.chaobrain.com/brainstate/apis/brainstate.html) — State classes, collections, tracing, hooks, and State-related errors.
- [`brainstate.graph` Module](https://brainx.chaobrain.com/brainstate/apis/graph.html) — Graph nodes, traversal, split/merge operations, graph definitions, edges, and references.
- [`brainstate.nn` Module](https://brainx.chaobrain.com/brainstate/apis/nn/index.html) — Modules, parameter containers, regularizers, layers, dynamics, delays, and metrics.
- [`brainstate.transform` Module](https://brainx.chaobrain.com/brainstate/apis/transform.html) — Gradients, JIT, mapping, control flow, checks, tracing, and IR tooling.
- [`brainstate.interop` Module](https://brainx.chaobrain.com/brainstate/apis/interop.html) — Flax and Equinox conversion APIs and interoperability errors.
- [`brainstate.random` Module](https://brainx.chaobrain.com/brainstate/apis/random.html) — Random State, key management, sampling, and probability distributions.
- [`brainstate.util` Module](https://brainx.chaobrain.com/brainstate/apis/util.html) — Mapping, filtering, pretty-printing, dataclass, cache, and dictionary utilities.
- [`brainstate.typing` Module](https://brainx.chaobrain.com/brainstate/apis/typing.html) — Public array, shape, filter, key, and PyTree type aliases.
- [`brainstate.mixin` Module](https://brainx.chaobrain.com/brainstate/apis/mixin.html) — Mixins, parameter descriptors, modes, and batching/training markers.
- [`brainstate.environ` Module](https://brainx.chaobrain.com/brainstate/apis/environ.html) — Environment contexts for time, fitting mode, precision, platform, and defaults.
