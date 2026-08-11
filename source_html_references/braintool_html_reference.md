# BrainTools Central API Documentation Sitemap

Purpose: choose the official BrainTools API page that directly supports cognitive-task construction, connectivity, initialization, simulation input, integration, training, analysis, persistence, or visualization.

## Main Namespace

- [`braintools` Module](https://brainx.chaobrain.com/braintools/apis/braintools.html) - Main public API hub for the BrainTools utility modules.

## Cognitive Tasks and Connectivity

- [`braintools.cogtask` Module](https://brainx.chaobrain.com/braintools/apis/cogtask.html) - Constructs composable cognitive tasks from phases, control flow, feature encodings, output labels, and pre-built task families.
- [`braintools.conn` Module](https://brainx.chaobrain.com/braintools/apis/conn.html) - Builds point-neuron and multicompartment connectivity with basic, spatial, composable, and unit-aware patterns.

## Simulation Setup and Dynamics

- [`braintools.init` Module](https://brainx.chaobrain.com/braintools/apis/init.html) - Initializes parameters with statistical distributions, variance scaling, orthogonal methods, spatial profiles, and composable strategies.
- [`braintools.input` Module](https://brainx.chaobrain.com/braintools/apis/input.html) - Generates composable or functional input currents, including basic signals, pulses, waveforms, and stochastic processes.
- [`braintools.quad` Module](https://brainx.chaobrain.com/braintools/apis/quad.html) - Applies JAX-friendly one-step integrators for ODEs, SDEs, DDEs, and implicit-explicit systems.

## Training and Analysis

- [`braintools.metric` Module](https://brainx.chaobrain.com/braintools/apis/metric.html) - Provides machine-learning losses and neuroscience metrics for spike trains, field potentials, correlations, and pairwise comparisons.
- [`braintools.optim` Module](https://brainx.chaobrain.com/braintools/apis/optim.html) - Provides gradient-based and black-box optimizers, learning-rate schedulers, clipping, and weight-decay utilities.
- [`braintools.surrogate` Module](https://brainx.chaobrain.com/braintools/apis/surrogate.html) - Supplies functional and class-based surrogate gradients for differentiating through discrete spike events.
- [`braintools.trainer` Module](https://brainx.chaobrain.com/braintools/apis/trainer.html) - Orchestrates JAX and BrainState training, validation, testing, prediction, callbacks, logging, data loading, distribution, and checkpointing.

## Persistence and Visualization

- [`braintools.file` Module](https://brainx.chaobrain.com/braintools/apis/file.html) - Loads and saves MATLAB data and serializes BrainState or BrainUnit-aware model checkpoints with MsgPack.
- [`braintools.visualize` Module](https://brainx.chaobrain.com/braintools/apis/visualize.html) - Visualizes neural activity, connectivity, statistics, trajectories, interactive dashboards, animations, and publication-ready figures.
