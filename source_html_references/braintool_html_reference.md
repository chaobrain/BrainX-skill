# BrainTools Central Documentation Sitemap

Purpose: choose the official BrainTools API page that directly supports cognitive-task construction, connectivity, initialization, simulation input, integration, training, analysis, persistence, or visualization.

## Table of contents

- [Root](#root)
- [Cognitive Tasks and Connectivity](#cognitive-tasks-and-connectivity)
- [Simulation Setup and Dynamics](#simulation-setup-and-dynamics)
- [Spike Data and Encoding](#spike-data-and-encoding)
- [Training, Fitting, and Analysis](#training-fitting-and-analysis)
- [Persistence, Trees, and Visualization](#persistence-trees-and-visualization)

## Root

- [BrainTools documentation](https://brainx.chaobrain.com/braintools/) — Main landing page for the BrainTools utility modules.
- [`braintools` Module](https://brainx.chaobrain.com/braintools/apis/braintools.html) — Main public API hub for the BrainTools utility modules.
- [Release notes](https://brainx.chaobrain.com/braintools/apis/changelog.html) — Check the release that introduced an API or changed its behavior.
- [Migration guide](https://brainx.chaobrain.com/braintools/migration.html) — Find replacements for deprecated imports and arguments.

## Cognitive Tasks and Connectivity

- [`braintools.cogtask` Module](https://brainx.chaobrain.com/braintools/apis/cogtask.html) — Constructs composable cognitive tasks from phases, control flow, feature encodings, output labels, and pre-built task families.
- [`braintools.conn` Module](https://brainx.chaobrain.com/braintools/apis/conn.html) — Builds point-neuron and multicompartment connectivity with basic, spatial, composable, and unit-aware patterns.

## Simulation Setup and Dynamics

- [`braintools.init` Module](https://brainx.chaobrain.com/braintools/apis/init.html) — Initializes parameters with statistical distributions, variance scaling, orthogonal methods, spatial profiles, and composable strategies.
- [`braintools.input` Module](https://brainx.chaobrain.com/braintools/apis/input.html) — Generates composable or functional input currents, including basic signals, pulses, waveforms, and stochastic processes.
- [`braintools.quad` Module](https://brainx.chaobrain.com/braintools/apis/quad.html) — Applies JAX-friendly one-step integrators for ODEs, SDEs, DDEs, and implicit-explicit systems.

## Spike Data and Encoding

- [`braintools.spike` Module](https://brainx.chaobrain.com/braintools/apis/spike.html) — Owns `SpikeTrains`, spike encoders (`LatencyEncoder`, `RateEncoder`, `PoissonEncoder`, …), and bitwise spike operations. Prefer this over deprecated top-level encoder aliases.

## Training, Fitting, and Analysis

- [`braintools.metric` Module](https://brainx.chaobrain.com/braintools/apis/metric.html) — Provides classification, regression, structured and ranking losses, plus pairwise vector comparisons.
- [`braintools.optim` Module](https://brainx.chaobrain.com/braintools/apis/optim.html) — Provides gradient-based and black-box optimizers, learning-rate schedulers, clipping, and weight-decay utilities.
- [`braintools.surrogate` Module](https://brainx.chaobrain.com/braintools/apis/surrogate.html) — Supplies functional and class-based surrogate gradients for differentiating through discrete spike events.
- [`braintools.trainer` Module](https://brainx.chaobrain.com/braintools/apis/trainer.html) — Orchestrates JAX and BrainState training, validation, testing, prediction, callbacks, logging, data loading, distribution, and checkpointing.
- [`braintools.fit` Module](https://brainx.chaobrain.com/braintools/apis/fit.html) — Fits brain models to data.
- [`braintools.analysis` Module](https://brainx.chaobrain.com/braintools/apis/analysis.html) — Owns spike-train, spectral, field, population, and statistical analysis. Use it in BrainTools 0.4 in place of deprecated analysis aliases under `braintools.metric`.

## Persistence, Trees, and Visualization

- [`braintools.file` Module](https://brainx.chaobrain.com/braintools/apis/file.html) — Loads and saves MATLAB data and serializes BrainState or BrainUnit-aware model checkpoints with MsgPack.
- [`braintools.tree` Module](https://brainx.chaobrain.com/braintools/apis/tree.html) — Manipulates JAX PyTrees with arithmetic, reshape, and conversion helpers used across the toolkit.
- [`braintools.visualize` Module](https://brainx.chaobrain.com/braintools/apis/visualize.html) — Visualizes neural activity, connectivity, statistics, trajectories, interactive dashboards, animations, and publication-ready figures.
