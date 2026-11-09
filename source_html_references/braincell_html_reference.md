# BrainCell Central Documentation Sitemap

Purpose: choose the official BrainCell HTML page that directly supports a modeling concept, workflow, example, solver decision, or file-format task.

## Table of contents

- [Root](#root)
- [Get Started](#get-started)
- [Concepts](#concepts)
- [Modeling Tutorials](#modeling-tutorials)
- [Numerical Integration](#numerical-integration)
- [Examples](#examples)
- [File Formats and IO](#file-formats-and-io)
- [API Reference](#api-reference)

## Root

- [BrainCell Documentation](https://brainx.chaobrain.com/braincell/) — Main BrainCell landing page for biophysically detailed single- and multicompartment neuron modeling.
- [Release notes](https://brainx.chaobrain.com/braincell/apis/changelog.html) — Check when a cellular model, solver, or public API changed before applying a live example to a pinned release.

## Get Started

- [Installation](https://brainx.chaobrain.com/braincell/getting_started/installation.html) — Installs BrainCell on JAX/brainstate for CPU, GPU, or TPU and states the Python version requirement.
- [Overview & Roadmap](https://brainx.chaobrain.com/braincell/getting_started/overview.html) — Maps what BrainCell does, how the pieces fit, and where to go next by task.
- [First Steps](https://brainx.chaobrain.com/braincell/getting_started/first_steps.html) — Runs a first complete Hodgkin–Huxley simulation with constant-current stimulation and spike plotting.

## Concepts

- [Architecture](https://brainx.chaobrain.com/braincell/concepts/architecture.html) — Explains declaration layers, discretization, runtime graphs, and JAX execution flow.
- [Units](https://brainx.chaobrain.com/braincell/concepts/units.html) — Explains physical units, conversions, dimensional checks, and quantity safety rules.
- [Cells](https://brainx.chaobrain.com/braincell/concepts/cells.html) — Compares point neurons and morphologically detailed cells sharing Hodgkin-Huxley machinery.
- [Morphology](https://brainx.chaobrain.com/braincell/concepts/morphology.html) — Defines neuronal branch geometry, topology, typed cables, and import sources.
- [Mechanisms](https://brainx.chaobrain.com/braincell/concepts/mechanisms.html) — Defines density and point mechanisms through declarative `paint` and `place` semantics.
- [Ions & Channels](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html) — Explains ion species, channels, reversal potentials, and current coupling logic.
- [Regions & Locsets](https://brainx.chaobrain.com/braincell/concepts/regions_locsets.html) — Teaches region and locset selectors for precise spatial targeting.
- [Discretization](https://brainx.chaobrain.com/braincell/concepts/discretization.html) — Explains control volumes, CV policies, and multicompartment discretization tradeoffs.
- [Integration](https://brainx.chaobrain.com/braincell/concepts/integration.html) — Explains differential equations, solver choices, stiffness, and differentiable integration.

## Modeling Tutorials

- [Tutorial Overview](https://brainx.chaobrain.com/braincell/tutorials/overview.html) — Maps the path from point Hodgkin-Huxley cells to multicompartment modeling.
- [Single Compartment Neurons](https://brainx.chaobrain.com/braincell/tutorials/single_compartment.html) — Builds point-neuron models from `HHTypedNeuron` foundations.
- [Channels](https://brainx.chaobrain.com/braincell/tutorials/channel.html) — Teaches existing, custom, Hodgkin-Huxley, and Markov channel patterns.
- [Ions](https://brainx.chaobrain.com/braincell/tutorials/ion.html) — Explains ion species, `MixIons`, Nernst templates, and calcium dynamics.
- [Morphology in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/morphology.html) — Builds morphologies from branches, points, topology, and geometry rules.
- [Region and Locset Filters](https://brainx.chaobrain.com/braincell/tutorials/filter.html) — Applies spatial filters for targeted mechanism placement.
- [Mechanisms in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/mech.html) — Covers density and point mechanisms, clamps, probes, `paint`, and `place`.
- [Cell in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/cell.html) — Turns morphologies into cells with CV policies and mechanism placement.
- [Point Tree Visualization](https://brainx.chaobrain.com/braincell/tutorials/vis.html) — Visualizes node trees, control volumes, and branch topology.
- [Single-cell Frontend](https://brainx.chaobrain.com/braincell/tutorials/single_cell_frontend.html) — Summarizes morphology ingestion, geometry/topology views, declarative regions, mechanisms, and model datasheets for the single-cell frontend.

## Numerical Integration

> Operational solver path. Distinct from the conceptual [Integration](https://brainx.chaobrain.com/braincell/concepts/integration.html) page under Concepts.

- [Integration Overview](https://brainx.chaobrain.com/braincell/integration/overview.html) — Explains how BrainCell turns channels, point neurons, and cables into stepped differential-equation systems.
- [Defining Differential Equations](https://brainx.chaobrain.com/braincell/integration/diffeq.html) — Expresses ODE systems with `DiffEqState` and `DiffEqModule`.
- [Choosing and Using Solvers](https://brainx.chaobrain.com/braincell/integration/solvers.html) — Discovers, selects, and applies `braincell.quad` integrators for stiff biophysical dynamics.
- [Advanced Integration](https://brainx.chaobrain.com/braincell/integration/advanced.html) — Covers advanced define-and-step topics beyond the basic solver workflow.

## Examples

- [Your First Hodgkin-Huxley Neuron](https://brainx.chaobrain.com/braincell/examples/hh_neuron_basics.html) — Builds, simulates, and plots one Hodgkin-Huxley neuron.
- [Calcium Channel Gating](https://brainx.chaobrain.com/braincell/examples/calcium_channel_gating.html) — Compares low- and high-threshold calcium-channel gating.
- [Frequency-Current Curve](https://brainx.chaobrain.com/braincell/examples/fi_curve.html) — Sweeps input current to measure firing-rate excitability curves.
- [Channel Ablation](https://brainx.chaobrain.com/braincell/examples/channel_ablation.html) — Removes potassium current to expose its repolarization role.
- [Spike-Frequency Adaptation](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html) — Demonstrates calcium-activated potassium adaptation currents.
- [T-Current Rebound](https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html) — Demonstrates post-inhibitory rebound bursting from T-type calcium current.
- [Integration Methods](https://brainx.chaobrain.com/braincell/examples/integration_methods.html) — Compares integration methods for Hodgkin-Huxley membrane-potential traces.
- [Thalamic Neurons](https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html) — Builds four thalamic neuron types and compares electrophysiological responses.
- [EI Network](https://brainx.chaobrain.com/braincell/examples/ei_network.html) — Builds an excitatory-inhibitory Hodgkin-Huxley network and analyzes spikes.

## File Formats and IO

- [IO Overview](https://brainx.chaobrain.com/braincell/file_formats/overview.html) — Surveys morphology loading, validation reports, reader options, and checkpoints.
- [SWC](https://brainx.chaobrain.com/braincell/file_formats/swc.html) — Loads SWC morphology files with structure identifiers and validation options.
- [Neurolucida ASC](https://brainx.chaobrain.com/braincell/file_formats/asc.html) — Loads Neurolucida ASC files with metadata, spines, and validation reports.
- [NeuroML2](https://brainx.chaobrain.com/braincell/file_formats/neuroml2.html) — Imports NeuroML2 morphologies while declaring mechanisms separately.
- [NeuroMorpho.Org](https://brainx.chaobrain.com/braincell/file_formats/neuromorpho.html) — Searches, downloads, caches, and parses reconstructed neurons.
- [Checkpointing](https://brainx.chaobrain.com/braincell/file_formats/checkpointing.html) — Saves and reloads processed morphologies and branches.

## API Reference

- [BrainCell Module](https://brainx.chaobrain.com/braincell/apis/braincell.html) — Generated public API hub for core BrainCell classes and functions; use only after choosing the relevant concept or tutorial page.
- [`braincell.morph` Module](https://brainx.chaobrain.com/braincell/apis/morphology.html) — Morphology, branch, and typed-section APIs.
- [`braincell.ion` Module](https://brainx.chaobrain.com/braincell/apis/braincell.ion.html) — Ion-species APIs.
- [`braincell.channel` Module](https://brainx.chaobrain.com/braincell/apis/braincell.channel.html) — Channel-model APIs.
- [`braincell.synapse` Module](https://brainx.chaobrain.com/braincell/apis/braincell.synapse.html) — Synapse APIs for detailed cells.
- [`braincell.mech` Module](https://brainx.chaobrain.com/braincell/apis/braincell.mech.html) — Declarative mechanism, clamp, and probe APIs.
- [`braincell.filter` Module](https://brainx.chaobrain.com/braincell/apis/filter.html) — Region and locset filter APIs.
- [`braincell.io` Module](https://brainx.chaobrain.com/braincell/apis/io.html) — Morphology IO and checkpoint APIs.
- [`braincell.quad` Module](https://brainx.chaobrain.com/braincell/apis/integration.html) — Numerical integrator / solver registry APIs.
- [`braincell.vis` Module](https://brainx.chaobrain.com/braincell/apis/vis.html) — Morphology and tree visualization APIs.
