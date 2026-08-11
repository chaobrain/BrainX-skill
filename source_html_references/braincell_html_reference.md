# BrainCell Central Documentation Sitemap

Purpose: choose the official BrainCell HTML page that directly supports a modeling concept, workflow, example, solver decision, or file-format task.

## Root

- [BrainCell Documentation](https://brainx.chaobrain.com/braincell/) — Main BrainCell landing page for biophysically detailed single- and multicompartment neuron modeling.

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

- [BrainCell Module](https://brainx.chaobrain.com/braincell/apis/braincell.html) — Generated public API hub for BrainCell classes and functions; use only after choosing the relevant concept or tutorial page.
