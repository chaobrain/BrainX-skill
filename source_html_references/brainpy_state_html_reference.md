# BrainPy-State Central Documentation Link Inventory
**Total sections:** 7
**Total entries:** 58

## Table of contents

- [Root / overview](#root--overview) — 1 entry
- [Get Started](#get-started) — 4 entries
- [Core Concepts](#core-concepts) — 7 entries
- [BrainPy-style Modeling](#brainpy-style-modeling) — 14 entries
- [NEST-Compatible Models](#nest-compatible-models) — 14 entries
- [API Category Pages](#api-category-pages) — 16 entries
## Get Started

> The beginner onboarding path: installation, first tour, and mental model.

- [Get Started](https://brainx.chaobrain.com/brainpy-state/get-started/index.html) — Landing page for the beginner path.
- [Installation](https://brainx.chaobrain.com/brainpy-state/get-started/installation.html) — Package installation options and setup guidance.
- [5-minute tour](https://brainx.chaobrain.com/brainpy-state/get-started/5-minute-tour.html) — Fast walkthrough of the core API through a compact runnable example.
- [The mental model](https://brainx.chaobrain.com/brainpy-state/get-started/mental-model.html) — Conceptual map for how state, modules, units, dynamics, and projections fit together.

---

## Core Concepts

> The conceptual backbone. These pages explain how BrainPy-State thinks about simulation objects, mutable state, synapses, differentiability, and learning.

- [Core Concepts](https://brainx.chaobrain.com/brainpy-state/concepts/index.html) — Hub page for the core conceptual chapters.
- [The state paradigm](https://brainx.chaobrain.com/brainpy-state/concepts/state-paradigm.html) — Explains explicit `State` as the central design pattern for mutable neural variables.
- [Physical units](https://brainx.chaobrain.com/brainpy-state/concepts/physical-units.html) — Explains unit-aware modeling with physical quantities.
- [Model anatomy](https://brainx.chaobrain.com/brainpy-state/concepts/model-anatomy.html) — Breaks a model into neuron, synapse, communication, output, state, and update pieces.
- [AlignPre / AlignPost — the keystone](https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html) — Explains the projection/state-alignment design that keeps synaptic memory efficient.
- [Differentiability](https://brainx.chaobrain.com/brainpy-state/concepts/differentiability.html) — Explains how spiking models remain compatible with JAX transforms and gradient-based learning.
- [Online learning](https://brainx.chaobrain.com/brainpy-state/concepts/online-learning.html) — Explains learning rules that update during simulation rather than only through offline backpropagation.

---

## BrainPy-style Modeling

> The native BrainPy-State path. This is the most important section if you want to build idiomatic BrainPy-State simulations and trainable SNNs.

- [BrainPy-style Modeling](https://brainx.chaobrain.com/brainpy-state/brainpy-style/index.html) — Hub page for the idiomatic BrainPy-style modeling track.
- [Tutorial 1 · Your first neuron](https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/01-first-neuron.html) — Builds and runs a first point-neuron model.
- [Tutorial 2 · Synapses and projections](https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/02-synapse-and-projection.html) — Introduces synaptic dynamics and projection wiring.
- [Tutorial 3 · An E/I balanced network](https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/03-ei-balanced-network.html) — Builds an excitatory/inhibitory balanced spiking network.
- [Tutorial 4 · Train a spiking network](https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html) — Shows gradient-based training through an SNN.
- [How-to guides](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/index.html) — Hub page for practical BrainPy-style recipes.
- [How to choose a neuron model](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-choose-neuron.html) — Decision guide for choosing IF/LIF/AdEx/HH/etc. based on simulation needs.
- [How to choose between COBA and CUBA synapses](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-coba-cuba-synapses.html) — Explains conductance-based versus current-based synaptic outputs.
- [How to add short-term plasticity](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-short-term-plasticity.html) — Recipe for adding STP/STD mechanisms to synapses.
- [How to add synaptic delays](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html) — Recipe for introducing delayed spike or synaptic communication.
- [How to reproduce a paper: gamma oscillation](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-reproduce-a-paper.html) — Paper-reproduction workflow centered on a gamma-oscillation model.
- [How to use surrogate gradients](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-surrogate-gradients.html) — Training recipe for non-differentiable spike functions using surrogate gradients.
- [How to add a trainable readout](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-readouts.html) — Adds readout layers for decoding spikes into trainable outputs.
- [How to train through long rollouts without exhausting memory](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html) — Memory-management strategy for long simulation/training rollouts.

---

## NEST-Compatible Models

> The compatibility path. This section matters if you want to port NEST/PyNEST-style models or understand how BrainPy-State maps NEST concepts into its own backend.

- [NEST-Compatible Models](https://brainx.chaobrain.com/brainpy-state/nest-style/index.html) — Hub page for the NEST-compatible model family and migration path.
- [One neuron](https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/01-one-neuron.html) — First NEST-style single-neuron simulation.
- [Populations and devices](https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/02-populations.html) — Creates populations and stimulation/recording devices.
- [Connect a network](https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/03-connect-network.html) — Connects NEST-style nodes into a network.
- [Record and analyze](https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/04-record-and-analyze.html) — Records simulation outputs and analyzes spike/trace data.
- [Model directory](https://brainx.chaobrain.com/brainpy-state/nest-style/models.html) — Directory of NEST-compatible neuron, synapse, plasticity, and device models.
- [Connectivity](https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html) — NEST-style connection rules and how they map into BrainPy-State.
- [Devices](https://brainx.chaobrain.com/brainpy-state/nest-style/devices.html) — Stimulators, generators, recorders, and detector devices.
- [Spatially-structured networks](https://brainx.chaobrain.com/brainpy-state/nest-style/spatial.html) — Spatial layers, distance kernels, masks, and spatial connection rules.
- [Porting walkthrough: Brunel, side by side](https://brainx.chaobrain.com/brainpy-state/nest-style/porting-walkthrough.html) — Side-by-side translation of a Brunel-style NEST model into BrainPy-State.
- [Semantic divergences](https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/index.html) — Explains where BrainPy-State intentionally differs from NEST semantics.
- [STDP parity: where state lives and how spikes pair](https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/stdp.html) — Deep dive on STDP state ownership and spike-pairing differences.
- [Validation status](https://brainx.chaobrain.com/brainpy-state/nest-style/validation-status.html) — Per-model validation/parity status against NEST-style behavior.
- [Integration categories](https://brainx.chaobrain.com/brainpy-state/nest-style/integration-categories.html) — Categorizes integration/numerical behavior across NEST-compatible models.

---

## API Category Pages

> High-level API category pages only. Generated per-object API pages are intentionally omitted to keep this file usable.

- [API Reference](https://brainx.chaobrain.com/brainpy-state/apis/index.html) — Top-level API reference hub.
- [Base Classes](https://brainx.chaobrain.com/brainpy-state/apis/base.html) — Base dynamics, neuron, and synapse classes.
- [BrainPy-style Neurons](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-neurons.html) — Native point-neuron model APIs.
- [BrainPy-style Synapses](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synapses.html) — Native synaptic dynamics APIs.
- [BrainPy-style Projections](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-projections.html) — Projection and communication APIs.
- [BrainPy-style Synaptic Outputs](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synouts.html) — Current/conductance output transformation APIs.
- [BrainPy-style Plasticity](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-plasticity.html) — Short-term plasticity and depression APIs.
- [BrainPy-style Readouts](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-readouts.html) — Readout/decoder APIs for spiking activity.
- [BrainPy-style Input Generators](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-inputs.html) — Spike trains, Poisson encoders, and input helpers.
- [NEST Base Classes](https://brainx.chaobrain.com/brainpy-state/apis/nest-base.html) — NEST-compatible base abstractions.
- [NEST-Compatible Neuron Models](https://brainx.chaobrain.com/brainpy-state/apis/nest-neurons.html) — NEST-compatible neuron model APIs.
- [NEST-Compatible Synapse Models](https://brainx.chaobrain.com/brainpy-state/apis/nest-synapses.html) — NEST-compatible synapse model APIs.
- [NEST-Compatible Plasticity Models](https://brainx.chaobrain.com/brainpy-state/apis/nest-plasticity.html) — NEST-compatible STP/STDP/plasticity APIs.
- [NEST-Compatible Devices](https://brainx.chaobrain.com/brainpy-state/apis/nest-devices.html) — NEST-compatible generators, recorders, and detectors.
- [NEST-Compatible Spatial Networks](https://brainx.chaobrain.com/brainpy-state/apis/nest-spatial.html) — Spatial layer, kernel, mask, and plotting APIs.
- [NEST-Compatible Network Builder](https://brainx.chaobrain.com/brainpy-state/apis/nest-network.html) — Network, builder, simulator, projection, and connection-rule APIs.

---

## Example Gallery Pages

> Keep the official gallery index pages, not every external script link. Use these when you need runnable examples after understanding the tutorial path.

- [BrainPy-style example gallery](https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html) — Gallery index for native BrainPy-style runnable examples, including balanced networks, oscillations, reproductions, and SNN training.

## BrainPy-style examples

### Balanced & E/I networks

- [brunel.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/brunel.py)
- [103_COBA_2005.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/103_COBA_2005.py)
- [104_CUBA_2005.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/104_CUBA_2005.py)
- [104_CUBA_2005_version2.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/104_CUBA_2005_version2.py)
- [102_EI_net_1996.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/102_EI_net_1996.py)
- [106_COBA_HH_2007.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/106_COBA_HH_2007.py)

### Oscillations & rhythms

- [107_gamma_oscillation_1996.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/107_gamma_oscillation_1996.py)
- [108_synfire_chains_199.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/108_synfire_chains_199.py)
- [109_fast_global_oscillation.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/109_fast_global_oscillation.py)
- [110_Susin_Destexhe_2021_gamma_oscillation_AI.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/110_Susin_Destexhe_2021_gamma_oscillation_AI.py)
- [111_Susin_Destexhe_2021_gamma_oscillation_CHING.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/111_Susin_Destexhe_2021_gamma_oscillation_CHING.py)
- [112_Susin_Destexhe_2021_gamma_oscillation_ING.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/112_Susin_Destexhe_2021_gamma_oscillation_ING.py)
- [113_Susin_Destexhe_2021_gamma_oscillation_PING.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/113_Susin_Destexhe_2021_gamma_oscillation_PING.py)
- [Susin_Destexhe_2021_gamma_oscillation.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/Susin_Destexhe_2021_gamma_oscillation.py)

### Reproductions & large models

- [203_cortical_model.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/203_cortical_model.py)
- [204_joglekar_2018_propagation.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/204_joglekar_2018_propagation.py)

### Training spiking networks

- [200_surrogate_grad_lif.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/200_surrogate_grad_lif.py)
- [201_surrogate_grad_lif_fashion_mnist.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/201_surrogate_grad_lif_fashion_mnist.py)
- [202_mnist_lif_readout.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/202_mnist_lif_readout.py)
















- [NEST-compatible example gallery](https://brainx.chaobrain.com/brainpy-state/examples/nest-gallery.html) — Gallery index for NEST-compatible runnable examples and PyNEST-style ports.

## NEST-compatible examples

### First steps: single & two neurons

- [one_neuron.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/one_neuron.py)
- [one_neuron_with_noise.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/one_neuron_with_noise.py)
- [twoneurons.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/twoneurons.py)
- [testiaf.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/testiaf.py)
- [if_curve.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/if_curve.py)
- [vinit_example.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/vinit_example.py)
- [balancedneuron.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/balancedneuron.py)
- [izhikevich.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/izhikevich.py)
- [hh_psc_alpha.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/hh_psc_alpha.py)
- [hh_phaseplane.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/hh_phaseplane.py)

### Neuron models

- [aeif_cond_beta_multisynapse.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/aeif_cond_beta_multisynapse.py)
- [brette_gerstner_fig_2c.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brette_gerstner_fig_2c.py)
- [brette_gerstner_fig_3d.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brette_gerstner_fig_3d.py)
- [gif_cond_exp_multisynapse.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/gif_cond_exp_multisynapse.py)
- [glif_cond_neuron.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/glif_cond_neuron.py)
- [glif_psc_neuron.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/glif_psc_neuron.py)
- [glif_psc_double_alpha_neuron.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/glif_psc_double_alpha_neuron.py)
- [mat_psc_exp.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/mat_psc_exp.py)
- [mc_neuron.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/mc_neuron.py)
- [two_comps.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/two_comps.py)
- [receptors_and_current.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/receptors_and_current.py)
- [wang_decision_making.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/wang_decision_making.py)

### Balanced & Brunel networks

- [brunel_alpha.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_alpha.py)
- [brunel_delta.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_delta.py)
- [brunel_exp_multisynapse.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_exp_multisynapse.py)
- [brunel_alpha_evolution_strategies.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_alpha_evolution_strategies.py)
- [brunel_siegert.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_siegert.py)
- [brette_et_al_2007.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brette_et_al_2007.py)
- [ei_clustered_network.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/ei_clustered_network.py)
- [sensitivity_to_perturbation.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/sensitivity_to_perturbation.py)
- [artificial_synchrony.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/artificial_synchrony.py)

### Devices: generators, recorders & detectors

- [multimeter_file.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/multimeter_file.py)
- [recording_demo.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/recording_demo.py)
- [repeated_stimulation.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/repeated_stimulation.py)
- [BrodyHopfield.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/BrodyHopfield.py)
- [sinusoidal_poisson_generator.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/sinusoidal_poisson_generator.py)
- [sinusoidal_gamma_generator.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/sinusoidal_gamma_generator.py)
- [pulsepacket.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/pulsepacket.py)
- [precise_spiking.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/precise_spiking.py)
- [correlospinmatrix_detector_two_neuron.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/correlospinmatrix_detector_two_neuron.py)
- [cross_check_mip_corrdet.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/cross_check_mip_corrdet.py)
- [CampbellSiegert.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/CampbellSiegert.py)

### Plasticity: STP, STDP & dendritic rules

- [evaluate_tsodyks2_synapse.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/evaluate_tsodyks2_synapse.py)
- [evaluate_quantal_stp_synapse.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/evaluate_quantal_stp_synapse.py)
- [iaf_tum_2000_short_term_depression.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/iaf_tum_2000_short_term_depression.py)
- [iaf_tum_2000_short_term_facilitation.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/iaf_tum_2000_short_term_facilitation.py)
- [clopath_synapse_spike_pairing.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/clopath_synapse_spike_pairing.py)
- [clopath_synapse_small_network.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/clopath_synapse_small_network.py)
- [urbanczik_synapse_example.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/urbanczik_synapse_example.py)

### Gap junctions

- [gap_junctions_two_neurons.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/gap_junctions_two_neurons.py)
- [gap_junctions_inhibitory_network.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/gap_junctions_inhibitory_network.py)

### Astrocytes & tripartite networks

- [astrocyte_single.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/astrocyte_single.py)
- [astrocyte_interaction.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/astrocyte_interaction.py)
- [astrocyte_small_network.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/astrocyte_small_network.py)
- [astrocyte_brunel_bernoulli.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/astrocyte_brunel_bernoulli.py)
- [astrocyte_brunel_fixed_indegree.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/astrocyte_brunel_fixed_indegree.py)

### Rate models & mean-field

- [lin_rate_ipn_network.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/lin_rate_ipn_network.py)
- [rate_neuron_dm.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/rate_neuron_dm.py)
- [gif_pop_psc_exp.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/gif_pop_psc_exp.py)
- [gif_population.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/gif_population.py)

### Spatially-structured networks

- [spatial_grid_iaf.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_grid_iaf.py)
- [spatial_gaussex.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_gaussex.py)
- [spatial_3d_gauss.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_3d_gauss.py)
- [spatial_gabor.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_gabor.py)
- [spatial_csa.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_csa.py)
- [csa_example.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/csa_example.py)

### Connectivity & introspection

- [synapsecollection.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/synapsecollection.py)
- [plot_weight_matrices.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/plot_weight_matrices.py)

### Applications: learning & large networks

- [pong.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/pong.py)
- [pong_networks.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/pong_networks.py)
- [pong_run.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/pong_run.py)
- [sudoku.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/sudoku.py)
- [sudoku_net.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/sudoku_net.py)
- [sudoku_puzzles.py](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/sudoku_puzzles.py)
