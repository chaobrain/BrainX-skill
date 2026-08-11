# BrainMass Central Documentation Sitemap

## Root

- [brainmass documentation](https://brainx.chaobrain.com/brainmass/) — Main BrainMass landing page; introduces the package, installation, features, and learning paths.

## Get Started

- [Installation](https://brainx.chaobrain.com/brainmass/getting_started/installation.html) — Package installation options for CPU, CUDA, TPU, visualization extras, and source development.
- [Quickstart](https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html) — Fast first workflow for running a BrainMass simulation.
- [Key Concepts](https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html) — Core mental model for units, Step models, Simulator, Network, noise, Fitter, and objectives.
- [Learning Paths](https://brainx.chaobrain.com/brainmass/getting_started/learning_paths.html) — Recommended study routes for beginners, empirical-data users, and model developers.

## Tutorials

- [Tutorials](https://brainx.chaobrain.com/brainmass/tutorials/index.html) — Landing page for the sequential tutorial path.
- [Your First Simulation](https://brainx.chaobrain.com/brainmass/tutorials/01_first_simulation.html) — First runnable example for defining a model, simulating it, and plotting results.
- [Models and Dynamics](https://brainx.chaobrain.com/brainmass/tutorials/02_models_and_dynamics.html) — Explains how neural-mass models define state variables and time evolution.
- [Noise and Stochastic Runs](https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html) — Adds stochastic noise processes to BrainMass simulations.
- [Building a Network](https://brainx.chaobrain.com/brainmass/tutorials/04_building_a_network.html) — Shows how to connect multiple neural-mass nodes with connectivity matrices and coupling.
- [Forward Models](https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html) — Maps hidden neural activity to observable BOLD, EEG, or MEG-like signals.
- [Fitting with Gradients](https://brainx.chaobrain.com/brainmass/tutorials/06_fitting_with_gradients.html) — Uses differentiable JAX optimization to fit model parameters.
- [Gradient-Free Fitting](https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html) — Fits model parameters using non-gradient optimization methods.
- [Training on Tasks](https://brainx.chaobrain.com/brainmass/tutorials/08_training_on_tasks.html) — Trains neural-mass-style networks on cognitive or sequence tasks.

## How-To Guides

- [How-To Guides](https://brainx.chaobrain.com/brainmass/howto/index.html) — Landing page for practical BrainMass workflow recipes.
- [Choose a Model](https://brainx.chaobrain.com/brainmass/howto/choose_a_model.html) — Helps select the right neural-mass model for a scientific or simulation goal.
- [Working with Units](https://brainx.chaobrain.com/brainmass/howto/work_with_units.html) — Explains unit-safe modeling with BrainUnit quantities.
- [Batch and Accelerate](https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html) — Uses batching, vectorization, and JAX acceleration for faster simulations.
- [Custom Coupling](https://brainx.chaobrain.com/brainmass/howto/custom_coupling.html) — Shows how to define custom coupling between network nodes.
- [Compose a Custom Objective](https://brainx.chaobrain.com/brainmass/howto/custom_objective.html) — Builds custom fitting losses and objective functions.
- [Run Parameter Sweeps](https://brainx.chaobrain.com/brainmass/howto/parameter_sweeps.html) — Sweeps parameters to inspect model regimes and sensitivity.
- [Analyze Results — FC / FCD / spectra](https://brainx.chaobrain.com/brainmass/howto/analyze_results.html) — Analyzes simulated signals with functional connectivity, FCD, and power spectra.

## Concepts

- [Concepts](https://brainx.chaobrain.com/brainmass/concepts/index.html) — Landing page for conceptual BrainMass modeling explanations.
- [What Is a Neural Mass Model?](https://brainx.chaobrain.com/brainmass/concepts/what_is_a_neural_mass_model.html) — Explains what neural-mass models represent biologically and computationally.
- [Why Differentiable?](https://brainx.chaobrain.com/brainmass/concepts/why_differentiable.html) — Explains why differentiable simulation matters for fitting, optimization, and learning.
- [Architecture Overview](https://brainx.chaobrain.com/brainmass/concepts/architecture_overview.html) — Describes how the main BrainMass components fit together.
- [Coupling and Delays](https://brainx.chaobrain.com/brainmass/concepts/coupling_and_delays.html) — Explains structural connectivity, coupling functions, and transmission delays.
- [From Activity to Signals](https://brainx.chaobrain.com/brainmass/concepts/from_activity_to_signals.html) — Explains how latent neural activity becomes BOLD, EEG, or MEG observations.

## Data-Driven Modeling

- [Data-Driven Modeling](https://brainx.chaobrain.com/brainmass/data_driven/index.html) — Main hub for fitting and training neural-mass networks against empirical data.
- [Data-Driven Modeling Roadmap](https://brainx.chaobrain.com/brainmass/data_driven/roadmap.html) — Roadmap for BrainMass data-driven modeling directions.

## Gallery — Model Zoo

- [Gallery](https://brainx.chaobrain.com/brainmass/gallery/index.html) — Landing page for runnable BrainMass examples and case studies.
- [Hopf Oscillator](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/hopf.html) — Demonstrates Hopf dynamics and oscillation onset.
- [Van der Pol Oscillator](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/van_der_pol.html) — Demonstrates nonlinear relaxation oscillations.
- [Stuart-Landau Oscillator](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/stuart_landau.html) — Demonstrates amplitude-phase oscillator dynamics near a Hopf bifurcation.
- [FitzHugh-Nagumo Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/fitzhugh_nagumo.html) — Demonstrates excitable two-variable neural dynamics.
- [Wilson-Cowan Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wilson_cowan.html) — Demonstrates excitatory-inhibitory population dynamics.
- [Jansen-Rit Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/jansen_rit.html) — Demonstrates cortical-column dynamics often used for EEG-like rhythms.
- [Wong-Wang Decision Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wong_wang.html) — Demonstrates decision-making dynamics with competing populations.
- [Montbrio-Pazo-Roxin Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/montbrio_pazo_roxin.html) — Demonstrates exact mean-field dynamics of QIF neuron populations.
- [Coombes-Byrne — Next-Generation Neural Mass](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/coombes_byrne.html) — Demonstrates next-generation mean-field neural-mass dynamics.
- [Larter-Breakspear Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/larter_breakspear.html) — Demonstrates nonlinear cortical excitatory-inhibitory dynamics.
- [Epileptor — Seizure Dynamics](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/epileptor.html) — Demonstrates seizure-like transitions and epileptiform dynamics.
- [Generic 2D Oscillator](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/generic_2d_oscillator.html) — Demonstrates a configurable two-dimensional oscillator.
- [Wong-Wang Excitatory-Inhibitory — Dynamic Mean Field](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wong_wang_exc_inh.html) — Demonstrates excitatory-inhibitory dynamic mean-field modeling.
- [Lorenz System](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/lorenz.html) — Demonstrates chaotic dynamical-system behavior.
- [Linear Node](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/linear.html) — Demonstrates a simple linear baseline node model.
- [Kuramoto Phase Oscillators](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/kuramoto.html) — Demonstrates synchronization in coupled phase oscillators.
- [Harmonic Oscillator Recurrent Network — HORN](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/horn.html) — Demonstrates harmonic-oscillator recurrent networks for sequence dynamics.

## Gallery — Case Studies

- [Resting-State MEG: A Whole-Brain Modeling Pipeline](https://brainx.chaobrain.com/brainmass/gallery/case_studies/resting_state_meg.html) — End-to-end whole-brain pipeline with connectome, network simulation, MEG forward model, and validation.
- [Fitting an EEG Model with Gradients](https://brainx.chaobrain.com/brainmass/gallery/case_studies/eeg_fitting.html) — Fits a neural-mass EEG model using gradient-based optimization.
- [Seizure Dynamics with the Epileptor](https://brainx.chaobrain.com/brainmass/gallery/case_studies/seizure_epileptor.html) — Case study for seizure-like dynamics using the Epileptor model.
- [Perceptual Decision-Making with Wong-Wang](https://brainx.chaobrain.com/brainmass/gallery/case_studies/decision_making.html) — Case study for decision-making dynamics using the Wong-Wang model.
- [Training a HORN Network on a Cognitive Task](https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html) — Trains a HORN network on a cognitive or sequential task.

## Central API Reference Pages

- [Neural Mass Models](https://brainx.chaobrain.com/brainmass/reference/models.html) — Central API page for oscillator, excitatory-inhibitory, cortical-column, seizure, decision, and whole-brain model classes.
- [Noise Processes](https://brainx.chaobrain.com/brainmass/reference/noise.html) — Central API page for Gaussian, white, OU, Brownian, and colored noise processes.
- [Coupling Mechanisms](https://brainx.chaobrain.com/brainmass/reference/coupling.html) — Central API page for coupling functions and classes used to connect neural-mass nodes.
- [Forward Models](https://brainx.chaobrain.com/brainmass/reference/forward.html) — Central API page for mapping neural activity to BOLD, EEG, MEG, and lead-field outputs.
- [Observation Models](https://brainx.chaobrain.com/brainmass/reference/observation.html) — Central API page for HRF kernels, BOLD observation models, and temporal averaging.
- [HORN Models](https://brainx.chaobrain.com/brainmass/reference/horn.html) — Central API page for Harmonic Oscillator Recurrent Network components.
- [Orchestration](https://brainx.chaobrain.com/brainmass/reference/orchestration.html) — Central API page for Network, Simulator, Fitter, FitResult, and objective functions.
- [Datasets](https://brainx.chaobrain.com/brainmass/reference/datasets.html) — Central API page for dataset registration, loading, connectome containers, signal containers, and task datasets.
- [Visualization](https://brainx.chaobrain.com/brainmass/reference/viz.html) — Central API page for plotting time series, phase portraits, connectivity, functional connectivity, and spectra.
- [Utilities & Types](https://brainx.chaobrain.com/brainmass/reference/utilities.html) — Central API page for helper functions and metadata types such as sigmoid, delay indexing, model listing, and ModelInfo.
