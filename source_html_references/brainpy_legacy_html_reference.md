# BrainPy Central Documentation Sitemap

Purpose: choose the official BrainPy HTML page that directly supports installation, core concepts, model building, simulation, training, analysis, numerical tooling, troubleshooting, or an exact module-level API task.

## Root

- [BrainPy Documentation](https://brainpy.readthedocs.io/index.html) - Main landing page for BrainPy brain-dynamics programming documentation.

## Quickstart
- [Simulating a Brain Dynamics Model](https://brainpy.readthedocs.io/quickstart/simulation.html) - Introduces the basic model-simulation workflow.
- [Training a Brain Dynamics Model](https://brainpy.readthedocs.io/quickstart/training.html) - Introduces the basic model-training workflow.
- [Analyzing a Brain Dynamics Model](https://brainpy.readthedocs.io/quickstart/analysis.html) - Introduces the basic model-analysis workflow.

## Core Concepts

- [Core Concepts](https://brainpy.readthedocs.io/core_concepts.html) - Hub for BrainPy's core principles and concepts.
- [Object-oriented Transformation](https://brainpy.readthedocs.io/core_concept/brainpy_transform_concept.html) - Explains object-oriented transformations in BrainPy programs.
- [Dynamical System](https://brainpy.readthedocs.io/core_concept/brainpy_dynamical_system.html) - Explains BrainPy's dynamical-system abstraction.

## BDP Tutorials

- [BDP Tutorials](https://brainpy.readthedocs.io/tutorials.html) - Hub for model-building, simulation, training, and analysis tutorials.

### Math Foundation

- [`Variable` and `BrainPyObject`](https://brainpy.readthedocs.io/tutorial_math/variables.html) - Introduces mutable variables and BrainPy object organization.
- [Control Flows for JIT Compilation](https://brainpy.readthedocs.io/tutorial_math/control_flows.html) - Uses BrainPy control-flow operations with JIT compilation.
- [NumPy-like Operations](https://brainpy.readthedocs.io/tutorial_math/Numpy_like_Operations.html) - Uses NumPy-style mathematical and array operations.
- [Dedicated Operators](https://brainpy.readthedocs.io/tutorial_math/Dedicated_Operators.html) - Introduces BrainPy-specific mathematical operators.
- [Einstein-style Array Operations](https://brainpy.readthedocs.io/tutorial_math/einops_in_brainpy.html) - Uses `ein_rearrange`, `ein_reduce`, and `ein_repeat` for array transformations.

### Model Building

- [Using Built-in Models](https://brainpy.readthedocs.io/tutorial_building/overview_of_dynamic_model.html) - Selects and uses BrainPy's built-in dynamical models.
- [Building Conductance-based Neuron Models](https://brainpy.readthedocs.io/tutorial_building/build_conductance_neurons_v2.html) - Builds conductance-based neuron models.
- [Phenomenological Synaptic Models](https://brainpy.readthedocs.io/tutorial_building/phenon_synapse_models.html) - Builds phenomenological synaptic models.
- [Kinetic Synaptic Models](https://brainpy.readthedocs.io/tutorial_building/kinetic_synapse_models.html) - Builds kinetic synaptic models.
- [Building Network Models](https://brainpy.readthedocs.io/tutorial_building/build_network_models.html) - Composes dynamical components into network models.
- [Customizing Neuron Models](https://brainpy.readthedocs.io/tutorial_building/customize_neuron_models.html) - Defines custom neuron models.
- [Customizing Synapse Models](https://brainpy.readthedocs.io/tutorial_building/customize_synapse_models.html) - Defines custom synapse models.
- [How to Customize a Synapse](https://brainpy.readthedocs.io/tutorial_building/how_to_customze_a_synapse.html) - Walks through a custom synapse implementation.

### Model Simulation

- [Simulation with `DSRunner`](https://brainpy.readthedocs.io/tutorial_simulation/simulation_dsrunner.html) - Configures and runs dynamical-system simulations with `DSRunner`.
- [Parallel Simulation for Parameter Exploration](https://brainpy.readthedocs.io/tutorial_simulation/parallel_for_parameter_exploration.html) - Runs parameter-exploration simulations in parallel.
- [Monitor Every Multiple Steps](https://brainpy.readthedocs.io/tutorial_simulation/monitor_per_multiple_steps.html) - Records simulation values at multi-step intervals.

### Model Training

- [Building Training Models](https://brainpy.readthedocs.io/tutorial_training/build_training_models.html) - Builds dynamical systems for training workflows.
- [Training with Offline Algorithms](https://brainpy.readthedocs.io/tutorial_training/offline_training.html) - Trains models with offline algorithms.
- [Training with Online Algorithms](https://brainpy.readthedocs.io/tutorial_training/online_training.html) - Trains models with online algorithms.
- [Training with Back-propagation Algorithms](https://brainpy.readthedocs.io/tutorial_training/bp_training.html) - Trains dynamical systems with backpropagation.
- [Introduction to Echo State Networks](https://brainpy.readthedocs.io/tutorial_training/esn_introduction.html) - Introduces echo-state-network construction and training.

### Model Analysis

- [Low-dimensional Analyzers](https://brainpy.readthedocs.io/tutorial_analysis/lowdim_analysis.html) - Uses BrainPy's low-dimensional analysis tools.
- [High-dimensional Analyzers](https://brainpy.readthedocs.io/tutorial_analysis/highdim_analysis.html) - Uses BrainPy's high-dimensional analysis tools.
- [Analysis of a Decision-making Model](https://brainpy.readthedocs.io/tutorial_analysis/decision_making_model.html) - Applies analysis tools to a decision-making model.

## BDP Toolboxes

- [BDP Toolboxes](https://brainpy.readthedocs.io/toolboxes.html) - Hub for BrainPy's detailed brain-dynamics modeling toolboxes.

### Differential Equations

- [Ordinary Differential Equation Solvers](https://brainpy.readthedocs.io/tutorial_toolbox/ode_numerical_solvers.html) - Selects and uses numerical solvers for ODEs.
- [Stochastic Differential Equation Solvers](https://brainpy.readthedocs.io/tutorial_toolbox/sde_numerical_solvers.html) - Selects and uses numerical solvers for SDEs.
- [Fractional Differential Equation Solvers](https://brainpy.readthedocs.io/tutorial_toolbox/fde_numerical_solvers.html) - Selects and uses numerical solvers for FDEs.
- [Delay Differential Equation Solvers](https://brainpy.readthedocs.io/tutorial_toolbox/dde_numerical_solvers.html) - Selects and uses numerical solvers for DDEs.
- [Joint Differential Equations](https://brainpy.readthedocs.io/tutorial_toolbox/joint_equations.html) - Combines coupled differential equations with `JointEq`.

### Modeling Utilities

- [Synaptic Connections](https://brainpy.readthedocs.io/tutorial_toolbox/synaptic_connections.html) - Constructs synaptic connectivity.
- [Synaptic Weights](https://brainpy.readthedocs.io/tutorial_toolbox/synaptic_weights.html) - Initializes and manages synaptic weights.
- [Inputs Construction](https://brainpy.readthedocs.io/tutorial_toolbox/inputs.html) - Constructs input currents and signals.
- [Gradient Descent Optimizers](https://brainpy.readthedocs.io/tutorial_toolbox/optimizers.html) - Selects and configures gradient-descent optimizers.
- [Surrogate Gradient](https://brainpy.readthedocs.io/tutorial_toolbox/surrogate_gradient.html) - Uses surrogate gradients for non-differentiable events.
- [State Saving and Loading](https://brainpy.readthedocs.io/tutorial_toolbox/state_saving_and_loading.html) - Saves and restores dynamical-system state.
- [State Resetting](https://brainpy.readthedocs.io/tutorial_toolbox/state_resetting.html) - Resets dynamical-system state between runs.

## Advanced Tutorials

- [Advanced Tutorials](https://brainpy.readthedocs.io/advanced_tutorials.html) - Hub for transformations, interoperability, contribution, and analyzer internals.
- [JIT Compilation with `BrainPyObject`](https://brainpy.readthedocs.io/tutorial_advanced/compilation.html) - Applies JIT compilation to object-oriented BrainPy programs.
- [Automatic Differentiation with `BrainPyObject`](https://brainpy.readthedocs.io/tutorial_advanced/differentiation.html) - Applies automatic differentiation to object-oriented BrainPy programs.
- [Use Flax Modules in BrainPy](https://brainpy.readthedocs.io/tutorial_advanced/integrate_flax_into_brainpy.html) - Integrates Flax modules into BrainPy programs.
- [Integrate BrainPy Models into Flax: LIF](https://brainpy.readthedocs.io/tutorial_advanced/integrate_bp_lif_into_flax.html) - Integrates a BrainPy LIF model into Flax.
- [Integrate BrainPy Models into Flax: ConvLSTM](https://brainpy.readthedocs.io/tutorial_advanced/integrate_bp_convlstm_into_flax.html) - Integrates a BrainPy ConvLSTM model into Flax.
- [Contributing to BrainPy](https://brainpy.readthedocs.io/tutorial_advanced/contributing.html) - Documents the contribution workflow.
- [How Low-dimensional Analyzers Work](https://brainpy.readthedocs.io/tutorial_advanced/advanced_lowdim_analysis.html) - Explains the implementation of low-dimensional analyzers.


## Central API Reference Pages

Generated class and function pages are intentionally omitted. Use one central API page per maintained module.

- [`brainpy` Module](https://brainpy.readthedocs.io/apis/brainpy.html) - Main namespace for differential integration, dynamical-system construction, simulation, training, and state helpers.
- [`brainpy.math` Module](https://brainpy.readthedocs.io/apis/math.html) - Arrays, transformations, environments, delays, random operations, sparse operations, and event operators.
- [`brainpy.dnn` Module](https://brainpy.readthedocs.io/apis/dnn.html) - Activation, convolution, dense, normalization, pooling, Flax-interoperation, and utility layers.
- [`brainpy.dyn` Module](https://brainpy.readthedocs.io/apis/dyn.html) - Ions, ion channels, neurons, synapses, projections, synaptic outputs, plasticity, and population-rate models.
- [`brainpy.integrators` Module](https://brainpy.readthedocs.io/apis/integrators.html) - Numerical solvers for ordinary, stochastic, delay, and fractional differential equations.
- [`brainpy.analysis` Module](https://brainpy.readthedocs.io/apis/analysis.html) - Low- and high-dimensional differential-equation analysis tools.
- [`brainpy.connect` Module](https://brainpy.readthedocs.io/apis/connect.html) - Connectivity constructors for linking neuron groups.
- [`brainpy.encoding` Module](https://brainpy.readthedocs.io/apis/encoding.html) - Encoders that convert input data into neural activity.
- [`brainpy.initialize` Module](https://brainpy.readthedocs.io/apis/initialize.html) - Basic, regular, random, and decay initializers plus initialization helpers.
- [`brainpy.inputs` Module](https://brainpy.readthedocs.io/apis/inputs.html) - Current-input construction functions.
- [`brainpy.losses` Module](https://brainpy.readthedocs.io/apis/losses.html) - Loss functions for model training.
- [`brainpy.measure` Module](https://brainpy.readthedocs.io/apis/measure.html) - Firing rates, correlations, functional connectivity, raster plots, and field-potential estimates.
- [`brainpy.optim` Module](https://brainpy.readthedocs.io/apis/optim.html) - Gradient-based optimizers and learning-rate schedulers.
- [`brainpy.running` Module](https://brainpy.readthedocs.io/apis/running.html) - Vectorized, parallel, process-pool, and CPU-parallel simulation helpers.
- [`brainpy.mixin` Module](https://brainpy.readthedocs.io/apis/mixin.html) - Shared mixins and protocols for parameter descriptions, containers, tree nodes, and aligned postsynaptic behavior.
- [`brainpy.state` Module](https://brainx.chaobrain.com/brainpy-state/apis/index.html) - State-based, differentiable spiking-network APIs exposed through `brainpy.state`.
