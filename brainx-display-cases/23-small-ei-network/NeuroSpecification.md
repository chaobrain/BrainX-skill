# NeuroSpecification

- **Request:** Build and simulate one recurrent excitatory/inhibitory spiking network with 20 neurons.
- **Scale:** Point-neuron network only; no ion channels, compartments, morphology, or population-rate state.
- **Population split:** 16 excitatory neurons followed by 4 inhibitory neurons in the shared population vector.
- **Neuron model:** `brainpy.state.LIFRef` with an explicit 5 ms refractory period, initialized at the reset voltage.
- **Synapses:** Probabilistic recurrent projections from each population to all 20 targets. Excitation uses a 0 mV reversal potential and inhibition uses -80 mV. Both use exponential conductance traces.
- **Protocol:** `dt=0.1 ms`, 100 ms simulation, and a constant 20 mA external current to every neuron.
- **Reproducibility:** Seed BrainState randomness with `1234` at the start of each `run()` call.
- **Outputs:** Time-major spike array `(1000, 20)`, simulation times, and a raster plot when run as a script.
- **Acceptance checks:** Imports use BrainX namespaces, all State is initialized before the rollout, projection updates occur before postsynaptic integration, and the returned spike array has the expected shape and boolean event semantics.
- **Scope boundary:** This is a compact teaching model. The random graph and drive are not calibrated to reproduce a published firing regime.
