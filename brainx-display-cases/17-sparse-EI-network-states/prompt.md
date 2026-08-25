# Sparse excitatory-inhibitory network states

## Prompt

Build a sparse recurrent spiking network with 10,000 excitatory and 2,500 inhibitory LIF neurons. Use fixed indegree corresponding to 0.1 connectivity, instantaneous voltage-jump synapses, a 1.5 ms delay, and external Poisson drive. Run four conditions by setting `(g, eta)` to `(3, 2)`, `(6, 4)`, `(5, 2)`, and `(4.5, 0.9)`, where `g` is the inhibitory-to-excitatory strength ratio and `eta` is the external rate relative to threshold rate. Determine whether the network enters synchronous regular, fast synchronous irregular, asynchronous irregular, and slow synchronous irregular states. For each condition, output a raster, population rate, E/I firing rates, ISI CV, power spectrum, and dominant population frequency. Use fixed seeds and repeated runs to test robustness.

## Expected BrainX Packages

- `brainpy-state`: build the LIF populations, delayed synapses, and Poisson input.
- `brainevent`: implement sparse event-driven recurrent communication.
- `brainunit`: enforce consistent voltage and time units.
- `brainstate`: manage and compile the network simulation.