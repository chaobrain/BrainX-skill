# Finding the Edge of Criticality

## Prompt

Start with a recurrent spiking network where a single spark usually fades away. Gradually strengthen excitation until sparks become neural avalanches and finally runaway activity. Across many network realizations, locate the narrow region where activity is most variable without becoming unstable.

## Expected BrainX Packages

- `brainpy-state`: define the recurrent excitatory and inhibitory spiking populations.
- `brainevent`: execute sparse spike propagation efficiently across large recurrent networks.
- `brainstate`: run long simulations with `for_loop` and batch coupling strengths, random seeds, and network realizations with `vmap`.
- `brainunit`: constrain membrane, synaptic, timing, and coupling parameters with explicit physical units.
