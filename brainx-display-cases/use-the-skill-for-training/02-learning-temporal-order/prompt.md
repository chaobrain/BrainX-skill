# Learning Temporal Order

## Prompt

Teach a small spiking circuit to recognize which of two tones came first, then reverse their order and show how the circuit relearns.

## Expected BrainX Packages

- `brainpy-state`: define the sensory neurons, spiking circuit, and output populations.
- `brainevent`: implement spike-driven communication and online synaptic plasticity that learns temporal order.
- `brainstate`: maintain neural and plasticity state, use `for_loop` across stimulus sequences and learning updates, and batch trials with `vmap`.
- `brainunit`: specify spike timing, delays, membrane parameters, and plasticity time constants with explicit units.
