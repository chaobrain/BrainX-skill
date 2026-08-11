# Memories Replaying During Sleep

## Prompt

Teach a spiking network the sequence of four places along a route. Let it run without external input during a sleep-like period and reveal whether the route replays forward or backward. Suppress replay in a matched group and compare how well each network recalls the route afterward.

## Expected BrainX Packages

- `brainpy-state`: define the place-cell populations and recurrent spiking network.
- `brainevent`: implement spike-driven synaptic transmission and activity-dependent plasticity for learning the route.
- `brainstate`: maintain neural and plasticity state, use `for_loop` across learning, sleep, and recall, and use `vmap` for matched replay and replay-suppression groups.
- `brainunit`: enforce consistent timing, voltage, conductance, and plasticity parameters.
