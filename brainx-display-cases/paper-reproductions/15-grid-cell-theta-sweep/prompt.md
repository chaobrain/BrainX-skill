# Alternating Theta Sweeps in a Direction-Grid Network

## Prompt

Develop a firing-rate model in which an internal direction signal organizes subsecond spatial sweeps during open-field navigation. Build a theta-modulated head-direction ring attractor with local recurrent excitation, global inhibition, sensory anchoring to the animal's head direction, and slow firing-rate adaptation. Couple it through a conjunctive direction-by-grid transformation to one or more two-dimensional toroidal grid-cell attractors. The conjunctive population may be represented by an effective projection, but the projection must convert the direction-ring state into a spatially shifted input rather than impose a decoded trajectory directly.

Simulate straight runs, changes in running speed, and turning in a two-dimensional arena. Decode internal direction from the ring population and position from grid-cell population activity within individual theta cycles. Determine whether firing-rate adaptation and theta modulation generate sweeps that alternate to the left and right of the animal's heading across successive cycles, and whether direction and position sweeps remain aligned.

Report all analysis definitions of the model. Produce clear figures and quantitative summaries showing representative population dynamics, left-right alternation relative to shuffled cycle order. Draw a 2D diagram showing a rat's running trajectory. At 10 selected time points along this trajectory, draw arrows (vectors) originating from the rat's position on the trajectory. The arrows should point in different directions representing decoded theta sweep directions at this time point.

## Expected BrainX Packages

- `brainmass`: implement the aggregate head-direction and grid-cell attractor dynamics.
- `brainstate`: manage recurrent state, seeded randomness, transformed time evolution, controls, and parameter sweeps.
- `brainunit`: keep time, theta frequency, speed, angle, and spatial scale dimensionally explicit.

## Reference
1. Vollan, A.Z., Gardner, R.J., Moser, MB. et al. Left–right-alternating theta sweeps in entorhinal–hippocampal maps of space. Nature 639, 995–1005 (2025). https://doi.org/10.1038/s41586-024-08527-1
2. Ji Z, Chu T, Wu S ...A systems model of alternating theta sweeps via firing rate adaptation Current Biology, 2025; 35, 709-722.e5