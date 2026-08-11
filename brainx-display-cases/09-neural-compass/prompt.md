# An Internal Neural Compass

## Prompt

Build a ring of head-direction neurons that holds an activity bump like an internal compass. Point it north, rotate the animal in darkness, and show whether the bump follows the turn. Then silence a wedge of the ring and test every starting direction to discover when the compass recovers or fails.

## Expected BrainX Packages

- `brainpy-state`: construct the head-direction neurons, recurrent ring connectivity, and velocity-driven inputs.
- `brainevent`: provide efficient spike-driven communication when the compass is implemented as a spiking attractor.
- `brainstate`: evolve the activity bump with `for_loop` and evaluate all initial headings and lesion conditions with `vmap`.
- `brainunit`: specify angular velocity, delays, time constants, and neural parameters with explicit units.
