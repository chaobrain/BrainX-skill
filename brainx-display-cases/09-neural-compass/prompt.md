# An Internal Neural Compass

## Prompt

Build a ring of head-direction neurons that holds an activity bump like an internal compass. Point it north, rotate the animal in darkness, and show whether the bump follows the turn. Then silence a wedge of the ring and test every starting direction to discover when the compass recovers or fails.

## Expected BrainX Packages

- `brainpy-state`: construct the head-direction neurons, recurrent ring connectivity, and velocity-driven inputs.
- `brainevent`: provide efficient spike-driven communication when the compass is implemented as a spiking attractor.
- `brainstate`: evolve the activity bump with `for_loop` and evaluate all initial headings and lesion conditions with `vmap`.
- `brainunit`: specify angular velocity, delays, time constants, and neural parameters with explicit units.

## Reference
1. Seelig, J., Jayaraman, V. Neural dynamics for landmark orientation and angular path integration. Nature 521, 186–191 (2015). https://doi.org/10.1038/nature14446
2. Kim SS, Rouault H, Druckmann S, Jayaraman V. Ring attractor dynamics in the Drosophila central brain. Science. 2017 May 26;356(6340):849-853. doi: 10.1126/science.aal4835. Epub 2017 May 4. PMID: 28473639.