# Sound Localization from Timing

## Prompt

Build a spiking network that tells whether a sound came from the left or right using only the tiny difference in when it reaches the two ears.

## Expected BrainX Packages

- `brainpy-state`: construct the auditory input neurons, coincidence detectors, and left-right readout populations.
- `brainevent`: deliver precisely timed spikes through efficient event-driven projections.
- `brainstate`: evolve the circuit with `for_loop` and use `vmap` to evaluate many interaural delays and sound directions together.
- `brainunit`: keep delays, time constants, voltages, currents, and conductances dimensionally correct.
