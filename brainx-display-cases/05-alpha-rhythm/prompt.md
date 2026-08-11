# Generating an Alpha Rhythm

## Prompt

Create a resting cortical circuit that produces an alpha-like brain rhythm, then weaken inhibition and show how the simulated EEG changes.

## Expected BrainX Packages

- `brainmass`: model interacting cortical populations and derive an interpretable population signal for the simulated EEG.
- `brainstate`: evolve population dynamics with `for_loop` and use `vmap` to compare inhibition strengths and initial conditions.
- `brainunit`: enforce consistent units for time constants, coupling strengths, firing rates, and external drive.
