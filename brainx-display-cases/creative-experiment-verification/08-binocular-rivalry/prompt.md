# Binocular Rivalry

## Prompt

Model two competing visual populations, one seeing vertical stripes and the other horizontal stripes. Present both continuously so perception alternates between them, then simulate many observers with different adaptation and noise levels and explain what controls how long each percept dominates.

## Expected BrainX Packages

- `brainmass`: represent the competing visual populations with interpretable population-level dynamics.
- `brainstate`: hold adaptation and stochastic state, run the dynamics with `for_loop`, and batch observers and parameter conditions with `vmap`.
- `brainunit`: enforce consistent units for time constants, input drive, adaptation, and coupling strength.
