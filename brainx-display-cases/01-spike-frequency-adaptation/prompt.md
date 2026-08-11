# Spike-Frequency Adaptation

## Prompt

Show why a neuron fires quickly at first but gradually slows during a steady input, then remove its adaptation current and reveal what changes.

## Expected BrainX Packages

- `braincell`: build a conductance-based cell with an explicit adaptation mechanism that can be removed for the comparison.
- `brainstate`: manage the cell state, evolve it with `for_loop`, and use `vmap` to compare input currents and adaptation strengths.
- `brainunit`: enforce consistent units for current, voltage, conductance, capacitance, and time.
