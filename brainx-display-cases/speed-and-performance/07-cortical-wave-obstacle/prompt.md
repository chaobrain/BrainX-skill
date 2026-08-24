# A Cortical Wave Meets an Obstacle

## Prompt

Create a sheet of excitatory and inhibitory neurons and trigger a brief spark at its left edge. Show the activity wave crossing the sheet, then place a silent circular patch in its path and reveal whether the wave bends around it, splits, or dies. Sweep the patch size and inhibition strength and summarize the outcomes in a phase map.

## Expected BrainX Packages

- `brainpy-state`: define the excitatory and inhibitory point-neuron populations and their projections.
- `brainevent`: implement sparse, event-driven spike communication across the sheet.
- `brainstate`: manage model state and use `for_loop` for time evolution plus `vmap` for the lesion-size and inhibition sweep.
- `brainunit`: keep time, voltage, current, conductance, and spatial parameters dimensionally consistent.
