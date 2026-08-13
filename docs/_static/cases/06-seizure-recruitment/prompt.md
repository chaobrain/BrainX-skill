# Seizure Recruitment Across Regions

## Prompt

Start a seizure-like burst in one brain region and show when it remains local and when it recruits neighboring regions.

## Expected BrainX Packages

- `brainmass`: represent regional population dynamics and coupling between brain regions.
- `brainstate`: run the evolving regional activity with `for_loop` and use `vmap` to sweep coupling strengths, propagation delays, and perturbation sizes.
- `brainunit`: keep regional time constants, coupling, delays, and stimulation parameters dimensionally consistent.
