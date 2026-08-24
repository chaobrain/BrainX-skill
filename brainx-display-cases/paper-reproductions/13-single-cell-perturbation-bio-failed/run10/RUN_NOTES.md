# Run notes

## Status

This retained run failed its causal-control assertion before analysis. Copying
and restoring State values across two transformed rollouts still did not yield
identical pre-photostimulation trajectories. No scientific influence result
from this run is valid.

## Frozen retry contract

This run retains the run9 scientific configuration and 32-trial tuning chunks.
It changes only matched-pair State checkpointing: every State leaf is copied
before the baseline rollout, then restored before the perturbation rollout.
This repairs the causal control identified by run9 without changing model,
stimulus, perturbation, seed, target, or analysis parameters.
