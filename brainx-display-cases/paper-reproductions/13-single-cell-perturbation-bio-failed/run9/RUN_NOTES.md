# Run notes

## Status

This retained run failed its causal-control assertion before analysis. The
baseline rollout mutated the live arrays referenced by the nominal State
snapshot, so restoration did not recover the identical pre-perturbation state.
No scientific influence result from this run is valid.

## Frozen retry contract

This run loads the scientific model and analysis from `../run8/v1_influence.py`.
It changes only the tuning rollout from one 64-trial batch to two sequential
32-trial batches. The complete 64-trial currents are generated before slicing,
so the run8 random-number stream, model parameters, connectivity, targets,
mapping trials, and statistical tests remain unchanged.
