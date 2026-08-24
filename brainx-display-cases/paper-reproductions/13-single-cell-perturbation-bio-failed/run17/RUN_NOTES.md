# Run notes

## Status

Completed successfully. All 512 causal pairs passed the external-input and
pre-photostimulation trajectory checks. None of the four requested influence
signatures reproduced, and the target spike dose missed the published benchmark.
The run is retained without parameter tuning. See `REPORT.md` and `results.json`;
the read-only statistical-support audit is in `../run18/`.

## Frozen retry contract

This run retains all frozen scientific and analysis parameters. It applies the
run16-validated BrainState lifecycle: allocate 32-lane State before transform
construction, reset before every rollout, and compare baseline/perturbation
lanes within one transformed call. Only the target photo-current mask differs
within each pair.
