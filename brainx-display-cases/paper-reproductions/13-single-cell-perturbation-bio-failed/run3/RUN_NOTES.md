# Run notes

## Status

This run is invalid and retained as failure evidence. The frozen full experiment stopped before accepting any scientific result because paired baseline and perturbation trajectories differed before the declared photostimulation onset.

The implementation's causal-control assertion raised:

```text
AssertionError: paired trajectories differ before photostimulation
```

No influence signature from this run should be interpreted. The external baseline and perturbation input arrays were intended to be identical before the intervention, so the next run must diagnose batch layout, State initialization/restoration, and onset indexing in a new directory. Do not rerun or overwrite this directory.

## Prompt

This run used `../prompt-experiment.md` unchanged.
