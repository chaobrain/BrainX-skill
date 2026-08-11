Implemented the complete BrainX experiment in [cortical_wave.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Nl7AhF/workspace/cortical_wave.py:1), with usage and phase definitions in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Nl7AhF/workspace/README.md:1).

The 36-condition sweep produced:

- 24 split/rejoin outcomes
- 5 one-sided bends
- 1 lesion-induced death
- All 6 matched controls crossed the sheet

Artifacts:

- [Wave snapshots](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Nl7AhF/workspace/results/wave_snapshots.png)
- [Phase map](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Nl7AhF/workspace/results/phase_map.png)
- [Phase metrics](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Nl7AhF/workspace/results/phase_metrics.npz)

Verified with an end-to-end run, compilation check, silent-patch invariant, phase-code assertions, and nonblank image checks.