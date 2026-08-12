Implemented the regional seizure-recruitment experiment in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.96NxMU/workspace/seizure_recruitment.py:41).

The sweep found:

- Local: `k=0.25`, delay `10 ms`, pulse `1.2`.
- Recruited: `k=0.50`, delay `2 ms`, pulse `0.8`, with sustained burst onsets at `419.6`, `475.7`, and `518.4 ms`.
- No-coupling and no-perturbation controls passed.
- Sampled delays were `2`, `6`, and `10 ms`; coupling was the dominant recruitment boundary in this phenomenological grid.

Outputs:

- [Summary figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.96NxMU/workspace/results/seizure_recruitment.png)
- [Numeric result bundle](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.96NxMU/workspace/results/seizure_recruitment.npz)
- [Usage notes](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.96NxMU/workspace/README.md:1)

Verified with a complete rerun, compilation check, fixed 20 ms sustained-LFP recruitment predicate, delay impulse check, artifact assertions, and visual inspection.