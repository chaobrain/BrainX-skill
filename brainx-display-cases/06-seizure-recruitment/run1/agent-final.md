Implemented the BrainX experiment in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.zdeAha/workspace/seizure_recruitment.py:1), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.zdeAha/workspace/README.md:1).

The 72-condition sweep uses BrainMass FHN regional dynamics, BrainState `for_loop` plus state-aware `vmap`, and BrainUnit timing/delay quantities. Results show:

- Local case peaks: `[0.928, 0.235, 0.037, 0.006]`
- Recruited case peaks: `[0.928, 0.922, 0.913, 0.907]`
- Recruited onsets at 4 ms edge delay: `[11.4, 17.0, 22.6, 28.3] ms`
- Increasing delay shifts onset later without changing eventual recruitment in this observation window.
- No-coupling and no-perturbation controls pass.

Generated artifacts:

- [Figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.zdeAha/workspace/outputs/seizure_recruitment.png)
- [Metrics CSV](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.zdeAha/workspace/outputs/seizure_recruitment_metrics.csv)
- [Complete NPZ data](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.zdeAha/workspace/outputs/seizure_recruitment_data.npz)

Verified with `MPLBACKEND=Agg python seizure_recruitment.py` and Python compilation checks.