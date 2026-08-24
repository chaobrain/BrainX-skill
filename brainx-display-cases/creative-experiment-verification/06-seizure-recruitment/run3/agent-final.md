Implemented the BrainX experiment in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Yrr2RF/workspace/seizure_recruitment.py), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Yrr2RF/workspace/README.md).

The 47-condition sweep shows:

- Local case (`k=0.25`, delay `6 ms`, perturbation `2.0`): only region 1 recruits at `110.9 ms`.
- Propagating case (`k=0.50`): regions recruit at `110.9`, `180.5`, and `251.2 ms`.
- All 18 full-recruitment conditions follow strict region 1 → 2 → 3 order.
- No-coupling and no-drive controls pass.

Generated artifacts:

- [Figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Yrr2RF/workspace/outputs/seizure_recruitment.png)
- [Result bundle](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.Yrr2RF/workspace/outputs/seizure_recruitment_results.npz)

Verified with syntax compilation, the complete mapped rollout, delay-phase assertions, finite-value checks, and artifact inspection. Parameters are explicitly labeled phenomenological rather than patient calibrated.