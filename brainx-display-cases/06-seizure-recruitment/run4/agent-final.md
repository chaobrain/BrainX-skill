Implemented the BrainX demonstration in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.g2nNtH/workspace/seizure_recruitment.py:24).

It uses BrainMass FitzHugh-Nagumo regional masses, BrainState `for_loop` and `vmap`, and BrainUnit quantities. Results:

- Weak coupling `k=0.2`: burst remains in Region 1.
- Stronger coupling `k=0.6`: Regions 1, 2, and 3 recruit at `20.7`, `27.9`, and `35.2 ms`.
- No-coupling and no-stimulation controls pass.

Artifacts:

- [Recruitment figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.g2nNtH/workspace/outputs/seizure_recruitment.png)
- [Numeric result bundle](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.g2nNtH/workspace/outputs/seizure_recruitment_results.npz)
- [Run documentation](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.g2nNtH/workspace/README.md:1)

Verified by running the complete 62-condition sweep, compilation checks, artifact assertions, delay-phase validation, and visual inspection. The calibrated regimes are phenomenological, not clinical predictions.