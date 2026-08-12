Implemented the BrainX seizure-recruitment study in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.bO29LC/workspace/seizure_recruitment.py:1).

Key result for pulse size `2.0`:

- `k=0`: burst remains local.
- `k=2`: first neighbor recruited.
- `k>=5`: both neighbors recruited.
- At `k=5`, `10 ms` delay: onsets were `111.8`, `122.6`, and `133.2 ms`.
- No-pulse and no-coupling controls behaved as expected.

The sweep uses BrainMass Epileptor regions and diffusive coupling, BrainState `for_loop` plus stateful `vmap`, and BrainUnit quantities for all physical times and delays.

Artifacts:

- [Figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.bO29LC/workspace/results/seizure_recruitment.png)
- [Numeric results](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.bO29LC/workspace/results/seizure_recruitment.npz)
- [README](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.bO29LC/workspace/README.md:1)

Verified with a complete rerun, finite-trajectory checks, delay-phase impulse test, causal controls, and artifact validation. Parameters and event criteria are explicitly labeled exploratory and outcome-calibrated.