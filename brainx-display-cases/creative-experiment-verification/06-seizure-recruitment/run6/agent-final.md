Implemented the BrainX seizure recruitment experiment in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.wHLl4I/workspace/seizure_recruitment.py).

It uses BrainMass Epileptor regions, BrainState `for_loop`/`vmap`, unit-aware delays and timescales, and matched controls. Results show:

- `k=0`: burst remains local.
- `k=20`: both neighbors recruit in order.
- Longer delays progressively postpone recruitment.
- No-stimulus control remains quiet.

Artifacts:

- [Figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.wHLl4I/workspace/outputs/seizure_recruitment.png)
- [Result bundle](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.wHLl4I/workspace/outputs/seizure_recruitment_results.npz)
- [README](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.wHLl4I/workspace/README.md)

Verified by running the complete 36-condition sweep, both controls, delay-phase test, artifact invariants, and Python compilation.