Implemented the BrainX seizure recruitment study in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.6rBb15/workspace/seizure_recruitment.py:1), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.6rBb15/workspace/README.md:1).

It uses BrainMass FitzHugh-Nagumo regional dynamics, BrainState `for_loop` and `vmap`, and BrainUnit quantities. Results:

- Local case: focus onset `21.5 ms`; neighbors not recruited.
- Stronger coupling: onsets `21.5`, `27.9`, and `34.9 ms` in causal order.
- No-stimulation and no-coupling controls pass.

Generated [figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.6rBb15/workspace/outputs/seizure_recruitment.png) and [complete sweep data](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.6rBb15/workspace/outputs/seizure_recruitment_results.npz). The script runs successfully with `python seizure_recruitment.py`.