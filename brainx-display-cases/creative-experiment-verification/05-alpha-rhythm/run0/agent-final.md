Created [alpha_rhythm.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.h0uvpB/workspace/alpha_rhythm.py) and generated [the comparison figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.h0uvpB/workspace/alpha_rhythm_inhibition.png).

The BrainMass Jansen-Rit circuit produces a stable 11 Hz alpha rhythm. Reducing inhibitory gain from `22 mV` to `15.4 mV` raises the mean EEG proxy from about `7.57 mV` to `8.79 mV`, while collapsing RMS amplitude from about `1.33 mV` to `0.02 mV`.

The implementation uses:

- `brainmass.JansenRitStep` with the interpretable `E - I` EEG proxy.
- `brainstate.transform.for_loop` for time evolution.
- `brainstate.transform.vmap` over both inhibition strengths and three matched initial conditions.
- `brainunit` quantities for time, firing rates, external drive, and postsynaptic gains.

Verified with deterministic repeated runs, scientific assertions, and `py_compile`. Run with `python alpha_rhythm.py`.