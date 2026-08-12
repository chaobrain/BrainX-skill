# Alpha rhythm under weakened inhibition

This example uses BrainMass's two-population Wilson-Cowan cortical circuit. The
baseline E/I loop is tuned phenomenologically to an alpha-like limit cycle; the
intervention changes only inhibitory feedback `wEI` from `12` to `8`. The
reported signal is `rE - rI`, an interpretable local population/EEG-like proxy,
not a calibrated scalp voltage.

BrainUnit carries physical time through the model (`dt`, duration, transient,
and E/I time constants). Firing rates and external drive carry hertz and cross
the model's normalized-rate boundary through one explicit `100 Hz` scale;
couplings are dimensionless Wilson-Cowan gains. BrainState maps complete
independent rollouts across both inhibition strengths and three initial
conditions with `vmap`; each rollout is evolved with `for_loop`.

Run:

```bash
MPLCONFIGDIR=/tmp/brainx-alpha-mpl python alpha_rhythm.py
```

The script prints peak frequency, RMS amplitude, and alpha-band power fraction
for both conditions, then writes `alpha_rhythm.png`. With the fixed parameters,
baseline inhibition sustains an approximately 10 Hz rhythm. Weakening
inhibition moves the circuit to a fixed point, so the oscillatory EEG-like
signal collapses. This is a model-level intervention result, not a general
claim about every biological form of disinhibition.
