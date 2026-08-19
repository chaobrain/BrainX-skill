# C. elegans body-wall muscle inference

This case implements the seven-dimensional Hodgkin-Huxley-type body-wall
muscle model from Du et al. (2025). Seven dimensions means membrane voltage,
five gates, and intracellular calcium; the membrane equation contains six
currents: EGL-19, SHK-1, SLO-2, Kr, NCA sodium, and leak.

Trace #9 (30 pA) is the only training observation. Traces #6-#8 (15, 20, and
25 pA) are held out until validation. The fit uses sequential rejection
approximate Bayesian computation over the six free parameters reported in the
paper. All simulations use BrainCell state and integration, BrainState's
transformed time loop, and BrainUnit quantities.

The executable equations follow the authors' Figure 4 notebook. In particular,
they retain its electrode-current calibration and effective reversal-potential
offsets, which differ in presentation from parts of the paper's prose appendix.

Run the complete case:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python celegans_muscle_inference.py
```

Outputs are written to `results/`: `report.json`, posterior samples, aligned
trace predictions, and `held_out_validation.png`.

## Result interpretation

The fitted model reproduces the stimulus-period spike count at every tested
current (3, 3, and 4 spikes at 15, 20, and 25 pA) and preserves the experimental
trend toward shorter interspike intervals as current increases. Peak voltages
are within 1.7 mV on all three held-out traces. It does not fully reproduce the
recordings: the first spike is 19-32 ms late, interspike intervals are 5.6-9.1 ms
short, and the 15 and 20 pA simulations each contain one post-stimulus rebound
spike that is absent experimentally. Treat the validation as qualitative support
for the current-response relationship, not a full waveform-level match.

Source: X. Du et al., "Biophysical modeling and experimental analysis of the
dynamics of C. elegans body-wall muscle cells," PLOS Computational Biology
21(1), e1012318 (2025), doi:10.1371/journal.pcbi.1012318.
