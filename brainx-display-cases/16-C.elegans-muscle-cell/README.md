# C. elegans body-wall muscle inference

This case implements the seven-dimensional Hodgkin-Huxley-type body-wall
muscle model from Du et al. (2025). Seven dimensions means membrane voltage,
five gates, and intracellular calcium; the membrane equation contains six
currents: EGL-19, SHK-1, SLO-2, Kr, NCA sodium, and leak.

Trace #9 (30 pA) is the only fitting observation. Traces #6-#8 (15, 20, and
25 pA) are held out until validation. The fit uses bounded sequential rejection
approximate Bayesian computation over six free parameters. This is a
simulation-based fitting method, but it is not the neural likelihood estimator
used in the paper. All simulations use BrainCell state and integration,
BrainState's transformed time loop, and BrainUnit quantities.

The executable equations retain the authors' electrode-current calibration and
effective reversal-potential offsets. The paper equation's voltage-dependent
SLO-2 factor is included; one source fitting helper omits that factor. The small
M-type current in the authors' helper is excluded because this case implements
the six currents requested here.

Run the complete case:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python celegans_muscle_inference.py
```

Outputs are written to `results/`: `report.json`, retained ABC samples,
synthetic-recovery results, aligned trace predictions, and
`held_out_validation.png`. The frozen production run and provenance are under
`runs/20260824T115614+0800-production-seed2025/`.

## Result interpretation

The best-fit calibration is `g_EGL-19=12.400 nS`, `g_SHK-1=45.000 nS`,
`g_leak=0.500 nS`, `C=16.953 pF`, `g_SLO-2=2.376 nS`, and
`V_shift=10.520 mV`. SHK-1 and leak land at their upper prior bounds, so these
values must not be interpreted as uniquely identified biological conductances.

The fitted model reproduces the stimulus-period spike count at every held-out
current (3, 3, and 4 spikes at 15, 20, and 25 pA) and preserves the experimental
trend toward shorter interspike intervals as current increases. It does not
reproduce the full recordings: held-out RMSE is 14.2-15.3 mV, correlations are
0.12-0.30, first spikes are 18-31 ms late, and the 15 and 20 pA simulations each
contain one post-stimulus spike absent experimentally. Treat the result as
qualitative protocol-level support, not waveform-level consistency.

Source: X. Du et al., "Biophysical modeling and experimental analysis of the
dynamics of C. elegans body-wall muscle cells," PLOS Computational Biology
21(1), e1012318 (2025), doi:10.1371/journal.pcbi.1012318. Author source:
https://github.com/XuexingDu/C.elegans-Muscle
