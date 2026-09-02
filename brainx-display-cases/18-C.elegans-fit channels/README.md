# C. elegans SHK-1 and EGL-19 channel fits

This case derives two phenomenological BrainCell HH channels only from the supplied Igor packed experiments. SHK-1 uses direct baseline-corrected WT minus `shk-1(lf)` traces. EGL-19 uses the packed graph's directly labeled WT calcium family; WT-minus-each-EGL-mutant controls change sign across the requested voltages and therefore cannot represent one consistent inward conductance.

With voltage in mV, time in ms, conductance in nS, and outward current in pA, the fitted SHK-1 model is

```text
I_SHK-1 = 25.38361 * n^2 * (V - (-67.96964))
dn/dt = (n_inf(V) - n) / tau_n(V)
n_inf(V) = 1 / (1 + exp(-(V - (-2.05093)) / 12.91469))
tau_n(V) = 1.41012 + 9.97732 * exp(-V / 27.01734)
```

The fitted EGL-19 model is

```text
I_EGL-19 = 19.06212 * m^4 * (V - 50.71993)
dm/dt = (m_inf(V) - m) / tau_m(V)
m_inf(V) = 1 / (1 + exp(-(V - (-8.55928)) / 12.89281))
tau_m(V) = 5.05831 + 13.91224 /
           (exp(-(V - (-16.14914)) / 5.56407)
            + exp((V - (-16.14914)) / 5.56407))
```

Each independent step initializes its gate at the fitted -60 mV steady state. The observation model subtracts the corresponding holding current because the experimental traces are baseline corrected. BrainCell's channel classes use its inward-positive `g * gate * (E - V)` convention internally.

Gate powers were selected from equal-parameter local trace comparisons: SHK-1 power 2 was best among 1-5, and EGL-19 power 4 was best among 1-4. Three of six full-budget `m^4h` candidates terminated successfully. The best successful extension reduced the robust objective by 3.18%, but adding two parameters per trace increased BIC by 33.60, so the final model remains activation-only.

The final production candidate is in `runs/20260902T132147+0800-production-seed20260902/`. SHK-1 aggregate waveform RMSE is 80.56 pA with 2.56-3.29% per-trace normalized RMSE. EGL-19 aggregate RMSE is 20.87 pA with 3.64-13.78% normalized RMSE; the smallest traces have the largest relative errors. Controlled recovery is accurate for most interior-domain cases, but some EGL time-constant decompositions remain boundary-equivalent. Leave-one-voltage-out errors are substantially larger, especially at edge voltages, so the fitted voltage functions should not be treated as validated predictors of unmeasured commands.

Run with:

```bash
MPLCONFIGDIR=/tmp/brainx-channel-fit-mpl \
conda run -n braincell-released python fit_channels.py
```

These are protocol-calibrated population-average fits over the measured voltage ranges, not unique molecular, stoichiometric, or single-cell parameter estimates.
