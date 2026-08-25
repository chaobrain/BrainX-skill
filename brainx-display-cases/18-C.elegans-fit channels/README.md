# C. elegans SHK-1 and EGL-19 channel fits

This case extracts voltage-clamp data from the supplied Igor packed experiments and fits BrainCell Hodgkin-Huxley channels. It uses the processed WT-minus-`shk-1(lf)` potassium waves for SHK-1 and the pharmacologically isolated WT calcium waves for EGL-19. Both models initialize at the -60 mV holding potential and reproduce 100 ms voltage steps.

## Fitted SHK-1 model

With voltage in mV, time in ms, conductance in nS, and outward current in pA:

```text
I_SHK-1 = 28.8853 * n^4 * (V - (-30))
dn/dt = (n_inf(V) - n) / tau_n(V)
n_inf(V) = 0.5 * (tanh((V + 14.2546) / 30.6764) + 1)
tau_n(V) = 0.98574 + 8.81633 / (1 + exp(V / 18.6053))
```

The fit covers 0, 20, 40, 60, 80, and 100 mV. Aggregate waveform RMSE is 88.08 pA; per-voltage normalized RMSE is 2.66-6.80% of trace range, and correlations are 0.929-0.984.

## Fitted EGL-19 model

```text
I_EGL-19 = 44.0794 * m^2 * h * (V - 60)
dm/dt = (m_inf(V) - m) / tau_m(V)
dh/dt = (h_inf(V) - h) / 3.55688

m_inf(V) = 1 / (1 + exp(-(V + 5.29805) / 9.13166))
tau_m(V) = 2.47297 + 24.1597 /
           (exp(-(V - 40) / 5.05626) + exp((V - 40) / 5.05626))
h_inf(V) = 0.0413925 + (0.959751 - 0.0413925) /
           (1 + exp((V + 50) / 31.0647))
```

The fit covers -20 through +40 mV in 10 mV increments. Aggregate waveform RMSE is 9.75 pA; per-voltage normalized RMSE is 4.98-8.13% of trace range. The activation function is supported within the measured voltage range. The fitted `tau_m` center and `h_inf` midpoint reached bounds, so do not extrapolate the EGL-19 kinetics or interpret the inactivation parameters as uniquely identified biology.

## Run

```bash
MPLCONFIGDIR=/tmp/brainx-channel-fit-mpl \
conda run -n braincell-released python fit_channels.py
```

The accepted numerical evidence and figures are in `runs/20260825T120001+0800-production-seed20260825/raw/`. `fit_channels.py` also exports reusable `SHK1Channel` and `EGL19Channel` BrainCell HH classes.

The source recordings are population averages from different cells. The fitted values are protocol-calibrated phenomenological parameters, not unique molecular or single-cell estimates.

Source context: Du et al., *PLOS Computational Biology* 21(1), e1012318 (2025), doi:10.1371/journal.pcbi.1012318.
