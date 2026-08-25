# Result assessment

## Deterministic outcome

The replacement production run completed with exit code 0 in 58.89 seconds. All 12 saved arrays are finite, all required files parse, and all four requested figures are nonblank.

| Channel | Aggregate RMSE | Per-trace normalized RMSE | Correlation | Assessment |
|---|---:|---:|---:|---|
| SHK-1 | 88.08 pA | 2.66-6.80% | 0.929-0.984 | Waveform and activation summaries supported over 0 to 100 mV. |
| EGL-19 | 9.75 pA | 4.98-8.13% | 0.624-0.957 | Waveform and activation summaries supported over -20 to +40 mV; low-amplitude traces have weaker correlation. |

Nominal exact-pipeline recovery reconstructed SHK-1 synthetic traces at 7.37 pA RMSE and EGL-19 exactly. This proves fitting mechanics at one nominal truth only; it does not establish full-domain identifiability.

## Claim-evidence matrix

| Claim | Evidence | Outcome |
|---|---|---|
| SHK-1 is represented by an `n^4` HH current under the requested protocol. | Six current overlays, `n_inf` points/function, `tau_n` points/function, and lifecycle tests. | Supported within the measured protocol. |
| EGL-19 is represented by an `m^2 h` HH current under the requested protocol. | Seven current overlays, activation points/function, activation-time-constant points/function, and lifecycle tests. | Supported within the measured protocol. |
| Fitted values are unique biological channel parameters. | Only genotype-average traces are available; no cell-level replicates or held-out voltages; EGL kinetic centers hit bounds. | Not supported. |
| The model extrapolates outside the measured voltages. | No out-of-range validation data. | Not supported. |

## Limitations

- Potassium observations are Igor-processed WT-minus-mutant averages, not paired-cell subtraction.
- EGL-19 mutants are partial loss-of-function, so isolated WT calcium current is the fitting target rather than a knockout subtraction.
- The source measurements did not subtract leak and do not include cell-level capacitance/replicate data in these files.
- EGL-19 `v_tau_m=40 mV` and `v_half_h=-50 mV` reached their fitting bounds. Report waveform prediction and activation only; treat inactivation and extrapolated kinetics as weakly identified.
