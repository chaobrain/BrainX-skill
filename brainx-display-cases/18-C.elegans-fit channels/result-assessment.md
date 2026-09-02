# Result assessment

## Deterministic outcome

Production run `20260902T132147+0800-production-seed20260902` completed with exit code 0 in 629.743 seconds. All 94 numerical arrays are finite, both JSON artifacts parse, seven tests pass, and all four requested figures are nonblank and visually inspected.

| Channel | Aggregate RMSE | Per-trace normalized RMSE | Correlation | Assessment |
|---|---:|---:|---:|---|
| SHK-1 | 80.56 pA | 2.56-3.29% | 0.979-0.994 | In-sample waveforms and model-conditioned gate summaries are supported over 0 to 100 mV. |
| EGL-19 | 20.87 pA | 3.64-13.78% | 0.939-0.989 | In-sample waveforms and model-conditioned gate summaries are supported over -20 to +40 mV; low-amplitude traces have larger relative error. |

All six observed starts terminated successfully for both channels. Two SHK starts and one EGL start reached the selected minima; poorer local minima and physical-bound hits remain archived. Across five noisy recovery truths, the largest median parameter error is 1.47% of range for SHK-1 and 0.50% for EGL-19. Maximum errors reach 13.2% and 12.6%, respectively, and two selected EGL fits use a time-function boundary despite sub-pA waveform errors. The protocol therefore identifies most tested interior cases but does not uniquely decompose every EGL time-constant parameterization.

Leave-one-voltage-out RMSE spans 93.19-245.75 pA for SHK-1 and 9.58-166.06 pA for EGL-19. All six starts terminate in every holdout fit, so these errors reflect predictive limitations of the fitted voltage-function family rather than an unverified reduced optimizer budget.

The optional EGL `m^4h` comparison closes at the full 1,200-iteration budget: three of six candidates terminate successfully. Its best successful robust objective is 3.18% lower, but BIC rises from 7979.82 to 8013.42 (`delta BIC = +33.60`) after accounting for 14 added local parameters. The activation-only `m^4` architecture is therefore selected by the locked penalized criterion.

## Claim-evidence matrix

| Claim | Evidence | Outcome |
|---|---|---|
| SHK-1 has a useful `n^2` HH representation under the requested protocol. | Power scores, six overlays, gate points/functions, BrainCell lifecycle/parity, recovery-domain, and holdout tests. | Supported as an in-sample phenomenological summary within 0-100 mV. |
| EGL-19 has a useful activation-only `m^4` HH representation under the requested protocol. | Power/structure scores, seven overlays, gate points/functions, BrainCell lifecycle/parity, recovery-domain, and holdout tests. | Supported as an in-sample phenomenological summary within -20 to +40 mV. |
| Either WT-minus-EGL-mutant family is a clean EGL-19 target over the full protocol. | Both difference families are positive at -20 to 0 mV and negative at +10 to +40 mV. | Rejected; directly labeled WT calcium traces are used. |
| Fitted values are unique biological channel parameters or activation powers are molecular stoichiometry. | Only genotype averages and one voltage protocol are available; no cell-level replicates or held-out protocol exist. | Not supported. |
| The voltage functions predict omitted or out-of-range voltages reliably. | Leave-one-voltage-out errors increase strongly at several voltages; no out-of-range validation data exist. | Not supported. |

## Limitations

- SHK subtraction combines independent WT and mutant population averages rather than paired cells.
- No cell capacitances or replicate traces are stored, so uncertainty and current density cannot be reconstructed.
- Reversal potentials are inferred near or outside a tested range and remain protocol-dependent.
- EGL traces retain small early clamp artifacts after the locked 1 ms exclusion and slow late deviations at some voltages; the minimal activation-only model does not reproduce those components.
- Local activation and time-constant points are model-conditioned trace fits, not directly measured gates. The SHK 80 mV activation estimate is 1.0072 and is plotted unclipped above the physical gate limit; the global sigmoid remains in [0, 1].
- The EGL -20 mV local activation time estimate is about 18.4 ms while the global voltage function is lower, so the curve is a compromise across traces rather than an exact interpolation of every local summary.
