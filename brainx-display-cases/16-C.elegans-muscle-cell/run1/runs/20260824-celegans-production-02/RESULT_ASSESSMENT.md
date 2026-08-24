# Deterministic result assessment

## Mechanical and optimization outcome

- Status: done, exit code 0, CPU `cpu:0`, duration 1,964.147 s; 15 JSON artifacts parse strictly and all 59 NPZ arrays are finite.
- All three observed starts reached the locked closure rule: seed 2025 plateaued at generation 90, seed 2026 reached estimator convergence at generation 84, and seed 2027 plateaued at generation 65.
- The unchanged training-only selection chose seed 2025 with objective 6.664992, an improvement of 0.302275 over iteration 1. Candidate evaluations were 2,548, 2,380, and 1,848 respectively.
- Full voltage, gate, and calcium State checks pass at nominal, lower-bound, and upper-bound parameters. No selected prediction spikes during 0-50 ms.

## Training and control

- Trace #8 / 25 pA: 6.206 mV RMSE, correlation 0.866, 4 observed versus 4 predicted spikes, and 4.9 ms first-spike error.
- Passive Trace #8: 17.179 mV RMSE and zero predicted spikes. The active model passes the locked passive comparison.

## Held-out prediction

| Trace | Current | RMSE (mV) | Correlation | Observed / predicted spikes | First-spike error (ms) |
|---|---:|---:|---:|---:|---:|
| #6 | 15 pA | 10.579 | 0.439 | 3 / 3 | 4.8 |
| #7 | 20 pA | 8.257 | 0.711 | 3 / 4 | 3.3 |
| #9 | 30 pA | 9.368 | 0.690 | 4 / 5 | 7.4 |

- Every held-out trace passes the count and latency thresholds. Predicted counts `[3, 4, 4, 5]` are nondecreasing and first-spike times `[110.9, 94.8, 85.8, 80.9]` ms are nonincreasing across 15-30 pA.
- All three independently fitted start vectors produce the same `[3, 4, 4, 5]` count trend, monotone latency, and a locked predictive pass. The recovered 15 pA response is therefore not peculiar to the selected seed.
- The locked overall result is `supported-under-tested-protocol`; this supports prediction for these four stimulation amplitudes and recording series only.

## Numerical convergence

- RK4 at 0.05 ms, downsampled at matched integration endpoints, preserves every selected spike count and first-spike time exactly at 0.1 ms reporting resolution.
- Coarse/fine waveform RMSE is 0.001334 mV at 15 pA and at most 0.000866 mV for the other protocols.

## Parameter recovery and claim boundary

- Sixteen predeclared Latin-hypercube truths and 48 unchanged-pipeline fits are retained with latent/noisy observations, truth objectives, histories, failures, profiles, boundary rates, and paired-error correlations.
- Twenty-seven of 48 recovery starts fail the closure rule, a 56.25% failure rate. Several parameters also exceed the predeclared normalized-error gate.
- All seven parameters remain `non-identifiable-under-this-protocol`, including parameters whose selected errors happen to pass the simple error thresholds. No fitted value receives mechanistic interpretation.

## Claim-evidence matrix

| Claim | Evidence | Outcome |
|---|---|---|
| Numerical and State validity | `metrics/metrics.json:full_state_checks`; `metrics/numerical_refinement.json` | Supported |
| Observed loss closure | `raw/fit_starts.json`; `raw/fitted_parameters.json` | Supported for all three starts |
| Active model improves over passive | `metrics/metrics.json:passive_trace_8_metrics` | Supported |
| Held-out consistency over 15-30 pA | `raw/per_start_predictions.npz`; `metrics/per_start_metrics.json` | Supported across all three starts |
| Individual parameter interpretation | `raw/recovery.json`; `raw/recovery_observations.npz`; `metrics/recovery_tradeoffs.json`; `raw/objective_profiles.json` | Withheld for all parameters |
| Canonical BrainTools Nevergrad parity | `../../artifacts/optimizer-boundary-iteration-2.json` | Unavailable because optional dependency is absent; valid SciPy boundary retained |

## Proposed next action

Submit iteration 2 and all unfavorable recovery evidence to the existing reviewer thread. Do not visualize and do not broaden the predictive or mechanistic claims.
