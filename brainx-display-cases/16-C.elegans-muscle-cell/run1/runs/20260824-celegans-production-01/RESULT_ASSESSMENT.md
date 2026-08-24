# Deterministic result assessment

## Mechanical outcome

- Status: done, exit code 0, CPU backend `cpu:0`, duration 120.117 s.
- All declared JSON and NPZ artifacts parse; all prediction arrays are finite.
- Lower- and upper-bound runs are finite; the zero-current control has no detected spikes.

## Training and control

- Selected optimizer start: seed 2025, objective 6.967267.
- Trace #8 / 25 pA: 6.563 mV RMSE, correlation 0.875, 4 observed versus 4 predicted spikes, 3.7 ms first-spike error.
- Passive Trace #8 control: 17.179 mV RMSE and zero spikes.
- The active model beats the passive control on the predeclared training RMSE comparison.

## Held-out prediction

| Trace | Current | RMSE (mV) | Correlation | Observed / predicted spikes | First-spike error (ms) |
|---|---:|---:|---:|---:|---:|
| #6 | 15 pA | 10.541 | 0.412 | 3 / 0 | unavailable |
| #7 | 20 pA | 8.443 | 0.682 | 3 / 3 | 0.9 |
| #9 | 30 pA | 9.742 | 0.648 | 4 / 5 | 6.5 |

Two of three held-out traces pass the spike-count and available latency thresholds, but the predicted spike counts `[0, 3, 4, 5]` do not match the requested consistency at 15 pA and first-spike monotonicity is undefined there. The locked overall result is `not-supported-under-locked-criteria`.

## Parameter recovery

- Recovery-supported under the predeclared normalized-error gate: `g_shk1`, `g_kr`, `g_na`, and `capacitance`.
- Non-identifiable under this protocol: `g_egl19`, `g_slo2`, and `g_leak`.
- Recovery is based on three noisy exact-pipeline synthetic truths and is evidence about this implementation/protocol, not proof of biological uniqueness.

## Claim-evidence matrix

| Claim | Evidence | Outcome |
|---|---|---|
| Numerical validity | `metrics/metrics.json`, `raw/predictions.npz` | Supported |
| Training fit | `metrics/metrics.json:trace_metrics.8`, `raw/fitted_parameters.json` | Supported |
| Active model improves over passive | `metrics/metrics.json:passive_trace_8_metrics` | Supported |
| Held-out consistency over 15-30 pA | `metrics/metrics.json:prediction_assessment` | Not supported |
| Individual parameter interpretation | `raw/recovery.json`, `metrics/metrics.json:parameter_recovery` | Mixed; limited to four parameters in this synthetic gate |

## Proposed next action

Submit the complete unfavorable iteration for independent review. Do not visualize or retune until the review verdict identifies whether the miss at 15 pA is an implementation defect, an underpowered fitting contract, or a valid scientific failure under the locked one-trace design.
