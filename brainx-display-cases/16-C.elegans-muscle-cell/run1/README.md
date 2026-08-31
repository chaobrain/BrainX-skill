# C. elegans muscle-cell HH inference

This run fits a BrainCell single-compartment conductance model to Trace #8
(25 pA) from `Fig4A-D.txt` and evaluates unchanged parameters on Trace #6
(15 pA), Trace #7 (20 pA), and Trace #9 (30 pA).

The model represents the six explicitly named currents: SHK-1, EGL-19, SLO-2,
Kr, Na, and leak. Because the supplied data do not include the reference
paper's supplementary equations, recording geometry, or a current-monitor
channel, this is a phenomenological HH model rather than an exact biological
reproduction.

## Accepted result

Iteration 2 completed on CPU in 1,964.147 seconds and received reviewer `PASS`.
All three observed-data starts met the locked loss-closure rule and independently
passed the held-out spike-count, latency, and monotonicity criteria. The selected
seed-2025 objective was 6.664992, improving on iteration 1 by 0.302275.

| Trace | Current | Split | RMSE (mV) | Observed / predicted spikes | First-spike error (ms) |
|---|---:|---|---:|---:|---:|
| #6 | 15 pA | held out | 10.579 | 3 / 3 | 4.8 |
| #7 | 20 pA | held out | 8.257 | 3 / 4 | 3.3 |
| #8 | 25 pA | fit | 6.206 | 4 / 4 | 4.9 |
| #9 | 30 pA | held out | 9.368 | 4 / 5 | 7.4 |

The accepted scientific claim is limited to prediction under the tested
15-30 pA protocol and supplied recording series. It does not establish the
biological fidelity of the channel kinetics or fitted parameter values.

## Parameter boundary

The selected parameter vector is retained as a predictive fit, but none of its
seven values receives mechanistic interpretation. Twenty-seven of 48 synthetic
recovery starts failed the closure rule, so all parameters remain
`non-identifiable-under-this-protocol`. The reviewer also noted that recovery
reused Trace #8's initial voltage rather than recomputing it from each noisy
synthetic observation; treat recovery as an approximate diagnostic.

## Key artifacts

- `NeuroSpecification.md`: locked data split, protocol, controls, and claim boundary.
- `brainmodeling-memory.md`: append-only modeling-loop record through reviewer acceptance.
- `reviews/iteration-2.md`: verbatim reviewer `PASS` report.
- `runs/20260824-celegans-production-02/`: immutable production configuration, provenance, logs, raw outputs, metrics, and manifest.
- `runs/20260824-celegans-production-02/RESULT_ASSESSMENT.md`: deterministic result and claim-evidence matrix.
- `artifacts/test-results.txt`: seven passing focused checks.

Visualization was intentionally not run.
