# BrainX acceleration audit: iteration 2

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| `cellegans_hh/model.py:simulate` | transformed time loop | BrainCell update runs inside `brainstate.transform.for_loop` | T | High | Low | Keep unchanged | High |
| `cellegans_hh/inference.py:InferenceProblem.loss_components` | native candidate ensemble | 28 candidates own independent voltage, gates, calcium, and physical parameters through `SingleCompartment.size` | E | High | Medium | Keep native batching | High |
| Host spike/objective reduction | discontinuous offline scoring | SciPy peak finding and objective bookkeeping consume completed traces | E | Low | Low | Keep outside transformed dynamics | High |
| Starts and recovery cases | immutable sequential runs | Each start/case has a distinct seed and artifact record | runs | Medium | Low | Keep sequential | High |

## Rewrite decision

No acceleration rewrite is scientifically warranted. The high-impact time and candidate axes are already BrainX-native and shape-stable. The iteration-2 additions are offline evidence collection or optional State returns and do not alter the production voltage-only fitting hot path.

## Validation

- Eight candidates x 1,000 steps: batched and scalar voltage differ by at most 3.814697e-6 mV.
- All returned gate and calcium trajectories preserve scalar/batch parity; the largest non-voltage absolute difference is 8.940697e-8.
- Warm batching is 8.26x faster than eight serial scalar rollouts on the recorded CPU environment.
- The 28-candidate SciPy boundary equals direct BrainCell objective evaluation exactly and returns finite one-loss-per-candidate output.
- Canonical `braintools.optim.NevergradOptimizer` construction reports the optional `nevergrad` dependency unavailable. No dependency was installed; this does not invalidate the current SciPy backend.

## Remaining risks

Full voltage histories remain necessary for the locked waveform and spike objective. The BrainTools Nevergrad implementation cannot be benchmarked in this environment, so the iteration retains the already validated SciPy optimizer boundary.
