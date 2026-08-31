# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Decision | Confidence |
|---|---|---|---|---|---|---|---|
| `celegans_model.rollout` | transformed fixed time loop | `brainstate.transform.for_loop` owns all 10,000 State updates | T | High | Low | retain | High |
| `fit_and_validate.build_objective` | whole-loss state-aware JIT | one `brainstate.transform.jit` wraps parameter application, reset, rollout, and reduction | T/P | High | Low | retain | High |
| held-out evaluation | small host protocol loop | four independent protocols, each produces a required full trace | B | Low | Low | retain for clarity | High |
| data/metrics/artifacts | host NumPy and Python | outside transformed execution | none | Low | Low | retain | High |

## Benchmark

- Device: CPU (exact device recorded by the production run).
- Representative workload: Trace 8, 25 pA, 10,000 steps at 0.05 ms, nominal conductances.
- Cold compiled objective: 0.319962 s.
- Warm objective calls: 0.003623, 0.003199, 0.003087, 0.003099, 0.003007 s.
- Median warm objective: 0.003099 s.
- Compiled loss: 251.125443 mV^2.
- Direct rollout loss: 251.116987 mV^2.
- Absolute difference: 0.008455 mV^2; relative difference 3.37e-5.
- Prediction finite: yes.

## Decision and parity

No code rewrite was applied in step 3. The implementation already uses the highest-impact time-axis transform and a stable whole-objective JIT boundary. Mapping four final protocols would add State-axis complexity to a low-frequency path, and checkpointing is unnecessary after switching away from reverse-mode gradients. The small compiled/direct difference is accepted as floating-point transform ordering and is negligible relative to the objective.

## Remaining risks

- Each newly constructed held-out cell has a cold compile cost; this affects wall time, not the scientific result.
- Full voltage history is required for waveform metrics, so summary-only scan output is inappropriate.
- The derivative-free search avoids long-rollout gradient memory but cannot establish parameter identifiability.
