# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| `BrunelLIF.update` | `already-fused` | One population-shaped leak, refractory mask, threshold, and reset; no scalar neuron loop | N | High | Low | Keep unchanged | High |
| `SparseEINetwork.update` | `already-fused` | Two `BinaryArray @ FixedNumPerPost` event products plus one population Poisson draw | N | High | Low | Keep unchanged | High |
| `build_runner` | Correct stable JIT boundary | One fixed-length State-aware `for_loop` is wrapped by one `brainstate.transform.jit` | T | High | Low | Keep unchanged | High |
| Panel loop | `loop-E-sweep` | Four independent parameter points, each with independent dynamical/RNG State and a 312.5 MB production spike history | E | Medium | High | Keep sequential; reset State and reduce each history immediately | High |
| Connectivity sampling | Host static setup | Exact unique fan-in and autapse exclusion are sampled once before transforms | N | Low | Low | Keep on host for inspectability | High |
| ISI analysis | Host offline analysis | Full spike history is required for per-neuron ISI CV and is consumed after the compiled run | N,T | Medium | Medium | Keep outside transforms; delete each history after reduction | High |

## Patch / rewrite plan

No acceleration rewrite is justified. Preserve exact fixed-fan-in BrainEvent storage, population State, and the complete compiled time loop. Do not map the four panels because independent delay, voltage, refractory, spike, and RNG State plus four full histories would multiply peak memory without changing the scientific workload.

The transform design follows the BrainState control-flow reference: `for_loop` owns time because iteration effects live in Module State, and one enclosing State-aware JIT owns the stable rollout. NumPy remains confined to static topology construction and post-run analysis.

## Validation plan

- Focused suite: `9 passed in 31.16s` on CPU.
- Representative workload: 800 E plus 200 I neurons, exact indegrees 80/20, 100 ms total, seed 1729, `(g, eta)=(5,2)`.
- Cold execution: 25.5102 s, including trace and compile.
- Warm execution after complete State/RNG reset: 2.5236 s.
- Output: shape `(1000, 1000)`, 8,793 spikes.
- Parity: spike histories are bit-identical and final membrane-voltage maximum absolute difference is `0.0 mV`.
- Production history size: `25,000 * 12,500` Boolean values, approximately 312.5 MB per panel, reduced and released sequentially.

## Remaining risks

- CPU-only production is long-running and sparse event cost depends on activity.
- Exact graph sampling uses a host loop once per source population; it is not on the simulation hot path.
- The full spike history is the main result buffer, but it is required by the locked per-neuron ISI-CV observable.
- Mapping panels or retaining several histories would unnecessarily increase peak memory.
