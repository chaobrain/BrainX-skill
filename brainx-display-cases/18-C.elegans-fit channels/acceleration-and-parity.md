# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Decision | Confidence |
|---|---|---|---|---|---|---|---|
| `fit_channels.py` trace simulation | `already-fused` | All voltages and time samples are broadcast in one NumPy operation per objective call. | B/T | High | Low | Keep vectorized pure functions. | High |
| SciPy optimizer | host black-box boundary | `least_squares` owns iterations and repeatedly calls a host NumPy residual. | P | High | Medium | Do not wrap in BrainState/JAX transforms. | High |
| SHK per-voltage extraction | small host loop | Six independent two-parameter robust fits run once. | B | Low | Low | Keep explicit loop for inspectable per-voltage evidence. | High |
| EGL multi-start selection | small host loop | Three deterministic starts run once and the finite minimum-cost result is selected. | E | Medium | Low | Keep explicit loop; candidate semantics remain independent. | High |
| EGL gate-point extraction | small host loop | Seven post-fit local summaries run once. | B | Low | Low | Keep explicit loop because it is analysis, not the objective hot path. | High |

## Patch / rewrite decision

No acceleration rewrite is warranted. The scientific model is small, deterministic, and vectorized over voltage and time. The remaining loops are optimizer control, independent starts, or post-fit evidence extraction. Moving them into BrainState transforms would not remove SciPy's host boundary and would complicate parameter and failure bookkeeping.

## Validation

- Four deterministic tests passed in 32.922 seconds on CPU.
- Vectorized prediction shapes exactly match observations: SHK-1 `(6, T)` and EGL-19 `(7, T)`.
- Gate State initialization, derivative direction, current units, and reversal behavior pass through the BrainCell channel classes.
- No RNG enters the objective; `brainstate.random.seed(20260825)` fixes any future BrainState initialization.
- Full histories are required for the requested overlays and residual metrics, so summary-only execution would discard required evidence.

## Remaining risks

- Robust least squares is CPU-bound and compile acceleration is not applicable at the host optimizer boundary.
- EGL-19 parameter tradeoffs remain possible even with a good waveform fit; synthetic recovery and claim classification must gate interpretation.
