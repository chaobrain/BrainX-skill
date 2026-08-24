# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| `simulate()` | transformed time loop | 5,000 updates execute through `brainstate.transform.for_loop` inside one `brainstate.transform.jit` rollout | T | High benefit already present | Low | Retain | High |
| `CelegansMuscleCell` | native candidate batch | `SingleCompartment.size` is the candidate count and all parameter/current arrays are lane-aligned | E | High benefit already present | Low | Retain; no extra `vmap` | High |
| `summarize_traces()` | host lane loop | Summary extraction runs after device-to-host conversion and includes variable-length spike times | E | Low relative to rollout | Low | Retain host analysis | High |
| `run_parameter_recovery()` | sequential fit loop | Each recovery case launches an independent full ABC fit | E | Moderate | High memory if mapped | Retain sequential cases | High |
| `simulate()` construction | repeated JIT closure | A new independent BrainCell is required for each candidate batch and State reset boundary | E | Moderate compile cost | High if parameters become mutable shared State | Retain for lifecycle clarity | Medium |

## Patch / rewrite plan

The implementation already uses the BrainState control-flow and JIT patterns routed by the acceleration skill. Candidate independence is represented by BrainCell's native `size` axis, so an additional stateful `vmap` would duplicate ownership and increase State-axis risk. No performance rewrite is accepted for iteration 1.

The implementation patch adds a variable unit-bearing `dt` argument only to support numerical parity evidence; it does not alter the 0.1 ms production path.

## Validation plan

- Expanded tests compare two batched candidate lanes against separate one-lane runs at `rtol=atol=1e-5`.
- A 64-lane, 5,000-step deterministic benchmark on CPU produced shape `(5000, 64)`: first call 6.210 s, repeated call 4.210 s, maximum absolute difference 0 mV.
- The smoke fit's 0.1 ms production rollout was compared with a 0.05 ms reference. RMSE was 0.230 mV, maximum instantaneous difference 2.952 mV, and all four protocol spike counts matched.
- The production report repeats parity checks using the selected fitted parameters.

## Remaining risks

- Constructing a new BrainCell/JIT closure for each independent ABC round retains several seconds of compile/dispatch overhead on CPU.
- Mapping multiple full recovery fits would multiply full-history memory and complicate State ownership; they remain sequential.
- Full voltage histories are required for the declared waveform metrics and trace artifacts, so summary-only scanning is not applicable.
- The run is CPU-only; no GPU acceleration claim is made.
