# BrainX acceleration and parity audit

| File/area | Pattern | Evidence | Decision |
|---|---|---|---|
| Global objectives | State-aware JIT | BrainTools calls a `brainstate.transform.jit` objective that constructs the BrainCell channel, resets its gate at -60 mV, writes the exact step solution, and reads `channel.current()`. | Keep; this is the production scientific path. |
| Voltage/time rollout | Vectorized batch/time axes | Every candidate evaluates all retained voltages and sampled times in one array expression with fixed shapes. | Keep; no Python time loop. |
| Fitting grid | Deterministic temporal reduction | Optimization uses 0.5 ms spacing, while full 0.1 ms SHK and 0.02 ms EGL histories remain available for metrics and plots. Kinetics are multi-millisecond, and final evaluation is full resolution. | Keep and record in the specification. |
| Experiment and recovery sweeps | Small host loops | Three final centers, two validation centers, voltage holdouts, and domain truths have different target arrays or shapes and must retain separate diagnostics. | Keep explicit for auditability. |
| Analytic helpers | Post-fit reporting path | Closed-form NumPy helpers are faster for simple analysis, but production objectives and saved predictions use BrainCell. | Require parity tests at nominal and near-boundary parameters. |

## Validation

- Compare analytic and BrainCell rollouts at fitted and near-boundary parameter sets with `rtol=2e-5`, `atol=3e-3 pA`.
- Test `init_state()` and `reset_state()` explicitly at the -60 mV holding potential.
- Preserve units through BrainCell current evaluation and convert to pA only at the observed-data boundary.
- Preserve full-resolution overlays, gate summaries, and errors after fitting on the reduced grid.

## Remaining performance boundary

BrainTools dispatches a host SciPy optimizer around compiled BrainState objectives. Compilation dominates short validation fits, while fitting every calcium sample would add substantial repeated work without increasing the temporal information relevant to the fitted time constants. Further acceleration would require changing the public optimizer interface or batching distinct optimization problems, neither of which is justified for this evidence-sized case.
