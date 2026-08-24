# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| `cellegans_hh/model.py:simulate` | transformed time loop | 5,000 membrane/channel/calcium updates per rollout use `brainstate.transform.for_loop` | T | High | Low | Keep the complete BrainCell update inside the transformed loop | High |
| `cellegans_hh/inference.py:InferenceProblem._losses` | candidate ensemble | Differential evolution evaluates 28 independent parameter candidates per generation | E | High | Medium | Use `SingleCompartment.size=(E,)` so every candidate owns independent voltage, gate, and calcium State | High |
| Offline spike metrics | host analysis | SciPy peak finding loops over completed voltage histories | E | Low | Low | Keep on host; it is discontinuous and outside the transformed simulation hot path | High |
| Production starts/recovery cases | causal host loop | Each start/case must retain a separate seed, log, and reset boundary | runs | Medium | Low | Keep sequential for immutable artifact identity and stable candidate shape | High |

## Rewrite decision

The implementation already applies the two supported high-impact changes: time is owned by `brainstate.transform.for_loop`, and candidates are represented by the BrainCell native `size` axis. No `vmap` is added because the owning package already supplies independent batched State. No gradient transform is added because the locked objective includes discontinuous spike detection and uses bounded differential evolution.

## State and shape contract

- Shared across candidate lanes: fixed reversal potentials, gate kinetic constants, stimulus waveform, integration step, and source data.
- Independent across lanes: voltage, spike State, all HH gates, intracellular calcium, and conductance/capacitance candidate values.
- Stable production shape: seven optimizer coordinates and 28 candidates (`popsize=4`) for every generation.
- Randomness: simulations are deterministic; only the host optimizer population uses the recorded start seed.

## Validation plan

- Compare a fixed eight-candidate batched run against eight scalar runs at every time point with 1e-5 mV absolute tolerance.
- Report cold time separately from warmed repetitions.
- Confirm output shapes and finite values.
- Retain the scalar/batch parity check in `tests/test_model.py`.

## Remaining risks

The complete voltage history is required by the waveform objective and spike detector, so its memory scales as `T x E`. Candidate batch shape changes would trigger recompilation; production fixes `popsize`. The optimizer and SciPy peak scoring remain host-side by design.
