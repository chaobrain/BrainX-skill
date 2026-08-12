# BrainX diagnosis: memories replaying during sleep, Run 2

## Evidence studied

- Task: `brainx-display-cases/11-sleep-memory-replay/prompt.md`.
- Generated artifacts: `README.md`, `sleep_replay.py`, `results/sleep_replay_metrics.json`, `results/sleep_replay_summary.png`, `agent-final.md`, `codex-events.jsonl`, and `codex-stderr.log`.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`, `skills/brainpy-state/SKILL.md`, `skills/brainevent/SKILL.md`, `skills/brainstate/SKILL.md`, and `skills/brainunit/SKILL.md`.
- Routed references: BrainEvent synaptic plasticity; BrainPy-State component selection and Braintools input currents; BrainState vmap, control-flow, and collective-model lifecycle operations.
- Closest compositions: `skills/brainevent/references/scripts/coba_ei_teaching.py`, `skills/brainpy-state/references/scripts/103_COBA_2005.py`, and `skills/brainpy-state/references/scripts/106_COBA_HH_2007.py`.
- Authoritative API pages: BrainEvent `BinaryArray`, dense pre/post plasticity operations, BrainPy-State `LIFRef` and `Expon`, and BrainState `vmap2`, `vmap_init_all_states`, and `for_loop` generated references.
- Reproduction: a disposable-copy run exited 0 and reproduced the archived JSON and PNG byte-for-byte. JSON SHA-256 is `84da5534f9b536fe5a400d1c3541dd2181d5e740f19bc172e4ae42d1ab52a479`; PNG SHA-256 is `f0edce8df268cbf1b8018d993b8165fbcadd55e37f5e40ea5201c657763567bd`.
- Independent branch audit: all eight mapped State paths were pair-equal before sleep (`cells.V`, `cells.last_spike_time`, both traces, synaptic current, both sleep counters, and weight); the two external-current tensors were identical and all zero. Only the intended recurrent gate differed.
- Sensitivity audit over recurrent scale `0.90, 0.95, 1.00, 1.05, 1.10 nA`: complete replay counts were `0, 0, 5, 5, 5`; replay/suppressed recall scores were `0/0, 0/0, 76/28, 76/64, 76/72`.

## Executive diagnosis

The implementation is BrainX-native, deterministic, unit-safe, correctly mapped across the two causal lanes, and visibly clear. It uses the requested packages for their owning responsibilities, compiles each phase through BrainState, retains learned versus transient State deliberately, saves condition-level event and recall evidence, and uses exactly one simple `plt.subplots(...)` composition.

The scientific conclusion is not yet evaluation-clean. The agent tuned recurrent topology and current scale on the displayed run, then introduced the 25 ms composite recall score after observing that both conditions already completed ordered recall. The result is also sensitive at the calibrated recurrent scale. In addition, a positive regression slope is not the full predicate for `A -> B -> C -> D`, and gating all recurrent transmission establishes recurrence-pathway causality rather than isolating complete replay events as the mediator. These are scientific-claim problems, not BrainX API failures.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| High | `codex-events.jsonl`, items 55-69; `sleep_replay.py:50`, `sleep_replay.py:62`, `sleep_replay.py:352` | Topology and recurrent scale were calibrated on the displayed evaluation; the 25 ms latency score was defined after both groups showed 100% completion. | The reported `76%` versus `28%` is a post-outcome assay, not prespecified evidence. | Freeze model and assay before evaluation, or calibrate separately and report held-out/nearby sensitivity. Report completion and A-to-D latency directly; do not promote a post-hoc composite score as confirmatory evidence. |
| High | `sleep_replay.py:335` | Direction is assigned solely from the sign of a fitted slope. | A non-monotonic event can be mislabeled forward or backward. | Require the full strict order `A < B < C < D` for forward and `A > B > C > D` for backward; label other complete events non-monotonic or simultaneous as appropriate. |
| Medium | `sleep_replay.py:248`, `sleep_replay.py:541`; figure title and agent final | The intervention gates the entire recurrent pathway during sleep. Event occurrence agrees with outcome, but the manipulation is not event-specific. | “Replay consolidates” exceeds what the intervention alone isolates; recurrent transmission/activity-dependent strengthening is the directly tested cause. | State the causal conclusion at the intervention level unless an event-specific manipulation or independent event-dose analysis isolates replay as mediator. |
| Medium | `sleep_replay.py:537` | The generated mechanism check compares only place-A spike totals, although all relevant mapped State and protocols must match at the causal branch. | A future unequal State path could silently invalidate the matched control while the check still passes. | Save or assert the complete pre-branch State/protocol equality audit, then vary only the declared intervention. |
| Low | `sleep_replay.py:424`; summary figure | The sleep panel shows only the replay-enabled lane. | The suppression result is not visually inspectable in the main summary even though it is present in JSON. | Show both sleep conditions with the same simple subplot layout, or label the panel explicitly as one condition and surface the suppressed event count beside it. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Point-neuron dynamics | One `brainpy.state.LIFRef` population with explicit unit-aware parameters and initializer | BrainPy-State `LIFRef` | Correct | None. |
| Recurrent temporal filtering | Current-valued `brainpy.state.Expon` with an `nA` initializer | BrainPy-State `Expon` | Correct and unit-safe | None. |
| Binary spike transmission | Previous spikes wrapped in `brainevent.BinaryArray` and multiplied by dense weighted connectivity | BrainEvent event boundary | Correct for this small dense network | None. |
| Persistent learned efficacy | `brainstate.LongTermState` | BrainState State roles | Correct | None. |
| STDP traces and sleep counters | `brainstate.ShortTermState` | BrainState State roles | Correct | None. |
| Event-triggered plasticity | Both dense post-triggered potentiation and pre-triggered depression operators | BrainEvent dense plasticity operations | Correct bidirectional trigger composition | Keep the full route-order scientific check separate from the operator mechanics. |
| Physical parameters | BrainUnit time, voltage, current, resistance, and unit-aware exponential decay | BrainUnit quantities and `u.math` | Correct | None. |
| Learning/recall protocols | Braintools `Constant` plus `u.math.concatenate`/`broadcast_to` | Braintools input and BrainUnit array mechanics | Correct time-major protocols | None. |
| Learning execution | One jitted `brainstate.transform.for_loop` | BrainState transforms | Correct | None. |
| Independent matched conditions | `vmap_init_all_states` and filter-based `vmap2` around the complete per-step transition | BrainState lifecycle and mapping | Correct; all eight State lanes independently verified | Persist the equality audit in the artifact. |
| Sleep execution | One jitted time loop with zero external current | BrainState `for_loop`/`jit` | Correct | None. |
| Recall boundary | Fresh mapped Module, then direct restoration of the single post-sleep weight State | BrainState lifecycle; direct State assignment | Correct and simpler than model-wide restoration for one State | Keep the shape check; no `assign_state_values` requirement is justified here. |
| Event decoding and recall statistics | NumPy host analysis after the simulation | Legitimate host boundary | Mechanically appropriate; predicates and assay timing are scientifically flawed | Fix the decision rules rather than forcing them into BrainX. |
| JSON persistence | Standard-library serialization | Legitimate host boundary | Correct and complete enough to reconstruct labels | Add pre-branch match evidence and calibration/sensitivity evidence. |
| Figure composition | One `plt.subplots(2, 2, ...)` call and basic `imshow`, `scatter`, `step`, `bar`, labels, and colorbar | High-level Matplotlib host boundary | Meets the absolute-simple-composition rule; 1980 x 1440 output is legible | Surface the sleep control without adding plotting infrastructure. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused. The direct assignment of `recall_net.weight.value` is appropriate because exactly one known State is restored and its shape is checked; model-wide `assign_state_values` would add bookkeeping without improving this case. NumPy decoding, JSON writing, and Matplotlib remain legitimate host boundaries.

The weaknesses are evaluation design and claim predicates. They should not be “fixed” by inventing a BrainX metric or adding lower-level infrastructure.

## Performance and code simplicity

- Learning, sleep, and recall each compile one logical State-aware rollout; there is no Python timestep loop.
- The matched causal groups run through one `vmap2` transition inside one time loop, with independent mapped neural, synaptic, trace, counter, and weight State.
- The small dense matrix is a reasonable plasticity representation; sparse storage would increase code without material benefit at 20 cells.
- Unit conversion occurs only at host analysis and display boundaries.
- The implementation has one Module and one direct phase flow. Helper functions correspond to protocols, assays, or output boundaries rather than abstraction layers.
- The figure obeys the requested absolute Matplotlib simplicity: exactly one `plt.subplots(...)` call and no `GridSpec`, manual axes, custom artists, projection axes, or style framework.

## Skill improvements

Make only a compact `brainx-general-guard` refinement and mirror it in `plan.md`:

1. Prohibit defining a metric, composite score, threshold, window, or displayed case after viewing intervention outcomes; require separate calibration or explicit held-out/nearby sensitivity.
2. Require the full stated sequence predicate for direction labels; a regression sign or proxy summary is insufficient.
3. Require causal wording at the intervention level unless a mediator-specific manipulation plus event-outcome concordance isolates the named event.
4. Say to verify and save all relevant State/protocol equality at a matched causal branch.

Do not edit BrainPy-State, BrainEvent, BrainState, or BrainUnit package skills from this run: their existing routing produced correct API selection and lifecycle composition.

## Checks for the next run

- The model, seeds, event definition, recall metric, thresholds/windows, and displayed cases are frozen before evaluation, or calibration uses separate data and reports held-out/nearby sensitivity.
- Forward requires `A < B < C < D`; backward requires `A > B > C > D`; all other complete orders receive a distinct label.
- The reported causal claim matches the recurrent-gate intervention unless replay mediation is independently isolated.
- Every relevant State path and protocol input is equal and saved before the two conditions branch; only the intended intervention differs.
- Condition-level replay events, weight changes, recall components, aggregation, and sensitivity evidence are preserved.
- The run still uses `LIFRef`, `Expon`, `BinaryArray`, both BrainEvent plasticity triggers, State roles, `for_loop`, and a complete mapped condition transition.
- The figure remains one `plt.subplots(...)` composition with only basic high-level Matplotlib methods and visibly includes or summarizes both conditions.
