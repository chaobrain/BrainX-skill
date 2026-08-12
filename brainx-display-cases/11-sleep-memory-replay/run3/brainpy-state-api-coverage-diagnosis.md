# BrainX diagnosis: memories replaying during sleep, Run 3

## Evidence studied

- Task: `brainx-display-cases/11-sleep-memory-replay/prompt.md`.
- Generated artifacts: `README.md`, `sleep_replay.py`,
  `sleep_replay_results.json`, `sleep_replay.png`, `agent-final.md`,
  `codex-events.jsonl`, and `codex-stderr.log`.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`, `skills/brainevent/SKILL.md`,
  `skills/brainstate/SKILL.md`, and `skills/brainunit/SKILL.md`.
- Routed references: BrainEvent synaptic plasticity; BrainPy-State component
  selection and input currents; BrainState vmap, control-flow, State graph, and
  collective-model lifecycle operations.
- Closest compositions: `skills/brainevent/references/scripts/coba_ei_teaching.py`,
  `skills/brainpy-state/references/scripts/103_COBA_2005.py`, and
  `skills/brainpy-state/references/scripts/106_COBA_HH_2007.py`.
- Authoritative API pages: BrainEvent `BinaryArray` and dense pre/post
  plasticity operations; BrainPy-State `LIFRef` and `Expon`; BrainState
  `vmap2`, `vmap_init_all_states`, `for_loop`, and State collection.
- Reproduction: a disposable-copy run exited 0 and reproduced the archived
  JSON and PNG byte-for-byte. JSON SHA-256 is
  `178297ab933aee9b728f733f5550bc3166e5597638d0cb176d452500b0c8680b`;
  PNG SHA-256 is
  `ddbbb387c48486b7a89a7e9eede36c1febc639bf49a0185ccb4e1f10d39289df`.
- Independent branch audit: all six mapped State paths were pair-equal before
  sleep: `cells.V`, `cells.last_spike_time`, `pre_trace`, `post_trace`,
  `recurrent_filter.g`, and `weight`. Both external-current lanes were equal
  and exactly zero, and both plasticity gates were equal. Only the intended
  recurrent gate differed (`1` versus `0`).
- Visual inspection: the 1800 x 1260 figure is legible, uses matched sleep
  axes, shows the boundary seed, and exposes both intervention conditions.

## Executive diagnosis

Run 3 is an acceptable endpoint. It is BrainX-native, deterministic,
unit-safe, and correctly maps two independent causal lanes through the complete
per-step transition. The agent calibrated the baseline while the lanes were
identical, froze its model and metrics, revealed suppression by changing only
the sleep recurrent gate, and preserved the resulting comparison without
retuning recall. It also replaced the earlier regression proxy with the full
strict route-order predicate and keeps the causal description at the recurrent
transmission intervention level.

The remaining limitations do not justify another skill edit. The baseline was
still calibrated on the displayed seed and regime without held-out or nearby
sensitivity, and the generated JSON does not persist every raw pre-branch State
or protocol tensor. The independent review establishes the match for this run,
but the artifact itself asserts only pre-sleep weight equality. The current
general guard already directs agents to separate calibration or report
sensitivity and to save complete branch evidence, so another wording change
would duplicate existing guidance.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| Medium | `codex-events.jsonl`, baseline calibration; `sleep_replay.py:40-45` | The recurrent scale and learning constants were calibrated on the displayed deterministic seed and regime, with no held-out or nearby sensitivity result. | The demonstration establishes the reported outcome for this frozen setup but does not establish robustness beyond it. | Treat the result as a calibrated phenomenological demonstration; for a stronger claim, calibrate separately and report held-out or nearby sensitivity without changing the displayed assay afterward. |
| Low | `sleep_replay.py:285-288`; `sleep_replay_results.json` | The artifact checks only pre-sleep weight equality and saves summary evidence rather than all pre-branch State paths and protocol tensors. | A reviewer can independently verify the branch, but the saved output alone cannot reconstruct the complete equality audit. | In future causal artifacts, enumerate, compare, and save every relevant State path and protocol input at the branch. |
| Low | `sleep_replay.py:173-182`; `sleep_replay_results.json` | Recall scoring intentionally excludes place A after the known A cue and scores strict B-C-D completion; replay therefore reports A as `null` while the suppressed lane later reports a spontaneous A event. | A reader could mistake the `null` for missing cue delivery rather than exclusion of the known cue from the completion assay. | Keep the metric, but name or document it explicitly as post-cue B-C-D completion when interpretation depends on the distinction. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Point-neuron dynamics | One unit-aware `brainpy.state.LIFRef` population with an explicit voltage initializer | BrainPy-State `LIFRef` | Correct | None. |
| Recurrent temporal filtering | Current-valued `brainpy.state.Expon` | BrainPy-State `Expon` | Correct and unit-safe | None. |
| Binary spike transmission | Previous spikes wrapped in `brainevent.BinaryArray` and multiplied by dense weights | BrainEvent event boundary | Correct for this small dense network | None. |
| Persistent learned efficacy | Dense weights in `brainstate.LongTermState` | BrainState State roles | Correct | None. |
| Transient learning traces | Pre- and postsynaptic traces in `brainstate.ShortTermState` | BrainState State roles | Correct | None. |
| Event-triggered plasticity | Dense post-triggered potentiation plus pre-triggered depression | BrainEvent dense plasticity operations | Correct bidirectional trigger composition | None. |
| Physical quantities | BrainUnit time, voltage, current, resistance, and unit-aware decay | BrainUnit quantities and `u.math` | Correct | None. |
| Wake and recall protocols | Complete time-major unit-aware arrays constructed before rollout | BrainUnit array mechanics; legitimate host protocol construction | Correct | Persist complete branch protocol evidence in stronger causal artifacts. |
| Independent matched conditions | `vmap_init_all_states` plus filter-based `vmap2` around the complete transition | BrainState lifecycle and mapping | Correct; all six mapped State lanes were independently verified | Save the path-level equality audit in the generated artifact. |
| Phase execution | One `brainstate.transform.for_loop` for each wake, sleep, and recall phase | BrainState transformed control flow | Correct | None. |
| Replay decoding | Exact onset extraction and strict `A < B < C < D` predicate | Legitimate NumPy host analysis | Correct | None. |
| Recall assay | Strict B-C-D prefix and completion latency after the known A cue | Legitimate NumPy host analysis | Correct but narrowly named | Make the post-cue boundary explicit in reporting. |
| JSON persistence | Standard-library serialization of condition summaries | Legitimate host boundary | Correct but not complete branch evidence | Save State/protocol equality and sensitivity evidence when required by the claim. |
| Figure composition | Exactly one `plt.subplots(2, 2, ...)` call with basic `scatter`, `bar`, labels, and legends | High-level Matplotlib host boundary | Correct, simple, and legible | None. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused. NumPy event decoding, JSON
serialization, and Matplotlib plotting are legitimate host boundaries. The
remaining issues concern scientific-evaluation evidence and reporting, not an
unselected BrainX abstraction.

## Performance and code simplicity

- Wake, sleep, and recall each use one State-aware transformed time loop; there
  is no Python timestep loop.
- Both causal lanes run through one `vmap2` transition with independently
  mapped neural, synaptic, trace, and weight State.
- Dense storage is appropriate for the 24-cell plastic network; sparse storage
  would add complexity without a material benefit.
- Unit conversion occurs only at analysis boundaries.
- Helpers represent protocols, assays, or output boundaries rather than an
  unnecessary framework.
- The figure uses absolutely simple Matplotlib composition: one
  `plt.subplots(...)` call and only basic high-level plotting methods.

## Skill improvements

Do not make another skill edit from Run 3. The current compact
`brainx-general-guard` refinement already covers its residual risks: freeze or
separately calibrate claims, save complete causal-branch evidence, apply the
full scientific predicate, identify external seeds, preserve categorical
evidence, and compose figures with absolutely simple Matplotlib code.

Do not edit BrainPy-State, BrainEvent, BrainState, or BrainUnit package skills:
their existing routing produced correct API selection, State lifecycle,
execution structure, units, and event-driven plasticity.

## Acceptance

Accept Run 3 and stop the Sleep Replay refinement. It materially improves on
Run 2 by freezing the intervention comparison before revealing suppression,
preserving the unretuned recall result, using exact sequence order, naming the
boundary seed, matching causal wording to recurrent transmission, showing both
conditions, and retaining a single simple Matplotlib composition.

Carry these checks into the next display case:

- Freeze parameters, seeds, metrics, thresholds, windows, and displayed cases
  before intervention outcomes, or separate calibration and report sensitivity.
- Verify and save every relevant State path and protocol input at a causal
  branch, then vary only the declared intervention.
- Apply the full claim predicate and preserve categorical evidence.
- Keep figures to one absolutely simple `plt.subplots(...)` composition unless
  the prompt explicitly requires a capability that it cannot express.
