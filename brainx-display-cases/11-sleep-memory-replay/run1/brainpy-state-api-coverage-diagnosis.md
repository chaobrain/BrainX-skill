# BrainX diagnosis: sleep memory replay Run 1

## Evidence studied

- Generated artifacts: `README.md`, `sleep_replay.py`, `test_sleep_replay.py`,
  `sleep_replay_results.png`, `sleep_replay_evidence.npz`, `agent-final.md`,
  `codex-events.jsonl`, `codex-stderr.log`, and `harness-metadata.txt`.
- Execution: the archived program was copied to a disposable directory and run
  with the fixed BrainX virtualenv and CPU JAX backend. All four test functions
  passed; the script reproduced two forward and zero backward detected events,
  zero control events, recall means `1.000` versus `0.333`, and byte-identical
  figure and evidence files.
- Independent audits: every State and protocol input at the sleep branch was
  compared within all eight matched pairs; detector and recall thresholds were
  varied; sleep seeds `0` through `15` were run without changing the model; and
  the rendered figure was inspected directly.
- Owning skills and references: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`, `skills/brainevent/SKILL.md`,
  `skills/brainstate/SKILL.md`, `skills/brainunit/SKILL.md`,
  `skills/brainevent/references/synaptic-plasticity.md`,
  `skills/brainpy-state/references/projection-patterns.md`,
  `skills/brainstate/references/collective_model_operations.md`,
  `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`,
  and `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md`.
- Closest executable examples:
  `skills/brainevent/references/scripts/coba_ei_teaching.py` for event-driven
  BrainPy communication and
  `skills/brainpy-state/references/scripts/sound_localization.py` for
  `vmap_init_all_states` + `vmap2` + `for_loop` composition.
- Authoritative API pages: BrainEvent `BinaryArray`, operations overview,
  `update_dense_on_binary_pre`, and `update_dense_on_binary_post`; BrainPy-State
  `LIFRef`, `Expon`, and `COBA`; BrainState `vmap2`, `for_loop`,
  `vmap_init_all_states`, and the Collective Operations guide.

## Executive diagnosis

Run 1 is a substantial implementation improvement. It uses the requested
BrainX packages coherently, maps all independent lanes through one stateful
transition, preserves units, applies bidirectional dense STDP through BrainEvent,
resets fast State between phases, saves per-lane evidence, and composes the
figure with one simple `plt.subplots(...)` call. The output is deterministic,
readable, and reproducible.

The central scientific conclusion is nevertheless too strong. A precomputed
stochastic current delivers `0.90 nA` three-step pulses to complete place
assemblies, so the sleep activity is externally seeded or evoked, not
endogenous. More importantly, the intervention disables all excitatory
recurrent transmission during sleep, while complete route events occur in only
two of eight enabled lanes. All eight enabled lanes still achieve perfect
recall, and five of sixteen unseen seeds produce no detected complete replay
while retaining a large recall advantage. The experiment therefore supports a
causal effect of recurrent sleep activity/plasticity on recall, not a causal
effect of detected complete route replay specifically.

The detector was changed to a `22 ms` gap after the target cascade was
inspected, the `50%` threshold is exactly the boundary at which events exist,
the recurrent gain and sleep-burst regime were selected on the displayed data,
and the plotted lane was selected because it contained an event. Recall is
robust to nearby activity thresholds, but the replay label lacks independent
calibration, held-out validation, and event-to-outcome concordance.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| High | `sleep_replay.py:198-216`; module docstring | Random, externally constructed assembly pulses are passed into the neuron as current but called intrinsic and endogenous. Random timing and the absence of an ordered route do not make supplied drive internal to the model. | The output overstates spontaneous or endogenous replay. | Call the activity externally seeded or evoked, or model the initiating fluctuation as internal network State without an external assembly-selective pulse. |
| High | `sleep_replay.py:364-371`, `393-431`; `README.md:17-21` | The intervention gates all excitatory recurrent transmission, not detected complete replay. Only lanes 0 and 12 contain complete events, yet every enabled lane has recall `1.0`; unseen seeds 2, 4, 5, 9, and 11 contain no complete events but retain a `+0.50` to `+0.67` recall advantage. | The causal result cannot be attributed specifically to complete forward/backward replay. Subsequence propagation or general recurrently driven plasticity is a sufficient alternative. | Either describe the result as recurrent-sleep-transmission causality, or use an event-specific perturbation and show per-lane replay occurrence/dose predicts later recall under a prespecified assay. |
| High | `codex-events.jsonl` items 67-76, 88-95; `sleep_replay.py:54`, `263-314` | Gain, sleep drive, detector gap, recall current, and displayed operating point were selected after inspecting the same outcomes. No held-out calibration set or prespecified source justifies them. | The displayed regime and qualitative event labels are circularly calibrated. | Freeze parameters and assays before evaluation, or calibrate on separate seeds/conditions and report held-out results plus sensitivity. Keep the phenomenological label. |
| Medium | `sleep_replay.py:263-296` | Replay detection exists only at assembly thresholds at or below `0.50`; raising the threshold to `0.51` removes both events. One event is found with gaps up to `16 ms`, while the second appears only at `22 ms`. | The count `2` is fragile to the exact post-outcome detector boundary. | Report the event count over a prespecified threshold/window range or establish the detector on synthetic/held-out data before the evaluated run. |
| Medium | `test_sleep_replay.py:31-39`; `README.md:17-18` | The artifact asserts matched current and weights but claims sleep inputs and neural State are matched. The independent audit found all seven State paths pair-equal, so the model is matched, but the shipped tests do not prove the full claim. | A later State addition or protocol change could silently invalidate the causal branch. | Snapshot and compare every State by absolute path plus every protocol tensor immediately before the branch; use `assign_state_values` and reject missing or unexpected paths for exact restoration. |
| Medium | `sleep_replay.py:457-541`; rendered PNG | The plotted enabled lane is selected after event detection, and the figure does not mark the four onsets that constitute the detected sequence. The trace contains many isolated and overlapping activations. | The visual emphasizes a favorable lane without exposing the event predicate or population incidence. | Prespecify the displayed lane or show all lanes/aggregate incidence, mark detected event onsets, and save the exact lane-selection rule and event membership. |
| Low | `sleep_replay_evidence.npz` | Spikes, weights, event labels, recall scores, gate, and `dt` are saved, but the actual sleep current, zero cue, assay parameters, selected display lane, and event onset membership are absent. | The saved evidence cannot reconstruct every causal-input and qualitative-label decision without rerunning code. | Persist all branch inputs, detector/recall settings, per-event onset indices, and display selection alongside the existing arrays. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Point-neuron dynamics | Unit-aware `brainpy.state.LIFRef` with explicit initializer | BrainPy-State `LIFRef` | Correct and canonical. | None. |
| Excitatory and inhibitory kinetics | `Expon` State bound through `COBA` outputs registered on the neuron | BrainPy-State `Expon`, `COBA`, and current-input registration | Correct for the custom dense recurrent network. | A projection could package this composition, but custom communication/plasticity gating makes the explicit form defensible. |
| Spike-driven recurrent communication | `BinaryArray(previous_spikes) @ dense_weight` | BrainEvent `BinaryArray` and dense event product | Correct for a 24-neuron dense, learned matrix. | None. |
| Pair-based online plasticity | Dense post-triggered potentiation followed by pre-triggered depression, bounded and masked | BrainEvent dense binary plasticity operators + BrainState State | Correct API family and event orientation. | Persist or test rule-specific weight changes by connection class. |
| Learned efficacy and traces | `LongTermState` for weights; `ShortTermState` for traces | BrainState semantic State roles | Correct. | None. |
| Ensemble initialization | `vmap_init_all_states(..., axis_size=16)` | BrainState collective initialization | Correct; all mutable State receives a lane axis. | Verify all mapped shapes and pair equality at the causal branch. |
| Matched condition execution | `vmap2` maps the complete per-step transition and every mutable State role | BrainState `vmap2` | Correct and materially satisfies the prompt. | None. |
| Time evolution | One `for_loop` per learning, sleep, and recall phase inside State-aware `jit` | BrainState `for_loop` and `jit` | Correct stable transform boundary with no Python timestep loop. | None. |
| Phase State restoration | Custom snapshot/restore over fast-State filters | BrainState `assign_state_values` is the public whole-graph restoration API | Behavior is correct here, but it is partial and duplicates the public path/mismatch contract. | Capture all States by path, deliberately preserve learned weights, restore through `assign_state_values`, and inspect both mismatch collections. |
| Time, voltage, current, conductance | BrainUnit quantities retained through simulation and converted only for host plotting/storage | BrainUnit | Correct. | Include units in saved evidence metadata for every physical array. |
| Random sleep protocol | Direct JAX keys and array construction outside the stateful rollout | Host/JAX boundary for a frozen protocol; BrainState randomness is preferred for model randomness | Legitimate as a presampled protocol, but it is external input and must be described that way. | Use explicit independent calibration/evaluation seeds and save the realized protocol. |
| Replay and recall scoring | NumPy host functions over completed spike arrays | Host analysis boundary; no owning BrainX API is required | Legitimate. | Prespecify/hold out the assays and persist their exact reductions. |
| Statistics and serialization | NumPy means/SD and compressed NPZ | Host boundary | Legitimate. | Report event incidence and paired uncertainty; save complete inputs and assay metadata. |
| Visualization | One `plt.subplots(2, 2)` and basic `plot`, `bar`, labels, legends | High-level Matplotlib host boundary | Correct, simple, and readable. | Mark event evidence and avoid outcome-selected display lanes. |

## Missing, bypassed, or misused BrainX APIs

### `brainstate.nn.assign_state_values`

Use it instead of the custom `restore_fast_state()` assignment loop when exact
path-aware restoration is required. The public API returns unexpected and
missing paths, which should be rejected. A partial restore remains valid only
when its omitted State policy is explicit and verified.

No BrainX API is missing for replay detection, recall scoring, paired summary,
NPZ serialization, or Matplotlib presentation. These are legitimate host-side
boundaries. Do not invent a BrainX analysis or plotting abstraction for them.

## Performance and code simplicity

The performance structure is strong: one mapped transition owns independent
conditions, one transformed loop owns time, and each phase has one JIT boundary.
The dense representation is appropriate for a `24 x 24` learned matrix. Host
loops construct only short protocols or analyze completed arrays; no Python
timestep loop bypasses BrainState.

The implementation is longer than an irreducible one-off demonstration because
it includes a result dataclass, custom snapshot helpers, several analysis
helpers, tests, a figure, and evidence serialization. Most of this supports the
prompt or validation. Replacing the restoration helpers with the public
collective API and saving a single structured evidence artifact would reduce
custom lifecycle code. The plot satisfies the refined simplicity rule: exactly
one `plt.subplots(...)` call and only basic axes methods.

## Skill improvements

Make one compact cross-package refinement in `brainx-general-guard` and mirror
it in `plan.md`:

1. Define external seeding by data flow, not by naming: supplied current,
   events, protocol tensors, and retained boundary State are external seeds even
   when stochastic, randomly located, or called intrinsic.
2. For an event-defined causal mediator, require per-condition concordance
   between measured event occurrence/dose and the downstream outcome; a broad
   pathway gate establishes only pathway-level causality.
3. Make independent calibration concrete but short: freeze model parameters,
   seeds, assay thresholds/windows, and displayed cases before evaluation, or
   use separate calibration data and report held-out sensitivity.

Do not add package-specific API guidance. Run 1 already selected and composed
the BrainPy-State, BrainEvent, BrainState, and BrainUnit APIs correctly.

## Checks for the next run

- Use the byte-identical prompt and the newly installed skill snapshot.
- Verify every State path and every protocol tensor is pair-equal immediately
  before the replay intervention; only the declared gate may differ.
- Call supplied assembly current seeded or evoked, regardless of random timing,
  or eliminate such current before claiming spontaneous/endogenous replay.
- Freeze model parameters, stochastic evaluation seeds, detector thresholds and
  windows, recall scoring, and display selection independently of evaluated
  outcomes; otherwise use separate calibration seeds and report held-out
  sensitivity.
- Require detected event incidence or dose to agree with per-lane recall before
  attributing recall to complete replay. Otherwise narrow the conclusion to the
  manipulated recurrent pathway.
- Save the realized branch inputs, all State snapshots, per-event onset members,
  assay settings, per-lane scores, and aggregation.
- Use BrainEvent plasticity and communication in the mapped State path,
  `vmap2` across independent conditions, `for_loop` across time, and BrainUnit
  quantities through the simulation.
- Create the figure with one `plt.subplots(...)` call and basic high-level
  Matplotlib methods; mark the exact event evidence without outcome-selecting a
  favorable lane.
