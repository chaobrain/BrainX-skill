# BrainX diagnosis: memories replaying during sleep

## Evidence studied

- Exact case prompt and every Run 0 artifact: `sleep_replay.py`,
  `test_sleep_replay.py`, `README.md`, the PNG, event stream, stderr, final
  response, and harness metadata.
- `skills/brainx-general-guard/SKILL.md`, `skills/brainpy-state/SKILL.md`,
  `skills/brainpy-state/references/projection-patterns.md`,
  `skills/brainevent/SKILL.md`,
  `skills/brainevent/references/synaptic-plasticity.md`, and
  `skills/brainstate/references/collective_model_operations.md`.
- Closest executable composition:
  `skills/brainevent/references/scripts/coba_ei_teaching.py`.
- Official BrainEvent synaptic-plasticity tutorial and operations API for the
  dense pre- and postsynaptic update operators, and the official BrainState
  API for `assign_state_values`.
- A disposable rerun under the required BrainX virtualenv. The three tests
  passed and the complete program reproduced forward/backward evidence
  `15/0`, exact pre-sleep weight matching, forward-edge changes
  `0.010/0.000`, and recall `0.625/0.000`.
- Independent audits of all pre-sleep State, per-lane sleep activity, learned
  directionality, and recall-gate robustness.

## Executive diagnosis

Run 0 is executable, BrainX-native, and mechanically well structured. It uses
unit-aware LIF and conductance dynamics, BrainEvent event communication and
dense bidirectional STDP, mapped independent State for 16 lanes, and one
compiled `for_loop` per phase. All matched pairs are numerically identical in
membrane, refractory, synaptic, trace, and weight State immediately before the
sleep intervention, although the program asserts only weight equality.

The network exhibits a forward event sequence in five of eight replay lanes
and none of the blocked lanes. Those five lanes selectively strengthen the
three forward edges and succeed at recall under the selected challenge. The
recall advantage is stable for recurrence gates `0.56` through `0.67`, absent
below that range because both groups fail, and absent at `0.68` and above
because both groups succeed.

The causal recall conclusion is nevertheless not cleanly established because
the `0.60` recall gate was selected after inspecting group separation. The
reported sleep activity is also seeded by an externally driven A spike on the
last wake step, so it is evoked continuation into a zero-input period rather
than demonstrated spontaneous replay. Finally, the archive saves only one
example lane in the figure and aggregate transition totals in console output;
it does not save the per-lane event or transition evidence needed to
reconstruct the direction label.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `sleep_replay.py:58`, `sleep_replay.py:366` | The recall recurrence gate was chosen after observing which value separates replay and blocked groups. | The recall benefit is an outcome-tuned assay rather than an independently specified causal test. | Pre-register the recall challenge, calibrate it on an independent cohort or control criterion, or report the complete gate-response curve and base the conclusion on its robust range. |
| P1 | `sleep_replay.py:375`, `sleep_replay.py:380` | The direction label aggregates all replay lanes, but only matched pair 1 is plotted and no per-lane event or transition table is saved. | A reader cannot reconstruct the reported `15/0` evidence or see that only five of eight lanes replay. | Save the first-spike times and forward/backward counts for every lane, and make the aggregation rule explicit. |
| P2 | `sleep_replay.py:151`, `sleep_replay.py:334` | The final wake protocol externally drives A, whose retained spike triggers the sleep sequence at the phase boundary. | “Replay during sleep” can be read as spontaneous or endogenous even though this result is evoked by a wake-boundary seed. | Call the result seeded or evoked replay, or remove the boundary drive and establish spontaneous initiation from a documented sleep mechanism. |
| P2 | `sleep_replay.py:330` | The program asserts only equal weights before intervention. | A future change could leave membrane, refractory, synaptic, or trace State unmatched while the control check still passes. | Validate every relevant pre-intervention State and protocol component, or construct both branches from one verified snapshot and verify the branch point. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical parameters and phase times | BrainUnit quantities and explicit conversion at plotting boundaries | BrainUnit | Correct | None. |
| Place-cell dynamics | `brainpy.state.LIFRef` with explicit initialization | BrainPy-State | Correct | None. |
| Recurrent conductance | `brainpy.state.Expon`, `COBA`, and bound postsynaptic current | BrainPy-State | Correct custom composition for plastic dense efficacy | None. |
| Spike communication | `brainevent.BinaryArray @ dense weight State` | BrainEvent | Correct for a four-cell fully plastic network | None. |
| Pair-based plasticity | Dense pre- and postsynaptic BrainEvent update operators with trace State | BrainEvent plus BrainState | Correct operator use and persistent topology | Document or test the chosen STDP sign convention. |
| Persistent weights and transient traces | `LongTermState` and `ShortTermState` | BrainState | Correct lifecycle roles | None. |
| Independent matched lanes | `vmap_init_all_states` and State-aware `vmap2` | BrainState | Correct; all mapped State has a lane axis | Verify full State equality at the branch point. |
| Learning, sleep, and recall evolution | One jitted `for_loop` wrapper reused across phases | BrainState | Correct and performant | None. |
| Recall reset | Full-path `assign_state_values` restore with retained `LongTermState` | BrainState | Correct workaround after mapped reset lost its lane axis | None. |
| Route and cue construction | Small host-side arrays converted once to unit-aware time-major inputs | Host/BrainUnit boundary | Correct | None. |
| Direction and recall scoring | NumPy first-spike reductions | Host scientific-analysis boundary | Mechanically correct, incompletely persisted | Save per-lane reductions and the aggregate rule. |
| Plotting | One `plt.subplots(1, 3, ...)` call with basic plotting methods | Host presentation boundary | Simple and readable | Show or save all-lane replay evidence rather than one example only. |
| Focused tests | Standard-library tests for direction order and recall prefix | Host verification boundary | Correct but too narrow | Add full-State matching, event persistence, and assay-robustness checks. |

## Missing, bypassed, or misused BrainX APIs

No required BrainX API is missing or misused. A packaged projection is not a
better replacement here because the small dense weights are modified online
by BrainEvent operators and then converted to conductance. Host NumPy analysis,
serialization of audit evidence, and Matplotlib are legitimate boundaries.

The implementation correctly avoids `vmap_reset_all_states` after verifying
that the selected model reset drops the lane axis, and uses the documented
full-path State restore instead. This is an important correct application of
the existing BrainState reference.

## Performance and code simplicity

- Independent lanes are mapped across the complete stateful transition; time
  is lowered through one transformed loop.
- Learning traversals are encoded before execution. The small Python loops
  construct an input protocol and are not simulation timestep loops.
- Dense four-by-four storage is simpler than sparse storage for this topology.
- Each figure is composed with exactly one `plt.subplots(...)` call and basic
  high-level Matplotlib methods.
- Separate compiled calls for learning, sleep, and recall preserve the
  intentional State lifecycle and are appropriate for this demonstration.

## Skill improvements

Refine only `brainx-general-guard` and the synchronized `plan.md` summary:

- require every scientifically relevant State and protocol value to match at
  a causal intervention branch, not only a convenient summary;
- require assay settings and decision rules to be fixed independently of the
  observed effect, or require held-out calibration or a robustness range;
- distinguish spontaneous or endogenous events from seeded, cued, or evoked
  continuation across a phase boundary;
- persist the exact per-condition evidence and aggregation behind qualitative
  labels.

No BrainPy-State, BrainEvent, BrainState, or BrainUnit package-skill edit is
justified because the package-specific API and lifecycle guidance worked.

## Checks for the next run

- Validate all pre-intervention neural, synaptic, trace, plasticity, and
  protocol State across matched branches.
- Describe replay as evoked when an external boundary event seeds it, or
  demonstrate spontaneous initiation without that event.
- Fix the recall assay independently of group outcomes, and report robustness
  if a calibrated challenge is necessary.
- Save per-lane first-spike times, direction counts, weight changes, and recall
  scores so every claim reconstructs from persisted evidence.
- Preserve the current BrainX mapping, State restore, unit, transformed-loop,
  and simple-plotting structure.
