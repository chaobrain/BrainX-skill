# BrainX diagnosis: Neural Compass Run 2

## Evidence studied

- Generated artifacts: `internal_neural_compass.py`,
  `test_internal_neural_compass.py`, `README.md`,
  `outputs/internal_neural_compass.png`, and
  `outputs/lesion_outcomes.csv`.
- Frozen-run evidence: `harness-metadata.txt`, `codex-events.jsonl`,
  `codex-stderr.log`, and `agent-final.md`.
- A clean disposable-copy execution with the required BrainX virtualenv. It
  reproduced the archived PNG and CSV byte for byte and reported a 7.97-degree
  dark-turn error, 0.00-degree stationary drift, 0.703 vector strength,
  0.000001 damaged-neuron activity, and 26 spared / 0 recovered / 46 failed
  headings.
- All five focused tests, executed directly because the run documents that
  `pytest` was unavailable.
- An independent reconstruction of every lesion predicate from the full
  traces and an independent audit of all 72 matched controls.
- `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`,
  `skills/brainpy-state/references/component-selection.md`,
  `skills/brainpy-state/references/projection-patterns.md`,
  `skills/brainpy-state/references/scripts/sound_localization.py`,
  `skills/brainevent/SKILL.md`, and
  `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`.
- Official routes indexed by
  `source_html_references/brainpy_state_html_reference.md`,
  `source_html_references/brainstate_html_reference.md`, and
  `source_html_references/brainevent_html_reference.md`: BrainPy-State neuron,
  synapse, projection, and synaptic-output APIs; BrainState Vectorization,
  Control Flow, and transform APIs; and BrainEvent event-array and matrix
  operation APIs.

No installed package source, signature, symbol inventory, docstring, or
internal implementation was used as modeling documentation.

## Executive diagnosis

Run 2 is a correct, deterministic, BrainX-native implementation of the
requested experiment. It uses a persistent wedge lesion, tests all 72 starting
headings against matched controls in one mapped simulation, accepts that no
heading recovered, encodes departure before sustained return, and labels the
phenomenological calibration honestly. The time loop, condition mapping,
event communication, units, and Matplotlib composition follow the intended
BrainX paths. The archived outputs reproduced byte for byte.

The remaining defect is auditability, not the current numerical result. The
classifier tests peak error, final-window maximum error, mean vector strength,
and mean rate ratio. The CSV substitutes mean final error for the tested
final-window maximum, while the figure plots that same proxy and omits vector
strength and rate ratio. Consequently, a reader cannot reconstruct every label
from the saved evidence. The current labels happen to be correct: independent
recomputation matched all 72 labels, and no heading sits in the proxy/exact
disagreement region.

The matched controls are also healthy but are not validated by the generated
program. Their maximum post-lesion heading error was `1.37e-5` degrees, minimum
final vector strength was `0.700`, and minimum final mean rate was `13.159`,
with no inactive lane. A future parameter change could invalidate a control
without stopping the program.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `internal_neural_compass.py:329`, `internal_neural_compass.py:421`, `internal_neural_compass.py:489` | `sustained_return` tests whether every final-window error is at most 10 degrees, equivalently the final-window maximum, but the CSV and categorical panel use the final-window mean. | The saved threshold comparison is not the classifier's actual predicate. A transient final-window violation could look recovered or spared in the figure even though the code labels it failed. | Compute named boundary reductions once. Save and plot peak error, final-window maximum error, final-window mean vector strength, and final-window mean rate ratio, with their exact thresholds. Feed those same values to the label predicate. |
| P1 | `internal_neural_compass.py:448` | The categorical figure omits the vector-strength and rate-ratio observables used by `reliable`. | The colors cannot be audited from the figure, especially in a future run where bump collapse rather than angular error causes failure. | Give every boundary reduction an aligned high-level Matplotlib axis and threshold before showing the categorical labels. Keep one `plt.subplots(...)` call and basic plotting methods. |
| P2 | `internal_neural_compass.py:366`, `internal_neural_compass.py:521` | The intervention is measured relative to each matched control, but the program never checks that every control itself retains the cued heading, bump strength, and nonzero activity. | A failed control could create a small relative error and a false spared/recovered label. This did not occur in Run 2, but the invariant is unprotected. | Validate every control lane independently before classifying its paired lesion lane; fail or mark the pair invalid when the control predicate fails. Add a focused test for an invalid control. |
| P2 | `README.md:37` | The prose definition of `spared` omits the classifier's sustained final return requirement. | The documented predicate is weaker than the implemented predicate and cannot fully explain a future failed label with peak error below 12 degrees. | State the complete predicate from the same named reductions used by classification, table, and figure. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, current, resistance, delay, and angular velocity | BrainUnit quantities; explicit conversion only for host-side timing and display | BrainUnit | Correct. Units survive model execution and are removed only at explicit analysis/presentation boundaries. | None. |
| Point-neuron dynamics | `brainpy.state.LIFRef` | BrainPy-State neurons | Correct low-cost refractory LIF choice for a phenomenological spiking ring. | None. |
| Recurrent temporal filtering and readout | `brainpy.state.Expon` | BrainPy-State synapses/readouts | Correct. One current filter follows the dynamically mixed recurrent product; a second dimensionless filter supports population-vector decoding. | None. |
| Voltage-independent recurrent current | `brainpy.state.CUBA` registered with `add_current_input()` | BrainPy-State synaptic outputs and neuron input binding | Correct for signed current weights. The custom three-kernel velocity mixture makes a direct component composition clearer than three redundant projections. | None. |
| Dense recurrent event communication | `brainevent.BinaryArray(delayed_spikes) @ weights` for three 72-by-72 kernels | BrainEvent event arrays and dense operations | Correct. The matrices are small and genuinely dense; sparse or generated connectivity would add complexity without a storage benefit. | None. |
| Axonal delay | A short boolean `HiddenState` history with an exact grid-aligned tap | Custom BrainState model State; BrainPy projection delay is the alternative for a single projection | Correct custom boundary because one delayed event vector feeds three dynamically mixed kernels. The focused impulse test verifies the tap convention. | None. |
| Cue, turn, and persistent lesion protocols | Complete time-major arrays constructed before the transformed loop | BrainUnit array construction plus pure JAX input preparation | Correct. The spatial cue and per-neuron lesion mask are task-specific and do not justify a named input-current wrapper. | None. |
| Independent headings and control/lesion conditions | `vmap_init_all_states` plus filter-based `brainstate.transform.vmap2` | BrainState State-aware vectorization | Correct. The mapping owns all 144 independent conditions and selects writable dynamical State by semantic role. | None. |
| Time evolution | One jitted `brainstate.transform.for_loop` | BrainState control flow and JIT | Correct. State owns recurrence, time is one transformed axis, and no Python timestep loop is used. | None. |
| Circular decoding and lesion predicates | Pure JAX reductions followed by host NumPy inspection | Scientific analysis host boundary | Correct ownership; no BrainX API owns this experiment-specific decoder or label definition. | Reuse one set of exact named reductions across classification, validation, CSV, and plot. |
| Matched control construction | Intact and lesioned lanes in the same mapped rollout | BrainState mapped execution plus scientific validation | Correct execution and pairing. | Add independent per-control validity checks before relative classification. |
| CSV serialization | Python `csv` | Host serialization boundary | Correct. | Save the exact tested reductions, not a mean-error proxy. |
| Figure composition | One `plt.subplots(2, 2, ...)` call, standard `plot`, `imshow`, `scatter`, spans, lines, labels, legends, and colorbars | High-level Matplotlib host presentation | Correctly simple and readable. No `GridSpec`, `add_subplot`, projection, custom artist, or manual axes placement is used. | Retain the simple composition while plotting all exact boundary reductions. |
| Focused validation | Five direct tests plus `validate_results()` | Host testing and scientific validation boundary | Core mechanics and classifier order are tested. | Add exact CSV/label reconstruction and invalid-control tests. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused.

- `brainpy.state.AlignPostProj` is not a required replacement for the direct
  recurrent composition. The model computes three event products, mixes them
  from instantaneous velocity, then applies one shared exponential filter and
  one CUBA output. Forcing three projections would duplicate synaptic State and
  obscure the intended current mixture.
- Sparse BrainEvent connectivity is not appropriate for the 72-by-72 smooth
  dense ring kernels.
- `braintools.input` is not needed for the custom heading-by-neuron cue tensor,
  velocity fraction, and persistent lesion mask; these arrays are constructed
  once outside the transformed step.
- BrainX does not own the circular decoder, experiment-specific predicates,
  CSV serialization, or custom presentation. Their host-side placement is
  legitimate.

## Performance and code simplicity

- One `vmap2` owns the independent condition axis; one `for_loop` owns time;
  one JIT owns the logical rollout.
- Mutable State is initialized per lane and selected by State role rather than
  rank or a coincidental shape.
- Dense event products are appropriate at this network size and connectivity
  density.
- Input arrays are built once before execution, and no Python timestep loop is
  present.
- The figure uses one `plt.subplots(...)` call and only basic high-level
  Matplotlib methods. Adding the missing boundary evidence should use a regular
  subplot grid, not `GridSpec`, projections, custom artists, or manual layout.

## Skill improvements

Change only `skills/brainx-general-guard/SKILL.md` and its matching `plan.md`
summary:

1. Require every matched control to pass its own baseline predicate before its
   intervention is interpreted.
2. For categorical maps, require the saved data and plotted threshold panels
   to use the exact reduction tested by each boundary, never a proxy summary.

Do not change BrainPy-State, BrainEvent, BrainState, or BrainUnit guidance. The
generated BrainX composition already follows their canonical contracts.

## Checks for the next run

- Run the byte-identical prompt with the frozen model, effort, virtualenv, and
  isolated skill snapshot.
- Require one healthy matched control for every tested heading and an explicit
  control-validity check before lesion labels are interpreted.
- Recompute every label from saved fields alone. The table must include peak
  error, final-window maximum error, final-window mean vector strength, and
  final-window mean rate ratio, or exact equivalent reductions.
- Require the figure to show each of those boundary values against its exact
  threshold before or alongside the categorical labels.
- Confirm that recovered still requires a measured departure followed by a
  sustained final return and that no outcome category is forced to appear.
- Preserve the persistent wedge lesion, all 72 headings, matched-condition
  mapping, one transformed time loop, BrainUnit quantities, and event-driven
  BrainEvent communication.
- Preserve absolutely simple plotting: one `plt.subplots(...)` call per figure
  and basic high-level plotting methods only.
- Run the focused tests and complete experiment, inspect the rendered figure,
  and compare the new diagnosis against these checks.
