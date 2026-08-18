# BrainX diagnosis: cortical wave obstacle, Run 1

## Evidence studied

- Generated artifacts: `README.md`, `cortical_wave_obstacle.py`,
  `outputs/phase_map.csv`, and `outputs/cortical_wave_obstacle.png`.
- Execution: the archived entry point completed independently in 17.35 seconds
  in the required BrainX virtualenv. Its 25-row CSV and 2091 x 1554 PNG were
  byte-identical to the archived outputs. The full run reported 7 splits, 3
  bends, and 15 deaths.
- Visual inspection: the figure is readable and shows left-to-right activity,
  the lesion geometry, one lower-flank passage, and the categorical phase map.
  The intact sequence is fragmented and remains active after the primary front
  rather than showing a compact wave followed by quiescence.
- Independent scientific checks: the no-spark intact control produced zero
  spikes; first arrival was monotonic across all 28 columns; both flank masks
  contained 56-63 sites at every radius; every surviving condition reached a
  flank before the right edge; and the reference lesion reached the right edge
  at 55.2 ms.
- Matched intact controls: inhibition scales 0.40 and 0.48 reached the right
  edge with 96 and 65 late right-zone spikes. Scales 0.56, 0.64, and 0.72 did
  not reach the right edge without a lesion.
- Owning skills and routed material: `brainx-general-guard`, BrainPy-State,
  BrainEvent, BrainState, BrainUnit, BrainPy-State `component-selection.md`,
  `projection-patterns.md`, `braintools/connectivity.md`, and
  `braintools/input-current.md`; BrainEvent `sparse-formats.md` and
  `coba_ei_teaching.py`; BrainState `transformation-vmap-expansion.md`; and
  BrainUnit `array-creation.md`.
- Official contracts: BrainTools `Grid2d`, `ConnectionResult`, and input-current
  APIs; BrainEvent `coo2csr`, `CSR`, and `BinaryArray`; BrainPy-State `LIFRef`,
  `Expon`, and `COBA`; BrainState `vmap2`, `vmap_init_all_states`, `for_loop`,
  and `jit`; and BrainUnit grid, quantity, and array operations.

## Executive diagnosis

Run 1 fixes the main Run 0 topology failure. It selects the two-dimensional
neighborhood through BrainTools, preserves the `ConnectionResult` edge
permutation, and hands a valid CSR to BrainEvent. It also keeps independent
dynamical State per condition and executes the full sweep inside one compiled
time loop.

The direct-current refinement did not transfer. The agent opened
`braintools/input-current.md` but still rebuilt the spark with time comparisons
inside every step. The reference explains current construction but does not
show the BrainPy-State handoff from a generated time-major current array into
`for_loop`, leaving the final integration decision underspecified.

Scientific validity improved but is not yet strong enough. The geometry now
leaves symmetric routes and the realized bend trajectories are temporally
ordered. However, the classifier itself uses unordered late spike totals, the
figure compares different inhibition settings, and three of five inhibition
columns also die in matched intact controls. The phase map therefore mixes a
global inhibition failure with obstacle interaction and does not show a
continuous control-normalized transmission measure.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `cortical_wave_obstacle.py:195`, `cortical_wave_obstacle.py:283`, and `cortical_wave_obstacle.py:285` | The only intact lane uses inhibition 0.40, while the displayed reference lesion uses 0.48. No intact control is run across the inhibition sweep. | The visual comparison changes both lesion and inhibition. The 0.56-0.72 death columns are not obstacle effects because matched intact controls also die. | Run a no-lesion control at every inhibition value. Compare intact and lesioned snapshots at the same inhibition, and express transmission relative to the matched intact lane. |
| P1 | `cortical_wave_obstacle.py:238` through `cortical_wave_obstacle.py:259` | Bend/split labels use late aggregate flank and right-zone spike counts without requiring flank arrival before downstream arrival or continuity of one advancing front. | The archived trajectories happen to be ordered, but the code can silently label reverberant or disconnected activity as passage after a parameter change. | Classify from first-arrival or another time-resolved front metric. Require source-to-flank-to-downstream order and use the same metric in validation and figures. |
| P2 | `outputs/cortical_wave_obstacle.png` | The intact frames show fragmented branches and 93 spikes remain after 70 ms. The categorical phase map omits the saved right-zone counts and any matched-control normalization. | The figure supports recurrent fragmented propagation more clearly than a coherent cortical wave, and it hides the magnitude of transmission near the categorical boundary. | Add first-arrival, front-coherence, or post-front-quiescence evidence. Plot a continuous arrival, reach, or normalized transmission metric beside or beneath the labels. |
| P2 | `cortical_wave_obstacle.py:478` | Validation requires all three requested labels to appear but does not require matched-control separation, causal ordering, or a no-spark control. | Parameter tuning can satisfy the label inventory without strengthening the mechanism that the labels claim. | Validate quiet baseline, matched controls, ordered propagation, lesion silence, and metric/label agreement. Treat label diversity as an output summary, not a scientific invariant. |
| P2 | `README.md:3` | The model is presented as a cortical sheet without parameter provenance or a phenomenological teaching-model boundary. | Readers may interpret the tuned regime as biologically calibrated. | Label it as a phenomenological demonstration unless parameters are sourced, and state which conclusions the controls support. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical sheet coordinates | BrainUnit `arange`, `meshgrid`, quantities, and reshaping | BrainUnit | Correct | Keep the explicit `indexing="xy"` and row-major agreement with `Grid2d`. |
| Spatial topology | `braintools.conn.Grid2d(connectivity="moore", periodic=False)` | BrainTools connectivity | Correct and improved | Keep; it replaces Run 0's nested lattice loops with a named topology. |
| Topology-to-storage handoff | `coo2csr(pre_indices, post_indices, shape=result.shape)` and `result.weights[order]` | BrainEvent `coo2csr` and `CSR` | Correct and improved | Keep the mandatory permutation and shape/orientation check. |
| Binary spike communication | `BinaryArray(spikes) @ CSR` | BrainEvent | Correct | Keep presynaptic rows and postsynaptic columns. |
| Excitatory/inhibitory dynamics | Two `LIFRef` populations | BrainPy-State | Correct | Keep explicit initialization, units, refractory State, and one-step calls. |
| Four conductance paths | Shared CSR, `Expon`, `COBA`, and a small projection Module | BrainPy-State plus BrainEvent | Correct | Keep one synaptic State per target site and explicit reversal potentials. |
| Brief spark | `spark_on = (t >= start) & (t < stop)` inside `update()` | BrainTools input-current protocol | Bypassed owner | Generate the piecewise unit-aware current once under the active `dt`; pass its time-major samples through `for_loop` and apply the spatial mask by broadcasting. |
| Lesion geometry and silence | Unit-aware circular mask applied to emitted spikes and direct currents | BrainUnit plus model-specific logic | Correct for communication silence | Keep; independent checks found no raw excitatory or inhibitory spikes inside the reference lesion. |
| Tonic spatial bias | Unit-aware sinusoid over physical coordinates | BrainUnit plus model-specific logic | Legitimate model logic | Keep only with a disclosed phenomenological role and apply it identically to matched controls. |
| Sweep grid | BrainUnit meshgrid and flattened condition axes | BrainUnit | Correct | Keep explicit axis order. |
| Independent condition State | `vmap_init_all_states` and filtered `vmap2` State axes | BrainState | Correct | Keep `unexpected_out_state_mapping="raise"`. |
| Time evolution and compilation | Mapped complete step inside `for_loop`, enclosed by `jit` | BrainState | Correct and efficient | Keep one transformed time loop; do not add a Python timestep loop. |
| Outcome analysis | NumPy host-side late spike counts | Legitimate host boundary | Mechanically valid, scientifically incomplete | Replace unordered totals with ordered front metrics and matched-control normalization. |
| CSV persistence | Python `csv` | Legitimate host boundary | Correct | Keep and add the continuous control and transmission fields used by the phase map. |
| Figure generation | High-level Matplotlib over host arrays | Legitimate presentation boundary | Readable | Keep, but compare matched conditions and visualize the causal metric. |
| Runtime validation | Host assertions after the compiled rollout | Legitimate host boundary | Useful but outcome-targeted | Validate mechanism and controls instead of requiring every category by construction. |

## Missing, bypassed, or misused BrainX APIs

### `braintools.input.Constant`, `Section`, or `Step`

Use one of these APIs to construct the baseline-plus-spark protocol under the
same `brainstate.environ.dt` as the rollout. Generate the complete time-major
array once, then pass it as an `xs` argument to `brainstate.transform.for_loop`.
Apply the edge mask after the per-step sample is sliced. Do not recompute a
piecewise protocol from time predicates inside every model step.

The agent read the current reference, so routing alone is insufficient. The
BrainPy-local reference needs one explicit current-array-to-rollout handoff
that demonstrates this lifecycle and preserves units.

No material misuse was found in `Grid2d`, `ConnectionResult`, `coo2csr`, `CSR`,
`BinaryArray`, `LIFRef`, `Expon`, `COBA`, mapped State initialization, `vmap2`,
`for_loop`, `jit`, or BrainUnit quantities.

## Performance and code simplicity

The execution architecture is strong. Twenty-six independent condition lanes
share immutable topology while owning independent neuron and synapse State.
One mapped transition executes inside one 450-step transformed loop, and one
JIT boundary encloses the rollout. Host loops are limited to classification,
CSV output, and plotting. The independent entry point took 17.35 seconds with
a fresh Matplotlib font cache; subsequent trajectory diagnostics completed in
roughly three seconds.

The avoidable per-step logic is the spark predicate. A generated current array
would make protocol duration and `dt` alignment explicit and remove stimulus
construction from the model transition. Matched controls can remain additional
mapped lanes, so stronger scientific validation does not require repeated
Python simulations.

## Skill improvements

1. Keep the Run 1 BrainTools connectivity reference and routing unchanged; the
   agent followed them correctly.
2. Strengthen the BrainPy-State root and `component-selection.md` input route:
   generate direct current protocols once and pass time-major samples into the
   transformed rollout; do not rebuild named sections or pulses with time
   predicates inside the step.
3. Add one BrainPy-State handoff section to
   `references/braintools/input-current.md`. Show a minimal `Constant` or
   `Step` baseline-pulse-baseline protocol constructed under `dt`, and
   `for_loop(step, times, current)` slicing one unit-aware sample per step.
   Keep the shared authoring source synchronized.
4. Add compact cross-package scientific-claim validation to
   `brainx-general-guard/SKILL.md`: use matched controls across swept nuisance
   parameters, encode causal order in mechanistic labels, retain continuous
   evidence beneath categorical maps, and validate the mechanism rather than
   forcing requested categories to appear.
5. Update `plan.md` for the stronger input-current handoff and general-guard
   validation boundary. Do not change BrainEvent, BrainState, BrainUnit, or the
   successful connectivity reference.

## Checks for the next run

- Preserve the exact prompt bytes, model, effort, virtualenv, CLI, and isolation
  conditions.
- Retain `Grid2d` topology and the ordered `ConnectionResult` to BrainEvent CSR
  handoff with `weights[order]`.
- Generate the spark through a documented `braintools.input` API outside the
  per-step transition and pass the current array through `for_loop`.
- Keep independent dynamical State per mapped condition and one transformed
  time loop.
- Include a no-lesion control at every inhibition value and compare snapshots
  at one identical inhibition setting.
- Confirm that the no-spark control remains quiescent and every lesion remains
  silent.
- Derive bend, split, and death from ordered source-to-flank-to-downstream
  activity or an equivalent causal-front metric; validate metric/label
  agreement.
- Plot a continuous reach, arrival, delay, or matched-control transmission
  measure with the categorical labels, and verify that obstacle size changes it
  within inhibition regimes where the intact control propagates.
- Open the figure and confirm a coherent advancing front, readable physical
  geometry and axes, matched-condition comparison, and agreement with saved
  numerical metrics.
