# BrainX diagnosis: cortical wave meets an obstacle, Run 3

## Evidence studied

- Generated artifacts: `README.md`, `cortical_wave.py`, `agent-final.md`,
  `results/phase_metrics.npz`, `results/phase_map.png`, and
  `results/wave_snapshots.png`; harness metadata, stderr, and the complete JSONL
  event stream were also inspected.
- Execution: the archived source ran in a fresh temporary copy under
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx` and reproduced all
  three result files byte-for-byte. A separate diagnostic run verified zero E
  and I spikes without the spark, zero monitored E and I spikes inside every
  lesion, and finite upper- and lower-route arrivals for all 24 saved split
  cells.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`, `skills/brainstate/SKILL.md`,
  `skills/brainevent/SKILL.md`, and `skills/brainunit/SKILL.md`.
- Relevant references and executable compositions:
  `skills/brainpy-state/references/component-selection.md`,
  `skills/brainpy-state/references/braintools/connectivity.md`,
  `skills/brainpy-state/references/braintools/input-current.md`,
  `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`,
  `skills/brainevent/references/sparse-formats.md`,
  `skills/brainpy-state/references/scripts/sound_localization.py`, and
  `skills/brainevent/references/scripts/coba_ei_teaching.py`.
- Official contracts: BrainUnit Array Creation and generated `asarray` API;
  BrainTools `Grid2d` and `ConnectionResult`; BrainEvent `coo2csr`, `CSR`, and
  `BinaryArray`; BrainState `for_loop`, `vmap2`, and mapped-State
  initialization; BrainPy-State `LIFRef`, `Expon`, and `COBA`.

## Executive diagnosis

Run 3 is a reproducible, BrainX-native solution. It uses an explicit E/I LIF
sheet, spatial BrainTools topology, BrainEvent CSR communication, unit-aware
input construction, independent mapped dynamical State for all 36 conditions,
and one compiled time loop. The figures now show aligned intact/lesion
snapshots and every continuous measurement used by the saved phase labels. All
matched controls cross, the no-spark control is silent, and the saved sweep
contains one die, five bends, and 24 splits.

Two non-blocking residual risks remain. First, every saved split has two finite
flank arrivals, but the classifier keeps only the earliest arrival and does not
encode that redundant check. Second, Run 3 recovered from one incorrect
`asarray(..., unit=target)` attempt before producing the final valid source.
Neither affects the saved results, and Run 3 is accepted as the completion
checkpoint without a Run 4.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P2 | `cortical_wave.py`, `measure_and_classify()` | `route_first` stores `min(upper_first, lower_first)`, while the split predicate checks aggregate upper/lower counts and balance. All 24 saved split cells independently pass the stronger two-arrival check. | A future calibration could require a stricter per-flank timing predicate, but the current labels are supported. | Accepted residual risk for this completed demonstration. |
| P2 | `codex-events.jsonl`, failed item 37 and recovery item 39 | The first source version used `u.math.asarray([plain numbers], unit=u.mm)` and raised `UnitMismatchError`; the final source correctly uses `jnp.asarray(...) * u.mm`. | One intermediate execution failed, but the archived source and deterministic rerun are valid. | Accepted recovered error; no additional 07 refinement. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, current, conductance, and geometry | BrainUnit quantities and `u.math` operations | BrainUnit | Correct and dimensionally explicit after the constructor recovery. | Clarify plain-data unit attachment in the BrainUnit root skill. |
| Local two-dimensional topology | `braintools.conn.Grid2d(connectivity="moore", periodic=False)` | BrainTools connectivity | Correct high-level topology choice. The default result has `weights=None`, so an intentional scalar CSR value is valid. | None. |
| COO-to-CSR handoff | `brainevent.coo2csr()` plus public `CSR` | BrainEvent | Correct. The value permutation is irrelevant for a scalar value; indices and shape are retained. | None. |
| Sparse spike communication | `BinaryArray(previous_spikes) @ CSR` | BrainEvent | Correct event-driven source-to-target accumulation. | None. |
| E/I neuron dynamics | Two `brainpy.state.LIFRef` populations | BrainPy-State | Correct reduced point-neuron model for a phenomenological wave. | None. |
| Synaptic filtering and current | `Expon` plus `COBA`, registered with `add_current_input` | BrainPy-State | Correct conductance workflow and causal update order. | None. |
| Edge spark | BrainTools `Constant` generated once as time-major current | BrainTools input | Correct; the time loop slices one sample per step and spatial broadcasting applies the edge mask. | None. |
| Condition independence | `vmap_init_all_states` plus `vmap2` with explicit dynamical-State filters | BrainState | Correct; all 36 conditions own independent State lanes while topology is shared. | None. |
| Time evolution and compilation | One jitted `brainstate.transform.for_loop` | BrainState | Correct transformed rollout with no Python timestep loop. | None. |
| Silent intervention | Active masks suppress E and I monitored spikes and future event emission | BrainPy-State/BrainEvent composition | Correct functional silence; direct diagnostics found zero lesion spikes in both populations. | None. |
| Phase measurements and categorical analysis | NumPy reductions after the compiled rollout | Host boundary | Appropriate host-side scientific analysis. Matched-control normalization and all current continuous boundaries are saved and plotted. | Add per-route finite causal arrivals to the split predicate. |
| Plotting | Matplotlib snapshots and heat maps | Host boundary | Appropriate. Control and lesion use identical nuisance settings and aligned physical times; no overlap or blank panels were found. | Plot the added multi-route arrival boundary in Run 4. |
| Serialization | `numpy.savez_compressed` and PNG | Host boundary | Appropriate, deterministic, and byte-reproducible. | None. |

## Missing, bypassed, or misused BrainX APIs

No BrainX modeling API is missing from the final source. The recovered
`asarray` call was an API-selection error caused by an underspecified skill
boundary, not a missing package API. NumPy classification, NPZ serialization,
and Matplotlib presentation are legitimate host boundaries; no official BrainX
API should replace them.

## Performance and code simplicity

The expensive scientific path is correctly structured as one 36-lane mapped
State rollout inside one compiled time loop. Sparse event communication avoids
dense sheet matrices, and plotting and classification run only after device
execution. The 750-by-36-by-1232 spike monitors are justified by the requested
snapshots and spatial phase analysis. No additional abstraction or batching
change is warranted.

## Skill improvements

Run 3 justifies no further 07-specific skill expansion. Keep the scientific
validation rules already added to `skills/brainx-general-guard/SKILL.md`, but
condense their wording without weakening matched-control, causal-order,
complete-predicate, or boundary-visualization requirements. Do not change
BrainPy-State connectivity, input-current, transform, or projection guidance;
Run 3 validates those paths.

## Completion checks

- The exact 836-byte prompt and all frozen harness conditions remain unchanged.
- Plain spatial parameter arrays use a valid unit-attachment path in the final
  archived source.
- The model keeps explicit E and I point-neuron populations, local BrainTools
  topology, BrainEvent sparse event communication, BrainTools time-major input,
  independent mapped State, and one transformed time loop.
- No-spark E and I activity remains exactly zero, and both populations remain
  silent inside the patch.
- Every inhibition value has a matched intact control, and control/intervention
  snapshots use aligned times and nuisance settings.
- Every saved `split` has finite upper and lower arrivals; `bend` and `die`
  predicates remain complete and mutually interpretable.
- Every continuous boundary used by every saved label is retained, serialized,
  and visualized; saved labels exactly equal a fresh recomputation from the
  saved measures.
- The sweep shows coherent crossing plus defensible bend, split, and die
  outcomes without forcing categories, and the source and all outputs reproduce
  deterministically.
