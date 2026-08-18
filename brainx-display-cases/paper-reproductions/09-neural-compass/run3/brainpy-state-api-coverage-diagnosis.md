# BrainX diagnosis: internal neural compass

## Evidence studied

- Exact case prompt and every Run 3 artifact: `neural_compass.py`,
  `test_neural_compass.py`, `README.md`, the PNG, CSV, JSON summary, event
  stream, stderr, final response, and harness metadata.
- `skills/brainx-general-guard/SKILL.md`, `skills/brainpy-state/SKILL.md`,
  `skills/brainpy-state/references/projection-patterns.md`,
  `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`,
  and `skills/brainevent/SKILL.md`.
- Closest executable composition:
  `skills/brainpy-state/references/scripts/sound_localization.py`.
- Official BrainPy-State API pages for `LIFRef`, `Expon`, and `CUBA`; the
  BrainState vectorization tutorial for `vmap2`, mapped State axes, and
  `vmap_init_all_states`; and the BrainEvent `BinaryArray`, dense-operation,
  and unit-aware-computation pages.
- A disposable rerun under the required BrainX virtualenv. The four tests
  passed; the complete program reproduced gain `0.941`, final error
  `5.35 deg`, and outcome counts `22/0/26`. The regenerated PNG, CSV, and JSON
  were byte-identical to the archive.
- Independent raw-array control audit: maximum intact tail error
  `0.00001366 deg`, minimum intact tail strength `0.80966485`, minimum intact
  tail activity `27.05871582`, and all 48 control-valid flags true.

## Executive diagnosis

Run 3 is scientifically coherent and BrainX-native. It represents the ring as
unit-aware LIF dynamics, routes binary spikes through BrainEvent, maps 96
independent intact and lesioned lanes with State-aware `vmap2`, and advances
time with one compiled `for_loop`. Every represented heading has an
independently validated matched control. Recovery requires a measured
departure followed by a fully valid final window, and the absent recovered
category is reported rather than forced.

The saved labels are numerically consistent: all 48 reconstruct from the saved
departure and sustained-return flags, and all 48 sustained-return flags
reconstruct from the three saved tail reductions. Of the 26 failures, 24 fail
the angular tail boundary and 2 fail only the relative-activity boundary.

The implementation meets the strengthened plotting constraint with one
`plt.subplots(...)` call and basic high-level methods. Three non-blocking
auditability gaps remain: the CSV does not persist every raw reduction behind
the departure and intact-control flags, the figure plots only the angular
classifier boundary, and the calibrated parameter regime is not marked as
phenomenological. The current general guard already states these requirements,
so another skill edit is not justified by this accepted checkpoint.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P2 | `neural_compass.py:389` | The CSV saves `departure_observed` and `intact_control_valid`, but omits minimum post-lesion strength, minimum post-lesion relative activity, maximum intact tail error, and minimum intact tail strength. | Labels reconstruct from saved booleans, but an external reader cannot independently recompute every boolean from continuous evidence. | If revising this artifact, save each exact post-lesion and intact-control reduction used by the predicates. |
| P2 | `neural_compass.py:431` | The outcome panel plots maximum matched error and its threshold but omits bump-strength and relative-activity reductions and thresholds. | Two low-relative-activity failures are colored correctly but their failing boundary is not visible. | If revising this artifact, expose all three boundary reductions with their exact thresholds while retaining one simple `plt.subplots(...)` composition. |
| P3 | `README.md:3` | The recurrent weights, cue, wedge, and decision thresholds are calibrated but not labeled as a phenomenological demonstration. | Readers may mistake the successful regime for a sourced biological parameterization. | State once that the regime is phenomenological unless sources are added. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, current, resistance, and angular velocity | BrainUnit quantities and explicit `to_decimal(...)` boundaries | BrainUnit | Correct | None. |
| Point-neuron dynamics | `brainpy.state.LIFRef` with an explicit voltage initializer | BrainPy-State | Correct | None. |
| Recurrent and velocity synaptic filtering | Two `brainpy.state.Expon` modules | BrainPy-State | Correct | None. |
| Current deposition | `brainpy.state.CUBA`, `add_current_input`, and `bind_cond` | BrainPy-State | Correct custom composition for the dynamically scaled velocity branch | A native projection would be preferable only if it preserved the same dynamic gain and State ownership more simply. |
| Ring event communication | `brainevent.BinaryArray @` unit-bearing dense ring matrices | BrainEvent | Correct; dense storage is appropriate for a fully weighted 48-cell ring | None. |
| Model graph and mutable dynamics | Registered `brainstate.nn.Module` children and BrainPy-State State | BrainState | Correct | None. |
| Independent conditions | `vmap_init_all_states` plus `vmap2` over semantic hidden and short-term State | BrainState | Correct; intact and lesioned conditions own separate dynamical State | None. |
| Time evolution | One jitted `brainstate.transform.for_loop` | BrainState | Correct and performant | None. |
| Cue, velocity, and lesion protocols | Time-major BrainUnit/JAX arrays built once before the loop | BrainUnit plus valid array boundary | Correct | None. |
| Circular population-vector decoding | NumPy complex reduction | Host-side scientific analysis boundary | Correct | None. |
| Matched-control validation and classification | NumPy reductions over reproduced trajectories | Host-side scientific analysis boundary | Correct at runtime; persistence is incomplete | Save every raw predicate reduction. |
| CSV and JSON | Python `csv` and `json` | Host serialization boundary | Correct | Add the omitted audit columns if the artifact is revised. |
| Figure | One `plt.subplots(3, 1, ...)` call with `imshow`, `plot`, `scatter`, spans, line, legend, and colorbar | Host presentation boundary | Simple and high quality; classifier evidence is incomplete | Add the two omitted classifier reductions without introducing manual layout machinery. |
| Focused tests | `unittest` for circular decoding, branch cuts, recovery ordering, and low relative activity | Host verification boundary | Correct but narrow | Add a persisted-CSV predicate reconstruction test if this artifact is revised. |

## Missing, bypassed, or misused BrainX APIs

No required BrainX API is missing or misused. The explicit `Expon` and `CUBA`
composition bypasses a packaged projection class, but the velocity branch
applies a time-varying angular gain between communication and current binding;
the custom composition expresses that mechanism directly and remains entirely
inside BrainPy-State and BrainEvent APIs.

Population-vector decoding, categorical reductions, CSV/JSON writing, and
Matplotlib are legitimate host boundaries. No official BrainX API owns the
task-specific circular classifier or its report format.

## Performance and code simplicity

- One mapped transition owns independent conditions and one transformed loop
  owns time; no Python timestep loop is present.
- Dynamical State is mapped by semantic role rather than inferred from shape.
- The 48-by-48 dense recurrent matrices are small and scientifically dense;
  sparse or generated connectivity would add machinery without reducing the
  meaningful cost.
- Rotation and lesion sweeps compile separately because their lane counts and
  protocol shapes differ. This is appropriate for a one-off demonstration.
- Plotting uses exactly one `plt.subplots(...)` call. It does not use
  `plt.figure`, `GridSpec`, `add_subplot`, projections, custom artists, or
  manual axes placement.
- The command-line output-directory option is minor optional structure, but it
  does not obscure the single canonical execution path.

## Skill improvements

No further skill change is justified. The latest `brainx-general-guard`
already requires independent matched-control validation, exact saved and
plotted boundary reductions, departure before sustained recovery, no forced
category, phenomenological labeling for unsourced calibration, and one simple
`plt.subplots(...)` composition. The remaining gaps are failures to apply
existing guidance, not missing guidance.

## Checks for the next task

- Validate every control independently before accepting intervention results.
- Save enough continuous reductions to recompute every control, departure,
  sustained-return, and outcome predicate without trusting stored booleans.
- Plot every classifier reduction against its exact threshold.
- Require measured departure before sustained recovery and accept an empty
  requested category.
- Mark unsourced calibrated regimes as phenomenological.
- Compose each Matplotlib figure with one `plt.subplots(...)` call and basic
  high-level plotting methods unless the prompt explicitly requires more.
