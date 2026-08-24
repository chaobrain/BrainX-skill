# BrainX diagnosis: binocular rivalry

## Evidence studied

- Frozen prompt: 718 UTF-8 bytes, SHA-256
  `b8e98eb70a623270c82a1894d6b8229c9b8ca2f71034ca336433c41c24f477b1`.
- Run artifacts: `binocular_rivalry.py`, `README.md`, the PNG and NPZ results,
  `agent-final.md`, the complete JSONL event stream, stderr, and harness metadata.
- Independent execution with the required BrainX virtualenv in
  `/tmp/case08-run0-review.wAKeZZ`: exit 0; all 17 NPZ fields and the compressed
  NPZ SHA-256 reproduced exactly.
- Visual inspection: the 2070 x 756 PNG is nonblank, legible, unclipped, and
  shows alternating population activity plus the labeled adaptation/noise map.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainmass/SKILL.md`, `skills/brainstate/SKILL.md`,
  `skills/brainunit/SKILL.md`, and their routed sweep, simulator, model-library,
  vmap, environment, unit-boundary, and visualization references.
- Closest executable composition:
  `skills/brainmass/references/scripts/wong-wang-decision-making.py`.
- Authoritative contracts: generated API pages for `WongWangStep`,
  `OUProcess`, `brainstate.transform.vmap`, `brainstate.transform.for_loop`,
  `brainstate.nn.exp_euler_step`, `vmap_init_all_states`, and BrainUnit quantity
  conversion.

## Executive diagnosis

Run 0 is executable, deterministic, dimensionally consistent, genuinely
state-vmapped across 300 independent observer lanes, and visually successful.
Its equal-drive, adaptation-augmented competition produces plausible switching.

The central scientific defect is disclosure. The event log shows repeated
outcome inspection followed by selection of the model, cross-inhibition,
adaptation range, noise range, threshold, and displayed condition. In
particular, the lower adaptation bound was moved from 0.17 to 0.18 nA because
0.18 made every observer alternate. The artifact instead records only
`"parameters fixed before the saved sweep"` and presents the selected grid as
the study result. It neither labels the result outcome-calibrated/exploratory
nor saves the observed neighboring failure boundary.

The code also overloads `WongWangStep.compute_inputs(..., noise_1_val=...,
noise_2_val=...)` with `noise - adaptation`. The arithmetic is correct, and the
subsequent public `phi`, derivative, and integration calls are valid, but the
named noise arguments no longer contain only noise. Current BrainMass guidance
does not teach the important boundary: `update(coherence=...)` accepts bounded
sensory evidence in `[-1, 1]`, while arbitrary population-specific adaptation
belongs as an explicit current modification after `compute_inputs()`.

Existing sweep guidance already requires outcome-calibration disclosure,
nearby sensitivity, mechanism controls, grid contracts, code version, and
random-design metadata. Those omissions are artifact failures, not new general
skill gaps. The surgical skill refinement should cover only the missing
Wong-Wang extension boundary.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `binocular_rivalry.py:28-29`, metadata at `:255`, event items 36-74 | The saved grid and operating regime were selected after viewing outcomes, including choosing 0.18 nA because all observers alternated, but metadata says only that parameters were fixed before the saved sweep. | The result reads as pre-specified even though it is outcome-calibrated; the 100% alternation and monotonic grid partly reflect selection. | Label the study exploratory/outcome-calibrated, record every tuned setting, and save the nearby 0.17 nA boundary or run a separately declared confirmation seed/grid. |
| P1 | `README.md:43-48`, `agent-final.md` | The response attributes dominance duration to adaptation and noise and extrapolates effects of recurrence, inhibition, and input imbalance without matched zero-adaptation, zero-noise, or varied-coupling/input controls. | Positive-grid associations support sensitivity within the selected regime, not the full causal or extrapolated claims. | Append controls to the same mapped call, save their tags and results, and restrict claims to tested interventions. |
| P1 | `binocular_rivalry.py:61-85` | Adaptation is passed through arguments documented as population noise, while `update(coherence=...)` cannot express that current because coherence is bounded sensory evidence. | The mechanism is numerically valid but semantically obscured and easy to implement incorrectly with out-of-range coherence. | Call `compute_inputs()` with valid coherence and actual noise, modify returned currents explicitly with adaptation, then use `phi`, `dS1_dt`/`dS2_dt`, and `exp_euler_step`. |
| P2 | `binocular_rivalry.py:180-190`, `:217-219`, figure | `condition_interval_s` is total analysis time divided by the number of segments, so first and last censored segments contribute to a quantity labeled mean dominance interval; completed episode means are saved but not mapped or plotted. | The displayed statistic is a reciprocal switch-count summary, not the mean of observed complete dominance durations, and can conceal censoring near sticky regimes. | Plot completed-episode duration with counts/coverage, or relabel the existing reduction exactly and show the censoring fraction beside it. |
| P2 | NPZ metadata | The bundle omits explicit `grid_shape`, `grid_axis_order`, code version, outcome-calibration fields, and the independent-key design. | Consumers must infer layout and cannot audit provenance or stochastic design directly. | Store these fields explicitly as required by the existing sweep reference. |
| P2 | `binocular_rivalry.py:325-340`, PNG | Only means are displayed across 12 observers per condition. | The figure does not show stochastic uncertainty or whether a mean is supported evenly across observers. | Retain observer values, and add an uncertainty reduction or state that the heatmap is descriptive. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Aggregate competing populations | `brainmass.WongWangStep` | BrainMass model library | Correct model family and scale. | Keep it; state that the adaptation augmentation is phenomenological. |
| Equal continuous sensory evidence | `compute_inputs(coherence=0.0)` | `WongWangStep.compute_inputs()` / `update()` | Correct bounded coherence value. | Keep coherence separate from current-valued adaptation. |
| Population-specific adaptation current | `brainstate.HiddenState` and subtraction through `noise_*_val` | BrainState State plus public Wong-Wang current/transfer/derivative APIs | State ownership is correct; current injection is semantically hidden in noise arguments. | Subtract adaptation explicitly from returned total currents. |
| Correlated stochastic current | `brainmass.OUProcess` | BrainMass noise process | Correct package API and time-valued decay. | Pass the OU values alone through `noise_1_val` and `noise_2_val`. |
| Wong-Wang transfer and gating step | `phi`, `dS1_dt`, `dS2_dt`, `exp_euler_step`, clipping | Public BrainMass methods plus BrainState integration | Valid custom extension because stock `update()` has no arbitrary current arguments. | Preserve this minimal public-method composition and document why it is necessary. |
| Physical units | BrainUnit quantities through simulation, explicit `to_decimal()` at host boundaries | BrainUnit | Correct. | Keep units intact until plotting/serialization/statistics. |
| Parameter grid construction | Magnitudes through `jnp.meshgrid`, then quantities restored | JAX at a documented dimensionless grid boundary | Legitimate host/array boundary. | Save shape and axis-order metadata. |
| Observer State initialization | `vmap_init_all_states(..., axis_size=300)` | BrainState | Correct independent State-lane initialization. | Keep. |
| Stateful observer batching | Exact writable State instances in `brainstate.transform.vmap` | BrainState `vmap` instance contract | Correct and fulfills the prompt; all mutated dynamical States are declared. | Keep; record that random keys split independently across lanes. |
| Time execution | One `brainstate.transform.for_loop` | BrainState | Correct; no Python timestep loop. | Keep. |
| Randomness | One seed before mapped stochastic execution | BrainState randomness and vmap key splitting | Reproducible and independently drawn, proven by exact rerun. | Describe the independent-key design in metadata. |
| Dominance classification | NumPy hysteresis/hold-last-state analysis | Legitimate host-side scientific analysis | Mechanically coherent, but outcome calibration and boundary censoring are not handled transparently. | Freeze/disclose the rule and retain exact decision evidence and censoring summaries. |
| Condition aggregation | NumPy reshape and means | Legitimate host-side statistics | Correct layout in code, incomplete serialized contract and uncertainty. | Save grid shape/order and uncertainty. |
| Serialization | `np.savez_compressed` with JSON metadata | Legitimate host boundary | Numeric bundle is deterministic and mostly self-describing. | Add provenance, calibration, control, grid, and random-design fields. |
| Visualization | One `plt.subplots()` call, line plot, discrete `imshow` | High-level Matplotlib host boundary | Simple, readable, and appropriate for the requested figure. | Keep; label the duration reduction exactly and optionally show uncertainty/censoring. |

## Missing, bypassed, or misused BrainX APIs

### `WongWangStep.update(coherence=...)`

Use this only for motion-coherence evidence in `[-1, 1]`. It cannot represent an
arbitrary adaptation current. Run 0 discovered this contract late: clipping its
initial adaptation-as-coherence design eliminated switching. The final direct
current extension is necessary.

### `WongWangStep.compute_inputs()`, `phi()`, `dS1_dt()`, and `dS2_dt()`

These public methods are the correct extension boundary when a mechanism adds
population-specific current before the transfer function. `compute_inputs()`
should receive actual noise in its named noise arguments; adaptation should
modify the returned total currents explicitly before `phi()` and the gating
derivative step. No higher-level BrainMass runner exposes this custom current
path while also satisfying the prompt's explicit stateful `vmap` requirement.

### Mechanism controls

No missing API caused their omission. The existing BrainMass sweep reference
already requires zero-mechanism controls to be appended to the same flattened
mapped call and tagged. Run 0 did not follow it.

### Host boundaries

NumPy dominance analysis, aggregation, NPZ serialization, textual reporting,
and high-level Matplotlib plotting have no more appropriate package-owned API
for this custom rivalry metric. They should remain host-side rather than being
wrapped in invented BrainX APIs.

## Performance and code simplicity

- The scientific step is mapped across all 300 independent lanes and executed
  through one transformed time loop; there is no Python simulation loop.
- The run is exactly reproducible across independent processes. Every NPZ leaf
  and the compressed archive hash match.
- Stacking both full activity and adaptation trajectories costs roughly twice
  the required time-major output memory even though only one downsampled
  adaptation trace is saved. Returning only required full observables or a
  sampled diagnostic would reduce memory, but the current 40 s study completes
  quickly and is not blocked by this cost.
- The observer-wise Python analysis loop is an acceptable host boundary for 300
  traces. Scientific clarity matters more here than compiling a one-off ragged
  episode parser.
- The plotting path follows the guard's simple-figure contract exactly: one
  `subplots()` call and basic high-level methods.

## Skill improvements

- Do not edit `brainx-general-guard`: its calibration, controls, validation,
  State, and host-boundary invariants already diagnose the failures.
- Add one compact section to
  `skills/brainmass/references/modellibrary.md` that distinguishes bounded
  Wong-Wang coherence from population-specific currents and shows the public
  `compute_inputs -> explicit current modification -> phi -> derivative ->
  exp_euler_step` extension path.
- Add the exact generated `WongWangStep` page to that reference's official
  sources. Do not expand the root skill; it already routes model-specific
  decisions to `modellibrary.md`.

## Checks for the next run

1. Use `WongWangStep` or justify another aggregate competition model.
2. Keep `coherence` within `[-1, 1]`; represent adaptation as an explicit
   population-specific current, not as coherence or mislabeled noise.
3. Use public BrainMass methods for any adaptation-augmented Wong-Wang step and
   validate its State and unit contracts.
4. Map the complete stateful observer transition with BrainState `vmap`, run
   time with one `for_loop`, and verify independent lane State and noise.
5. Label all outcome-selected model parameters, grid axes, threshold, seed, and
   displayed condition as exploratory/outcome-calibrated; save nearby
   sensitivity or separately declared confirmation.
6. Append and save matched no-adaptation and no-noise controls before claiming
   those mechanisms control switching; do not claim untested recurrence,
   inhibition, or input effects.
7. Distinguish complete dominance episodes from boundary-censored segments and
   label every plotted reduction exactly.
8. Store grid shape/order, controls, units, code version, random-key design,
   monitor phase, and the complete decision rule in the numeric bundle.
9. Save observer-level evidence and show uncertainty or support counts for grid
   summaries.
10. Independently rerun the script, compare every saved field, and visually
    inspect the final figure.
