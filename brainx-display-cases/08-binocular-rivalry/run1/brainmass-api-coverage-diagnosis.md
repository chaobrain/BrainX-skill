# BrainX diagnosis: binocular rivalry, Run 1

## Evidence studied

- Frozen prompt: 718 UTF-8 bytes, SHA-256
  `b8e98eb70a623270c82a1894d6b8229c9b8ca2f71034ca336433c41c24f477b1`.
- Run artifacts: `binocular_rivalry.py`, `README.md`, the PNG and NPZ results,
  `agent-final.md`, the complete JSONL event stream, stderr, harness metadata,
  and the archived Python bytecode created during the run.
- Independent execution with the required BrainX virtualenv in
  `/tmp/case08-run1-review.8EPDKj`: exit 0; all 30 NPZ fields, dtypes, shapes,
  values, field order, and compressed NPZ bytes reproduced exactly. Both PNGs
  are 2610 x 756, nonblank, legible, unclipped, and visually equivalent. Their
  renderer output is not pixel-identical, but every plotted numeric input is
  present in the byte-identical NPZ.
- Independent in-memory reconstruction of every saved observer- and
  condition-level metric from the script's returned trajectories: exact match
  for `switch_count`, `mean_complete_duration_s`, `vertical_fraction`,
  `median_duration_s`, `switch_rate_per_min`, `locked_fraction`, and
  `undecided_fraction`, using 4,276 complete dominance episodes.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainmass/SKILL.md`, `skills/brainstate/SKILL.md`,
  `skills/brainunit/SKILL.md`, and their routed model-library, sweep, custom
  rollout, vmap, environment, randomness, and unit-boundary references.
- Closest executable composition:
  `skills/brainmass/references/scripts/wong-wang-decision-making.py`.
- Authoritative contracts: generated API pages for `WongWangStep`,
  `brainstate.transform.vmap`, `brainstate.transform.for_loop`,
  `brainstate.nn.exp_euler_step`, BrainState initialization and randomness,
  and BrainUnit quantity conversion.

## Executive diagnosis

Run 1 materially resolves Run 0's transferable BrainMass failure. It keeps
bounded evidence at `coherence=0.0`, passes only stochastic current through the
named noise arguments, subtracts adaptation explicitly from the returned
population currents before `phi()`, and advances the public gating derivatives
with exponential Euler. The complete 160-observer simulation is genuinely
state-vmapped, time is carried by one transformed loop, physical units remain
intact to explicit host boundaries, and the exact numerical rerun establishes
reproducibility.

The scientific disclosure is also substantially better. The artifact labels
the regime phenomenological and outcome-calibrated, evaluates a held-out final
seed, separates zero-adaptation and zero-noise grid controls, distinguishes
perfectly symmetric undecided lanes from decided-but-unswitched lanes, excludes
both boundary-censored dominance intervals from complete-episode durations,
and preserves the representative decision and adaptation traces. Its held-out
results support the limited claims that stronger adaptation shortens dominance
at weak noise, stronger noise increases switching, and equal stimulation has no
orientation bias in this selected regime.

Residual weaknesses concern auditability and claim scope: the NPZ does not
store the raw complete episode durations needed to reconstruct its pooled
median, explicit grid shape/axis order, a code version, or the random-key
design; eight replicates are summarized without uncertainty; the mechanism
language invokes competition without a matched no-additional-coupling control;
and the script reimplements an OU process that BrainMass already supplies.
Existing noise, guard, and sweep guidance already requires the named process,
controls, provenance, layout contracts, random-design disclosure, continuous
evidence, and retained per-condition evidence. These are remaining artifact
failures, not a new transferable skill gap. No further skill edit or Run 2 is
justified.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P2 | `binocular_rivalry.py:47-64` | The script manually implements two OU recurrences with `HiddenState` even though `brainmass.OUProcess` supports standalone direct sampling and unit-aware `sigma` / `tau`. | The dynamics are correct but duplicate a package-owned noise process and add State and equations that the model code need not own. | Construct one two-channel `brainmass.OUProcess`, initialize it with the observer State, and call it inside the custom step; pass its two returned currents unchanged to `compute_inputs()`. |
| P2 | `binocular_rivalry.py:141-151`, `:193-198`, NPZ | The condition median pools 4,276 complete episode durations, but the NPZ saves only observer means and the final pooled median, not the raw durations or per-condition complete-episode counts. | The reported median is correct but cannot be reconstructed from the saved numeric bundle alone, and support differs silently across conditions. | Save long-form complete durations with observer/condition identifiers or fixed-shape duration histograms plus counts; retain the exact pooled reduction and support count. |
| P2 | `binocular_rivalry.py:25-39`, `:243-281`, NPZ | The bundle omits explicit `grid_shape`, `grid_axis_order`, code version, and the random-key design. | A consumer can infer the current layout and seed from code, but cannot audit those contracts from the artifact alone or distinguish common-noise from independent-lane sampling. | Store the shape, ordered axis names, script or repository revision, and an explicit statement that stateful `vmap` splits the BrainState random stream independently across observer lanes. |
| P2 | `binocular_rivalry.py:30-31`, `:66-82`, `:399-424`, README | The explanation attributes rivalry to mutual competition, including an added cross-population current, but there is no matched run with `RIVALRY_COUPLING=0`; the zero-adaptation and zero-noise lanes test different mechanisms. | The simulation demonstrates sensitivity to adaptation and noise within a competition model but does not isolate the contribution of the added rivalry-coupling term. | Append a tagged no-additional-coupling condition to the same mapped execution and save its observables before attributing switching specifically to that term. |
| P3 | `binocular_rivalry.py:37-39`, `:177-208`, PNG | Each condition has eight replicates, while the figure and report show only pooled medians and mean switch rates without uncertainty or support counts. | The heatmaps are descriptive and may overstate precision, especially in weak-noise conditions with few complete episodes. | Save replicate or bootstrap uncertainty and complete-episode counts, or label the displayed grids explicitly as descriptive summaries. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Aggregate competing populations | `brainmass.WongWangStep` | BrainMass model library | Correct population scale and model family. | Keep; preserve the phenomenological-extension disclosure. |
| Equal sensory evidence | `compute_inputs(coherence=0.0)` | `WongWangStep.compute_inputs()` / stock `update()` contract | Correct bounded coherence and equal evidence. | Keep coherence separate from current-valued mechanisms. |
| Population-specific current extension | Explicit binocular drive, cross-population coupling, and adaptation applied to returned currents | Public Wong-Wang `compute_inputs()`, `phi()`, and derivative methods | Correctly follows the refined extension boundary. | Keep; add a no-additional-coupling control for the scientific claim. |
| Correlated stochastic current | Explicit OU `HiddenState` recurrence passed alone as `noise_1_val` / `noise_2_val` | `brainmass.OUProcess` used as a standalone process | Correct stochastic semantics and units, but it duplicates a package-owned process; direct process output composes with the explicit current path. | Replace the recurrence and its two States with one two-channel `OUProcess`; record mapped random-key design in metadata. |
| Wong-Wang gating integration | `dS1_dt`, `dS2_dt`, `brainstate.nn.exp_euler_step`, and clipping | Public BrainMass derivatives plus BrainState integration | Correct custom extension without copied model equations. | Keep. |
| Adaptation dynamics | Two unit-aware `HiddenState` values updated inside the complete step | BrainState State plus BrainUnit | Correct custom model mechanism and State ownership. | Keep. |
| Hysteretic percept | `ShortTermState` retaining the previous decision inside the mapped rollout | BrainState State plus task-specific decision logic | Correct fixed-shape stateful readout. | Save the complete decision rule as structured metadata, not only its threshold. |
| Physical units | BrainUnit quantities through dynamics; `to_decimal()` at analysis, plotting, and serialization boundaries | BrainUnit | Correct. | Keep explicit target units at every raw boundary. |
| Cohort construction | `jnp.meshgrid(..., indexing="ij")` and flattened condition arrays | JAX at a documented dimensionless grid boundary | Correct ordering in code. | Save explicit grid shape and axis-order fields. |
| State initialization | Model and custom States constructed inside each mapped observer function | BrainState transformation lifecycle | Correct independent per-lane State construction. | Keep. |
| Stateful batching | `brainstate.transform.vmap(simulate_observer)` over all 160 observers | BrainState `vmap` | Correct complete-operation mapping; this is not merely mapped setup or analysis. | Keep; document independent mapped RNG semantics. |
| Time execution | One `brainstate.transform.for_loop` under `environ.context(dt=...)` | BrainState control flow and environment | Correct, with no Python timestep loop. | Keep. |
| Randomness | One held-out seed before mapped stochastic execution | BrainState random stream | Exactly reproducible across independent processes. | Save the calibration/evaluation seed roles and key-splitting design explicitly. |
| Dominance classification and censoring | NumPy host analysis after the rollout | Legitimate host-side scientific analysis | Correctly distinguishes undecided traces and removes first/last censored intervals. | Save the raw complete episodes or sufficient reconstructive evidence. |
| Condition aggregation | NumPy observer loop, concatenation, medians, and means | Legitimate host-side statistics | Correct calculations; support and uncertainty are not retained. | Save counts and uncertainty beside every grid summary. |
| Serialization | `np.savez_compressed` with 30 named fields | Legitimate host boundary | Deterministic and substantially self-describing. | Add reconstructive episode evidence, layout, version, random design, and control tags. |
| Visualization | One `plt.subplots()` call with a trace and two discrete heatmaps | High-level Matplotlib host boundary | Simple, readable, accurate, and visually reproducible. | Optionally display support or uncertainty; do not imply interpolation between sampled cells. |

## Missing, bypassed, or misused BrainX APIs

### Wong-Wang public extension APIs

No Wong-Wang API remains misused. Run 1 follows the refined public sequence:
`compute_inputs()` with bounded coherence and actual stochastic current,
explicit population-current modification, `phi()`, `dS1_dt()` / `dS2_dt()`,
and `brainstate.nn.exp_euler_step()`. Stock `update()` cannot express the added
population-specific adaptation and cross-coupling currents, so the lower-level
public composition is justified.

### BrainMass noise processes

`brainmass.OUProcess` owns the mean-reverting stochastic drive and may be called
directly to obtain explicit current values. Run 1 should replace lines 47-64
with one two-channel process, initialize its State with the observer, and pass
the two sampled outputs through `noise_1_val` and `noise_2_val`. This preserves
the explicit adaptation-current composition without hand-writing the OU
equation. The existing `noiseprocesses.md` reference already teaches direct
process generation, so this bypass does not justify new skill text.

### Mechanism controls and result provenance

No missing BrainX API caused the absent no-additional-coupling control,
uncertainty, episode serialization, or metadata. The current sweep reference
already instructs agents to append matched controls to the same mapped arrays,
retain reconstructive evidence, save grid shape and axis order, record code
version, and state the stochastic design.

### Host boundaries

NumPy episode parsing and aggregation, NPZ serialization, textual reporting,
and high-level Matplotlib plotting are appropriate host boundaries. Do not
invent BrainX wrappers for these task-specific responsibilities.

## Performance and code simplicity

- One stateful `vmap` owns all 160 independent observer lanes, while one
  `for_loop` owns all 30,000 time steps. No Python loop performs simulation
  updates.
- `jax.block_until_ready(outputs)` establishes a real completion boundary
  before host analysis. The independent rerun reproduced the full 358 KiB NPZ
  byte for byte.
- The rollout returns four complete time-major trajectories. These are required
  for the requested activity trace, percept episodes, and adaptation evidence;
  retaining them is reasonable for this 160 x 30,000 demonstration.
- Replacing two custom OU `HiddenState` values and their recurrence with one
  BrainMass process would reduce mechanism code without changing the rollout
  structure or returned evidence.
- The Python observer and condition loops handle ragged episode durations after
  device execution. This is a clear host-side boundary and does not warrant a
  more complex transformed ragged representation.
- The plotting path uses one `plt.subplots()` call and basic high-level methods.
  Both independent renders are visually equivalent and preserve requested
  figure quality despite non-identical rasterization bytes.

## Skill improvements

- Keep the Run 0 refinement in
  `skills/brainmass/references/modellibrary.md`: Run 1 demonstrates that its
  bounded-coherence/current-extension decision boundary changes implementation
  behavior correctly.
- Do not edit `skills/brainx-general-guard/SKILL.md`, the BrainMass root skill,
  BrainState, BrainUnit, `parameter-sweeps-and-regime-analysis.md`, or
  `noiseprocesses.md`, or `plan.md`. Their existing package-ownership, direct
  noise-process, control, calibration, evidence-retention, layout,
  random-design, and provenance invariants already diagnose every residual
  issue.
- Do not add duplicate warnings or launch Run 2 from the same skill snapshot.
  The latest diagnosis exposes no new transferable skill gap requiring another
  refinement checkpoint.

## Checks for a hypothetical next run

No next numbered run is required for this case. If a future, independently
justified refinement changes relevant guidance, verify that its fresh run:

1. Preserves the explicit public Wong-Wang current-extension sequence and all
   unit, State, `vmap`, and `for_loop` contracts from Run 1.
2. Uses `brainmass.OUProcess` directly instead of reimplementing its recurrence.
3. Saves raw complete episode durations or equivalent reconstructive evidence,
   per-condition support, and uncertainty.
4. Stores `grid_shape`, `grid_axis_order`, code version, seed roles, and the
   mapped random-key design.
5. Adds a tagged no-additional-coupling condition to the same stateful mapped
   path before attributing rivalry specifically to the added cross-coupling.
6. Repeats the independent field-by-field rerun and visual inspection.
