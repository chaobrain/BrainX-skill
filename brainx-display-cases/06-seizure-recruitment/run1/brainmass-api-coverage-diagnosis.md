# BrainX diagnosis: seizure recruitment across regions, Run 1

## Evidence studied

- Exact prompt: `brainx-display-cases/06-seizure-recruitment/prompt.md` (560
  bytes; SHA-256
  `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`, `README.md`,
  `outputs/seizure_recruitment.png`,
  `outputs/seizure_recruitment_data.npz`,
  `outputs/seizure_recruitment_metrics.csv`, `agent-final.md`, and the complete
  evaluator event log.
- Harness metadata: `gpt-5.6-sol`, `xhigh`, Codex CLI
  `0.147.0-alpha.1.2`, macOS Seatbelt host-read isolation, and exit code 0.
- Independent execution with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`: exit
  code 0; local peaks were `[0.928, 0.235, 0.037, 0.006]`, recruited peaks
  were `[0.928, 0.922, 0.913, 0.907]`, and the representative recruited
  onsets were `[11.4, 17.0, 22.6, 28.3] ms`.
- Independent numeric inspection: all 74 trajectories and peaks were finite;
  trajectory shape was `(74, 1200, 4)`; the 72 grid lanes and two control
  lanes were aligned; finite onset values matched recruitment labels exactly;
  all 32 all-region cases had strict region 0 -> 1 -> 2 -> 3 onset order; and
  no-coupling and no-perturbation controls passed.
- Independent temporal inspection: every labeled event happened to remain
  above threshold for at least 18 samples (`1.8 ms`), but the implemented
  classifier did not require this duration and would accept one sample.
- Independent CSV inspection: 296 rows, one per condition and region, with
  aligned condition type, coordinates, peaks, labels, and onsets.
- Independent full-resolution figure inspection: every panel, label, marker,
  discrete regime cell, and colorbar was visible and unclipped.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainmass/SKILL.md`, `skills/brainstate/SKILL.md`, and the BrainUnit
  skill used by the evaluator.
- BrainMass references: `modellibrary.md`, `coupling-network-api.md`,
  `parameter-sweeps-and-regime-analysis.md`,
  `batch-transform-acceleration.md`, and `visualization-analysis-api.md`.
- BrainState references: `transformation-vmap-expansion.md`,
  `brainstate-control-flow-patterns.md`, and the root State lifecycle and
  environment sections.
- Closest executable BrainMass examples:
  `scripts/seizure-epileptor-case-study.py` and
  `scripts/resting-state-meg-whole-brain-pipeline.py`.
- Official BrainMass pages: model selection, FitzHugh-Nagumo and Epileptor
  model pages, coupling and delays, custom coupling, parameter sweeps, central
  model API, central coupling API, and orchestration API.
- Official BrainState pages: delay protocol, `brainstate.nn.Delay`,
  vectorization, control flow, and the generated `for_loop` and `vmap` APIs.

## Executive diagnosis

Run 1 fixes the consequential Run 0 API and timing defect. It replaces the
custom ring buffer with a fixed-capacity `brainstate.nn.Delay`, constructs and
initializes it under the same `dt` used for execution, inserts before retrieval,
and proves the exact phase with an impulse assertion. Direct functional
coupling remains justified because coupling and the independent focal drive
must be summed on the same first FHN input channel. The complete condition is
mapped with state-aware `vmap`, all timesteps run through `for_loop`, State is
independent per lane, and the causal controls share that mapped path.

The artifact is scientifically explicit that FHN is a phenomenological
seizure-like regional model rather than a clinical or Epileptor seizure model.
Its directed chain, finite delay, local versus recruited cases, continuous
peaks and onsets, strict representative onset order, and controls support a
demonstration of excitable propagation.

One important scientific defect remains: the README calls the response a
"burst," but the label is `any(trace >= threshold)`. A single threshold sample
would therefore qualify as a recruited burst. The current trajectories happen
to stay above threshold for at least `1.8 ms`, so the displayed conclusion is
not numerically overturned, but that fact is accidental rather than enforced
by the declared predicate. The NPZ also omits the code/version identifier
required by the existing sweep guidance, and the onset panel connects four
sampled delay coordinates with lines that imply unsampled continuity.

## Run 1 compared with Run 0

| Concern | Run 0 | Run 1 | Assessment |
|---|---|---|---|
| Delay owner | Custom ring buffer | `brainstate.nn.Delay` | Improved; package-owned State replaces custom infrastructure. |
| Delay phase | Every label was one `dt` short | Update then `retrieve_at_step(d)`, with impulse check | Corrected. |
| Same-channel drive | Direct sum, but the `Network` boundary was unexplained | Direct sum follows explicit skill routing | Improved and justified. |
| Stateful sweep | Complete-condition `vmap` and `for_loop` | Complete-condition `vmap` and `for_loop` | Preserved. |
| Recruitment predicate | Required a sustained threshold window | Accepts any one threshold crossing | Regressed and must be corrected. |
| Controls | No stimulation and no coupling | No perturbation and no coupling in the same mapping | Preserved. |
| Graph semantics | README incorrectly called a symmetric graph directed | Matrix is genuinely directed and documented as `W[target, source]` | Corrected. |
| Continuous evidence | Peaks and onsets retained | Complete traces, peaks, and onsets retained | Improved. |
| Metadata | Missing model/protocol/connectivity/version | Adds model, protocol, connectivity, timing, and unit-named fields; version still missing | Improved but incomplete. |
| Delay presentation | Irregular samples appeared as continuous heatmap extent | Exact discrete heatmap columns, but onset samples are joined by lines | Improved but still over-interpolated. |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `seizure_recruitment.py:113-124`; `README.md:3-5,33-35` | `jnp.any(above_threshold, axis=1)` treats one threshold sample as a recruited burst, while the prose claims a burst. | A transient numerical crossing can change the categorical regime map and onset order without satisfying the claimed event. | Define a minimum consecutive duration before inspecting the map, detect the first qualifying window, save that duration, and derive the label and onset from the full predicate. |
| P2 | `seizure_recruitment.py:349-369` | The NPZ has no code version, repository revision, or explicit artifact/schema version. | The data identifies the model but not the implementation revision that produced it. | Store a repository commit when meaningful; otherwise store an explicit script or artifact version. |
| P3 | `seizure_recruitment.py:241-256` | Four sampled delays at `1`, `4`, `8`, and `12 ms` are joined by line segments. | The panel visually implies behavior between unsampled delays. | Plot markers without connecting lines, or sample densely enough to justify interpolation and state that assumption. |

FHN remains a defensible model choice for this exact prompt. It would become a
model-selection error only if the artifact claimed seizure onset/offset
mechanisms, Epileptor dynamics, physiological calibration, or clinical
validity.

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Represent regional fast-slow excitability | `brainmass.FitzHughNagumoStep` with explicit `tau`, `V`, and `w` initialization | `brainmass.FitzHughNagumoStep` | Correct for a deterministic phenomenological seizure-like event. | Keep the model boundary explicit. |
| Represent directed regional wiring | Constant `(4, 4)` JAX matrix with `W[target, source]` comment | Host-authored structural data consumed by BrainMass coupling | Correct; it encodes only 0 -> 1 -> 2 -> 3. | Keep and serialize it. |
| Store delayed source activity | Fixed-capacity `brainstate.nn.Delay` | `brainstate.nn.Delay` | Correct package-owned State and initialization path. | Keep. |
| Convert physical delay to retrieval offset | Unit-aware delay divided by fixed unit-aware `DT`, rounded to traced integer | BrainUnit arithmetic plus `Delay.retrieve_at_step` | Correct for the exactly representable sampled delays. | Keep declared delay coordinates aligned to `dt`. |
| Define delay phase | Insert current source, then retrieve offset `d`; separate three-step impulse check | `Delay.update`, `Delay.retrieve_at_step`, `brainstate.transform.for_loop` | Correct and independently verified. | Keep the focused assertion. |
| Compute inter-region input | `brainmass.additive_coupling` | `brainmass.additive_coupling` | Correct high-level stateless coupling kernel. | Keep. |
| Combine coupling and focal perturbation | Add both values before FHN's first input | Direct composition required by `Network` positional input ownership | Correct; `Network` cannot sum an independent caller value onto its coupling-owned first channel. | Keep direct composition. |
| Initialize each condition | Construct model and call `brainstate.nn.init_all_states` inside `run_condition` | BrainState collective lifecycle API | Correct and gives each mapped lane independent State. | Keep construction and initialization inside the final `dt` scope. |
| Scope `dt`, `i`, and `t` | Nested `brainstate.environ.context` | `brainstate.environ.context` | Correct; delay construction, initialization, and execution share `dt`. | Keep. |
| Run time evolution | `brainstate.transform.for_loop(step, arange(N_STEPS))` | `brainstate.transform.for_loop` | Correct; State effects remain in the transformed time loop. | Keep. |
| Sweep all conditions and controls | One `brainstate.transform.vmap(run_condition)` over 74 complete conditions | `brainstate.transform.vmap` | Correct state-aware batching; controls use the same path. | Keep. |
| Synchronize before host conversion | `jax.block_until_ready(traces)` | JAX execution boundary | Correct before NumPy serialization and plotting. | Keep. |
| Define recruitment | Threshold, `any`, first crossing | Legitimate host/JAX scientific predicate boundary | Incomplete for a claimed burst because duration is absent. | Encode the full sustained-event predicate and derive onset from it. |
| Enforce route order | Strictly increasing onsets for one representative spread case | Host/JAX scientific validation boundary | Correct for that case; all complete-recruitment grid cases also pass independently. | Apply or save the strict predicate wherever a propagated label is reported. |
| Validate causal controls | No-coupling and no-perturbation lanes with assertions | Host-side scientific validation boundary | Correct and mapped with the experiment. | Keep. |
| Retain continuous evidence | Full traces, peaks, and onset arrays | Host/JAX analysis boundary | Correct and stronger than storing labels alone. | Keep. |
| Preserve grid axes | `meshgrid(indexing="ij")`, flatten for mapping, reshape for display | BrainUnit/JAX array construction and host presentation | Correct coordinate ordering. | Keep. |
| Plot traces and regime map | One high-level Matplotlib `subplots` figure | Legitimate host presentation boundary | Clear and unclipped; no BrainMass categorical regime plot owns this layout. | Remove interpolation implied by connected sparse delay samples. |
| Serialize numeric outputs | Compressed NPZ plus long-form CSV | Legitimate host serialization boundary | Correct format and alignment; version metadata is incomplete. | Add explicit code/artifact version. |
| Document scientific boundary | README labels FHN as deterministic and phenomenological | Host reporting boundary | Correct and appropriately limited. | Make the burst predicate match the prose. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API remains bypassed or misused in the Run 1 simulation.

### `brainstate.nn.Delay`

Run 1 now uses this API correctly. Capacity is fixed by `MAX_DELAY`, the object
is constructed and initialized under `DT`, the source is inserted exactly once
per step before retrieval, and the mapped condition varies only the traced
integer retrieval offset. The impulse check proves that offset three returns
the impulse three completed updates later. This resolves the Run 0 bypass and
off-by-one error.

### `brainmass.Network`

`Network` is still intentionally absent. It inserts coupling as the node's
first positional input and forwards caller inputs afterward. Because FHN uses
that first channel for both regional coupling and the independent focal drive,
the artifact must retrieve delayed activity, apply
`brainmass.additive_coupling`, sum the drive, and call the node directly.

### `brainmass.Simulator`

`Simulator` is not missing. The prompt explicitly requires `for_loop` and
`vmap` over coupling, delay, and perturbation, while the same-channel
composition requires a custom State-aware step. BrainState is therefore the
correct lower-level execution owner.

### Temporal event analysis

No BrainX API owns the scientific definition of a sustained burst. JAX or
host-side array logic is the legitimate boundary, but it must implement the
complete declared predicate. `jnp.any()` is sufficient only when the claim is
"ever crossed the threshold," not when the label is "burst" or "sustained
recruitment."

## Performance and code simplicity

- One state-aware `vmap` owns every independent condition and control; one
  `for_loop` owns all 1200 timesteps. There is no Python simulation loop.
- Fixed delay capacity keeps State shape static across mapped lanes. Only the
  integer read offset varies, avoiding tracer-dependent allocation.
- Model construction inside the mapped function gives each lane independent
  dynamical and delay State without a separate manual State-axis declaration.
- `jax.block_until_ready` occurs before host conversion. NumPy and CSV are used
  only for serialization, validation, and plotting.
- Full trajectories cost about 1.4 million values but are justified here by
  the requested representative traces and by the need to audit a categorical
  temporal predicate. A larger production sweep could return fixed-shape
  summaries and rerun selected traces.
- The custom Module contains only the scientific composition that `Network`
  cannot express. No custom delay, integrator, State container, or coupling
  equation remains.
- The figure uses one `plt.subplots` call and basic plotting methods. Removing
  line interpolation from the four-point delay panel reduces implied claims
  without adding complexity.

## Skill improvements

Make one surgical change in
`skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`:

1. State that a categorical time-resolved regime must implement the full
   temporal predicate named by the claim. Distinguish a one-sample crossing
   from a sustained event, require the minimum consecutive duration to be
   fixed and saved, and derive onset from the first qualifying window.
2. State that sparse sampled coordinates must be plotted as samples; do not
   connect or interpolate them unless that inference is scientifically
   justified.
3. Clarify that the required code-version metadata is an explicit repository
   revision or script/artifact version, not the model name or filename.

Do not edit `brainx-general-guard`. It already requires each claim's full
temporal predicate, continuous boundary evidence, matched controls, explicit
sampled-point claims, and BrainX-native execution. Do not edit the BrainMass
root, model library, coupling reference, BrainState, or BrainUnit guidance;
Run 1 follows those contracts.

## Checks for the next run

- The generated entry point executes independently and all numeric and visual
  outputs are inspectable.
- The artifact distinguishes phenomenological FHN propagation from Epileptor,
  physiological, or clinical seizure claims.
- Delay history uses a documented BrainX delay abstraction. Construction,
  initialization, and execution share one `dt`; capacity stays static; and an
  impulse check proves the exact update/retrieval phase.
- Same-channel coupling plus focal drive uses justified direct composition;
  `brainmass.additive_coupling` owns the weighted regional input.
- `brainstate.transform.vmap` covers complete independent conditions and
  controls, while `brainstate.transform.for_loop` owns every timestep.
- The recruitment label fixes and enforces a minimum consecutive duration;
  onset is the first qualifying sustained window, not the first isolated
  threshold sample.
- Propagated cases enforce strict region 0 -> 1 -> 2 -> 3 onset order.
- No-perturbation and no-coupling controls execute in the same mapped path.
- Complete traces or the exact tested reduction, continuous peaks, onsets, and
  categorical labels remain aligned for every condition and region.
- Saved data includes coordinate arrays, units, model identity, connectivity,
  fixed model/protocol parameters, `dt`, duration, threshold, minimum event
  duration, and an explicit repository or artifact code version.
- Sparse delay coordinates appear as sampled points or discrete columns, with
  no unjustified connecting interpolation or continuous interval claim.
