# BrainX diagnosis: seizure recruitment across regions, Run 7

## Evidence studied

- Exact prompt: `brainx-display-cases/06-seizure-recruitment/prompt.md` (560
  bytes; SHA-256
  `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`, `README.md`,
  `results/seizure_recruitment.npz`,
  `results/seizure_recruitment.png`, `agent-final.md`, and the complete
  event log. No generated test file exists.
- Harness metadata: `gpt-5.6-sol`, `xhigh`, Codex CLI
  `0.147.0-alpha.1.2`, macOS Seatbelt host-read isolation, and exit code 0.
  The repository, normal skill homes, all earlier evaluation stages, failed
  launches, and smoke stage were denied; only the current staged skill snapshot
  and empty workspace were visible.
- Independent execution from a disposable copy with the required BrainX
  virtualenv: compilation and entry-point execution pass; all 28 archived NPZ
  fields exactly match the independent rerun, including intentional `NaN`
  onsets.
- Numeric inspection: 48 grid conditions plus two same-path controls; original
  grid shape `(4, 3, 4)` and ordered coupling, delay, and perturbation axes
  saved explicitly; trajectories shaped `(50, 2500, 3)`; 18 focal, three
  partial, and 27 fully routed grid conditions.
- Independent startup check: Epileptor `x1` and delay prehistory both
  initialize to `-1.5`, so the first diffusive current is exactly zero at
  every saved coupling strength.
- Event-log inspection: the agent outcome-calibrated the event rule, coupling
  family, `x0`, grid axes, and displayed cases after inspecting results. It
  tested nearby coupling values, rejected unstable additive coupling, retained
  a finite diffusive range, and used the no-pulse control to move all regions
  into a non-autonomous baseline. Both README and NPZ explicitly label the
  event rule and sampled regimes exploratory and outcome-calibrated.
- Full-resolution figure inspection: the four panels are legible and
  unclipped. The heatmap and traces use the same perturbation size; the traces
  expose the repeated fast excursions measured by the cumulative-occupancy
  predicate; the timing panel reports measured lags at the three sampled delays.
- Owning guidance: BrainX general guard, BrainMass, BrainState, and BrainUnit
  root skills; BrainMass `modellibrary.md`, `coupling-network-api.md`,
  `parameter-sweeps-and-regime-analysis.md`, and
  `batch-transform-acceleration.md`; BrainState delay, control-flow, and
  vectorization references.
- Closest executable examples: BrainMass
  `seizure-epileptor-case-study.py` and
  `resting-state-meg-whole-brain-pipeline.py`.
- Authoritative contracts: BrainMass `EpileptorStep`, `Network`, functional
  coupling, and parameter-sweep API pages; BrainState `Delay`,
  `retrieve_at_step`, `for_loop`, `vmap`, and initialization API pages
  indexed by `source_html_references/brainmass_html_reference.md` and
  `source_html_references/brainstate_html_reference.md`.

## Executive diagnosis

Run 7 satisfies the Run 6 edit specification and is an acceptable final
exploratory checkpoint. It saves a complete grid reconstruction contract,
states exactly which analysis choices were outcome-calibrated, retains the
continuous quantity behind every label, and avoids confirmatory or
patient-specific claims. The resulting demonstration directly shows focal,
partial, and ordered downstream recruitment, and both same-path controls pass.

The BrainX composition is correct and efficient. `EpileptorStep` owns regional
dynamics, the functional diffusive kernel owns coupling on the same channel as
the independent pulse, fixed-capacity `Delay` owns history, one complete
`vmap` owns all independent conditions and controls, and one `for_loop`
owns every timestep.

Several metadata and hardening details remain: the bundle does not persist
delay prehistory/phase, Epileptor input gains, integration method, an explicit
code version, or a saved time axis, and the entry point does not assert neutral
startup current. These do not invalidate the deterministic demonstration:
`dt`, duration, and monitor phase reconstruct time; source code establishes
the other settings; and independent review verifies zero startup diffusion.
The installed references already require these fields and check, so repeating
their wording would not improve the skill.

## Run 7 compared with Run 6

| Concern | Run 6 | Run 7 | Assessment |
|---|---|---|---|
| Grid reconstruction | Flattened condition tuples only | `grid_shape` plus named ordered axes and flattened tuples | Fixed. |
| Calibration disclosure | Generic phenomenological label | Event rule and sampled regimes explicitly outcome-calibrated and exploratory | Fixed. |
| Event evidence | Continuous sustained-window margin | Continuous cumulative-occupancy margin | Preserved for the declared exploratory rule. |
| Event rule | Predeclared continuous `x1 > 0` for 20 ms | Outcome-calibrated 2 ms cumulative positivity within 40 ms | Not confirmatory, but accurately disclosed and visually auditable. |
| Controls | No coupling and no stimulus pass | No coupling and no pulse pass; no-pulse control also drove baseline correction | Preserved and used more effectively. |
| Grid sensitivity | Coarse requested regimes without explicit status | Focal, partial, and routed conditions across four coupling and four pulse values | Improved exploratory sensitivity. |
| Mechanism metadata | Delay lifecycle, gains, integration, and schema saved | Several mechanism fields omitted | Minor regression; already covered by installed guidance. |
| Figure | Six panels over three pulse slices | Four aligned panels for one pulse, including route lags | Simpler and clearer for the prompt. |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P3 | `seizure_recruitment.py:39-42,110-140`; README and NPZ analysis status | The cumulative-occupancy event rule was selected after inspecting generated trajectories. | It cannot support a confirmatory seizure-detection claim, and another threshold/window could change boundary labels. | Keep the current exploratory label and continuous margins. Freeze the rule before independently declared confirmation if a confirmatory claim is later needed. |
| P3 | `seizure_recruitment.py:64-80,170-192,354-389`; NPZ | Delay prehistory/capacity/phase, Epileptor gains, integration method, time axis, and explicit code version are not all persisted. | Exact reconstruction requires the script even though the saved grid and labels are self-describing. | Persist those already-guided fields in a future production artifact; no new skill wording is needed. |
| P3 | `seizure_recruitment.py:64-80,195-213` | Matching delay and `x1` initializers make startup diffusion neutral, but the entry point does not assert the first coupling current is zero. | A later initializer change could introduce a silent startup transient. | Add the focused neutral-startup assertion already required by `coupling-network-api.md`. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Regional seizure dynamics | Three-region `brainmass.EpileptorStep` with region-wise `x0` and enabled first-channel gains | `brainmass.EpileptorStep` | Correct aggregate seizure model and input path. | Persist the non-default gains. |
| Directed anatomy | `CONNECTIVITY[target, source]` chain | Host-authored structure consumed by BrainMass coupling | Correct and saved with its convention. | Keep. |
| Same-channel pulse and coupling | Functional `brainmass.diffusive_coupling`, then add the focal pulse | BrainMass functional coupling boundary | Correct because `Network` would forward a caller input after coupling rather than sum it onto the same first channel. | Keep. |
| Delayed source history | Fixed-capacity `brainstate.nn.Delay`, insert then retrieve traced integer offset | `brainstate.nn.Delay` | Correct capacity and phase composition; exact impulse check passes. | Save capacity, prehistory, and phase. |
| State initialization | Construct and initialize a fresh Module inside each mapped condition | BrainState Module lifecycle | Correct independent State ownership. | Keep. |
| Time execution | One `brainstate.transform.for_loop` over 2,500 steps | BrainState control flow | Correct State-aware rollout and post-update monitoring. | Keep. |
| Sweep execution | One complete `brainstate.transform.vmap(run_condition)` over 50 lanes | BrainState vectorization | Correct stateful mapping of all grid conditions and controls. | Keep. |
| Physical quantities | BrainUnit for `dt`, duration, pulse timing, delay, and analysis windows | BrainUnit | Correct; documented dimensionless model inputs stay plain arrays. | Keep. |
| Event classification | JAX fixed-window cumulative occupancy with saved continuous maximum | Legitimate pure-array scientific boundary | Mechanically correct for its explicit exploratory predicate. | Freeze before independent confirmation. |
| Routed order | All regions classified plus strict focus-to-neighbor onset order | Scientific analysis boundary | Correct and directly supports propagation. | Keep. |
| Controls | Tagged no-coupling and no-pulse conditions in the same mapped path | Scientific design boundary | Correct and independently asserted. | Keep. |
| Grid construction | `u.math.meshgrid(indexing="ij")`, flatten, append controls | BrainUnit plus host parameter-design boundary | Correct execution and reconstruction contract. | Keep. |
| Persistence | Compressed NPZ with full trajectories, grid contract, labels, margins, protocol, and analysis status | Host serialization boundary | Deterministic and adequate for this demonstration. | Add the residual mechanism metadata for production reuse. |
| Presentation | One `plt.subplots` figure with a discrete regime map, measured delay lags, and matched traces | Host presentation boundary | Clear, simple, and aligned with the numeric bundle. | Keep. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused. `brainmass.Network` cannot
compose an independent drive onto the same first Epileptor input as its
coupling, and `brainmass.Simulator` does not own the required complete
stateful mapping over traced delay offsets. Direct functional coupling,
`brainstate.transform.vmap`, and `brainstate.transform.for_loop` are
therefore justified. Classification, grid reporting, serialization, and
presentation are legitimate host boundaries.

## Performance and code simplicity

- One complete `vmap` owns all 48 grid conditions and both controls.
- One `for_loop` owns all 2,500 State transitions in every mapped lane.
- Fixed delay capacity keeps State shape static across all traced offsets.
- All 50 trajectories are finite and the independent rerun reproduces every
  saved field exactly.
- One script, one numeric bundle, one figure, and one short README satisfy the
  request without extra framework or test scaffolding.
- The four-panel figure uses one `plt.subplots` call and aligns the displayed
  pulse, local trace, recruited trace, and timing comparison.

## Skill improvements

No further skill edit is justified by Run 7. The latest installed guidance
already requires:

- explicit exploratory/outcome-calibrated disclosure and nearby sensitivity;
- a dedicated grid shape plus named ordered axes;
- continuous evidence behind categorical labels;
- neutral delay prehistory and a zero-startup-current check;
- delay lifecycle, mechanism, timing, unit, seed, and code-version metadata.

Run 7 follows the first three and leaves only minor instances of the latter two.
Repeating existing rules would add duplication rather than a new transferable
decision boundary. Do not edit `brainx-general-guard`, the BrainMass root,
BrainMass references, BrainState, or BrainUnit from this diagnosis.

## Checks for the completed refinement

- Entry point compiles and executes independently with the required BrainX
  virtualenv.
- All 28 numeric fields reproduce exactly on an independent rerun.
- Delay capacity is static, update precedes retrieval, and the exact impulse
  phase test passes.
- Initial diffusive coupling is independently verified as exactly zero at
  every coupling axis value.
- All 48 grid conditions and both controls share one complete `vmap`;
  `for_loop` owns every timestep.
- Grid shape, named ordered axes, flattened coordinates, and tags reconstruct
  the exact `(4, 3, 4)` mapped layout.
- The NPZ retains the exact categorical predicate inputs, continuous margin,
  onsets, full trajectories, controls, and explicit outcome-calibrated status.
- Focal, partial, and fully routed outcomes appear; no-coupling stays focal and
  no-pulse stays quiet.
- Post-update timing is consistent, and measured route lags increase with the
  sampled propagation delay.
- The full-resolution figure is legible, unclipped, and consistent with the
  saved numeric evidence.
