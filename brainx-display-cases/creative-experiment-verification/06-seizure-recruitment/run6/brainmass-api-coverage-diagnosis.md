# BrainX diagnosis: seizure recruitment across regions, Run 6

## Evidence studied

- Exact prompt: `brainx-display-cases/creative-experiment-verification/06-seizure-recruitment/prompt.md` (560
  bytes; SHA-256
  `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`, `README.md`,
  `outputs/seizure_recruitment_results.npz`,
  `outputs/seizure_recruitment.png`, `agent-final.md`, and the complete
  event log. No generated test file exists.
- Harness metadata: `gpt-5.6-sol`, `xhigh`, Codex CLI
  `0.147.0-alpha.1.2`, and exit code 0.
- Independent execution from a disposable copy with the required BrainX
  virtualenv: compilation and entry-point execution pass; all 16 archived NPZ
  fields exactly match the independent rerun, including intentional `NaN`
  onsets.
- Numeric inspection: 36 grid conditions plus two same-path controls; 6,000
  post-update samples from `0.2` through `1200.0 ms`; trajectories shaped
  `(38, 6000, 3)`; four local, six partial, three routed, and 25
  no-sustained-burst conditions across the complete condition table.
- Independent startup check: Epileptor `x1` and all delay prehistory entries
  initialize to `-1.5`, so every first delayed-source minus current-target
  term is exactly zero before multiplication by connectivity or coupling.
- Event-log inspection: the agent fixed `x1 > 0` continuously for 20 ms
  before simulation outcomes, then outcome-calibrated regional timescales,
  perturbation values, and the coupling grid until both requested regimes
  appeared. The run is therefore exploratory parameter calibration under a
  predeclared classifier, not an independently confirmed regime map.
- Full-resolution figure inspection: all panels are legible and unclipped.
  The top row presents categorical counts at discrete grid cells, representative
  traces show local and routed cases, and the onset panel shows measured
  region-wise timing at the three sampled delays without implying interpolation
  between delays.
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

Run 6 resolves both consequential Run 5 scientific defects. Delay prehistory is
neutral relative to the diffusive equation, and the sustained event predicate is
fixed before model or grid calibration. The implementation now uses the correct
BrainMass abstractions and BrainState lifecycle: one complete mapped condition
owns independent State, one transformed loop owns time, delay capacity is
static, update/retrieval phase is tested, continuous classification evidence is
saved, strict routed order is enforced, and both causal controls pass.

Two P2 reproducibility disclosures remain. The result bundle saves only flattened
condition coordinates, not the original `(4, 3, 3)` grid shape and named
ordered axis arrays needed to reconstruct the mapped layout without source code.
It also labels the run only as a deterministic phenomenological demonstration,
although the event log shows that timescales, parameter axes, and displayed
regimes were selected after viewing outcomes. This is valid exploratory work,
but the artifact must say `outcome-calibrated` or `exploratory` and should
provide nearby sensitivity or independently declared confirmation before a
robust or confirmatory claim.

## Run 6 compared with Run 5

| Concern | Run 5 | Run 6 | Assessment |
|---|---|---|---|
| Delay prehistory | Zero against initial `x1=-1.5`; nonzero startup diffusion | Prehistory and current source both `-1.5`; zero startup diffusion | Fixed. |
| Predicate selection | Observable and duration selected after outcome inspection | `x1 > 0` for 20 ms fixed before calibration | Fixed for classifier selection. |
| Parameter selection | Outcome-guided but obscured by pre-specification wording | Classifier fixed, then timescales and grid outcome-calibrated | Scientifically coherent exploratory design, but disclosure remains too vague. |
| Continuous decision evidence | Not saved | `max_above_in_window_ms` saved for every condition and region | Fixed. |
| Mechanism metadata | Several fields absent | Coupling, connectivity convention, delay capacity/prehistory/phase, integration, units, and timing saved | Fixed. |
| Grid reconstruction | Shape and ordered axes saved | Only flattened coordinates saved | Regressed; explicit grid contract is missing. |
| Sparse-delay presentation | Discrete cells | Discrete cells plus measured route-wise timing at each sampled delay | Correct. |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P2 | Event log items 54-85; `README.md:25-27`; NPZ `metadata_json.calibration_status` | The classifier was predeclared, but regional timescales, perturbation values, coupling values, and displayed regimes were selected after inspecting outcomes. The metadata says only `deterministic phenomenological demonstration`. | A consumer cannot distinguish an outcome-calibrated exploratory map from a predeclared or independently confirmed result. The artifact supports this deterministic demonstration, but not an unqualified robust-regime claim. | Record parameter-selection status explicitly as `exploratory` or `outcome-calibrated`, identify what was tuned, and provide nearby sensitivity or independently declared confirmation before a confirmatory or robust claim. |
| P2 | `seizure_recruitment.py:237-255`; NPZ | The numeric bundle omits `grid_shape` and named ordered arrays for the coupling, delay, and perturbation axes. Flattened per-condition columns preserve sampled tuples but not the declared reshape contract. | Reconstructing the `(coupling, delay, perturbation)` tensor and its axis order requires reading the script and inferring `meshgrid(indexing="ij")`. | Save `grid_shape`, `coupling_axis`, `delay_axis_ms`, and `perturbation_axis` as dedicated fields in addition to flattened condition columns and tags. |
| P3 | `seizure_recruitment.py:77-95,397-428` | Neutral startup coupling is established by matching initializers but is not asserted in the generated entry point. | A later change to either baseline could silently reintroduce an artificial startup transient even while the independent review passes. | Add a focused zero-startup-current assertion when timing conclusions are sensitive to delayed coupling initialization. This is a useful artifact check; existing skill guidance already requires it, so no further skill edit is needed. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Regional seizure dynamics | Three-region `brainmass.EpileptorStep` bank with explicit `x0`, gains, timescales, and initial `x1` | `brainmass.EpileptorStep` | Correct seizure-specific aggregate model. | Keep; label its outcome-calibrated parameterization. |
| Directed chain structure | `CONNECTIVITY[target, source]` | Host-authored structure consumed by BrainMass coupling | Correct and explicitly documented. | Keep. |
| Same-channel drive and coupling | Functional `brainmass.diffusive_coupling`, then add the focal perturbation before the first Epileptor input | BrainMass functional coupling boundary | Correct because `Network` would forward caller inputs after coupling rather than sum an independent drive onto the same first channel. | Keep. |
| Delayed history | Fixed-capacity `brainstate.nn.Delay`, insert before traced retrieval | `brainstate.nn.Delay` | Correct static-capacity and phase composition. | Add the already-guided neutral startup assertion. |
| State initialization | Construct model inside one condition and call `brainstate.nn.init_all_states` | BrainState Module lifecycle | Correct independent State ownership. | Keep. |
| Time execution | One `brainstate.transform.for_loop` over 6,000 indices | BrainState control flow | Correct State-aware transformed loop. | Keep. |
| Independent sweep execution | One complete `brainstate.transform.vmap(run_condition)` across 38 lanes | BrainState vectorization | Correct; simulation State, not only input construction or scoring, is mapped. | Keep. |
| Physical quantities | BrainUnit for `dt`, duration, delays, timescales, and stimulation timing; declared unitless model inputs at their documented boundary | BrainUnit | Correct. | Keep. |
| Sustained-event classification | Pure JAX fixed-window reduction over saved `x1` | Legitimate host/pure-array scientific boundary | Correct predicate and continuous margin. | Keep. |
| Routed propagation | All regions recruited plus strict focus-to-neighbor onset order | Scientific analysis boundary | Correct and directly supports the claim. | Keep. |
| Mechanism controls | Tagged no-coupling and no-stimulus lanes in the same mapped path | Scientific design boundary | Correct independent controls. | Keep. |
| Grid construction | `meshgrid(indexing="ij")`, flatten, and append controls | Host parameter-design boundary | Correct execution layout; saved reconstruction metadata is incomplete. | Persist the original grid shape and named axes. |
| Numeric persistence | Compressed NPZ with trajectories, labels, continuous margins, coordinates, timing, connectivity, and metadata | Host serialization boundary | Deterministic and strong aside from grid and calibration disclosures. | Add only the missing fields and explicit selection status. |
| Presentation | One `plt.subplots` figure with discrete heatmaps, trajectories, and onset comparisons | Host presentation boundary | Clear and scientifically scoped. | Keep. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused. `brainmass.Network` cannot add
an independent perturbation to the same first Epileptor input as its coupling,
so direct functional coupling is the correct package-owned composition.
`brainmass.Simulator` does not own this mapped variable-delay retrieval path,
so `brainstate.transform.vmap` around a complete condition and
`brainstate.transform.for_loop` over time are justified. Classification,
parameter-table construction, serialization, and presentation are legitimate
host boundaries.

## Performance and code simplicity

- One complete `vmap` owns all 36 grid conditions and both controls.
- One `for_loop` owns all 6,000 State transitions in every mapped lane.
- Fixed delay capacity preserves mapped State shape while retrieval offset
  varies.
- The result retains all trajectories because the prompt requires showing when
  recruitment occurs; the continuous decision margin prevents categorical
  labels from becoming opaque.
- The independent rerun exactly reproduces all 16 archived fields.
- Adding four grid fields and a precise calibration-status field requires no
  new abstraction, simulation pass, or change to the scientific path.

## Skill improvements

Make one surgical refinement to
`skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`:

1. Require a dedicated grid-shape field and named ordered axis fields in the
   numeric result bundle; flattened condition coordinates and tags do not
   replace that reconstruction contract.
2. Require parameters, grids, displayed cases, or other scientific settings
   selected after inspecting outcomes to be labeled explicitly
   `exploratory` or `outcome-calibrated`. Require nearby sensitivity or
   independently declared confirmation before a robust or confirmatory claim,
   even when the classifier itself was fixed before calibration.

Do not edit `brainx-general-guard`, the BrainMass root, `modellibrary.md`,
`coupling-network-api.md`, BrainState, or BrainUnit. Their current guidance
already covers model choice, input gains, outcome calibration, neutral
prehistory, exact delay phase, mapped control composition, and continuous
classifier evidence.

## Checks for the next run

- Entry point and any generated tests execute independently with the required
  BrainX virtualenv; every numeric and visual artifact is inspected.
- `Delay` has fixed capacity, uses the rollout `dt`, inserts before
  retrieval, and passes an exact impulse phase test.
- Delay prehistory is neutral for the actual coupling equation; zero startup
  coupling is asserted and the policy is saved.
- Grid conditions and controls share one complete `vmap`; `for_loop` owns
  every timestep.
- The observable, direction, threshold, duration, and route predicate are fixed
  before outcome inspection, or every changed component is declared
  exploratory.
- Every outcome-tuned model parameter, protocol value, grid axis, or displayed
  case is explicitly labeled exploratory or outcome-calibrated; nearby
  sensitivity or independent confirmation supports any robustness claim.
- Sustained events and strict route order are asserted; no-coupling is
  source-only and no-stimulus has no event.
- Post-update sample zero and onset values both start at one `dt`.
- The NPZ stores `grid_shape`, named ordered axis arrays, flattened
  coordinates and tags, continuous margins, labels, explicit units, model and
  mechanism metadata, connectivity and convention, delay lifecycle metadata,
  predicate settings, monitor phase, seed, and code version.
- Sparse sampled coordinates remain discrete unless their interval is actually
  resolved or a declared interpolation model supports connecting them.
