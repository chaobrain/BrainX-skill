# BrainX diagnosis: seizure recruitment across regions, Run 2

## Evidence studied

- Exact prompt: `brainx-display-cases/06-seizure-recruitment/prompt.md` (560
  bytes; SHA-256
  `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`,
  `test_seizure_recruitment.py`, `README.md`,
  `seizure_recruitment.png`, `agent-final.md`, and the complete evaluator event
  log.
- Harness metadata: `gpt-5.6-sol`, `xhigh`, Codex CLI
  `0.147.0-alpha.1.2`, macOS Seatbelt host-read isolation, and exit code 0.
- Independent execution with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`: exit
  code 0; the selected local case recruited only region 0, while the selected
  spread case recruited regions 0 -> 1 -> 2 -> 3 at
  `[6.4, 9.5, 12.8, 16.5] ms`.
- Independent `unittest` execution: both the exact three-step delay-phase test
  and local/ordered-recruitment sweep test passed.
- Independent syntax compilation: both generated Python files passed.
- Independent numeric inspection: sweep axes were `(5, 4, 4)` and complete
  trajectories were `(5, 4, 4, 800, 4)`; every trajectory was finite; region
  recruitment counts were 1, 2, or 4; and all 28 all-region cases had strict
  region 0 -> 1 -> 2 -> 3 onset order.
- Independent control execution outside the delivered sweep: `k=0` with a
  pulse recruited only the source, and zero pulse with strong coupling
  recruited no region. These mechanisms work, but neither control is included
  in the mapped result or delivered evidence.
- Independent artifact inventory: the run contains no CSV, NPZ, JSON, or other
  numeric result bundle. The complete sweep and derived values exist only
  during execution.
- Independent full-resolution figure inspection: trace panels, sampled regime
  cells, blank non-recruited cells, axes, labels, and colorbars were visible and
  unclipped. Sparse delay values are displayed as discrete columns without
  connecting interpolation.
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
- Official BrainMass pages: model selection, FitzHugh-Nagumo and Epileptor,
  coupling and delays, custom coupling, parameter sweeps, central model API,
  central coupling API, and orchestration API.
- Official BrainState pages: delay protocol, `brainstate.nn.Delay`,
  vectorization, control flow, and generated `for_loop` and `vmap` APIs.
- Official BrainUnit guidance: unit-aware `meshgrid`, `diff`, `arange`,
  conversion, and array-function boundaries.

## Executive diagnosis

Run 2 preserves the corrected BrainX execution architecture and fixes the Run 1
scientific predicate. It uses a fixed-capacity `brainstate.nn.Delay` under one
`dt`, inserts before retrieval, verifies exact phase by impulse, composes
`brainmass.additive_coupling` with a same-channel focal drive, maps complete
independent conditions with state-aware `vmap`, and owns all timesteps with
`for_loop`. Unit-aware grid, range, difference, and conversion operations keep
physical time intact.

The recruitment label now requires `V >= 0.5` continuously for `1 ms`, fixed
before inspecting the sweep. Onset is the start of the first qualifying
window. The directed matrix and every all-region result enforce strict source
to neighbor order. The model is correctly labeled as a deterministic
phenomenological FHN demonstration rather than an Epileptor or clinical model.

Two coupled evidence failures remain. First, the 80-lane mapped sweep contains
only positive coupling and positive pulse values. No-coupling and no-pulse
controls therefore do not share the transformed path even though the artifact
attributes propagation to coupling and identifies the pulse as its external
seed. Second, the run saves only a PNG. The README says peak and onset evidence
is retained for every region, but full peaks are not derived and no grid,
trajectory, onset, label, predicate, control, connectivity, timing, or code
version is serialized. Re-execution can regenerate values, but that is not a
preserved result bundle.

## Run 2 compared with Run 1

| Concern | Run 1 | Run 2 | Assessment |
|---|---|---|---|
| Delay API and phase | Correct `Delay` plus impulse assertion | Correct `Delay` plus dedicated unit test | Preserved and better tested. |
| Same-channel composition | Correct direct additive composition | Correct direct additive composition | Preserved. |
| Stateful sweep | 72 grid lanes plus two controls in one `vmap` | 80 grid lanes, no controls | Core mapping preserved; controls regressed. |
| Event predicate | Any one threshold crossing | At least 1 ms continuously above threshold | Corrected. |
| Route order | Strict for one declared spread case; all complete cases passed independently | Strict selection and test; all complete cases pass independently | Improved. |
| Sparse delays | Four points joined by lines | Four explicit image columns | Corrected. |
| Continuous evidence | Full trajectories, peaks, and onsets saved | Full trajectories/onsets exist only in memory; representative peaks printed | Regressed. |
| Result bundle | NPZ and long-form CSV, with most metadata | PNG only | Regressed substantially. |
| Version metadata | Missing from NPZ | No numeric metadata bundle exists | Unresolved. |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `seizure_recruitment.py:42-44,116-132,251-262`; `test_seizure_recruitment.py:33-44` | Every mapped lane has positive coupling and a positive source pulse. Neither a no-coupling nor a no-pulse control runs through `vmap`. | The artifact attributes neighbor recruitment to coupling and the event to the pulse without preserving matched mechanism controls under the same execution path. | Append tagged zero-coupling and zero-pulse lanes to the same mapped call, derive the same sustained metrics, assert their expected outcomes, and save them with the grid. |
| P1 | `seizure_recruitment.py:251-262`; `README.md:15-18,28-32` | Only the figure is saved. Full continuous evidence, labels, coordinates, and metadata disappear after execution even though the README says they are retained. | The categorical map and claims cannot be audited from delivered numeric artifacts; exact results depend on rerunning code and cannot be tied to one artifact version. | Save a self-describing NPZ or equivalent containing grid and control coordinates, condition tags, traces or the exact tested reduction, peaks, labels, onsets, threshold, minimum duration, model/connectivity/protocol timing, units, and code version. |
| P3 | `seizure_recruitment.py:135-151` | The first qualifying local and spread cases are selected after viewing the same sweep used for display, with no sensitivity summary around those exact representatives. | Representative values are illustrative calibrated points, not held-out evidence or robust interval estimates. | Keep the phenomenological calibration label and report the surrounding sampled regime map, as the figure already does; do not generalize beyond sampled coordinates. |

FHN remains appropriate for the exact "seizure-like burst" request because the
artifact limits its claims to phenomenological excitable regional propagation.

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Represent regional excitability | `brainmass.FitzHughNagumoStep` | `brainmass.FitzHughNagumoStep` | Correct for the stated phenomenological boundary. | Keep. |
| Represent directed wiring | Constant `W[target, source]` chain | Host-authored structural data consumed by BrainMass coupling | Correct and explicit. | Save it with results. |
| Store delayed activity | Fixed-capacity `brainstate.nn.Delay` | `brainstate.nn.Delay` | Correct package-owned delay State. | Keep. |
| Define delay phase | Update then retrieve; exact three-step impulse test | `Delay.update`, `Delay.retrieve_at_step`, `for_loop` | Correct and directly verified. | Keep. |
| Compute regional input | `brainmass.additive_coupling` | `brainmass.additive_coupling` | Correct high-level functional kernel. | Keep. |
| Sum focal drive on the same input | Direct addition before FHN call | Direct composition required by `Network` first-input ownership | Correct. | Keep. |
| Initialize independent conditions | Construct and initialize model inside `run_condition` | BrainState Module lifecycle API | Correct per-lane State ownership. | Keep. |
| Scope physical time | Environment context plus BrainUnit `arange` | `brainstate.environ`, `brainunit.math` | Correct. | Keep. |
| Run all timesteps | One `brainstate.transform.for_loop` | `brainstate.transform.for_loop` | Correct. | Keep. |
| Construct unit-aware parameter grid | `u.math.meshgrid` | BrainUnit array creation | Correct after the evaluator caught raw-JAX incompatibility. | Keep. |
| Sweep complete conditions | One `brainstate.transform.vmap(run_condition)` | `brainstate.transform.vmap` | Correct for grid lanes but incomplete for controls. | Append controls before the same mapped call and tag them. |
| Classify sustained events | Fixed-duration `jax.lax.reduce_window` predicate | Legitimate scientific JAX boundary | Correct; no BrainX API owns the event definition. | Save the tested duration, threshold, reduction, and results. |
| Compare onset order | `u.math.diff` on quantity onsets | BrainUnit unit-aware math | Correct after replacing raw `jnp.diff`. | Keep. |
| Select representative cases | Host/JAX search over labels and onset order | Legitimate scientific analysis boundary | Correct for sampled illustrative cases. | Preserve indices/coordinates and nearby grid evidence. |
| Preserve continuous evidence | Full activity and onsets in memory; representative peaks printed | Host/JAX analysis and serialization boundary | Incomplete persistence. | Derive peaks for every lane and serialize them with labels/onsets or save full traces. |
| Validate mechanisms | Local/spread and delay-phase tests only | Host-side scientific validation boundary | Incomplete; causal controls are absent. | Test zero coupling and zero pulse through the same mapped path. |
| Plot scientific results | One high-level Matplotlib figure | Legitimate host presentation boundary | Clear, discrete, and unclipped. | Keep. |
| Serialize results | PNG only | Host serialization boundary | Inadequate for a parameter-sweep result. | Add a self-describing numeric bundle. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX simulation API is missing or misused. The remaining defects
are scientific-design and host-persistence responsibilities.

### `brainstate.nn.Delay`

Run 2 uses `Delay` correctly. Capacity is static, construction and
initialization occur under the execution `dt`, update precedes retrieval, and
the independent impulse test proves the convention.

### `brainmass.Network`

`Network` remains intentionally absent. It owns the first node input for
coupling and would forward an independent supplied value to the next positional
input. Direct composition is required to sum coupling and the focal drive on
FHN's first input.

### `brainmass.Simulator`

`Simulator` is not missing. The prompt explicitly requires `for_loop` and
`vmap`, and the same-channel custom composition needs one explicit stateful
step. BrainState correctly owns execution.

### Controls and serialization

No BrainX API owns causal control selection or NPZ/CSV persistence. Construct
control coordinates with BrainUnit/JAX arrays, execute them through the same
BrainState mapping, and serialize only after transformed execution at a clear
host boundary.

## Performance and code simplicity

- The 80 complete stateful conditions run in one `vmap`; every 800-step rollout
  runs in one `for_loop`. No Python simulation loop exists.
- Delay State shape stays static across all mapped lanes, while only retrieval
  offset varies.
- Complete trajectories occupy about one million float values and are already
  needed for sustained-window classification and representative traces. Saving
  them is reasonable at this scale; a larger sweep could save the exact window
  reduction, peaks, labels, and onsets, then rerun selected traces.
- The second `vmap` over the pure sustained classifier is valid but optional;
  raw `jax.vmap` or direct batched reduction would also be a pure-array
  boundary. It is not a correctness issue.
- The figure uses one `plt.subplots` call and basic plotting. Discrete image
  columns faithfully show sampled coordinates.
- Adding two tagged control lanes and one compressed result write does not
  require a new abstraction or custom infrastructure.

## Skill improvements

Make one surgical change in
`skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`:

1. Require causal baseline and mechanism controls to be appended to the same
   mapped parameter arrays when a regime claim attributes an outcome to a
   drive, coupling, intervention, or other varied mechanism. Tag controls and
   derive the same metrics rather than executing them on a separate path.
2. Make result persistence operational: before plotting or reporting, save a
   self-describing numeric bundle with grid ordering, control tags, continuous
   evidence, labels, full predicate settings, model/protocol/connectivity,
   physical timing and units, and explicit repository or artifact version.
   State that a figure, stdout, or in-memory value does not retain a sweep.

Do not edit `brainx-general-guard`: it already requires independent controls in
the same mapped path, complete per-condition evidence, continuous values behind
categorical maps, and full temporal predicates. Do not edit BrainMass model or
coupling guidance, BrainState, or BrainUnit; Run 2 follows those contracts.

## Checks for the next run

- The generated entry point and focused tests execute independently; every
  visual and numeric artifact is inspectable.
- FHN is labeled as phenomenological seizure-like regional propagation, not an
  Epileptor, physiological, calibrated, or clinical seizure mechanism.
- Delay uses a documented BrainX abstraction under one `dt`, with static
  capacity and an exact phase check.
- Same-channel coupling and focal drive use justified direct composition with
  `brainmass.additive_coupling`.
- Complete grid conditions plus no-coupling and no-pulse controls run through
  the same state-aware `vmap`; `for_loop` owns all timesteps.
- Recruitment requires a predeclared minimum consecutive duration, and onset
  is the first qualifying window.
- Propagated labels enforce strict region 0 -> 1 -> 2 -> 3 order.
- The no-coupling lane does not recruit neighbors; the no-pulse lane recruits
  no region.
- Continuous peaks, onsets, labels, and either full traces or the exact tested
  temporal reduction are saved for grid and control lanes.
- The numeric bundle stores ordered coordinates, condition tags, units, model
  identity, connectivity, fixed model/protocol parameters, `dt`, duration,
  threshold, minimum event duration, and explicit code/artifact version.
- Sparse delay coordinates remain discrete sampled points or cells without
  unjustified interpolation.
