# BrainX diagnosis: seizure recruitment across regions, Run 3

## Evidence studied

- Exact prompt: `brainx-display-cases/creative-experiment-verification/06-seizure-recruitment/prompt.md` (560
  bytes; SHA-256
  `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`, `README.md`,
  `outputs/seizure_recruitment_results.npz`,
  `outputs/seizure_recruitment.png`, `agent-final.md`, and the complete
  evaluator event log.
- Harness metadata: `gpt-5.6-sol`, `xhigh`, Codex CLI
  `0.147.0-alpha.1.2`, macOS Seatbelt host-read isolation, and exit code 0.
- Independent execution with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`: exit
  code 0; the selected local case recruited only region 1 at `110.9 ms`, the
  selected spreading case recruited regions 1 -> 2 -> 3 at
  `[110.9, 180.5, 251.2] ms`, and the no-coupling and no-drive controls
  produced local and no-event outcomes respectively.
- No separate generated test entry point exists. The entry point independently
  executed its delay impulse assertion, calibrated-case assertions, strict
  onset-order assertion, control assertions, and finite-value check.
- Independent result-bundle comparison: all 31 NPZ arrays and scalar fields
  from the rerun exactly match the archived bundle, including floating arrays
  with intentional `NaN` onsets for unrecruited regions.
- Independent numeric inspection: 45 grid conditions plus two tagged controls
  produced trajectories shaped `(47, 15000, 3)`; all trajectory values were
  finite; regime counts were 1 no-event, 23 local, 5 partial, and 18 ordered
  full-recruitment conditions; every full-recruitment condition had strict
  region 1 -> 2 -> 3 onset order.
- Independent predicate reconstruction from the saved `x1`, threshold,
  minimum duration, and `dt`: every saved recruited mask and onset was
  reproduced. Post-update sample times span `0.1` through `1500.0 ms`.
- Independent full-resolution figure inspection: all panels, labels, sampled
  regime cells, traces, onset markers, stimulus windows, legends, and colorbar
  were visible and unclipped. The three latency samples at delays 2, 6, and
  10 ms are connected by lines despite the sparse sampling.
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
- Official BrainMass pages: Epileptor case study, generated `EpileptorStep`
  API, coupling and delays, generated `diffusive_coupling` API, custom
  coupling, parameter sweeps, and orchestration.
- Official BrainState pages: delay protocol, generated `brainstate.nn.Delay`,
  vectorization, generated `vmap`, control flow, generated `for_loop`, and
  generated `init_all_states` APIs.

## Executive diagnosis

Run 3 satisfies the core scientific and BrainX execution contract. It uses
three healthy `EpileptorStep` regions with explicit `Kvf` and `Ks` input gains,
a directed structural chain, fixed-capacity `brainstate.nn.Delay`, direct
`diffusive_coupling` plus a same-channel focal drive, one state-aware `vmap`
over every grid and control condition, and one `for_loop` over all timesteps.
The delay buffer is constructed, initialized, and executed under one `dt`;
update precedes retrieval; an exact impulse assertion locks the phase; and the
prehistory equals the node's initial `x1`, avoiding a startup coupling
transient.

The scientific result is also well supported. Recruitment requires `x1 > 0`
continuously for 1 ms, all propagated labels enforce strict route order, both
mechanism controls use the same mapped path, full trajectories and continuous
boundary evidence are saved, and the calibrated regime is explicitly labeled
phenomenological rather than patient calibrated. Independent execution exactly
reproduces the archived bundle.

Two presentation and persistence gaps remain. The latency panel joins three
sparse delay samples, visually asserting behavior between unmeasured values.
The NPZ is much improved but not fully self-describing: it omits the inactive
`Kf` input gain, coupling family, delay capacity and prehistory, integration
method, and seed. Those values materially distinguish the mechanism and exact
execution but currently require reading the script or relying on defaults.

## Run 3 compared with Run 2

| Concern | Run 2 | Run 3 | Assessment |
|---|---|---|---|
| Regional mechanism | Four-region phenomenological FitzHugh-Nagumo chain | Three-region Epileptor chain with enabled fast and slow input gains | More package- and domain-specific while remaining explicitly phenomenological. |
| Delay API and phase | Correct fixed-capacity `Delay` plus impulse test | Correct fixed-capacity `Delay`, matched prehistory, and impulse assertion | Preserved and startup behavior improved. |
| Stateful sweep | Complete positive grid only | Complete grid plus tagged no-coupling and no-drive controls in one `vmap` | Corrected. |
| Event predicate | Sustained threshold predicate | Sustained threshold predicate with post-update onset alignment | Preserved and timestamp semantics improved. |
| Route order | Strict four-region order | Strict three-region order in every full-recruitment lane | Preserved. |
| Numeric evidence | No numeric result bundle | Exact reproducible NPZ with full trajectories, predicates, labels, controls, and most metadata | Corrected substantially; a few mechanism settings remain implicit. |
| Sparse delay presentation | Discrete image columns | Discrete map rows, but the latency panel connects three samples | Regressed in one panel. |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P2 | `seizure_recruitment.py:368-387`; `outputs/seizure_recruitment.png` | Each latency series connects only three sampled delay coordinates with a line. No interpolation model or dense sampling supports values between 2, 6, and 10 ms. | The panel visually implies a continuous latency curve and trend over unsampled delays, exceeding the evidence retained by the sweep. | Show each delay as an unconnected sampled point, for example with `scatter()` or `plot(..., linestyle="none", marker="o")`; connect samples only after adding and justifying sufficient resolution or an interpolation model. |
| P2 | `seizure_recruitment.py:243-280`; `outputs/seizure_recruitment_results.npz` | The numeric bundle omits `Kf=0`, the diffusive coupling identity, `MAX_DELAY`, delay prehistory `-1.5`, integration method `exp_euler`, and seed 0. These are mechanism- or execution-defining settings even though several use defaults. | The bundle cannot independently distinguish this exact input path, delay startup, or numerical configuration from another run with the same coordinates and labels. Reproduction still succeeds only because the script is preserved beside it. | Save every mechanism-defining constructor choice and execution setting, including relevant inactive/default gains, coupling family, delay capacity and initializer, integration method, seed, and explicit unit labels. |

The selected local and spreading points are calibrated from the displayed grid,
not held-out estimates. The figure shows nearby sampled sensitivity and both
README and bundle label the values phenomenological, so no stronger correction
is required.

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Represent regional seizure dynamics | `brainmass.EpileptorStep` with healthy `x0` and driven input gains | `brainmass.EpileptorStep` | Correct model family and input path. | Preserve and record all relevant gains/settings. |
| Represent directed wiring | Constant `W[target, source]` chain | Host-authored structural data consumed by BrainMass coupling | Correct and saved. | Keep. |
| Store delayed activity | Fixed-capacity `brainstate.nn.Delay` | `brainstate.nn.Delay` | Correct package-owned delay State. | Preserve and save capacity/prehistory metadata. |
| Define delay phase | Update then retrieve; exact three-step impulse assertion | `Delay.update`, `Delay.retrieve_at_step`, `for_loop` | Correct and directly verified. | Keep. |
| Prevent startup coupling | Delay prehistory `-1.5` matches initial `x1` | `Delay(..., init=...)` | Correct. | Save the initializer value. |
| Compute regional coupling | `brainmass.diffusive_coupling` | `brainmass.diffusive_coupling` | Correct high-level functional kernel and matrix orientation. | Save the coupling family. |
| Sum focal drive on the same input | Direct addition before `EpileptorStep.update` | Direct composition required by `Network` first-input ownership | Correct. | Keep. |
| Initialize independent conditions | Construct and initialize the complete Module inside `run_condition` | BrainState Module lifecycle API | Correct per-lane ownership and independently executable. | Keep. |
| Scope physical time | BrainState environment contexts with BrainUnit quantities | `brainstate.environ`, BrainUnit quantities | Correct. | Keep explicit unit metadata at serialization. |
| Run all timesteps | One `brainstate.transform.for_loop` | `brainstate.transform.for_loop` | Correct State-driven loop. | Keep. |
| Sweep independent conditions | One `brainstate.transform.vmap(run_condition)` | `brainstate.transform.vmap` | Correct complete-operation mapping; all grid and controls share it. | Keep. |
| Classify sustained events | Fixed-duration `jax.lax.reduce_window` predicate | Legitimate scientific JAX boundary | Correct; no BrainX API owns the event definition. | Keep. |
| Enforce route order | Host/JAX comparisons over unit-aware onsets | Legitimate scientific analysis boundary | Correct and independently reconstructed. | Keep. |
| Validate mechanisms | Tagged zero-coupling and zero-drive lanes plus assertions | Host-side scientific validation boundary | Correct; controls share model and metric paths. | Keep. |
| Preserve continuous evidence | Full `x1`, LFP, peaks, onsets, window hits, masks, and labels in NPZ | Host serialization boundary | Strong and exactly reproducible. | Add the remaining mechanism/execution metadata. |
| Plot categorical map | Discrete `imshow` cells at saved coordinates | High-level Matplotlib host boundary | Correct and faithful to sparse sampling. | Keep. |
| Plot latency sensitivity | `plot()` with markers and connecting lines | High-level Matplotlib host boundary | Scientifically overstated between sparse samples. | Use unconnected markers. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX simulation API is missing or misused.

### `brainmass.Network`

`Network` is intentionally absent. Its coupling occupies the node's first
positional input, while a caller-supplied value would be forwarded as the
second input. Direct composition is required to sum coupling and the pulse on
Epileptor's first input channel.

### `brainmass.Simulator`

`Simulator` is not missing. The prompt explicitly requires `for_loop` and
`vmap`, delay retrieval varies inside the mapped operation, and same-channel
composition needs one custom stateful step. BrainState correctly owns this
execution path.

### `brainstate.nn.Delay`

The artifact uses `Delay` correctly. Capacity is static, one `dt` governs
construction through execution, update precedes retrieval, prehistory prevents
false startup input, and the impulse assertion proves exact offset semantics.

### Analysis and serialization

JAX reduction, NumPy serialization, and Matplotlib presentation are legitimate
host boundaries. The remaining issues concern how those boundaries represent
sampling and metadata, not a missing BrainX API.

## Performance and code simplicity

- All 47 complete stateful conditions execute in one `vmap`; every 15,000-step
  rollout executes in one `for_loop`. There is no Python simulation loop.
- Delay State capacity remains static across mapped lanes while only the
  retrieval offset changes.
- Full trajectories occupy about 10 MB compressed and are necessary to audit
  the temporal predicate and representative traces at this sweep size.
- Raw `jax.vmap` over the pure post-simulation classifier is a valid array-only
  boundary and does not duplicate State-aware execution.
- The figure uses one `plt.subplots` call and basic high-level plotting. Remove
  line interpolation from the sparse latency panel without adding layout
  machinery.
- Adding scalar/string metadata to the existing NPZ requires no new
  abstraction or file.

## Skill improvements

Make two narrow clarifications in
`skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`:

1. Make sparse-coordinate presentation executable: default to discrete cells,
   `scatter`, or marker-only plots; permit connected lines only when the
   sampling resolution or a declared interpolation model supports inference
   between coordinates.
2. Define mechanism-complete persistence: save every setting required to
   distinguish the simulated mechanism and reproduce execution, including
   inactive/default gains that govern selected input channels, coupling
   identity, delay capacity/prehistory, integrator, seed, and explicit units
   when those concerns apply.

Do not edit `brainx-general-guard`: it already requires sampled-point claims,
full predicates, same-path controls, and per-condition evidence. Do not edit
BrainMass model/coupling guidance, BrainState, or BrainUnit; Run 3 follows
their contracts.

## Checks for the next run

- The generated entry point and any generated tests execute independently with
  the required BrainX virtualenv; every numeric and visual artifact is
  inspectable.
- Epileptor input gains are enabled deliberately, and unsourced calibrated
  values remain labeled phenomenological rather than clinical.
- Delay uses fixed-capacity `brainstate.nn.Delay` under one `dt`, inserts before
  retrieval, initializes neutral prehistory, and proves exact phase by impulse.
- Coupling and same-channel drive use justified direct functional composition.
- The complete grid plus no-coupling and no-drive controls share one state-aware
  `vmap`; `for_loop` owns all timesteps.
- Recruitment requires the declared sustained duration, and propagated labels
  enforce strict source -> neighbor onset order.
- Controls independently produce source-only and no-event outcomes.
- Continuous peaks, onsets, masks, labels, and either trajectories or the exact
  tested reduction are saved for every lane.
- The numeric bundle records coordinates, tags, units, model identity, all
  mechanism-defining parameters and gains, coupling/connectivity, delay
  capacity/prehistory, timing, integrator, seed, predicate, and code/artifact
  version.
- Every sparse delay display uses discrete cells or unconnected markers unless
  denser sampling or an explicit interpolation analysis justifies lines.
