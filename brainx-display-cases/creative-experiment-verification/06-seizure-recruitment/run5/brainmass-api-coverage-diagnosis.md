# BrainX diagnosis: seizure recruitment across regions, Run 5

## Evidence studied

- Exact prompt: `brainx-display-cases/creative-experiment-verification/06-seizure-recruitment/prompt.md` (560
  bytes; SHA-256
  `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`, `README.md`,
  `results/seizure_recruitment.npz`,
  `results/seizure_recruitment.png`, `agent-final.md`, and the complete event
  log. No separate generated test entry point exists.
- Harness metadata: `gpt-5.6-sol`, `xhigh`, Codex CLI
  `0.147.0-alpha.1.2`, macOS Seatbelt host-read isolation, and exit code 0.
- Independent execution from a disposable copy with the required BrainX
  virtualenv: compilation and entry-point execution pass; all 35 archived NPZ
  fields exactly match the independent rerun, including intentional `NaN`
  onsets.
- Numeric inspection: 36 grid conditions plus two same-path mechanism
  controls; result masks shaped `(38, 3)`; original grid shape `(4, 3, 3)` and
  ordered axes preserved; representative trajectories shaped
  `(2, 12000, 3)`; post-update time spans `0.1` through `1200.0 ms`.
- Independent delay-history check: Epileptor `x1` initializes to `-1.5` in all
  regions, but the delay history initializes to `0.0`. At `k=0.75`, that
  mismatch creates startup diffusive currents `[0.0, 1.125, 1.125]`; matching
  history to `-1.5` gives exactly zero startup coupling.
- Neutral-history comparison: replacing only the delay initializer with
  `Constant(-1.5)` preserves all 114 categorical region labels, but changes
  mutually recruited onsets by as much as `9.2 ms` and peak `x1` by as much as
  `0.31`.
- Event-log inspection: the evaluator first used a single-sample `x1 > 0`
  predicate, selected a 20 ms duration after inspecting representative event
  lengths, and then changed the observable to `LFP < 0` after the sustained
  `x1` rule failed the desired route ordering. The README nevertheless says
  the final rule was defined before the sweep.
- Full-resolution figure inspection: trajectories, post-update onset lines,
  regime cells, labels, axes, and explanatory text are clear and unclipped.
  Sparse delay coordinates are represented as discrete cells rather than
  connected samples.
- Owning guidance: BrainX general guard, BrainMass, BrainState, and BrainUnit
  root skills; BrainMass `modellibrary.md`, `coupling-network-api.md`,
  `parameter-sweeps-and-regime-analysis.md`, and
  `batch-transform-acceleration.md`; BrainState vectorization, control-flow,
  and delay guidance.
- Closest executable examples: BrainMass seizure Epileptor case study and
  delayed whole-brain pipeline.
- Official contracts: BrainMass model and coupling API pages; BrainState delay,
  `Delay`, `vmap`, `for_loop`, and initialization API pages indexed by the
  repository source inventories.

## Executive diagnosis

Run 5 fixes every Run 4 presentation, timing, and grid-reconstruction problem.
It uses `EpileptorStep`, fixed-capacity `brainstate.nn.Delay`,
`brainmass.diffusive_coupling`, one complete `vmap` for the grid and controls,
and one `for_loop` for time. Post-update timestamps and onsets are aligned,
both controls pass, strict routed order is asserted, and the numeric bundle is
deterministic and substantially more self-describing.

Two scientific-evidence problems remain. First, zero delay prehistory is not
neutral for diffusive coupling around the initialized Epileptor state; it
injects a startup transient and materially shifts continuous results. Second,
the final event observable and duration were selected after outcome inspection
but are described as pre-specified. That makes this an exploratory calibrated
classifier, not confirmatory evidence for a predicate fixed before the sweep.
The NPZ also retains labels and onsets without the per-condition continuous
sustained-window margin that directly produced each label, and it omits several
explicit mechanism metadata fields.

## Run 5 compared with Run 4

| Concern | Run 4 | Run 5 | Assessment |
|---|---|---|---|
| Model and input gains | Phenomenological FHN | Epileptor with nonzero `Kvf` and `Ks` | More seizure-specific and correctly driven. |
| Post-update time | One `dt` early | `0.1` through `1200.0 ms`; onsets shifted consistently | Fixed. |
| Sparse delays | Three delays connected by lines | Discrete regime cells | Fixed. |
| Grid reconstruction | Shape and ordered axes absent | Shape, axes, flattened coordinates, and tags saved | Fixed. |
| Delay capacity and phase | Capacity absent | Capacity saved and phase impulse asserted | Improved, but exact phase metadata is not persisted. |
| Delay prehistory | Zero, matching zero-initialized FHN | Zero against Epileptor `x1=-1.5` | Regressed scientifically; startup coupling is not neutral. |
| Predicate | Fixed sustained `V` rule | Sustained LFP rule chosen after inspecting alternatives | Execution is sustained, but the evidentiary claim is post hoc. |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `seizure_recruitment.py:54-58`; NPZ `delay_prehistory`; startup coupling | Delay history is filled with `0.0` while the coupled source `x1` initializes to `-1.5`. Zero is not neutral for diffusive coupling; it produces downstream startup currents up to `1.125` in the sampled grid. | The run begins with an artificial network perturbation. It leaves labels unchanged here but shifts onsets by up to `9.2 ms` and peak `x1` by `0.31`. | Initialize history to the coupled source State's baseline or warm the model consistently before classification. Verify that the first delayed diffusive current is zero and save the baseline or warm-up policy. |
| P1 | Event log items 45-55; `README.md:17-20`; classifier constants | The 20 ms duration was selected after viewing event lengths, and `LFP < 0` was selected after a sustained `x1` rule failed route order, yet the artifact says the predicate was defined before the sweep. | Outcome-guided classifier selection can manufacture the desired separation and overstates confirmatory evidence. | Label this result exploratory, or freeze the final predicate and rerun it unchanged on held-out conditions or a separately declared confirmation run. Never describe an outcome-tuned rule as pre-specified. |
| P2 | `seizure_recruitment.py:122-148,312-351`; NPZ | The bundle stores labels, onsets, threshold, duration, and peak `x1`, but not the continuous quantity that shows how close every region was to satisfying the sustained-LFP predicate. | A consumer cannot audit near-boundary negative labels or distinguish a 19.9 ms excursion from no event without regenerating all trajectories. | Save a per-condition/per-region continuous decision margin such as longest qualifying duration or maximum fixed-window hit count, with its units and exact reduction identity. |
| P2 | NPZ metadata | Coupling kernel and connectivity convention, integration method, delay update/retrieval phase, and explicit dimensionless units for coupling and perturbation are not stored. | Reconstructing the exact mechanism still requires reading the script. | Save compact explicit fields for these mechanism-selecting settings; retain the existing schema version as the artifact code version. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Regional seizure dynamics | `brainmass.EpileptorStep` with region-wise `x0`, `Kvf=1`, and `Ks=0.2` | `brainmass.EpileptorStep` | Correct seizure-specific model and enabled first-channel input path. | Keep. |
| Directed wiring | `W[target, source]` chain | Host-authored structure consumed by BrainMass coupling | Correct matrix orientation in code. | Persist the convention. |
| Delayed history | Fixed-capacity `brainstate.nn.Delay` | `brainstate.nn.Delay` | Correct API and static capacity; initializer is not neutral for the coupled State. | Match prehistory to source baseline or use a declared warm-up. |
| Coupling | `brainmass.diffusive_coupling` | BrainMass functional coupling kernel | Correct direct kernel for same-channel coupling plus pulse. | Keep and persist kernel identity. |
| Independent conditions | Model construction and initialization inside `run_condition` | BrainState Module lifecycle | Correct independent State ownership. | Keep. |
| Time execution | One `brainstate.transform.for_loop` | BrainState control flow | Correct transformed loop and post-update monitoring. | Keep. |
| Sweep execution | One `brainstate.transform.vmap` over grid and controls | BrainState vectorization | Correct complete-operation mapping. | Keep. |
| Unit boundaries | BrainUnit for `dt`, duration, stimulus timing, and delay; plain arrays for documented dimensionless parameters | BrainUnit | Correct. | Persist explicit dimensionless unit labels. |
| Sustained classification | `jax.lax.reduce_window` over LFP | Legitimate pure-array scientific boundary | Mechanically correct; selected post hoc and continuous margin not saved. | Declare exploratory/confirmatory status and save the decision margin. |
| Causal controls | Tagged no-coupling and no-perturbation lanes in the same `vmap` | Scientific design boundary | Correct. | Keep. |
| Numeric persistence | Compressed NPZ | Host serialization boundary | Deterministic and strong, with the listed metadata/evidence gaps. | Add only the missing decision evidence and mechanism fields. |
| Presentation | Matplotlib trajectories and cell maps | Host presentation boundary | Clear, discrete, and scientifically scoped. | Keep. |

## Missing, bypassed, or misused BrainX APIs

No higher-level BrainX orchestration API should replace the custom rollout.
`Network` cannot add an independent pulse to the same first Epileptor input,
and `Simulator` does not own the mapped variable-delay retrieval required by
the prompt. The material defect is the initializer supplied to the correct
`Delay` API, not a missing API. Classification, persistence, and presentation
are legitimate host boundaries.

## Performance and code simplicity

- Thirty-eight complete State-owning conditions execute in one `vmap`; all
  12,000 steps execute in one `for_loop`.
- Fixed delay capacity keeps mapped State shapes static while the retrieval
  offset varies.
- The deterministic rerun reproduces every saved field exactly.
- Matching delay history to the initial source State and returning one
  continuous classifier margin require no new abstraction or additional
  simulation pass.

## Skill improvements

1. Refine `skills/brainmass/references/coupling-network-api.md` to define neutral
   delay prehistory by the coupling equation: for diffusive coupling, match the
   delayed source baseline to the current target baseline or use an explicit
   warm-up, then assert zero startup coupling.
2. Refine
   `skills/brainmass/references/parameter-sweeps-and-regime-analysis.md` to
   distinguish pre-specified confirmation from outcome-guided exploratory
   calibration. Require a tuned predicate to be labeled exploratory or frozen
   and evaluated on independent confirmation evidence.
3. In the same sweep reference, make the continuous evidence requirement
   concrete for sustained events and require explicit mechanism metadata rather
   than relying on names embedded in source code.

Do not edit `brainx-general-guard`, the BrainMass root, BrainState, or BrainUnit:
the remaining failures are package-specific delayed-coupling and regime-evidence
decisions.

## Checks for the next run

- Entry point and any generated tests execute independently with the required
  BrainX virtualenv; every numeric and visual artifact is inspected.
- `Delay` has fixed capacity, uses one `dt`, inserts before retrieval, and passes
  the exact impulse phase test.
- Delay prehistory matches the coupled source baseline or follows a declared
  warm-up; initial diffusive coupling is zero and the policy is saved.
- Grid and controls share one complete `vmap`; `for_loop` owns all timesteps.
- The event predicate is fixed before outcome inspection, or the artifact is
  explicitly exploratory and uses separately declared confirmation evidence.
- Sustained events and strict route order are asserted; no-coupling is
  source-only and no-perturbation has no event.
- Post-update sample zero is `dt`, and saved onset values use the same phase.
- The NPZ stores grid shape, axes, flattened coordinates/tags, units,
  connectivity and convention, coupling kernel, non-default model parameters,
  delay capacity/prehistory/phase, monitor phase, integrator, predicate and
  continuous decision margin, seed, and artifact version.
- Sparse delay samples remain discrete cells or unconnected markers.
