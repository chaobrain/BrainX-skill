# BrainX diagnosis: seizure recruitment across regions, Run 4

## Evidence studied

- Exact prompt: `brainx-display-cases/creative-experiment-verification/06-seizure-recruitment/prompt.md` (560
  bytes; SHA-256
  `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`, `README.md`,
  `outputs/seizure_recruitment_results.npz`,
  `outputs/seizure_recruitment.png`, `agent-final.md`, and the complete event
  log. No separate generated test entry point exists.
- Harness metadata: `gpt-5.6-sol`, `xhigh`, Codex CLI
  `0.147.0-alpha.1.2`, macOS Seatbelt host-read isolation, and exit code 0.
- Independent execution with the required BrainX virtualenv: exit code 0; the
  local case recruited only region 1, the spreading case recruited regions
  1 -> 2 -> 3 at saved onsets `[20.7, 27.9, 35.2] ms`, and the no-coupling and
  no-stimulus controls produced source-only and no-event outcomes.
- Independent result-bundle comparison: all 31 arrays and scalar fields from
  the rerun exactly match the archived NPZ, including intentional `NaN` onsets.
- Numeric inspection: 60 grid conditions plus two controls; recruitment masks
  shaped `(62, 3)`; representative trajectories shaped `(4, 2000, 3)`;
  finite peak and sustained-window evidence; explicit model scope, units,
  coupling identity, seed, integration method, and delay phase/prehistory.
- Timing inspection: `for_loop` returns the State after each update, but saved
  representative times start at `0.0 ms` and end at `199.9 ms`; onsets use the
  unshifted first qualifying array index. The first saved activity therefore
  actually belongs at `0.1 ms`, and every saved onset is one `dt` early.
- Full-resolution figure inspection: the local/recruited traces, stimulus
  windows, threshold, legends, sampled regime cells, labels, and colorbar are
  legible and unclipped. The recruitment-timing panel still connects the three
  delay samples at 2, 6, and 10 ms.
- Owning guidance: BrainX general guard, BrainMass, BrainState, and BrainUnit
  root skills; BrainMass `modellibrary.md`, `coupling-network-api.md`,
  `parameter-sweeps-and-regime-analysis.md`, and
  `batch-transform-acceleration.md`; BrainState vectorization and control-flow
  references.
- Closest executable examples: BrainMass seizure Epileptor case study and
  resting-state delayed-network pipeline.
- Official contracts: FitzHugh-Nagumo generated API, BrainMass coupling API,
  BrainState delay protocol and generated `Delay`, `vmap`, `for_loop`, and
  `init_all_states` APIs.

## Executive diagnosis

Run 4 retains a correct, simple BrainX architecture and improves Run 3's
persistence substantially. It uses a documented phenomenological
`FitzHughNagumoStep` regional model, fixed-capacity `brainstate.nn.Delay`,
`brainmass.additive_coupling`, one complete state-aware `vmap` for 60 grid
conditions and two matched controls, and one `for_loop` for every timestep.
Physical time and explicit dimensionless input contracts use BrainUnit. The
delay phase is locked by an impulse assertion, delay prehistory matches the
initial State, recruitment requires `V >= 0.5` continuously for 1 ms, and the
positive case enforces strict region 1 -> 2 -> 3 order. Scientific wording is
properly limited to a phenomenological seizure-like demonstration.

Three residual evidence problems remain. Post-update output is labeled one
integration step early in both saved times and onset values. The timing panel
still connects only three sparse delay samples. The NPZ saves flattened
condition columns but not the original grid shape or ordered axis arrays, and
it names zero delay prehistory without saving the fixed delay capacity. Thus a
consumer cannot reconstruct the mapped grid or exact delay configuration from
the numeric bundle alone.

## Run 4 compared with Run 3

| Concern | Run 3 | Run 4 | Assessment |
|---|---|---|---|
| Model | Driven Epileptor | Driven FitzHugh-Nagumo | Both valid; Run 4 correctly limits the cheaper model to phenomenological seizure-like recruitment. |
| Controls and mapped execution | 45 grid + 2 controls in one `vmap` | 60 grid + 2 controls in one `vmap` | Preserved. |
| Predicate and route order | Sustained predicate with strict order | Sustained predicate with strict order | Preserved. |
| Post-update timestamps | Corrected to start at one `dt` | Start at zero and onsets use zero-based sample indices | Regressed. |
| Mechanism metadata | Most settings, but several implicit | Adds units, model scope, seed, method, coupling, prehistory, and phase | Improved substantially. |
| Grid reconstruction | Shape and ordered axes saved | Only flattened coordinate columns saved | Regressed. |
| Sparse timing presentation | Three samples connected | Three samples connected | Unresolved. |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `seizure_recruitment.py:116-119,135-136,332`; NPZ `onset_ms`, `representative_time_ms`; figure trace/timing panels | `for_loop` records State after each update, but time sample 0 is labeled `0.0 ms`, and first qualifying windows are multiplied by `dt` without the post-update `+1` offset. | Every reported onset and saved trace time is one `dt` early, so the protocol and event evidence are physically misaligned. | Save post-update times as `(arange(n_steps) + 1) * dt` and compute qualifying onset as `(first_start + 1) * dt`; state the monitor phase in metadata. |
| P2 | `seizure_recruitment.py:211-225`; figure recruitment-timing panel | Three sampled delays are joined by continuous lines. | The figure implies timing values between unmeasured delays without sufficient resolution or an interpolation model. | Use `scatter()` or `plot(..., linestyle="none", marker="o")`; connect only after the sampling design supports the interpolation. |
| P2 | `seizure_recruitment.py:334-371`; NPZ | The numeric bundle omits the original `grid_shape`, ordered coupling/delay/size axis arrays, and `MAX_DELAY` capacity. | Flattened rows cannot be reconstructed into the exact mapped grid without reading code, and the fixed delay configuration is incomplete. | Save the grid shape, each ordered axis array with units, and delay capacity. Continue saving per-condition columns and control tags. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Regional excitable dynamics | `brainmass.FitzHughNagumoStep(tau=20 ms)` | `brainmass.FitzHughNagumoStep` | Correct for explicitly phenomenological seizure-like events. | Keep. |
| Directed wiring | Saved `W[target, source]` chain | Host-authored structural data consumed by BrainMass | Correct. | Keep. |
| Delayed history | Fixed-capacity `brainstate.nn.Delay` with zero prehistory | `brainstate.nn.Delay` | Correct phase and neutral startup. | Save capacity numerically. |
| Regional coupling | `brainmass.additive_coupling` | `brainmass.additive_coupling` | Correct direct kernel for same-channel drive composition. | Keep. |
| Independent conditions | Model constructed and initialized inside `run_condition` | BrainState Module lifecycle | Correct independent State ownership. | Keep. |
| Time execution | One `brainstate.transform.for_loop` | `brainstate.transform.for_loop` | Correct loop, but post-update output phase is mislabeled. | Shift saved sample/onset coordinates by one `dt`. |
| Sweep execution | One `brainstate.transform.vmap` | `brainstate.transform.vmap` | Correct complete-operation mapping for grid and controls. | Keep. |
| Unit boundaries | BrainUnit time and explicit `UNITLESS` inputs converted at documented raw model boundary | BrainUnit | Correct. | Keep. |
| Sustained classification | JAX minimum reduction over fixed windows | Legitimate pure-array scientific boundary | Correct predicate, incorrect reported time coordinate. | Preserve reduction and fix phase. |
| Causal controls | Tagged no-coupling and no-stimulus lanes | Scientific design boundary | Correct and same-path. | Keep. |
| Numeric persistence | Compressed NPZ with continuous summaries and representative traces | Host serialization boundary | Strong but incomplete grid/delay reconstruction metadata. | Add axes, shape, capacity, and monitor phase. |
| Presentation | One `plt.subplots` figure with basic Matplotlib | Host presentation boundary | Clear except for sparse-line interpolation. | Use marker-only delay samples. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused. `Network` is intentionally
absent because coupling and the independent pulse must sum on FHN's first
input. `Simulator` is intentionally absent because the prompt requires the
explicit mapped custom rollout and variable delay retrieval. Analysis,
serialization, and presentation are legitimate host boundaries.

## Performance and code simplicity

- Sixty-two complete State-owning simulations execute in one `vmap`; all 2,000
  timesteps execute in one `for_loop`.
- Fixed delay capacity prevents mapped shape specialization while retrieval
  offset varies by condition.
- Only four representative trajectories are saved; every condition retains
  the exact continuous peak and sustained-window floor used for labels. This is
  a compact and auditable persistence design.
- Correcting timestamps, changing one plotting keyword, and adding a few NPZ
  fields need no new abstraction.

## Skill improvements

Make three compact additions in
`skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`:

1. Add an explicit marker-only plotting form for sparse coordinates so the
   directive cannot be satisfied by markers plus connecting lines.
2. State the post-update sampling rule: when a custom step updates State then
   returns it, sample index zero is at one `dt`; shift both saved time arrays
   and onset indices accordingly. Save the monitor phase.
3. Require original grid shape and ordered axis arrays alongside flattened
   condition columns, plus fixed delay capacity when delay retrieval varies.

Do not require exhaustive serialization of every untouched model default.
Record non-default or mechanism-selecting parameters and version the code or
artifact schema; the purpose is reproducible decision evidence, not a duplicate
constructor catalogue.

Do not edit `brainx-general-guard`, coupling/model guidance, BrainState, or
BrainUnit: their current rules already own the relevant selection and execution
contracts.

## Checks for the next run

- Entry point and generated tests execute independently with the BrainX
  virtualenv; numeric and visual artifacts are inspectable.
- The model's scientific scope and calibrated status are stated accurately.
- Fixed-capacity `Delay` uses one `dt`, neutral prehistory, insert-before-read,
  and exact impulse validation.
- Grid and both causal controls share one complete state-aware `vmap`;
  `for_loop` owns time.
- Sustained recruitment and strict route order are enforced.
- Saved post-update time begins at `dt`, ends at duration, and onset values
  match that phase.
- The result bundle stores grid shape, ordered axes, flattened coordinates and
  tags, units, model/coupling/connectivity, non-default mechanism settings,
  delay capacity/prehistory/phase, timing, predicate, seed, integrator, and
  artifact version.
- Sparse delay samples render as discrete cells or unconnected markers.
