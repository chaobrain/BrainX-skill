# BrainX diagnosis: alternating theta sweeps in a direction-grid network

## Evidence studied

- Original 2,348-byte `prompt.md` with SHA-256 `8c684671317ad1b54e59afd3ec167c1c24df6baa59d3f108af5588fb38cf4fed`.
- Generated `README.md`, `theta_sweep_network.py`, `test_theta_sweep_network.py`, `summary.json`, cycle CSV, NPZ evidence, and all three PNG figures.
- A clean full execution with `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python theta_sweep_network.py`; it exited 0 and reproduced the archived summary.
- Direct execution of all six focused test functions; all passed. `pytest` is absent from the virtualenv.
- Array inspection of every NPZ field; all 25 arrays had the documented shapes and finite values.
- `skills/brainx-general-guard/SKILL.md`, `skills/brainmass/SKILL.md`, `skills/brainstate/SKILL.md`, and `skills/brainunit/SKILL.md`.
- BrainMass references `modellibrary.md`, `simulator-input-monitor-api.md`, `parameter-sweeps-and-regime-analysis.md`, `visualization-analysis-api.md`, and the Wilson-Cowan and Hopf scripts.
- BrainState references `transformation-vmap-expansion.md`, `simulation-environment.md`, and `randomness-and-reproducibility.md`.
- BrainUnit reference `quantity-inspection-and-conversion.md`.
- Official API pages for `brainmass.Simulator`, `brainstate.transform.vmap`, and `brainstate.transform.for_loop`, plus the package documentation inventories in `source_html_references/`.

## Executive diagnosis

The artifact runs quickly and deterministically, implements all requested mechanisms, produces the requested three navigation regimes and 10-vector trajectory figure, and uses a genuinely distributed direction-to-grid field rather than commanding a decoded path. The custom aggregate model is justified because BrainMass has no public head-direction ring or toroidal grid step.

The strongest scientific claims are still exploratory. Matched controls are run through separate compiled calls and only their scalar reductions are saved, so the claimed adaptation-theta-coupling mechanism cannot be audited from the archive. The parameters, one initialization seed, one integration step, and displayed regime have no held-out confirmation. The shuffle p-value establishes cycle-order structure only, not robustness across model realizations.

BrainX coverage is partial. BrainState State and transforms are used correctly enough to execute, but BrainMass appears only in one plotting call even though `brainmass.Simulator` explicitly accepts any `brainstate.nn.Module` and owns exactly the loop rebuilt here. BrainUnit quantities are present at the protocol boundary, then time constants, angles, positions, speeds, and grid scales are converted to bare convention-dependent numbers before the model update.

This is not a fully controlled refinement checkpoint under `how-to-refine-skill.md`. The requested orchestrator subagent shared the repository, could read prior runs, wrote directly into `run2`, produced no event log, and changed a skill file while running. Preserve this folder as the requested `run2`, but do not treat it as leakage-free evidence equivalent to the documented CLI harness.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `theta_sweep_network.py:916-923`, `theta_sweep_network.py:996-1030` | Baseline and three causal controls run as separate simulations, while the NPZ saves time-resolved baseline evidence but no control trajectories, protocol arrays, or State observables. | The summary bars cannot verify that only the declared intervention changed or explain why each control lost alternation. The mechanism attribution is not reproducible from archived evidence. | Map baseline, no-adaptation, no-theta, and no-coupling through one fixed-shape condition axis with identical initial State. Save per-condition ring rate, adaptation, grid rate, decoded sweeps, protocol inputs, and reductions. |
| P1 | `README.md:8-25`, `theta_sweep_network.py:27-31`, `theta_sweep_network.py:943-961` | A hand-tuned deterministic regime is evaluated at one model seed, one `dt`, one theta frequency, and four sparse adaptation values; no held-out or nearby confirmation is reported. | The result demonstrates existence for one calibrated realization, not a robust mechanism. The minimum shuffle p-value can look more inferential than the evidence warrants. | Label the regime outcome-calibrated, freeze it, then confirm across independent initialization seeds and nearby `dt`, theta frequency, anchoring, recurrent gain, and adaptation values. Report uncertainty over realizations separately from the cycle-order shuffle. |
| P2 | `theta_sweep_network.py:290-327`, `theta_sweep_network.py:147-156`, `theta_sweep_network.py:464-483` | Position is advanced to `(i+1)dt`, theta drive and environment time use `i*dt`, and the recorded sample is timestamped `(i+1)dt`. Phase-pi samples are therefore labeled at 1.052 s while the applied drive phase is pi at 1.050 s. | Position, theta forcing, and post-update observation are misaligned by one 2 ms step, weakening exact within-cycle phase claims. | Choose one monitor convention. Supply position at step start and record at step end, or drive all inputs at the same post-update time. Save both input phase and observation time, and sample phase using that declared convention. |
| P2 | `README.md:21-25`, `results/summary.json` | The no-adaptation condition has no valid ring sweeps but its high directional alignment is reported alongside meaningful baseline alignment. | Alignment with a near-zero ring offset does not show aligned theta sweeps and can overstate preserved sweep coupling. | Require both the predeclared ring-angle and grid-length validity thresholds for every cycle-level alignment summary, report the valid-cycle fraction, and omit alignment when either sweep is absent. |
| P2 | `test_theta_sweep_network.py:34-64` | Most mechanism tests assert values already stored in `summary.json` rather than recomputing the simulation or reductions. | Tests can pass with stale or manually inconsistent artifacts. | Add one quick recomputation test, independently recalculate metrics from the NPZ, and test control evidence once it is archived. |
| P3 | `results/navigation_and_controls.png` | The adaptation panel overlays degrees with `30 x alternation` on one axis. | The arbitrary scale invites visual comparison between quantities with different meanings. | Use a second axis or separate panels, and show uncertainty or replicate points once confirmation runs exist. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Aggregate direction and grid dynamics | Custom `DirectionGridStep` subclass of `brainstate.nn.Module` | Custom BrainMass-scale Module; no public ring/grid model exists | Legitimate custom scientific mechanism | Keep the custom equations, but run the Module with `brainmass.Simulator`. |
| Dynamical State | Six `brainstate.HiddenState` values | `brainstate.HiddenState` | Correct | Keep; save matched control State evidence. |
| Numerical integration | Hand-coded exponential leak factors with bare millisecond constants | `brainstate.nn.exp_euler_step` or a documented custom exact step | Scientifically plausible but unit-light and duplicated | Use unit-bearing time constants and the documented integrator when it preserves the intended held-target update. |
| Recurrent circular and toroidal convolution | Dimensionless JAX FFTs | Custom model logic; `brainunit.fft` if quantities are retained | Valid custom high-performance mechanism | Keep JAX FFT only after an explicit dimensionless boundary. |
| Standard rollout | Nested `environ.context`, `for_loop`, and `jit` in `simulate()` | `brainmass.Simulator.run()` | Bypasses the owning orchestrator | Use callable tuple inputs, returned-output monitoring, unit-aware duration, default JIT, and returned `ts`. |
| Matched causal conditions | Four sequential host calls | BrainState mapped condition axis or BrainMass native batch when semantics match | Misses same-path causal execution | Map the complete independent condition rollout once and save every lane. |
| Adaptation sweep | `brainstate.transform.vmap` over complete custom rollouts | `brainstate.transform.vmap` | Correct fixed-shape mapping; shared initial perturbation is deliberate | Return fixed scientific summaries when full trajectories are not needed; declare shared initialization in metadata. |
| Random initialization | `brainstate.random.seed` before construction | `brainstate.random.seed` | Correct reproducibility, insufficient robustness | Add independent confirmation streams or seeds. |
| Physical protocol | BrainUnit quantities in `make_protocol()` | BrainUnit quantities and `u.math` | Good at construction | Retain quantities through model inputs and use explicit conversion only at JAX/NumPy host boundaries. |
| Model time, angles, speed, and scale | Early `to_decimal()` conversion and bare normalized values | BrainUnit | Partial | Make time constants, phase, speed, position, and scale contracts unit-bearing or explicitly typed normalized quantities. |
| Decoding and statistics | NumPy host analysis | Legitimate host boundary | Appropriate | Preserve units in saved metadata and independently recalculate archive metrics. |
| Shuffle null | NumPy RNG and permutation loop | Legitimate host statistics boundary | Correct for cycle-order structure | Distinguish its p-value from uncertainty across seeds and parameters. |
| Visualization | One `brainmass.viz.plot_timeseries` call plus Matplotlib | `brainmass.viz`, then Matplotlib host boundary | Appropriate for custom multi-panel scientific figures | Keep; revise the mixed-scale adaptation panel. |
| Serialization | CSV, JSON, compressed NPZ | Legitimate host boundary | Appropriate but incomplete | Save matched control trajectories, units, timing convention, and code revision. |

## Missing, bypassed, or misused BrainX APIs

### `brainmass.Simulator`

`Simulator` accepts a `brainstate.nn.Module`, drives tuple-valued callable inputs, records the return of `update()`, owns `dt`, initialization, `for_loop`, JIT, and post-update timestamps, and returns a JAX-pytree dictionary. It should replace `simulate()`'s custom orchestration. The custom equations do not justify rebuilding the runner.

### BrainState mapped condition execution

The existing adaptation `vmap` proves the model can map complete fixed-shape runs. Extend the same condition axis to the baseline and causal controls, varying only adaptation, theta, and coupling. This is necessary for both performance and matched causal evidence.

### BrainUnit quantities

`to_decimal()` is valid at an external raw-array boundary, but here it removes units before most model calculations. Keep `tau_u`, `tau_a`, `tau_z`, `tau_b`, position, speed, grid scale, frequency, and observation time as quantities through the scientific operation. Use explicit conversion only where a dimensionless FFT kernel, host statistic, plot, or serializer requires it.

### `brainstate.nn.exp_euler_step`

This API can own exponential-Euler stepping under the active unit-aware `dt`. Use it if its diagonal-linearized semantics match the intended update. If the held-target closed form is retained instead, document that deliberate solver choice and still express every time constant as a BrainUnit quantity.

## Performance and code simplicity

The implementation avoids Python timestep loops and uses FFT convolution, so the full analysis runs in about five seconds. Its largest avoidable cost is repeated construction and compilation of identical straight-run graphs for the four controls, followed by a separate mapped adaptation sweep. One mapped control-and-sweep path would reduce compilations and enforce causal matching.

At 1,048 lines, the script is longer than the scientific contract requires. `Simulator` would remove the custom rollout and timestamp reconstruction. A structured condition table would replace repeated simulation calls. Saving only the exact evidence needed for claims would keep the current 4.2 MB archive manageable while adding the missing controls.

The figures are readable, unclipped, and visibly nonblank. The dedicated trajectory figure contains exactly 10 numbered vectors. The adaptation panel should not scale an alternation score by 30 to share an angle axis.

## Skill improvements

Do not edit `brainx-general-guard`: it already requires same-path controls, retained evidence, high-level orchestration, unit preservation, and honest claim boundaries. The run ignored existing guidance rather than exposing a new guard gap.

Add one sentence to `skills/brainmass/references/simulator-input-monitor-api.md` during the next actual refinement: `Simulator` accepts custom aggregate `brainstate.nn.Module` models, not only public `*Step` classes, so a missing catalogue model does not justify rebuilding the standard loop. No broader skill rewrite is warranted.

## Checks for the next run

1. Launch from a disposable workspace and isolated skill snapshot with the exact prompt bytes; archive an event log and verify no skill file changed during the run.
2. Execute all straight baseline and causal controls through one mapped or native batched path with byte-identical protocol inputs and initial State.
3. Save time-resolved ring rate, ring adaptation, grid rate, decoded outputs, and protocol inputs for every control.
4. Align position, theta phase, environment time, and post-update timestamps to one documented convention; assert the phase-pi sample time exactly.
5. Confirm alternation and alignment across independent initialization seeds, at least one smaller `dt`, and nearby theta/adaptation/recurrent/anchor settings selected before confirmation.
6. Recalculate every summary from the saved evidence in tests; do not assert only precomputed JSON values.
7. Use `brainmass.Simulator` for the custom Module and retain BrainUnit quantities through the scientific model boundary.
8. Preserve all three requested navigation regimes, the shuffle comparison, representative population dynamics, and the dedicated 10-vector trajectory figure.
