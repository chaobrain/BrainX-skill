# BrainX diagnosis: seizure recruitment across regions, Run 0

## Evidence studied

- Exact prompt: `brainx-display-cases/06-seizure-recruitment/prompt.md` (560 bytes; SHA-256 `eb726d96bf1cb1baac11f39113fce2faf2dab096fa97e01f71c89938f13bd139`).
- Generated artifacts: `seizure_recruitment.py`, `README.md`, `outputs/seizure_recruitment.png`, `outputs/seizure_recruitment_results.npz`, `agent-final.md`, and the complete evaluator event log.
- Independent execution with `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`: exit code 0; representative onsets were `[21.5, inf, inf] ms` and `[21.5, 27.9, 34.9] ms`.
- Independent syntax check with `python -m py_compile`: passed.
- Independent numeric inspection: sweep shapes were `(5, 4, 11, 3)` for onset and peak arrays and `(5, 4, 11)` for recruitment labels; all peaks were finite; no-stimulation conditions had label 0; no-coupling conditions had label 0 or 1 only.
- Independent full-resolution figure inspection: traces, threshold, labels, categorical maps, axes, and colorbar were visible and unclipped.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`, `skills/brainmass/SKILL.md`, `skills/brainstate/SKILL.md`, and the BrainUnit skill used by the evaluator.
- BrainMass references: `modellibrary.md`, `simulator-input-monitor-api.md`, `coupling-network-api.md`, `parameter-sweeps-and-regime-analysis.md`, `batch-transform-acceleration.md`, and `visualization-analysis-api.md`.
- BrainState references: `transformation-vmap-expansion.md`, `brainstate-control-flow-patterns.md`, and the State lifecycle sections of the root skill.
- Closest executable BrainMass examples: `scripts/seizure-epileptor-case-study.py` and `scripts/resting-state-meg-whole-brain-pipeline.py`.
- Official BrainMass pages: `EpileptorStep`, `FitzHughNagumoStep`, `Network`, `additive_coupling`, model selection, coupling and delays, custom coupling, parameter sweeps, and orchestration.
- Official BrainState pages: `brainstate.nn.Delay`, `brainstate.transform.for_loop`, `brainstate.transform.vmap`, the delay protocol, and vectorization guidance.
- Focused API experiments: a fixed-capacity `brainstate.nn.Delay` constructed under the same `dt` context as initialization and execution accepted traced integer retrieval steps inside a complete vmapped FHN rollout; the official update-then-`retrieve_at_step(d)` impulse pattern returned the impulse exactly `d` completed updates later.

## Executive diagnosis

Run 0 is executable, visually clear, dimensionally explicit, and scientifically honest about being a deterministic phenomenological demonstration. `FitzHughNagumoStep` is defensible for the exact request's “seizure-like burst” wording because it supplies fast-slow excitable population dynamics and the artifact does not claim Epileptor or clinical seizure mechanisms. The mapped three-parameter sweep, sustained-burst predicate, ordered recruitment rule, matched no-stimulation/no-coupling controls, continuous peak/onset evidence, and high-level BrainMass plotting are all substantive strengths.

The main implementation defect is the custom delay history. It duplicates `brainstate.nn.Delay` and its read-before-write indexing makes every positive labeled delay one recorded timestep too short after the history fills. At `dt = 0.1 ms`, a requested `4.0 ms` delay reads the post-update sample at `t - 3.9 ms`. This is small numerically but directly corrupts the swept physical variable and is difficult to notice from the figure.

The skill gap is broader than this one artifact. Current guidance documents `Network` and a direct prefetch path but does not state the node-input ownership boundary: `Network` always inserts coupling as the first positional node input and forwards supplied inputs only after it. A one-input model therefore cannot receive coupling and an independent focal drive on that same channel through `Network`. The current direct-construction route also does not show how to keep delay capacity static while varying retrieval delay under `vmap`, so the agent hand-wrote a ring buffer. The Epileptor model guidance likewise omits that `x1_inp` and `x2_inp` are scaled by `Kvf`/`Kf`, `x1_inp` also reaches `z` through `Ks`, and all three gains default to zero; this omission caused the agent to misinterpret an inert input trial as absence of an input contract.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `seizure_recruitment.py:71-86` | The history stores post-update state at slot `n`, but step `n` reads slot `n - delay_steps` before writing. The desired state at physical time `n*dt - delay` is in slot `n - delay_steps - 1`. | Every positive labeled delay is one `dt` shorter than requested after startup; onset and regime boundaries are associated with the wrong physical delay. | Use `brainstate.nn.Delay` and retrieve by physical time, or correct and explicitly verify the manual phase convention if no package API can represent it. |
| P2 | `seizure_recruitment.py:248-268` | The saved sweep omits the model identity, connectivity matrix, duration, stimulus start/stop, integration method, and code/version identifier. | The numeric archive cannot independently reconstruct the declared experiment from its own metadata. | Save model identity, fixed protocol/model parameters, structural connectivity, timing, units, and a code/version identifier with the coordinate and result arrays. Existing sweep guidance already requires this. |
| P2 | `README.md:6-7`, `seizure_recruitment.py:25-31` | The README calls the regional input “directed,” but the actual chain connectivity is symmetric. | The description overstates directionality and can make temporal order look structurally imposed when it emerges from an undirected chain seeded at one endpoint. | Call it a bidirectional chain, or use a genuinely directed matrix and state the source-to-target convention. |
| P3 | `seizure_recruitment.py:216-231` | Four irregularly spaced delay samples `[1, 4, 8, 12] ms` are rendered as equal-height image rows over one continuous extent. | The map visually implies uniform and continuously filled delay support that was not sampled. | Use cell edges derived from the actual coordinates, or plot the four sampled delay rows explicitly and avoid interval claims. |

The FHN model choice is not itself a scientific error. It would become one only if the result were called an Epileptor seizure, a physiological seizure mechanism, or patient/clinical evidence. The artifact consistently says “seizure-like,” “phenomenological,” and “illustrative.”

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Represent excitable regional dynamics | `brainmass.FitzHughNagumoStep` with explicit initializers and unit-aware `tau` | `brainmass.FitzHughNagumoStep` | Correct for a phenomenological seizure-like burst. | Keep, unless the claim changes to seizure onset/offset or Epileptor mechanisms. |
| Represent the three-region graph | Constant `(3, 3)` JAX array | BrainMass `Network` or functional coupling input; host-authored structural data | Correct host-authored toy connectivity, but its symmetric semantics should be stated. | Preserve `W[i,j] = source j -> target i` explicitly. |
| Compute inter-region input | `brainmass.additive_coupling` | `brainmass.additive_coupling` | Correct high-level functional kernel for direct weighted drive. | Keep. |
| Store and retrieve delayed regional activity | Custom `HiddenState`, `ShortTermState`, modulo pointer, and indexed update | `brainstate.nn.Delay` | Bypassed API and off-by-one physical timing. | Construct one fixed-capacity `Delay` under `dt`, initialize the intended prehistory, insert the current source once per step, then call `retrieve_at_step(traced_delay_steps)`. |
| Combine coupling and focal stimulus on FHN's first input | `node.update(regional_input + stimulus)` | Direct composition is required because `Network` owns the first node input | Semantically correct composition after the coupling calculation. | Retain the sum but use `Delay` for its source history. |
| Initialize all State per condition | `brainstate.nn.init_all_states(model)` | BrainState collective lifecycle API | Correct. Model construction and initialization occur inside the independent mapped condition. | Keep under the same `dt` context as `Delay` construction. |
| Run simulation time | `brainstate.transform.for_loop(step, indices)` | `brainstate.transform.for_loop` | Correct. All iteration effects live in registered State and outputs stack time-major. | Keep. |
| Scope `dt`, `i`, and `t` | Nested `brainstate.environ.context` | `brainstate.environ.context` | Correct. | Include `Delay` construction in the `dt` scope. |
| Sweep coupling, delay, and perturbation | Unit-aware `meshgrid`, flatten, `brainstate.transform.vmap(summarize_condition)`, reshape | BrainUnit array creation plus BrainState state-aware `vmap` | Correct complete-condition mapping. It batches the stateful rollout, not only input construction or scoring. | Keep static model/Delay capacity across mapped lanes and vary only delay retrieval. |
| Define sustained burst onset | JAX boolean mask, cumulative window counts, `any`, `argmax` | Legitimate scientific analysis logic; no BrainX API owns this predicate | Correct and transform-safe. The `(onset_index + 1) * dt` timestamp matches post-update samples. | Keep; save the threshold and duration as done. |
| Enforce recruitment order | Explicit finite and strict onset comparisons | Legitimate scientific analysis logic | Correct full predicate for the three-node chain. | Keep strict focus -> Neighbor 1 -> Neighbor 2 order. |
| Retain continuous evidence | Peak and onset arrays for every region/grid point | Host/JAX analysis boundary | Correct and important; it keeps the categorical map auditable. | Keep. Consider retaining the exact tested time reduction if the predicate becomes more complex. |
| Verify matched controls | No-stimulation and no-coupling slices plus representative assertions | Host-side validation boundary | Correct, though metadata is incomplete. | Keep controls in the same mapped path and save their fixed protocol details. |
| Plot representative trajectories | `brainmass.viz.plot_timeseries` | `brainmass.viz.plot_timeseries` | Correct highest-level scientific plotting API. | Keep. |
| Plot categorical regime slices | High-level Matplotlib `imshow` | Host presentation boundary | Legitimate because BrainMass has no categorical three-parameter regime-map API. | Represent sampled delay coordinates without implying uniform spacing. |
| Serialize results | `numpy.savez_compressed` | Host serialization boundary | Correct host boundary. | Add complete experiment metadata. |
| Report results | README and stdout formatting | Host reporting boundary | Correct and concise. | Correct the graph direction description. |

## Missing, bypassed, or misused BrainX APIs

### `brainstate.nn.Delay`

Replace the custom `history`/`pointer` State and modulo indexing. `Delay(target_info, time=max_delay, init=...)` owns fixed-capacity short-term history, initialization, retrieval, and write progression. Construct it while the final `dt` is active because capacity is discretized at construction. Keep `time=max_delay` static so vmapped model construction does not derive buffer shape from a tracer; convert each mapped physical delay to an integer step and pass only that tracer to `retrieve_at_step()`.

Insert the source State at the current simulation time before retrieval. Then step 0 is current and step `d` is exactly `d` completed updates earlier. This applies directly to the generated workflow and eliminates the timing defect. Focused independent experiments confirmed that a `Delay` created inside `with brainstate.environ.context(dt=DT)` can be initialized, written in `for_loop`, and queried with traced integer delay steps inside `brainstate.transform.vmap`; an impulse check confirmed the exact nonzero latency.

### `brainmass.Network` input forwarding

`Network.update(*node_inputs)` is not generally missing from the artifact; it is semantically insufficient for this exact one-input composition. The official contract inserts coupling as the first positional node input and forwards `*node_inputs` after it. This supports an independently driven second or later node channel, but it cannot add a separate focal drive to the same first channel of `FitzHughNagumoStep.update(V_inp=None, w_inp=None)`.

The direct functional coupling path is therefore justified. The skill should state this boundary so an agent does not repeatedly test `Network` inputs or abandon high-level delay APIs along with the high-level orchestrator.

### Epileptor input and coupling gains

The evaluator initially tried `EpileptorStep` and concluded that its external input did not perturb the model. The official API instead states that `x1_inp` and `x2_inp` exist but enter through `Kvf` and `Kf`; `x1_inp` also reaches the slow permittivity path through `Ks`. Those gains default to zero. A delayed Epileptor `Network` therefore needs an explicit nonzero scientifically selected gain for its first coupling current to affect `x1` and/or `z`.

Use heterogeneous `x0` when the focus is autonomous epileptogenic tissue. Use time-varying `x1_inp` or `x2_inp` only when the scientific perturbation is an input and choose the corresponding gain explicitly. Do not infer that the model lacks inputs from a default-gain trial.

### `Simulator`

`Simulator.run` is not missing. The exact prompt explicitly requires `for_loop` and `vmap` over coupling, delay, and perturbation, and the custom same-channel sum is outside `Network`'s input composition. The explicit BrainState rollout is justified once the package-owned delay replaces the manual buffer.

## Performance and code simplicity

- The complete stateful condition is mapped, and time is owned by one `for_loop`; there is no Python timestep loop or per-grid host loop.
- Constructing the model inside the mapped function gives each condition independent State. It is consistent with the repository's parameter-sweep pattern.
- The custom delay adds two State objects, modulo arithmetic, indexed writes, and a phase convention that `brainstate.nn.Delay` already owns. Replacing it removes the most error-prone infrastructure without changing scientific behavior or output structure.
- The separate vmapped representative rerun duplicates two grid conditions. This is a modest cost and keeps complete traces out of the large sweep. It is reasonable for figure quality.
- Pure JAX onset analysis inside BrainState `vmap` is acceptable because it is fused with the stateful simulation summary and has no mutable State.
- NumPy conversion happens only after transformed execution for validation, plotting, and serialization; those are legitimate host boundaries.
- `brainmass.viz.plot_timeseries` is used for trajectories; custom Matplotlib is limited to threshold decoration and the categorical map.

## Skill improvements

Make only these surgical BrainMass reference changes:

1. In `references/modellibrary.md`, add the Epileptor input-gain decision boundary: `x1_inp` is scaled by `Kvf` and also reaches `z` through `Ks`; `x2_inp` is scaled by `Kf`; all default to zero. Route autonomous focal tissue to heterogeneous `x0` and transient input perturbations to explicit nonzero gains.
2. In `references/coupling-network-api.md`, state that `Network` inserts coupling as the first node input and forwards caller inputs after it. State when this permits an independent drive and when same-channel coupling plus drive requires direct composition.
3. In the same coupling reference, add the smallest canonical fixed-capacity `brainstate.nn.Delay` workflow for varying delay under `vmap`: construct under the final `dt`, keep capacity static at the maximum delay, initialize prehistory explicitly, insert the current source, retrieve with the mapped integer delay step, apply the functional BrainMass coupling kernel, and verify the phase with an impulse assertion.
4. State that varying delay by constructing a delayed `Network` inside `vmap` can make buffer capacity depend on a tracer. Keep capacity static and vary retrieval, or map only parameters that do not change State shape.

Do not edit `brainx-general-guard`: its existing rules already require the highest-level semantically valid package API, exact temporal propagation predicates, continuous evidence behind categorical maps, matched controls, external-seed identification, and static stateful mapping semantics. Do not add another generic reproducibility rule: `parameter-sweeps-and-regime-analysis.md` already requires model identity, fixed parameters, timing, seed, and code version alongside stored results.

## Checks for the next run

- The generated entry point executes independently and all figures and numeric outputs are inspectable.
- A focal seizure-like event is explicitly seeded, and the artifact distinguishes a phenomenological FHN event from an Epileptor seizure mechanism.
- If Epileptor is used, the artifact selects and reports `x0`, `Kvf`, `Kf`, and `Ks` according to the intended focus and coupling path; it does not expect default-zero gains to transmit input.
- If coupling and a focal drive share a one-input model channel, the artifact uses justified direct composition rather than claiming `Network` forwards both onto that channel.
- Delay history uses `brainstate.nn.Delay` or another documented BrainX delay abstraction. Construction, initialization, and execution use the same `dt`; capacity is static across mapped conditions; integer retrieval steps derive from the declared physical delay.
- A focused timing check shows that a requested `d` retrieves the state at `t - d`, with no one-step phase error and with explicit startup history.
- `brainstate.transform.vmap` covers the complete independent coupling-delay-perturbation condition, and `for_loop` owns all timesteps.
- The recruitment predicate requires a sustained event and strict focus -> first neighbor -> second neighbor onset order in physical recorded time.
- No-stimulation and no-coupling controls execute in the same mapped path, and the supplied focus is identified as the external seed.
- The categorical map retains continuous onset and peak evidence for every region and grid point.
- Saved data includes coordinate arrays, units, model identity, connectivity, fixed model/protocol parameters, `dt`, duration, and a code/version identifier.
- Any graph direction claim matches `W[i,j] = source j -> target i`; sampled delay coordinates are plotted without implying unsupported continuous intervals.
