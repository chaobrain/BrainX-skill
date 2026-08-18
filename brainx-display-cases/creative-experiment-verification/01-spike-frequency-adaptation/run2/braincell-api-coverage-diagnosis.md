# BrainX diagnosis: spike-frequency adaptation, run 2

## Evidence studied

- Exact prompt: 570 bytes, SHA-256
  `0cb46165ece3d0f84a614490795d33a44bee2af68efeb133d641a0ae5f842bf1`.
- Run conditions: fresh empty workspace, fresh temporary `CODEX_HOME`, refined
  skill snapshot, designated `.venv-brainx`, byte-identical prompt through
  stdin, ephemeral Codex CLI process, and explicit `xhigh` reasoning.
- Generated artifacts: `README.md`, `spike_frequency_adaptation.py`,
  `spike_frequency_adaptation.png`, compiled bytecode, `agent-final.md`, and
  `codex-events.jsonl`.
- Event trace: the implementation turn completed successfully. Its only failed
  command was `git status` in the intentionally non-Git disposable workspace;
  model construction, repeated executions, assertions, and compilation passed.
- Independent execution: a copied workspace passed under
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`.
  The regenerated PNG had the same SHA-256 as the archived PNG:
  `c7007444d4b9fed2a1911e9fd76e29dd6c7313c9f257fa6ead92dea822cf735b`.
- Visual inspection: the 2160 x 1296 RGBA figure is nonblank, legible, and
  consistent with the printed spike and ISI metrics.
- Skills and routes: `brainx-general-guard`, `braincell`, `brainstate`, and
  `brainunit`; BrainCell channel, ion, and adaptation references; the revised
  adaptation script; and BrainState simulation, control-flow, and vectorization
  references.
- Authoritative mechanisms and APIs: `braincell.SingleCompartment`,
  `braincell.MixIons`, `braincell.ion.CalciumDetailed`,
  `braincell.channel.AHP_De1994`, `brainstate.transform.for_loop`, and
  `brainstate.transform.vmap`.

## Executive diagnosis

Run 2 passes the consequential scientific and API checks. It explicitly labels
the hybrid HH/Ca/AHP cell as a teaching model, changes only `g_AHP` for the
ablation, uses BrainCell's native two-dimensional condition axis, and reserves
nested `vmap` for a real per-trace analysis rather than manufacturing the model
grid.

At the matched `10 uA/cm^2` input, the zero-AHP lane emits 46 spikes and changes
from an `11.80 ms` first ISI to an `11.05 ms` last ISI, a ratio of `0.94`. The
strong present-AHP lane emits 27 spikes and changes from `11.96 ms` to
`20.52 ms`, a ratio of `1.72`. The same direction holds at all three currents.
The intermediate lane using the source example's conductance
(`g_AHP=0.3 mS/cm^2`) shows only a mild effect, while the `1.0 mS/cm^2`
teaching value produces the clear demonstration; the sweep makes that
sensitivity visible instead of hiding the calibration.

No P0 or P1 problem remains in the generated artifact. The artifact could make
parameter provenance more explicit and should not clip sub-one ratios at a
heatmap lower bound of one, but neither issue invalidates the controlled
within-model conclusion. Post-run validation did find that the skill's own
canonical adaptation script, unlike the Run 2 artifact, still emitted six
pre-stimulus spikes; that justified one final narrow skill refinement.

## Run 1 to run 2 comparison

| Check | Run 1 | Run 2 | Assessment |
|---|---|---|---|
| Model status | Hybrid parameters were insufficiently qualified | Module and README call it a teaching model, not a biological-cell reproduction | Improved |
| AHP ablation | Only `g_AHP` changed | Only `g_AHP` changes across the strength axis | Preserved and clearer |
| Condition batching | Nested `vmap` manufactured a grid before native BrainCell batching | BrainUnit broadcasting constructs parameter arrays; `SingleCompartment(size=(3, 3))` owns model lanes | Corrected |
| `vmap` purpose | Grid construction and reduction | One per-trace spike-summary callable is mapped over current and strength axes | Corrected |
| Baseline | Zero-current model spiked before stimulation | Uniform holding current and `-75 mV` initialization produce zero pre-stimulus spikes | Improved |
| Mechanism visibility | Voltage, calcium, ISI, and rate views | Voltage, dynamic calcium, full ISI trajectories, and a labeled sensitivity grid | Improved |
| Reproducibility | Entry point passed | Entry point passes and regenerates a byte-identical PNG | Preserved |

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P2 | `spike_frequency_adaptation.py:1-6`; `README.md:10-12` | The artifact identifies a calibrated teaching model but does not link the mechanism and ablation sources or state which numerical values are calibrated. | A reader can interpret the within-model result correctly but cannot reconstruct the provenance boundary from the artifact alone. | Include the adaptation and channel-ablation source URLs and label `g_AHP=1.0 mS/cm^2`, calcium `tau=80 ms`, and `CaL=5.0 mS/cm^2` as teaching values. |
| P3 | `spike_frequency_adaptation.py:272-279` | The heatmap fixes `vmin=1.0` although zero-AHP ratios are `0.93-0.95`. | Sub-one values are color-clipped even though their text labels remain correct. | Let normalization include the finite data minimum or use a centered normalization around one. |

The one spike outside the nominal stimulus mask occurs at `550.90 ms`, after
the input switches off, in one strong-AHP lane. It is a state-dependent offset
transient, not spontaneous pre-stimulus activity; pre-stimulus spike count is
zero and the reported ISIs exclude it.

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Isopotential cell | `AdaptingCell(SingleCompartment)` | BrainCell `SingleCompartment` | Correct | Keep. |
| Voltage and capacitance initialization | Unit-aware values plus `braintools.init.Constant` | BrainUnit and BrainTools initialization | Correct | Keep the teaching-model label. |
| Sodium and potassium spiking | Fixed ions plus `Na_HH1952` and `K_HH1952` | BrainCell ions and channels | API-correct | Do not imply a named phenotype. |
| Dynamic calcium | `CalciumDetailed` plus `CaL_IS2008` | BrainCell ion and channel | Correct ownership and units | Cite calibrated values in reusable artifacts. |
| Ca-activated K current | `MixIons(k, ca)` plus `AHP_De1994` | BrainCell mixed-ion composition | Correct ion order and mechanism | Keep. |
| Controlled ablation | Array-valued `g_ahp` includes zero, sourced, and teaching values | BrainCell channel parameterization | Correct | Keep all other lane parameters matched. |
| Condition sweep | Broadcast strength/current arrays and `size=(3, 3)` | BrainCell native condition batching plus BrainUnit broadcasting | Correct and simpler than Run 1 | Keep. |
| State initialization | `cell.init_state()` inside the `dt` context | BrainCell lifecycle | Correct | Keep. |
| Stimulus and holding current | Unit-aware `u.math.where` | BrainUnit math | Correct | Keep identical across lanes. |
| Time evolution | One `brainstate.transform.for_loop` | BrainState State-aware control flow | Correct | Keep; no Python timestep loop. |
| Run environment | Scoped `dt` and per-step `t` contexts | BrainState environment | Correct | Keep. |
| Monitoring | Return voltage, `ca.Ci.value`, and spikes from the step | BrainCell and BrainState State | Correct and sufficient | Calcium is a gating signal, not direct AHP-current measurement. |
| Condition analysis | Nested `brainstate.transform.vmap` over one spike-train summarizer | BrainState mapping requested explicitly by the case | Meaningful mapped computation; no duplicate model batching | Without the explicit BrainState requirement, raw JAX mapping or direct axis reductions would also be valid for this pure-array function. |
| Unit conversion | `.to_decimal(...)` only at reporting and plotting boundaries | BrainUnit | Correct | Keep target units explicit. |
| Spike times and ISIs | JAX reductions in mapped analysis; NumPy `diff` for plotting | Legitimate array and host-analysis boundaries | Correct | Keep edge handling for fewer than two spikes. |
| Validation | Matched-current assertions over ISI ratios | Host validation boundary | Correct and causal within the model | Optionally assert no pre-stimulus spikes. |
| Plotting | High-level Matplotlib | Host presentation boundary | Legible and scientifically consistent | Include sub-one ratios in color normalization. |
| Documentation | README plus module docstring | Host reporting boundary | Correctly says teaching model | Add exact provenance links and calibrated-value disclosure. |

## Missing, bypassed, or misused BrainX APIs

No consequential BrainX API is missing, bypassed, or misused in the final
artifact.

`SingleCompartment(size=condition_shape)` now owns the independent model lanes,
as required by the revised BrainCell guidance. Nested `vmap` no longer creates
axes that BrainCell already owns; it maps the explicitly requested comparison
over complete spike traces. Because that analysis function is pure and has no
State effects, raw `jax.vmap` or direct reductions would be a valid default in a
task that did not explicitly request BrainState `vmap`.

The code does not monitor AHP current directly. This is not an API-coverage
failure: the causal claim follows from the controlled zero-conductance
intervention, dynamic-calcium trace, and ISI response. The prose correctly says
calcium gates the AHP mechanism rather than claiming the calcium panel measures
the current.

## Performance and code simplicity

- One transformed rollout evolves 60,000 timesteps for nine independent cells;
  no Python timestep loop or per-condition model loop is present.
- State outputs have the expected shapes: time `(60000,)`, voltage/calcium/spike
  `(60000, 3, 3)`, and summary `(3, 3, 4)`.
- Native BrainCell batching removes Run 1's mapped grid-construction function.
- The analysis maps one compact callable over nine traces. At this scale its
  cost is negligible compared with the ODE rollout.
- The saved figure is deterministic in the review environment and needs no
  custom HTML or low-level rendering infrastructure.
- The generated `__pycache__` is incidental compilation output, not part of the
  scientific deliverable; preserving it is appropriate for an unchanged run
  archive.

## Skill improvements

The Run 1 refinements produced the intended Run 2 behavior:

1. `brainx-general-guard` now rejects transforms used only to manufacture axes
   already owned by a package Module.
2. BrainCell now teaches multidimensional native condition batching, controlled
   `g_AHP=0` ablation, provenance boundaries, and explicit teaching-model labels.
3. BrainState now distinguishes `vmap` and `vmap2` State-axis contracts and
   defers to owning-package batching.
4. The canonical adaptation reference and script expose zero, sourced, and
   stronger teaching AHP values so sensitivity is visible.

Run 2 then supplied one additional reusable lesson. Its agent detected
spontaneous pre-stimulus firing and added an identical holding current to every
condition. Directly checking the skill's canonical script confirmed six
pre-stimulus spikes. The reference and script now require a disclosed, matched
holding baseline when a calibrated hybrid is spontaneously active, initialize
this teaching model at `-75 mV`, and assert zero pre-stimulus spikes. The revised
script passes with the same Run 2 metrics and zero pre-stimulus events.

Do not add another package-skill rule for the two residual artifact
presentation issues; the existing provenance guidance already covers the
important decision, and heatmap normalization is ordinary host-side plotting
judgment.

The repository-level `how-to-refine-skill.md` was separately refined from this
case workflow to specify a fresh `codex exec` process, temporary `CODEX_HOME`,
exact stdin prompt bytes, explicit `xhigh` effort, frozen environment settings,
post-exit artifact archival, and the distinction between conversation isolation
and filesystem isolation.

## Checks for a future run

- Keep the 570-byte prompt and its SHA-256 unchanged.
- Keep the model, `xhigh` reasoning, BrainX virtualenv, CLI version, tools, and
  isolation controls fixed.
- Preserve native BrainCell condition batching and a one-parameter AHP ablation.
- Require zero pre-stimulus spikes and exclude offset transients from ISIs.
- Keep the teaching-model label and add exact source/calibration notes if the
  artifact is intended for reuse beyond this case.
- Require entry-point success, matched metrics, a nonblank inspected figure,
  and unchanged raw artifacts before reviewer files are added.
