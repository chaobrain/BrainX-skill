# BrainX diagnosis: spike-frequency adaptation

## Evidence studied

- Exact prompt: 570 bytes, SHA-256 `0cb46165ece3d0f84a614490795d33a44bee2af68efeb133d641a0ae5f842bf1`.
- Run conditions: isolated empty workspace, repository skill snapshot, designated `.venv-brainx`, fresh ephemeral agent, and explicit `xhigh` reasoning.
- Generated artifacts: `README.md`, `spike_frequency_adaptation.py`, `spike_frequency_adaptation.png`, `agent-final.md`, and `codex-events.jsonl`.
- Execution: the unchanged entry point passed under `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`; the 1960 x 1279 figure was opened and inspected.
- Skills and routes: `brainx-general-guard`, `braincell`, `brainstate`, `brainunit`, BrainCell ion/channel/adaptation references and scripts, and BrainState control-flow/vectorization references.
- Official examples: Spike-Frequency Adaptation, Channel Ablation, Frequency-Current Curve, and the single-compartment HH workflow.
- Authoritative APIs: `braincell.SingleCompartment`, `braincell.MixIons`, `braincell.ion.CalciumDetailed`, `braincell.channel.AHP_De1994`, `brainstate.transform.for_loop`, `brainstate.transform.vmap`, and `brainstate.transform.vmap2` generated pages.

## Executive diagnosis

The final artifact runs and demonstrates a causal within-model contrast: at 10 uA/cm^2, zeroing only `g_AHP` changes 45 nearly tonic spikes into 27 adapting spikes, with the last ISI increasing from about 11 ms to 20.52 ms. Units, State evolution, channel ownership, and output boundaries are correct.

The main weakness is scientific provenance. To make the effect conspicuous, the run replaced the official thalamic-relay template with a hybrid HH/Ca/AHP model and selected `CaL=5.0 mS/cm^2`, calcium `tau=80 ms`, and `g_AHP=1.0 mS/cm^2`; these are respectively 10x, 8x, and 3.3x the matching values in the official adaptation example. The final comparison still isolates AHP conductance, but it should be described as a calibrated teaching model rather than an intact literature model.

The implementation also uses nested `vmap` calls to manufacture a Cartesian grid and reduce spike trains even though BrainCell already vectorizes independent conditions through `SingleCompartment(size=...)`. The official F-I example establishes native `size` batching as the canonical sweep. BrainState `vmap` remains appropriate only when it maps a real callable that the owning package does not already batch.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `spike_frequency_adaptation.py:30-70` | The final cell mixes classical HH channels with an IS2008 calcium channel and a De1994 AHP channel, then outcome-calibrates three parameters without provenance or a sensitivity statement. | The causal AHP result is valid for this constructed model, but the labels and prose can be mistaken for a literature-grounded intact neuron. | Preserve one coherent published template for scientific claims, or label the hybrid explicitly as a calibrated teaching model and report the parameter source/sensitivity. |
| P2 | `spike_frequency_adaptation.py:97-99, 238-240` | `1.0 mS/cm^2` is called "AHP intact" although the official example uses `0.3 mS/cm^2`. | "Intact" implies a canonical baseline that was not established. | Use "AHP present" for a calibrated value, or retain the official template/value for an intact-versus-zero comparison. |
| P2 | `README.md:4-6`, figure calcium panel | Calcium is shown as the recruiting signal, but AHP current itself is not monitored; calcium also changes after ablation because spike count changes. | The panel supports mechanism plausibility but does not independently quantify the outward current. | Base the causal conclusion on the one-parameter ablation and ISIs; describe calcium as the feedback signal, not direct measurement of AHP current. |
| P2 | `spike_frequency_adaptation.py:26-27, 133-152` | Early/late rates include fixed windows and are secondary to the full ISI trajectory. | A window choice can change the reported rate drop. | Keep ISI progression and spike counts as primary checks; state window bounds beside any rate comparison. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Isopotential conductance cell | `AdaptingCell(SingleCompartment)` | BrainCell `SingleCompartment` | Correct | Keep. |
| Membrane capacitance and voltage initialization | Unit quantities plus `braintools.init.Constant` | BrainUnit + BrainTools init | Correct | Keep density parameters consistent. |
| Na/K spike generator | `SodiumFixed`, `PotassiumFixed`, `Na_HH1952`, `K_HH1952` | BrainCell ions/channels | API-correct, scientifically hybridized with later mechanisms | State provenance or use a coherent template. |
| Dynamic calcium | `CalciumDetailed` + `CaL_IS2008` | BrainCell ion/channel | Correct mechanism ownership | Do not tune unrelated calcium kinetics only to force the desired plot without disclosure. |
| Calcium-activated K current | `MixIons(k, ca)` + `AHP_De1994` | BrainCell mixed-ion ownership | Correct | Preserve ion order and expose `g_AHP`. |
| Controlled ablation | Set only `g_AHP=0` in one condition | Official channel-ablation pattern | Correct | Teach this directly in the adaptation reference/script. |
| Condition sweep | Nested `vmap` creates current/conductance grids; `size=(3, 3)` runs all cells | BrainCell native `size` batching; BrainUnit broadcasting | Simulation batching is correct; grid `vmap` is unnecessary | Build broadcastable condition arrays directly and reserve `vmap` for an unowned mapped computation. |
| State initialization | `cell.init_state()` inside the `dt` context | BrainCell lifecycle | Correct | Keep one shared initialization for matched lanes. |
| Stimulus protocol | `u.math.where` over unit-aware time | BrainUnit math | Correct | Keep onset/offset explicit. |
| Time evolution | `brainstate.transform.for_loop` | BrainState control flow | Correct | Keep; State carries cell dynamics. |
| Environment | nested `brainstate.environ.context(dt=...)` and `context(t=...)` | BrainState environment | Correct | Keep. |
| Voltage/spike/calcium monitoring | Return State values from `step` | BrainState/BrainCell State | Correct | Keep only observables needed for the claim. |
| Firing-rate reduction | Nested pure `vmap` plus `u.math.sum` | BrainUnit reductions; BrainState `vmap` optional | Correct result, over-composed | Reduce over the known time axis directly or use one meaningful mapped analysis function. |
| Unit conversion | `.to_decimal(...)`/`.in_unit(...)` at reporting boundary | BrainUnit | Correct | Keep target units explicit. |
| Spike-time and ISI analysis | NumPy masks and `diff` after conversion | Legitimate host analysis boundary | Correct | Keep; no BrainX API must replace ordinary host statistics. |
| Assertions | host assertions over ISIs/rates plus unit conversion checks | Legitimate validation boundary | Good | Add parameter-provenance/coherence check to the workflow guidance. |
| CLI and paths | `argparse`, `Path` | Host boundary | Correct | Keep only if reusable output control is desired. |
| Plotting | high-level Matplotlib | Host presentation boundary | Correct and visually valid | Simplify legend/series encoding if revising the artifact. |
| README/reporting | Markdown and `print` | Host boundary | Correct | Label calibrated versus literature-grounded models precisely. |

## Missing, bypassed, or misused BrainX APIs

### BrainCell native condition batching

`SingleCompartment(size=condition_shape)` already represents independent cells. The official F-I example uses this path for current sweeps. The run ultimately uses it, but first wraps a tuple-returning identity function in two `vmap` transforms merely to create arrays. Replace that construction with BrainUnit array creation/broadcasting and let BrainCell own simulation batching.

### `brainstate.transform.vmap` versus `vmap2`

The current generated APIs define two contracts:

- `vmap(..., in_states=..., out_states=...)` declares mapped State instances and raises on an undeclared batched write.
- `vmap2(..., state_in_axes=..., state_out_axes=..., unexpected_out_state_mapping=...)` selects State by filters and supports automatic output-axis inference.

The repository root skill incorrectly says `in_states`/`out_states` are undocumented. Correct that decision boundary. Neither API is needed around a BrainCell condition axis already represented by `size`.

### Adaptation-specific ablation workflow

The BrainCell adaptation reference teaches construction but not the requested causal test. Add the official channel-ablation pattern: expose AHP `g_max`, compare zero against a sourced present value with all other mechanisms and inputs fixed, and verify both ISI progression and spike/rate change.

## Performance and code simplicity

- The simulation uses one transformed 60,000-step rollout for nine cells and does not use a Python timestep loop.
- BrainCell native vectorization is the correct high-level performance structure.
- `build_parameter_grid()` and nested mapped rate reducers add transform tracing and 35 lines without adding model parallelism.
- The 369-line artifact is substantially larger than the official adaptation and ablation examples combined. CLI, four-panel plotting, reporting, and validation are legitimate output features, but the canonical skill script should remain smaller.
- A cold review run reproduced all outputs. Font-cache initialization dominated wall time; do not attribute that one-time host setup to BrainX execution.

## Skill improvements

1. In `brainx-general-guard`, forbid adding a named transform solely to construct axes or satisfy a package checklist when the owning Module already exposes the batch axis.
2. In `braincell/SKILL.md`, state that `size` may be a multidimensional condition shape and array-valued parameters/inputs map directly to those independent cells.
3. In `braincell/references/mixions-for-adaptation.md`, add a controlled AHP-ablation workflow, parameter-provenance rule, primary ISI checks, and a precise route to the runnable script.
4. Refine `braincell/references/scripts/spike_frequency_adaptation.py` to compare AHP present versus zero while preserving one model template.
5. In `brainstate/SKILL.md` and its vmap reference, correct the generated `vmap`/`vmap2` State-axis contracts and state the owning-package batching boundary.
6. Synchronize the affected BrainCell and BrainState decisions in `plan.md`.

## Checks for the next run

- The agent uses a coherent sourced template or explicitly labels any calibrated hybrid and its parameter changes.
- AHP removal changes only `g_AHP`; initial State, stimulus, solver, and every other mechanism remain matched.
- The present-AHP condition has a materially increasing ISI trajectory; the zero-AHP condition is substantially more tonic or fires materially faster late in the stimulus.
- `MixIons(k, ca)` order, dynamic calcium, units, initialization, and `for_loop` remain correct.
- BrainCell `size` owns independent condition batching; no `vmap` is added merely to create a Cartesian grid.
- If State-aware mapping is genuinely needed, `vmap` uses `in_states`/`out_states` and `vmap2` uses `state_in_axes`/`state_out_axes` according to the generated API pages.
- The entry point runs in the designated virtualenv and the generated figure is legible and scientifically consistent with printed metrics.
