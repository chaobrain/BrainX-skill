# BrainX diagnosis: finding the edge of criticality

## Evidence studied

- Exact case prompt and every archived Run 1 artifact: `edge_of_criticality.py`,
  `test_edge_of_criticality.py`, `README.md`, the result files, PNG, event
  stream, stderr, final response, and harness metadata.
- Current `skills/brainx-general-guard/SKILL.md`, plus the complete BrainPy-State,
  BrainEvent, BrainState, and BrainUnit root skills.
- Routed references for projection alignment, JITC connectivity variants,
  mapped State, and transformed control flow.
- Closest executable compositions:
  `skills/brainevent/references/scripts/coba_ei_teaching.py`,
  `skills/brainevent/references/scripts/102_EI_net_1996.py`, and
  `skills/brainpy-state/references/scripts/103_COBA_2005.py`.
- Official API routes indexed by the BrainPy-State, BrainEvent, and BrainState
  HTML inventories for `LIFRef`, `Expon`, JITC matrices, `vmap2`, collective
  State initialization, and `for_loop`.
- A clean rerun under the required BrainX virtualenv. All four result files
  reproduced byte-for-byte with SHA-256 hashes
  `9d6347dc2204e0140f6399ba2d8edef76b872aa9a32f648c313b3d413339d05b`,
  `7f33bec78b85b94ab5175d773fbd6cafe8b413f98dc8a0050699e908182726e4`,
  `64cd30599cc4a4de50124b5d1fed93c578c06fe48773c08f72bec25897c11149`,
  and `d43c206b1fb07de8408298b94b405defee3eadf647f8f3a310796c900e5f271a`
  for the NPZ, aggregate CSV, realization CSV, and PNG respectively.
- Direct execution of all four focused test functions. The archived binned
  counts reconstruct all 320 realization rows and every aggregate
  susceptibility and instability fraction.

## Executive diagnosis

Run 1 materially improves on Run 0. It preserves the BrainX-native mapped
simulation, saves reconstructable realization-level evidence, labels the
parameters phenomenological, rejects a singleton as a region, and uses one
simple `plt.subplots(...)` composition. The archived results are deterministic:
the sampled optimum is `0.750`, and the final classifier returns the two-point
interval `0.745` to `0.750`.

Run 1 is not accepted as an independent held-out interval result. The first
held-out pass contained no region and selected `0.750` as its optimum. After
viewing that result, the agent added midpoint samples `0.745` and `0.755`, then
reran the same base seed `90210`; the second pass produced the reported
interval. Thus the added displayed cases were calibrated on the evaluation
outcomes, and the README's claim that the default grid is held out from the
calibration used to choose it is false for those two points. This failure is
already covered by the restored freeze-or-separate-calibration rule.

The implementation also omits the no-spark control from the mapped lanes, and
the heatmap renders a nonuniform coupling grid with uniformly spaced image
rows. Existing control-path and accurate-figure guidance already covers both
problems. Run 1 exposes no new transferable guard gap and therefore justifies
no further skill edit or Run 2.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `codex-events.jsonl` items 76, 81-84; `edge_of_criticality.py:50`; `README.md:28` | The first held-out result selected the final midpoint samples, which were evaluated with the same seed ensemble and then described as held out. | The reported `0.745-0.750` interval is an adaptive calibration result, not an independent evaluation of a frozen grid. | Preserve `0.750` as the first held-out sampled optimum, or freeze the refined grid and evaluate it once with new independent realization seeds. Archive and identify calibration and evaluation evidence separately. |
| P2 | `edge_of_criticality.py:157` | Every mapped lane receives the spark; no matched no-spark condition is executed or saved. | The result shows post-spark dynamics but does not independently establish the claimed otherwise-quiescent baseline at each coupling and realization. | Add spark and no-spark conditions to the same mapped lane structure, save both, and limit causal wording if only the intervention is measured. |
| P2 | `edge_of_criticality.py:323`; `criticality_summary.png` | `imshow(..., extent=...)` spaces image rows uniformly although the coupling values are nonuniform. | The heatmap y geometry visually misrepresents distances between sampled couplings, especially around the refined interval. | Use a high-level plotting method that accepts explicit nonuniform coupling coordinates, while retaining the single `plt.subplots(...)` composition. |
| P3 | `edge_of_criticality.py:147` | The spatial spark mask is rebuilt inside every mapped timestep from fixed lane targets. | The compiled loop carries avoidable per-step construction and obscures the fixed protocol input. | Construct the lane-specific spark masks once before the time loop and pass the precomputed protocol into the mapped step. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, and current | BrainUnit quantities with explicit conversion at analysis boundaries | BrainUnit | Correct | None. |
| Excitatory and inhibitory point neurons | Two initialized `brainpy.state.LIFRef` populations | BrainPy-State | Correct | None. |
| Recurrent synaptic filtering | Four `brainpy.state.Expon` modules | BrainPy-State | Correct explicit current-based composition | A projection wrapper is optional and would not fix the evidence failures. |
| Sparse recurrent communication | Positive `JITCUniformR` E-to-E weights and signed `JITCScalarR` E/I pathways with stable seeds | BrainEvent | Correct generated row-oriented event communication | None. |
| Matched graphs across coupling values | The same realization seed is paired across coupling lanes while only E-to-E scale changes | BrainEvent plus host protocol construction | Correct common-random-number design | Persist calibration and evaluation seed sets separately. |
| Independent coupling-realization State | `vmap_init_all_states` plus filter-selected `vmap2` State axes | BrainState | Correct semantic State mapping | Add the independent control condition to the same mapped operation. |
| Time evolution | One jitted `for_loop` over time and the spark envelope | BrainState | Correct and performant | Precompute the fixed spatial spark masks outside the step. |
| Lane-output layout | One host transpose and reshape after the transformed rollout | Host array boundary | Correct | None. |
| Avalanche and stability analysis | NumPy binning, contiguous-run extraction, and fixed threshold reductions | Host scientific-analysis boundary | Appropriate and fully defined | Keep calibration and evaluation reductions separate. |
| Realization evidence | Per-realization stability rows plus archived binned population counts | Host evidence and serialization boundary | Passing; all aggregates reconstruct | Add explicit seed and intervention columns when controls are introduced. |
| Region classification | Positive finite susceptibility, instability cap, near-peak threshold, and at least two adjacent rows | Host scientific-analysis boundary | Correctly rejects a singleton, but the final points were chosen adaptively | Confirm the frozen refined interval on new seeds. |
| CSV and NPZ output | Python `csv` and NumPy compressed storage | Host serialization boundary | Correct | None. |
| Figure composition | One `plt.subplots(2, 1, ...)` call with basic high-level methods | Host presentation boundary | Simple and legible; heatmap coordinates are inaccurate | Plot the nonuniform grid with explicit coordinates. |
| Focused verification | Four direct unit tests and artifact reconstruction | Host verification boundary | Passing, but no calibration-separation or evidence-reconstruction test is archived | Add checks for frozen protocols and aggregate reconstruction in the next implementation. |

## Missing, bypassed, or misused BrainX APIs

No required BrainX modeling or execution API is missing or misused.
BrainPy-State owns the point-neuron and synaptic dynamics, BrainEvent owns the
generated sparse event communication, BrainState owns mapped State and time
execution, and BrainUnit owns the physical quantities. NumPy analysis, CSV/NPZ
serialization, and Matplotlib remain legitimate host boundaries.

`AlignPostProj` could package the communication, synapse, and target roles, but
the direct `BinaryArray @ JITC*` composition follows the routed BrainEvent E/I
teaching pattern and supports lane-specific generated weights. Replacing it
would add structure without correcting the scientific failures.

## Performance and code simplicity

- The final run maps 320 independent coupling-realization lanes through the
  complete stateful transition and compiles one 600-step time loop.
- JITC connectivity regenerates matched sparse graphs from compact seeds rather
  than materializing dense matrices for every lane.
- Host loops are confined to post-simulation scientific analysis; there is no
  Python timestep loop.
- The fixed spatial spark mask can move outside the step, but this is a small
  local simplification rather than a skill-level API gap.
- The figure satisfies the one-`plt.subplots(...)` rule and is readable at the
  saved resolution. Correcting its nonuniform y coordinates requires no layout
  scaffolding or relaxation of the simple-Matplotlib rule.

## Skill improvements

Make no guard or package-skill edit from Run 1.

- The current freeze-or-calibrate-separately rule already prohibits selecting
  displayed midpoint cases from evaluation outcomes and reusing the same
  evaluation seeds for the resulting claim.
- The current requirement to run independent controls through the same mapped
  path already covers the missing no-spark lanes.
- The current visualization section already requires accurate axes while using
  one simple `plt.subplots(...)` composition.
- The BrainPy-State, BrainEvent, BrainState, and BrainUnit guidance already
  produced the intended BrainX-native architecture.

The latest run therefore identifies implementation failures covered by the
cumulative baseline, not a genuinely transferable gap. Semantic append-only
policy forbids duplicate wording, and there is no edit specification for a
Run 2 snapshot.

## No-edit guard invariant audit

| Current invariant | Run 1 evidence | Status | Guard action |
|---|---|---|---|
| Freeze settings and displayed cases before outcomes, or separate calibration and report held-out or nearby sensitivity. | The held-out outcome selected two final displayed midpoint cases, then the same seeds were reused. | Violated by artifact; retained in guard. | None. |
| Validate and save matched controls at every nuisance setting through the same mapped or batched path. | The mapped lanes include only the spark condition. | Violated by artifact; retained in guard. | None. |
| Preserve per-condition evidence plus aggregation. | The realization CSV and binned NPZ reconstruct all aggregate rows. | Satisfied. | None. |
| Claim a region only when multiple samples resolve its extent. | The classifier rejects singleton sets and the final grid contains two eligible adjacent samples. | Mechanically satisfied; independent confirmation still fails. | None. |
| Label unsourced calibrated regimes phenomenological. | The README explicitly uses the phenomenological label. | Satisfied. | None. |
| Run BrainX-native mapped execution and keep Matplotlib composition absolutely simple without lowering figure quality. | `vmap2` and `for_loop` are correct; one `plt.subplots(...)` is used, but the nonuniform y coordinates are rendered uniformly. | Execution and simplicity satisfied; coordinate fidelity violated. | None. |

Every current invariant remains unchanged. No invariant is added, removed, or
combined after Run 1.

## Checks for a future implementation

- Freeze the complete refined coupling grid before evaluation and run it once
  on an unseen seed ensemble. Do not add displayed samples after seeing those
  outcomes unless the run is relabeled calibration.
- Save calibration and evaluation seed sets and evidence separately.
- Execute spark and no-spark conditions through the same mapped path and save
  both per-realization reductions.
- Preserve the realization CSV and raw binned counts so every aggregate remains
  reconstructable.
- Keep the singleton rejection, BrainX-native mapped simulation, unit-aware
  parameters, and single simple Matplotlib composition.
- Render the nonuniform coupling coordinates faithfully.
