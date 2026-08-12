# BrainX diagnosis: finding the edge of criticality

## Evidence studied

- Exact case prompt and every Run 0 artifact: `criticality_scan.py`,
  `test_criticality_scan.py`, `README.md`, both result directories, the PNG,
  event stream, stderr, final response, and harness metadata.
- `skills/brainx-general-guard/SKILL.md`, `skills/brainpy-state/SKILL.md`,
  `skills/brainpy-state/references/projection-patterns.md`,
  `skills/brainevent/SKILL.md`,
  `skills/brainevent/references/connectivity-variants.md`, and
  `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`.
- Closest executable compositions:
  `skills/brainevent/references/scripts/coba_ei_teaching.py`,
  `skills/brainpy-state/references/scripts/103_COBA_2005.py`, and
  `skills/brainevent/references/scripts/102_EI_net_1996.py`.
- Official BrainPy-State E/I, neuron, synapse, projection, and synaptic-output
  pages; BrainEvent JIT connectivity and sparse-data pages; and BrainState
  `vmap2` and collective-initialization contracts.
- A disposable rerun under the required BrainX virtualenv. The JSON, CSV, and
  PNG reproduced byte-for-byte with SHA-256 hashes
  `e8f9a7a15f442976f154ebdc2ae4fdd60cc08ca086dd5b0b900705e34ed7d75c`,
  `ccac7559dcb482c8dcd40bf8142aeadcdef193f448d7b20592b283276442b00e`,
  and `655236b5af9a2960afd4fbd4e26293a796c5db438e7e0393628a6a02ecb0ab67`.
  Both focused test functions passed directly; `pytest` is absent from the
  frozen virtualenv.
- A temporary diagnostic rerun at gain `2.1`. Fifteen realization sizes were
  between 1 and 5 spikes; one was 4,897 spikes, producing the reported CV.

## Executive diagnosis

Run 0 is executable, reproducible, BrainX-native, and structurally efficient.
It uses unit-aware LIF E/I populations, four reproducible JITC pathways,
independent mapped dynamical State across all gain-realization lanes, one
compiled time loop, and one simple `plt.subplots(...)` figure. Siemens-scale
weights are not by themselves an error: the routed canonical BrainEvent COBA
teaching model documents the same unit convention with `LIFRef`'s default
resistance.

The main scientific conclusion is overstated. The output calls `[2.1, 2.1]` a
critical region and shades it as an interval, although only one sampled gain
passes the region predicate. That point's CV of `3.97` is driven by one rare
4,897-spike realization among fifteen 1-to-5-spike outcomes. The archive saves
only aggregate rows, so a reader cannot inspect or replicate that distribution.

The event log also shows that exploratory quick and default runs were used to
choose the gain range and displayed transition. Therefore the final scan is a
calibrated demonstration, not an independently evaluated or pre-registered
result. Calibration separation and per-condition persistence are already
required by the cumulatively restored guard and do not justify duplicate
guidance. The only new transferable gap is that an interval or region requires
multiple resolved sample points; a singleton must be reported as a sampled
point or optimum and trigger finer sampling.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `criticality_scan.py:330`, `summary.json:4`, figure | The reported `critical_region` is `[2.1, 2.1]`; only one grid point passes the region predicate. | A sampled peak is presented as a resolved interval, and the shaded span visually implies extent the scan did not measure. | Report a selected sampled point, or refine the grid and claim a region only when multiple adjacent samples establish its extent. |
| P1 | `codex-events.jsonl`, `criticality_scan.py:51` | Exploratory outcomes determined the final scan range and displayed transition, while the README describes the settings as pre-registered. | Selection and evaluation reuse the same outcomes, so the peak and regime are calibrated rather than independently confirmed. | Label the scan calibration, then evaluate fixed settings on held-out realization seeds or report nearby-grid and seed sensitivity. |
| P1 | `criticality_scan.csv`, `summary.json` | The archive retains only aggregate metrics. At gain `2.1`, one 4,897-spike realization drives the CV while the other fifteen sizes are 1 to 5. | The peak cannot be audited, its uncertainty is unknown, and one rare event may dominate the selection. | Save per-gain, per-realization size, duration, late rate, and stability evidence plus the exact aggregation; replicate the peak on held-out seeds. |
| P2 | `criticality_scan.py:401` | The no-spark control runs only at the strongest gain, not at each swept gain in the same mapped intervention path. | The control establishes no spontaneous activity at one endpoint but does not preserve matched per-gain evidence. | Include spark and no-spark conditions across the gain-realization lanes, or explicitly limit the control claim to the tested endpoint. |
| P2 | `test_criticality_scan.py` | Tests verify point selection and flat-scan rejection but not singleton-region handling, persistence, or calibration separation. | The principal reporting error and auditability gap can regress undetected. | Test that singleton eligibility yields a point rather than an interval, and verify saved realization rows reconstruct every aggregate. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, current, conductance, and reversal potential | BrainUnit quantities with explicit decimal conversion at analysis boundaries | BrainUnit | Correct | None. |
| Excitatory and inhibitory point neurons | Two `brainpy.state.LIFRef` populations with explicit initialization | BrainPy-State | Correct | None. |
| Synaptic filtering and conductance current | Four `Expon` modules and four bound `COBA` outputs | BrainPy-State | Correct custom composition for four independently generated pathways | A packaged projection is optional, not required. |
| Sparse probabilistic communication | `BinaryArray @ JITCScalarR` with stable realization seeds | BrainEvent | Correct and reproducible for immutable random graphs | None. |
| Independent gain-realization lanes | `vmap_init_all_states` and filter-based `vmap2` over writable dynamical State | BrainState | Correct | Add the control condition to the same mapping. |
| Time evolution | One jitted `for_loop` over a time-major spark schedule | BrainState | Correct and performant | None. |
| Initial-voltage and seed construction | NumPy host arrays copied across paired gain lanes | Host initialization boundary | Correct common-random-number design | Persist exact per-lane inputs with results. |
| Avalanche extraction | NumPy binning and sequential quiet-window analysis after simulation | Host scientific-analysis boundary | Appropriate, but evidence is not persisted | Save the per-realization reductions. |
| Peak and region classification | Host reductions over stable lanes and contiguous eligible grid points | Host scientific-analysis boundary | Peak calculation is defined; region label is invalid for a singleton | Distinguish sampled point from resolved multi-point region. |
| CSV and JSON output | Python `csv` and `json` | Host serialization boundary | Correct APIs, incomplete scientific evidence | Add a per-realization table and calibration metadata. |
| Figure | One `plt.subplots(3, 1, ...)` call with basic methods | Host presentation boundary | Simple, readable, and reproducible; singleton shading misstates extent | Mark a point when no interval is resolved. |
| Focused verification | Two direct analysis tests | Host verification boundary | Passing but too narrow; `pytest` is unavailable | Add reporting and evidence-reconstruction tests runnable without extra dependencies. |

## Missing, bypassed, or misused BrainX APIs

No required BrainX API is missing or misused. BrainPy-State owns the neuron and
synapse dynamics, BrainEvent owns generated event communication, BrainState
owns mapped State and time execution, and BrainUnit owns physical quantities.
Host NumPy analysis, CSV/JSON serialization, and Matplotlib are legitimate
boundaries.

`AlignPostProj` could package each pathway, but the explicit JITC communication,
synapse, and output composition is already the canonical BrainEvent teaching
pattern and keeps the gain intervention direct. Replacing it would not correct
the scientific failures.

## Performance and code simplicity

- All 304 default gain-realization lanes are mapped through the complete
  stateful transition; one transformed loop owns 1,500 simulation steps.
- Connectivity is regenerated from compact seeds rather than materialized for
  every lane, and common realization seeds pair gain comparisons correctly.
- Host loops are limited to initialization and post-simulation analysis; there
  is no Python timestep loop.
- The simulation runs twice because the control is separate. Mapping the
  control condition with the intervention would improve both causal matching
  and reuse without changing the execution model.
- The figure uses exactly one `plt.subplots(...)` call and basic high-level
  plotting methods. Figure implementation complexity is not a problem.

## Skill improvements

The restored `brainx-general-guard` already covers calibration separation,
held-out or nearby sensitivity, independently validated matched controls,
saved per-condition evidence and aggregation, external seeding, and execution
of controls through the same mapped path. These Run 0 failures require no new
wording.

Add only this transferable invariant to `brainx-general-guard` and the matching
`plan.md` summary:

> Claim an interval or region only when sampling resolves its extent across
> multiple points; otherwise report a sampled point or optimum and refine the
> sampling.

No BrainPy-State, BrainEvent, BrainState, or BrainUnit package-skill edit is
justified.

## Guard invariant audit

This one-time repair compares the three historical guard revisions because
their meanings were accidentally replaced. After this repair, the current
guard is the baseline and earlier run folders are not reopened for edits.

| Existing invariant | Proposed final wording | Status | Evidence for addition |
|---|---|---|---|
| Validate the baseline and mechanism before tuning; mark unsourced calibrated regimes phenomenological. | Derive claims from observables that distinguish the claimed mechanism; validate the baseline and mechanism before calibration, and mark unsourced calibrated regimes phenomenological. | Retained and losslessly combined with the calibration rule. | None. |
| Freeze parameters, evaluation seeds, metrics or scores, thresholds or windows, and displayed cases before outcomes. | Freeze parameters, evaluation seeds, metrics or scores, thresholds or windows, and displayed cases before viewing intervention outcomes; otherwise calibrate separately and report held-out or nearby sensitivity. | Retained. | None. |
| Validate every matched control independently at each nuisance setting. | Validate each matched control independently, then compare and save control and intervention evidence at every nuisance setting and aligned physical time or event landmark; a normalized summary is not a substitute for those paired observables. | Losslessly combined with aligned paired evidence. | None. |
| Align matched observations by physical time or event landmark; summaries do not substitute for paired evidence. | Validate each matched control independently, then compare and save control and intervention evidence at every nuisance setting and aligned physical time or event landmark; a normalized summary is not a substitute for those paired observables. | Losslessly combined with independent matched-control validation. | None. |
| Verify and save every relevant State and protocol input at causal branches; vary only the intervention. | At each causal branch, verify and save every relevant State and protocol input, vary only the declared intervention, and preserve per-condition evidence plus its aggregation. | Losslessly combined with per-condition persistence. | None. |
| Limit causal wording to the intervention; require mediator manipulation and event-outcome agreement for mediation. | State causality at the intervention level; claim event mediation only when a mediator-specific manipulation and per-condition event dose agree with the outcome. | Retained. | None. |
| Require time-resolved source-to-route-to-target order for propagation. | Apply each claim's full temporal predicate: require time-resolved source-to-route-to-target order for propagation and exact element order for sequence direction rather than a regression sign or proxy. | Losslessly combined with exact sequence order. | None. |
| Require exact sequence-element order rather than a regression sign or proxy. | Apply each claim's full temporal predicate: require time-resolved source-to-route-to-target order for propagation and exact element order for sequence direction rather than a regression sign or proxy. | Losslessly combined with propagation order. | None. |
| Treat supplied drive or retained boundary State as an external seed, including stochastic drive. | Identify supplied drive or retained boundary State as an external seed even when stochastic. | Retained. | None. |
| Retain every continuous categorical-boundary observable and save the exact threshold reduction. | For categorical maps, retain every continuous boundary observable, plot and save the exact reduction tested at each threshold, and verify each label's full predicate. | Losslessly combined with the full categorical predicate. | None. |
| Require measured departure before sustained recovery; never force a requested category. | Require measured departure before sustained recovery, and never force a requested category to appear. | Retained. | None. |
| Run independent controls and mechanism checks in the same mapped or batched path. | Run independent controls and mechanism checks in the same mapped or batched path as the intervention. | Retained. | None. |
| No prior rule governs the minimum evidence needed to name a sampled interval or region. | Claim an interval or region only when sampling resolves its extent across multiple points; otherwise report a sampled point or optimum and refine the sampling. | New transferable invariant. | Run 0 reports `[2.1, 2.1]` as a critical region. |

## Checks for the next run

- Separate calibration from evaluation or explicitly report held-out and
  nearby gain/seed sensitivity.
- Save per-gain, per-realization avalanche size, duration, late rate, stability,
  seed, and intervention condition so every aggregate can be reconstructed.
- Report a singleton eligible gain as a sampled point, not a region; claim a
  region only from multiple samples that resolve its extent.
- Execute matched spark and no-spark controls through the same mapped path, or
  limit causal language to the actual endpoint control.
- Preserve the BrainX-native JITC communication, semantic State mapping, one
  compiled time loop, unit-aware parameters, and one simple Matplotlib figure.
