# pp-prop workflow

Use this reference when input/output-factorized `pp_prop` is preferable to D-RTRL, especially for recurrent SNNs and silent-delay tasks. Open `batching.md` for batching ownership and `ETP operators.md` when a custom projection does not participate.

## Choose pp-prop

`pp_prop` carries input-side and output-side eligibility factors with exponential smoothing instead of a parameter-dimensional trace.

| API or option | Description |
|---|---|
| `braintrace.pp_prop` | Use when factorized `O(B * (I + O))` trace storage is preferable to parameter-shaped storage. |
| `braintrace.ES_D_RTRL` | Use only for compatibility with historical code; prefer `pp_prop`. |
| `decay_or_rank=<float>` | Use a value in `(0, 1)` to set exponential decay directly; larger values retain longer history. |
| `decay_or_rank=<int>` | Use a positive integer to select `decay = (rank - 1) / (rank + 1)`; it does not allocate that many stored factors. |

Factorization discards correlations present in an unfactorized trace. Report `decay_or_rank` with results and validate the estimator against BPTT on a reduced matched model when gradient fidelity matters.

## Check the model integration before training

Use ETP-aware native projections with a public `brainspike.Neuron` subclass. Open `brainspike` → `references/native-training.md` and `brainspike` → `scripts/train_native_snn.py` for the verified recurrent SNN, compiled pp-prop learner, State resets, and sequence update. The reduced Docker check verifies compiler participation, finite nonzero recurrent/readout gradients, decreasing loss, and a matched BPTT gradient direction. Do not transfer that result to an arbitrary NEST Simulator graph or assume estimator equality.

## Validate silent-delay memory

A lower training loss does not prove that the first cue survived the delay; use a control that changes the claimed memory variable while preserving the evaluation path.

| Check | Required evidence |
|---|---|
| Silent interval | Assert that cue input is exactly zero during the declared delay. |
| Frozen evaluation | Fix evaluation trials, seeds, masks, and metrics before viewing the trained result. |
| First-cue ablation | Replace or permute only the first cue and rerun the same mapped rollout; accuracy should fall materially relative to intact trials. |
| Robustness | Prefer held-out stochastic cue realizations or a nearby longer delay when the task permits them. |
| Schedule | Report sequences and optimizer updates separately; one sequence-driver call followed by one update means one optimizer update per sequence. |

For a deterministic two-cue task with only four combinations, first-cue ablation is the minimum discriminating control. Perfect accuracy on the same four training combinations is insufficient by itself.

## Interpret compiler diagnostics

Use `algorithm.report.show(2)` and structured fields together. A parameter may have a rejected unbatched candidate and an included batched relation in the same report. Require zero errors and check that the intended recurrent path appears in `etrace_weights`; do not classify it from warning text alone.

Open `compiler_internal.md` only when structured report records cannot explain the relation. Open `ETP operators.md` for sparse input and structure contracts.

## Sources

- [pp-prop tutorial](https://brainx.chaobrain.com/braintrace/tutorials/pp_prop.html)
- [SNN online learning](https://brainx.chaobrain.com/braintrace/tutorials/snn_online_learning.html)
