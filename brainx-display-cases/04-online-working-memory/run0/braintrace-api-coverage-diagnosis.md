# BrainX diagnosis: online working memory

## Evidence studied

- Generated artifacts: `online_working_memory.py`, `agent-final.md`,
  `codex-events.jsonl`, `codex-stderr.log`, and `harness-metadata.txt` in this
  run folder.
- Independent execution with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`: the
  artifact completed in about 7.5 seconds, moving from loss `0.6781` and
  accuracy `0.500` to loss `0.2611` and accuracy `1.000`.
- Owning skills and routed references: `brainx-general-guard`,
  `brainpy-state`, `brainevent`, `braintrace`, `brainstate`, `brainunit`,
  `braintrace/references/pp_pprop workflow.md`, `batching.md`,
  `ETP operators.md`, `Drtrl.md`, `algorithm selection.md`,
  `compiler_internal.md`, and BrainState collective-operation and vmap
  references.
- Official BrainTrace `v0.2.4` source at tag
  `9491aaa4ea62ddbae306c92e52c229c23ffcb6ee`: `examples/pp_prop/02-neurons-alif-dms.py`,
  `03-neurons-gif-working-memory.py`, `05-batching-vmap.py`,
  `06-batching-batched.py`, `09-operator-sparse.py`, `_shared.py`,
  `examples/tests/test_compile_modes.py`, `docs/tutorials/pp_prop.md`,
  `docs/apis/algorithms.rst`, and `docs/apis/compiler.rst`.
- Exact `v0.2.4` contracts for `braintrace.compile`, `ETraceVjpAlgorithm.update`,
  `CompilationReport`, and `sparse_matmul`, plus the current release notes
  identifying sequence drivers as a BrainTrace `0.2.5` addition.
- Frozen compatibility row: BrainX `v2026.7.9` bundles BrainTrace `0.2.4`.

## Executive diagnosis

The artifact is executable, unit-aware, batched, sparse, and demonstrably
trainable, but the skill snapshot directed it toward BrainTrace `0.2.5`
sequence APIs that are absent from the bundled `0.2.4` environment. The agent
therefore replaced the unavailable driver with a manual per-step gradient
loop, a gradient-based evaluation probe, manual mapped-State zeroing, and an
optimizer update at every report step. Official `0.2.4` instead compiles once
with `braintrace.compile(...)`, differentiates one learner call per step, uses
`brainstate.transform.scan(...)` to accumulate gradients through time, and
updates the optimizer once after the sequence.

The final in-sample accuracy does not by itself establish working memory. All
training and evaluation trials are the same deterministic four cue pairs, and
the artifact provides no first-cue ablation, delayed-state evidence, noisy
evaluation, or changed-delay test. The compiler does include the recurrent,
cue, and readout weights through batched ETP relations; duplicate unbatched
path warnings are not evidence that those parameters are excluded. However,
`report.show(1)` hides the diagnostics needed to make that distinction.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `online_working_memory.py:95-126`, `162-183`, `223-244` | Training and evaluation reuse the same fixed four cue combinations without a cue-dependence or robustness control. | Perfect accuracy can show task fitting, but it does not demonstrate that information from the first cue is retained through the silent delay. | Freeze evaluation before training and add at least a first-cue ablation; also prefer noisy held-out trials or a nearby longer delay. Report chance-level or substantially degraded ablated accuracy alongside intact accuracy. |
| P1 | `online_working_memory.py:185-221`, `234-239` | `N_UPDATES` counts sequences, but the optimizer updates at every one of the ten report steps. | “320 online updates” describes 3,200 optimizer updates and obscures the actual learning schedule. | Accumulate per-step online gradients across one sequence and call `optimizer.update(...)` once; otherwise label both sequence count and optimizer-step count exactly. |
| P2 | `online_working_memory.py:188-216` | The loss is zero before the report window, but learner and eligibility State advance throughout the cue and delay. This is scientifically sensible, yet the implementation does not state the distinction between trace evolution and loss application. | Readers may interpret zero masked loss as no online computation during the memory interval. | Keep the full temporal scan, state explicitly that every step advances the model and trace, and apply the task loss only at the declared report steps. |
| P2 | `online_working_memory.py:242-244` | Assertions check finite losses and final in-sample accuracy only. | A second-cue-only, fixed-pattern, or otherwise non-memory solution is not rejected. | Assert the silent interval, intact performance, and a predeclared ablation gap or delayed-state decoding criterion. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, resistance, current, and neuron parameters | BrainUnit quantities and `u.math.arange` | BrainUnit | Correct. | Preserve units through model execution. |
| Point-neuron dynamics and surrogate spike derivative | `brainpy.state.LIF` with `braintools.surrogate.ReluGrad` | BrainPy-State and BrainTools | Correct package ownership. | Preserve. |
| Sparse recurrent topology | Host NumPy samples fixed-degree targets, then constructs `brainevent.CSR` | BrainEvent plus legitimate host-side topology generation | Correct. | Preserve deterministic construction and CSR structure. |
| Sparse recurrent transmission | `braintrace.sparse_matmul(previous_spikes, weight_data, sparse_mat=CSR)` | BrainTrace ETP operator plus BrainEvent representation | Correct final call. The event log shows that wrapping `previous_spikes` in `brainevent.BinaryArray` failed. | Keep `x` as a raw numeric or boolean array and require `sparse_mat` to be a BrainEvent `DataRepresentation`. |
| Cue projection and leaky readout | `braintrace.nn.Linear` and `LeakyRateReadout` | BrainTrace prebuilt layers | Correct. | Preserve. |
| Independent batch lanes | Explicit `brainstate.nn.Map` around the model | BrainTrace `compile(..., vmap=True)` or native batched compilation | Functional but unnecessarily exposes mapped lifecycle hazards. | Prefer compile-owned vmap for per-lane State, or use native batched State when the model supports it; use exactly one mapping owner. |
| Model and trace compilation | Direct `braintrace.pp_prop(mapped_model)` then `compile_graph(...)` | `braintrace.compile(model, braintrace.pp_prop, example_step, batch_size=..., vmap=...)` | Bypasses the canonical one-call initializer, guardrail, and reporting path. | Compile once with the one-call API and the real per-step shape. |
| Compiler inspection | `learner.report.show(1)` | `CompilationReport.show(2)`, `counts`, `etrace_weights`, `excluded_weights`, and diagnostics | Incomplete. The visible warnings mix excluded unbatched candidates with included batched relations. | Show level 2 and assert no errors plus the intended recurrent/cue parameter participation from structured fields. |
| Independent-sequence reset | Manually zero every mapped `HiddenState`, then call `learner.reset_state()` | BrainTrace learner reset plus BrainState shape-preserving mapped reset or exact State restoration | Necessary workaround for the selected explicit `Map`, but it duplicates lifecycle logic. A direct `reset_all_states(mapped_model)` collapsed the leading lane axis during agent exploration. | Use compile-owned vmap and its mapped reset path; verify every mapped dynamical State shape before and after reset. Keep the existing BrainState collective-operation warning authoritative. |
| Evaluation rollout | Per-step dummy squared loss under `brainstate.transform.grad`, with logits copied through `ShortTermState` | Plain learner calls inside `brainstate.transform.for_loop`/`scan`; host metric after rollout | Misuses differentiation to recover outputs because the unavailable `etrace_evolve()` was expected. | Run a loss-free forward loop directly, retaining only the outputs required for the report metric. |
| Per-step online gradient | Builds `brainstate.transform.grad(...)` inside the step and calls the learner once | `brainstate.transform.grad` around exactly one learner call | Semantically valid for BrainTrace `0.2.4`, but transform construction should be hoisted. | Construct the gradient function once outside the scan body. |
| Temporal gradient accumulation | `brainstate.transform.for_loop` returns step gradients that are immediately applied | `brainstate.transform.scan` with an explicit gradient carry | Incorrect optimizer schedule and avoidable logic. | Sum masked per-step gradients in the scan carry and update once after the full sequence. |
| Repeated training sequences | Outer `brainstate.transform.for_loop` | BrainState transformed loop | Correct for fixed repeated sequences, subject to correct reset and one update per sequence. | Preserve or use a small host loop if each call is one compiled sequence step and boundary resets must remain explicit. |
| Classification loss and accuracy | BrainTools softmax cross entropy; JAX reductions and argmax | BrainTools metric plus legitimate array/host analysis | Correct. | Preserve and add ablation/robustness metrics. |
| Text output and assertions | Python formatting and JAX assertions | Host boundary | Appropriate. | Correct the update count and report the memory control. |

## Missing, bypassed, or misused BrainX APIs

### `braintrace.compile(...)`

Use the one-call compiler instead of manually constructing `pp_prop`, mapping
the model, initializing mapped State, and calling `compile_graph`. In
BrainTrace `0.2.4`, `compile` initializes State, builds the graph, checks that
at least one ETP relation exists, attaches a report, and can own per-sample
mapping through `vmap=True`.

### `brainstate.transform.scan(...)`

Use `scan` because the sequence needs an ordinary explicit gradient carry.
Each body call must differentiate exactly one learner call, add its gradient to
the carry only when the report mask applies, and return any needed output. The
optimizer update belongs after the scan, not inside it.

### Structured `CompilationReport` inspection

Use `report.show(2)` and inspect `counts`, `etrace_weights`,
`excluded_weights`, and diagnostic records. The Run 0 report lists recurrent,
cue-projection, and readout parameters as associated with hidden groups even
though warnings also describe rejected duplicate paths. A warning string
alone is therefore insufficient to decide participation.

### Unavailable sequence drivers

Do not use `etrace_grad`, `etrace_evolve`, `SequenceDriverMixin`, or
`ETraceVmap` for the repository's BrainX `v2026.7.9` target. Those APIs begin
in BrainTrace `0.2.5`, while the compatibility matrix freezes `0.2.4`.
Teaching them as the canonical path directly caused the Run 0 workaround.

### `braintrace.sparse_matmul(...)`

The final artifact uses this correctly. The skill must state its complete
boundary: `x` is a raw numeric or boolean array, `weight_data` holds trainable
nonzero values, and `sparse_mat` is a BrainEvent `DataRepresentation` such as
`brainevent.CSR`. `brainevent.BinaryArray` is not the required `x` wrapper.

## Performance and code simplicity

- Compilation occurs once, and the final artifact runs quickly, but explicit
  `Map` plus manual State traversal adds lifecycle machinery that
  compile-owned vmap can remove.
- Constructing `brainstate.transform.grad` inside both evaluation and training
  loop bodies is avoidable. Hoist the training transform and remove the
  evaluation gradient entirely.
- Updating the optimizer at each report step performs ten times more optimizer
  work than the printed schedule claims and changes the algorithm from the
  official sequence-accumulation workflow.
- The full 60-step sequence is small. A scan with one gradient pytree carry and
  optional logits is sufficient; no custom sequence-driver abstraction is
  needed in `0.2.4`.
- Fixed-degree topology construction, label construction, report aggregation,
  accuracy calculation, textual reporting, and assertions are legitimate host
  boundaries.

## Skill improvements

- Do not edit `skills/brainx-general-guard/SKILL.md`; its package routing,
  State-lifecycle, transformed-loop, batching-owner, and scientific-control
  rules already cover the cross-cutting failures.
- Refine `skills/braintrace/SKILL.md` around the BrainTrace `0.2.4` lifecycle:
  one-call compilation, level-2 structured inspection, one differentiated
  learner call per step, scan accumulation, and one optimizer update per
  sequence.
- Replace `0.2.5` sequence-driver guidance in `Drtrl.md`,
  `pp_pprop workflow.md`, `algorithm selection.md`, `batching.md`,
  `pre-built-braintrace-layer.md`, `custom algorithms.md`, and
  `compiler_internal.md` with the `0.2.4` workflow or a precise version
  boundary.
- Add the complete sparse operator input contract to `ETP operators.md`.
- Align `plan.md` and `source_html_references/braintrace_html_reference.md`
  with the bundled `0.2.4` release and its tagged sources. Do not describe
  current `0.2.5` pages as contracts for the frozen package.
- Do not duplicate BrainState's existing warning about mapped reset shape
  collapse; route to that reference when a manual mapping path requires it.

## Checks for the next run

1. The artifact compiles with BrainX `v2026.7.9` / BrainTrace `0.2.4` and uses
   no unavailable sequence-driver API.
2. It calls `braintrace.compile(...)` once with the actual per-step batch and
   feature shape, with exactly one mapping owner.
3. It uses `report.show(2)` or equivalent structured checks, reports zero
   compiler errors, and confirms the intended recurrent parameter is in
   `etrace_weights`.
4. Every time step advances the learner; one gradient transform wraps exactly
   one learner call; a `scan` accumulates masked per-step gradients; and the
   optimizer updates once per sequence.
5. Independent-sequence reset clears both model and eligibility State without
   changing mapped State shapes.
6. `sparse_matmul` receives a raw numeric/bool spike array and a BrainEvent
   sparse representation.
7. Printed sequence and optimizer-update counts are exact.
8. The run verifies the silent input interval and adds a predeclared first-cue
   ablation or comparably discriminating memory control. Intact performance
   must exceed the control by a meaningful margin.
9. Losses and gradients remain finite, final intact accuracy is at least
   `0.75`, and the artifact remains substantially simpler than Run 0.
