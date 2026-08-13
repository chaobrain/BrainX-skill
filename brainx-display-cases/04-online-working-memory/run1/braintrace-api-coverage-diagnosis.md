# BrainX diagnosis: online working memory

## Evidence studied

- Run 1 artifacts: `online_working_memory.py`, `agent-final.md`,
  `codex-events.jsonl`, `codex-stderr.log`, and `harness-metadata.txt`.
- Byte-identity metadata: 733 prompt bytes, SHA-256
  `629e6b6f36083f51bf808a1e8c0014a272222b993511097c7068261728e00119`,
  `gpt-5.6-sol`, xhigh reasoning, Codex CLI `0.147.0-alpha.6.5`, and exit
  code 0, matching Run 0 conditions.
- Independent default execution with the frozen BrainX virtualenv: 7.2 seconds,
  final loss `0.4458`, intact accuracy `1.000`, and first-cue-ablated
  accuracy `0.500`, exactly reproducing the archived result.
- Independent compiler and lifecycle probe: report counts were two hidden
  groups, four ETP relations, zero excluded weights, zero warnings, and zero
  errors; `('recurrent', 'comm', 'weight')` was explicitly present; all ten
  mapped State shapes survived reset; every leading lane axis remained 32;
  and `running_index` reset to zero.
- Refined BrainTrace root skill and routed `pp_prop`, batching, D-RTRL, ETP
  operator, compiler, algorithm-selection, and BrainState lifecycle guidance.
- Official BrainTrace `v0.2.4` delayed-match, working-memory, batching, sparse
  operator, one-call compiler, algorithms API, and compiler-report sources
  listed in the Run 0 diagnosis.

## Executive diagnosis

Run 1 resolves the consequential Run 0 failures. It uses APIs available in
BrainTrace `0.2.4`, compiles the unwrapped model once with compile-owned vmap,
passes raw spikes and BrainEvent CSR structure to `sparse_matmul`, advances the
learner exactly once at every time step, carries summed online gradients
through `brainstate.transform.scan`, and applies one optimizer update per
sequence. It replaces fixed repeated training cases with reproducibly sampled
trial streams and adds a frozen exhaustive evaluation batch plus a first-cue
ablation.

The intact-to-ablated accuracy change from `1.000` to chance (`0.500`) provides
direct evidence that the first cue affects the report after a verified 30 ms
zero-input interval. The compiler probe independently confirms that the sparse
recurrent parameter participates in eligibility traces and that mapped reset
preserves every State axis. Run 1 is simpler in ownership and execution despite
adding the scientific control and a small command-line boundary.

One minor reporting issue remains: `etrace_parameters=4` is the number of
compiled ETP relations, not the number of unique parameter paths; the input
weight appears twice and there are three unique paths. This does not affect
training or the recurrence claim because the named sparse recurrent path was
verified separately.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P3 | `online_working_memory.py:228-245` | Evaluation exhausts the four deterministic cue pairs but does not test another initialization seed or a nearby delay. | The artifact demonstrates the requested mechanism for the fixed reproducible run, not convergence robustness across training realizations or a resolved delay range. | For a study making robustness or capacity claims, freeze multiple seeds and several nearby delays before training and retain per-condition results. No such broader claim is made here. |
| P3 | `online_working_memory.py:249-280` | `etrace_parameters` reports `len(report.etrace_weights)`, which counts four relations rather than three unique paths. | A reader could interpret the value as a unique parameter count. | Label it `etrace_relations`, or deduplicate the first tuple element before reporting unique paths. |

No P0, P1, or P2 scientific problem remains. The silent interval, intact
result, matched ablation, and exact trial labels are all checked on the same
mapped evaluation path.

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, current, voltage, resistance, and time constants | BrainUnit quantities and unit ratios for step counts | BrainUnit | Correct. | Preserve. |
| Point-neuron memory dynamics | `brainpy.state.ALIF` with membrane and adaptation State | BrainPy-State | Correct and scientifically appropriate for delay bridging. | Preserve. |
| Sparse recurrent structure | Deterministic ring-like fan-out encoded as `brainevent.CSR` | BrainEvent | Correct fixed explicit connectivity. | Preserve; random or generated connectivity is unnecessary for this demonstration. |
| Sparse recurrent ETP operation | Raw spike array, trainable nonzero data, and BrainEvent CSR passed to `braintrace.sparse_matmul` | BrainTrace plus BrainEvent | Correct complete contract. | Preserve. |
| Synaptic current and projection scheduling | `AlignPostProj`, `Expon`, and `CUBA` feed ALIF dynamics | BrainPy-State | Correct package-owned composition. | Preserve. |
| Input and readout parameter paths | `braintrace.nn.Linear` and `LeakyRateReadout` | BrainTrace | Correct ETP-aware layers. | Preserve. |
| Independent batch State | `braintrace.compile(..., batch_size=B, vmap=True)` | BrainTrace compile-owned vmap | Correct single owner. | Preserve. |
| Graph construction | One `braintrace.compile` call with a batched one-step example | BrainTrace | Correct and available in `0.2.4`. | Preserve. |
| Compiler validation | Structured error and nonempty-participation checks | BrainTrace `CompilationReport` | Correct minimum runtime guard; independent review confirms the named recurrent path, no exclusions, and no warnings/errors. | For a reusable test, assert the named recurrent path in the artifact rather than only nonempty participation. |
| Mapped State reset | `brainstate.transform.vmap(in_states=learner.states('new'))` around collective reset | BrainState and compile-owned mapped State | Correct. Independent review confirms all shapes and `running_index`. | Preserve. |
| Per-step online gradient | Hoisted `brainstate.transform.grad` around exactly one learner call | BrainState plus BrainTrace custom VJP | Correct. | Preserve. |
| Temporal accumulation | Explicit gradient carry in `brainstate.transform.scan` | BrainState | Correct for `0.2.4`. | Preserve. |
| Parameter schedule | Gradient clipping and one Adam update after each sequence | BrainState/BrainTools | Correct; `updates=160` equals 160 sequence-level optimizer updates. | Preserve. |
| Sequence stream execution | Jitted `for_loop` over eight causally sequential training sequences, called in a small host chunk loop | BrainState plus host orchestration | Correct. Weight and optimizer State carry between sequences while dynamical State resets. | Preserve. |
| Trial sampling | Pure JAX random construction outside the stateful neural path | Legitimate host/data boundary | Correct and reproducible from explicit keys. | Preserve. |
| Frozen evaluation cases | Exhaustive four-pair deterministic batch | Legitimate task protocol boundary | Correct for a two-cue binary task. | Preserve. |
| Silent-delay assertion | Exact zero-input slice check | Host/JAX validation boundary | Correct. | Preserve. |
| First-cue ablation | Zero only the first cue, rerun the same mapped evaluation, retain labels and second cue | Scientific control on the same BrainX path | Correct matched intervention. | Preserve. |
| Metrics and reporting | BrainTools cross entropy; JAX aggregation; Python CLI and text | BrainTools plus legitimate host boundary | Correct except relation-count label. | Rename the reported compiler count. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing, bypassed, or misused.

- `braintrace.compile` owns initialization, mapping, graph construction, and
  the report.
- `brainstate.transform.grad` and `scan` implement the official BrainTrace
  `0.2.4` online-gradient composition.
- `braintrace.sparse_matmul` receives the correct raw array and BrainEvent
  structure rather than a `BinaryArray` wrapper.
- `brainstate.transform.vmap` owns the lane-wise reset of compile-created
  mapped State.
- BrainTools owns the optimizer, gradient clipping, surrogate loss support,
  and classification metric.

Unique ETP-path reporting is ordinary structured-data presentation rather than
a missing API. Use `report.etrace_weights` directly and deduplicate its path
field when a unique count is desired.

## Performance and code simplicity

- The default 160-sequence run completes in 7.2 seconds and reproduces exactly.
- Compilation, the per-step gradient transform, reset mapping, training-stream
  JIT, and evaluation JIT are each constructed once.
- The scan carries one gradient pytree and returns only step losses during
  training. Evaluation retains outputs because report-window logits and the
  ablation metric require them.
- The outer host loop chunks sequential parameter updates without placing a
  Python loop over neural time steps. This is an appropriate host boundary.
- Run 1 removes Run 0's explicit `Map`, manual hidden-State traversal,
  dummy-gradient evaluation probe, per-report-step optimizer updates, and
  fixed repeated training set.
- The CLI exposes only three useful evaluation controls and one seed. It does
  not add configuration scaffolding beyond the demonstration's needs.

## Skill improvements

- Keep the current Run 0 refinements to `skills/braintrace/SKILL.md` and the
  BrainTrace references. Run 1 demonstrates that the `0.2.4` release boundary,
  one-call compiler, compile-owned vmap, scan accumulation, sparse contract,
  structured diagnostics, and first-cue control change agent behavior in the
  intended direction.
- Keep the aligned `plan.md` and
  `source_html_references/braintrace_html_reference.md` release boundary.
- Do not edit `skills/brainx-general-guard/SKILL.md`, BrainPy-State,
  BrainEvent, BrainState, or BrainUnit. Their existing package ownership,
  State, mapped reset, units, transformed execution, and causal-control rules
  cover all remaining considerations.
- Do not add a new warning solely for the relation-count label. The current
  BrainTrace root already requires inspection of `etrace_weights`,
  `excluded_weights`, and diagnostics, and the residual issue is artifact
  presentation rather than a transferable API gap.
- No Run 2 is justified by this diagnosis.

## Checks for a hypothetical next run

No next numbered run is required. If a future independently justified change
touches this workflow, require it to:

1. Preserve the BrainTrace `0.2.4` one-call compile, mapped reset, one-call step
   gradient, scan accumulation, and one-update-per-sequence schedule.
2. Assert the named sparse recurrent parameter path in `etrace_weights`, zero
   compiler errors, and mapped State shape preservation.
3. Keep raw spike arrays and a BrainEvent `DataRepresentation` separate in
   `sparse_matmul`.
4. Preserve the exact silent interval and matched first-cue ablation on the
   same mapped evaluation path.
5. Report unique ETP parameter paths separately from ETP relation count.
6. Add predeclared training seeds or nearby delays only when making robustness
   or memory-capacity claims beyond this demonstration.
