# BrainX closed-loop brain modeling workflow plan

## Get started

Read `brainmodeling-memory.md` first when it exists. Choose exactly one entry
case.

| Entry case | Condition | Start position |
|---|---|---|
| `fresh-new` | No memory or prior loop artifact exists. | Create memory and start step 0. |
| `resume` | Memory or any prior loop artifact exists. | Verify recorded artifacts and continue from the first unfinished action in the recorded step. |

Recover missing or inconsistent state inside `resume` by comparing memory with
the existing specification, code, run artifacts, and review output. Do not add a
third entry case or repeat completed experiments and reviews solely because the
agent session changed.

Use this complete append-only memory contract:

```markdown
# Brain modeling memory

## Checkpoint
- Iteration: <integer>
- Step: 0 | 1 | 2 | 3 | 4 | 5 | 6

### Artifacts
- `<path or stable identifier>`: <contents and status>

### Important milestones
- <decision, review outcome, blocker, or completed result with its evidence pointer>
```

At `fresh-new`, create only the file title and start iteration 1 at step 0. After
every step, including a blocked step, append one checkpoint. `Iteration` and `Step` are metadata
for the recorded step; `Artifacts` and `Important milestones` are its content.
Never overwrite an earlier checkpoint. Append corrections with the same metadata
and identify the checkpoint they supersede. On `resume`, read checkpoints in
order, verify the latest artifact pointers, and continue after the latest valid
checkpoint.

```text
fresh-new -> step 0 -> step 1 -> step 2 -> step 3 -> step 4 -> step 5
                              ^                          |
                              |-------- REFUSE ----------|

step 5 PASS -> step 6 -> complete
```

Keep full code, logs, results, and reviews in their owning artifacts and point to
them from the checkpoint that records the step.

## Step 0: Specification

Inspect the researcher request and supplied data before writing
`NeuroSpecification.md`. Keep the specification short and use these three
sections:

```markdown
# NeuroSpecification

- Status: draft | locked
- Researcher approval:

## Researcher request

## Inspected data contract

## Acceptance boundary
```

The three sections must establish the modeling question and requested outputs,
the inspected shapes/axes/units and preprocessing/data-to-model mapping, and the
evidence that distinguishes success, failure, and an inconclusive result.

Ask only for missing choices that change the model, data interpretation, or
acceptance boundary. Lock the specification with researcher approval before
step 1.

**Result:** locked `NeuroSpecification.md`; memory points to step 1.

## Step 1: Deep BrainX study

Study the relevant BrainX skills and APIs before implementing.

1. Open `brainx-general-guard` and identify every represented biological scale.
2. Read the complete owning modeling skills: `braincell`, `brainpy-state`, and/or
   `brainmass` according to the selected scales.
3. Open `brainunit`, `brainstate`, `brainevent`, and `braintrace` only when their
   capabilities participate in the model.
4. Follow the selected skills into every reference and canonical script that can
   change this implementation.
5. Trace exact APIs, construction, initialization, State and data flow, update
   order, execution, outputs, and validation.
6. Record the selected skills, opened resources, API choices, lifecycle,
   invariants, implementation design, and required optional coverage in memory.

Do not implement during this step.

**Result:** grounded BrainX study record and implementation design; memory points
to step 2.

## Optional training and fitting coverage

Training and fitting are optional coverage additions for steps 2–5. They are
not entry cases or standalone stages.

| Specification mode | Coverage through steps 2–5 |
|---|---|
| `forward-simulation` | Open neither optional workflow. |
| `task-training` | Open `references/training-workflow.md` at step 2 and keep it active through implementation, acceleration, experiment execution, and review. |
| `parameter-fitting` | Open `references/parameter-fitting-workflow.md` at step 2 and keep it active through implementation, acceleration, experiment execution, and review. |
| `hybrid` | Open `references/training-workflow.md` and `references/parameter-fitting-workflow.md` at step 2, then keep their objectives, State lifecycles, evidence, and review checks distinct through step 5. |

A step is incomplete until both its modeling requirements and every active
training/fitting requirement are satisfied. A step-5 refusal returns with the
same coverage unless the locked specification changes outside this loop.

## Step 2: Implementation

Implement the BrainX model and experiment from the locked specification and the
step-1 study record.

- Open the selected training or parameter-fitting reference before coding.
- Implement model dynamics, data processing, initialization, protocol, inputs,
  controls, observation mapping, metrics, artifacts, and active optional
  workflows through the owning BrainX APIs.
- Preserve units, State lifecycle, update order, randomness, and biological
  scale.
- Add focused component, equation, shape, unit, initialization/reset, control,
  observable, and metric checks.
- Run only small checks needed to establish readiness for acceleration and the
  experiment runner.

On `REFUSE`, apply the review findings here, then repeat steps 3–5.

**Result:** BrainX-native model and experiment code; memory points to step 3.

## Step 3: Acceleration

Open `brainx-acceleration`, apply its relevant workflow, and keep active
training/fitting coverage in force. Establish a representative baseline before
changing performance code and require parity in scientific outputs, State,
units, randomness, and declared numerical tolerances.

For training, also preserve loss, gradients, parameter updates, reset behavior,
and checkpoint meaning. For fitting, also preserve the objective, parameter
order and units, recovery behavior, and inference outputs.

**Result:** accelerated code with parity evidence, or an explicit decision that
no acceleration change is justified; memory points to step 4.

## Step 4: Experiment runner

Hand the locked specification, implementation, acceleration evidence, exact run
configuration, seeds, expected outputs, artifact locations, and active optional
coverage to `references/run-experiment.md`, then hand the launched run to
`references/monitor-experiment.md`.

Follow that reference through target-device smoke, immutable production or
replication launch, monitoring, and artifact collection.

**Result:** inspectable configuration, provenance, logs, raw results, metrics,
optional training/fitting evidence, and artifact paths; memory points to step 5.

## Step 5: MCP Codex review

Use the configured BrainX Codex MCP server. Treat
`mcp-servers/codex/system-prompt.md` as the sole review contract; the proxy
injects it as `base-instructions` into every fresh `mcp__codex__codex` call.

Start one fresh call per completed iteration and send the locked specification,
study record, code and tests, acceleration evidence, run configuration, raw
results, metrics, deterministic result assessment, claim-evidence matrix,
proposed next action, and active training/fitting evidence. Preserve the raw
response and returned `threadId`. Do not start a review with `codex-reply`, do not
duplicate the system prompt in the request, and do not substitute self-review.
If the configured MCP server is unavailable, checkpoint step 5 as blocked and
preserve the experiment artifacts.

| Review outcome | Transition |
|---|---|
| `REFUSE` | Preserve all findings, increment the iteration, return to step 2, and repeat steps 2–5. |
| `PASS` | Preserve the accepted code/result scope and continue to step 6. |

No other review outcome or transition belongs to the loop.

## Step 6: Visualization

Hand only review-passed specification, code, run artifacts, results, and accepted
scope to `references/visualization-workflow.md`.

This visualization instruction will be authored later. Do not invent its
mechanics in the loop. Until the file exists, checkpoint step 6 as blocked and
preserve the accepted artifacts.

**Result:** figures linked to accepted runs; memory marks the loop complete.

## 6. Implementation and optional workflow routing

### Package routing

Do not add a serial "study all BrainX skills" stage. Invoke
`brainx-general-guard`, identify every explicitly represented scale, and open only
the owning skills and references.

| Existing skill | Role in the loop |
|---|---|
| `brainx-general-guard` | Classify represented scales and enforce BrainX-native implementation. |
| `brainx-install` | Own installation, compatibility, version, and environment repair. |
| `brainunit` | Own dimensional safety and unit-aware external boundaries. |
| `brainstate` | Own mutable state, model graphs, environments, randomness, and transforms. |
| `braincell` | Own ions, channels, compartments, morphology, and cellular dynamics. |
| `brainevent` | Own event representations, sparse communication, connectivity, and event plasticity. |
| `brainmass` | Own aggregate population, regional, network, and whole-brain dynamics. |
| `brainpy-state` | Own point-neuron and spiking-network dynamics, projections, and training APIs. |
| `braintrace` | Activate only for long temporal training where BPTT memory is the bottleneck. |
| `brainx-acceleration` | Own the representative baseline, profiling, performance changes, and parity proof. |

Record a route such as:

```yaml
represented_scales: [point-neuron, network]
required_skills: [brainx-general-guard, brainunit, brainstate, brainpy-state]
conditional_skills: [brainevent]
excluded_skills:
  braincell: no channel or compartment mechanism represented
  brainmass: no aggregate population state represented
  braintrace: no temporal-training memory bottleneck
```

### BrainX-native subflow rule

Treat every subflow as a composition layer over the active BrainX model and its
routed package skills. A subflow may add training, inference, execution,
performance, or figure decisions; it must not replace BrainX state, units,
dynamics, or transforms with a generic framework workflow.

| Subflow | Required BrainX composition |
|---|---|
| Forward simulation | Construct and evolve the model through the routed scale packages, `brainstate`, and `brainunit` where quantities are physical. |
| Task training | Select BrainState parameter state, preserve BrainX state/reset semantics and units, use the owning model APIs, and activate `braintrace` only for its declared temporal-memory case. |
| Parameter fitting | Keep the simulator BrainX-native; preserve BrainUnit quantities and BrainState lifecycle through objectives, recovery, and external inference boundaries. |
| Acceleration | Apply BrainX/JAX transformations with explicit state, unit, event, solver, and stochastic parity. |
| Experiment execution | Launch the BrainX environment and model entry point with declared state initialization, randomness, units, devices, and artifacts. |
| Visualization | Read BrainX quantities and scale-specific outputs without silently stripping units, axes, event semantics, or provenance. |

Each subflow must record the BrainX skills and references it consumes. Reject an
implementation that could run unchanged as a generic PyTorch, JAX, SciPy, or
inference recipe with BrainX appearing only at the simulator call boundary.

### Optional workflows

Keep the decision boundary in the root and the full workflow in one reference.

| Execution mode | Root action | Exact route |
|---|---|---|
| `forward-simulation` | Implement simulation, controls, observation, and metrics directly; open neither optional workflow | None. |
| `task-training` | Preserve parameter/state/reset/split/checkpoint invariants, then open the training workflow | `references/training-workflow.md`. |
| `parameter-fitting` | Preserve units/observation/recovery/identifiability invariants, then open the fitting workflow | `references/parameter-fitting-workflow.md`. |
| `hybrid` | Open both in the phase order locked by the specification; keep objectives, splits, checkpoints, and gates separate | The training and parameter-fitting references. |

Keep task training and parameter fitting as modeling-loop references. Nest the
complete BrainCell HH fitting script under the parameter-fitting reference.


## 8. Experiment execution

### Experiment execution reference boundary

Keep `references/run-experiment.md` as a focused modeling-loop reference because
process execution, checkpointing, environment handling, and immutable run
capture vary independently from the scientific loop. It must not contain a
scientific iteration or result acceptance policy.

Keep post-launch monitoring, stopping, retry, and collection in the single
`references/monitor-experiment.md` companion. Do not split either workflow into
additional Markdown references.

Make the runner BrainX-aware: it launches the routed model under the declared
BrainState environment, initialization, randomness, units, precision, and device
contract. It is not a generic shell-job runner.

### Run levels

| Level | Purpose | Gate |
|---|---|---|
| `smoke` | Verify environment, imports, device, construction, and output paths | Mechanical checks pass. |
| `production` | Execute the locked experiment at declared scale | Implementation is ready and optional acceleration parity passes. |
| `replication` | Confirm seeds, subjects, protocols, or controls | Production artifacts are valid. |

### Immutable run snapshot

Before launch, record:

- run and parent-run IDs;
- specification ID, version, and hash;
- code commit and dirty-diff hash;
- exact config and command;
- data and processed-artifact hashes;
- BrainX, JAX, Python, accelerator, and dependency versions;
- device, platform, and precision;
- random seeds and determinism expectation;
- checkpoint source and compatibility evidence;
- expected outputs, resource estimate, and kill conditions.

Any scientific config change creates a new run ID. Do not edit `RUN_SPEC.md` or
`config.json` after launch.

### Monitoring and retry

- Monitor process liveness, device use, disk, checkpoint freshness, and logs.
- Monitor only scientific health signals declared by the active workflow.
- Stop deterministic invalidity such as NaN/Inf, corrupt output, impossible state,
  confirmed leakage, or a broken metric.
- Wait when evidence is only noisy or insufficient.
- Retry transient failure with the same scientific config within budget.
- Return to loop implementation when an OOM or precision workaround changes
  semantics.
- Collect inspectable artifacts from stopped and failed runs when safe.

Mark completion `done`, not scientifically `accepted`.

## 9. Integrated result review and Codex quality gate

Keep deterministic assessment in the modeling loop. Keep the reviewer behavior
and structured output contract only in `mcp-servers/codex/system-prompt.md`; the
MCP proxy injects it into every fresh `codex` call. Keep only invocation mechanics
and the artifact-path packet in `brainx-modeling-loop/SKILL.md`. Do not create a
review skill, review prompt reference, or second copy of the system prompt.

### Mental model

Judge the run against criteria locked before production. Separate raw observation,
interpretation, alternative explanation, and the next discriminating action.

### Deterministic evidence synthesis

1. Verify artifact hashes, config identity, exclusions, completion, and metric
   recomputation where feasible.
2. Load the exact specification version and hash used by every reviewed run.
3. Build a raw comparison table across conditions, controls, seeds, and accepted
   baselines.
4. Compute primary and secondary summaries with the locked aggregation and
   uncertainty rules.
5. Check numerical stability, units, boundaries, seed sensitivity, and outliers.
6. Apply training leakage/checkpoint checks when the training reference was used.
7. Apply fitting recovery/identifiability/predictive checks when the fitting
   reference was used.
8. Evaluate every claim against its minimum evidence.
9. List alternative explanations and whether controls discriminate them.
10. Propose one result category and one legal next action for Codex to audit.

### Result categories

| Category | Meaning | Proposed next action |
|---|---|---|
| `supported` | Required evidence passes for the scoped claim | Ask Codex to accept or identify remaining defects. |
| `partially-supported` | A narrower claim passes with explicit limits | Ask Codex whether narrowed evidence is sufficient. |
| `refuted` | A valid run contradicts the hypothesis under declared conditions | Ask Codex whether the negative conclusion is rigorous enough to close. |
| `inconclusive` | A valid run cannot distinguish leading explanations | Ask Codex whether another in-spec iteration would add useful evidence. |
| `invalid` | Code, data, numerical, leakage, metric, or provenance defects prevent interpretation | Propose revision; Codex identifies the minimum validity fix. |

### Claim-evidence matrix

| Claim ID | Locked wording | Required evidence | Observed evidence | Run/artifact | Verdict | Scope limit |
|---|---|---|---|---|---|---|

Do not mark support from a secondary metric, best seed, or post hoc subset unless
the locked specification explicitly permits it.

### Integrated review artifact

```markdown
# Iteration review

## Contract and provenance
- Specification ID/version/hash:
- Run IDs:
- Code/data/environment identity:

## Artifact-integrity checks
## Raw comparison table
## Primary result and uncertainty
## Controls and failure checks
## Training or fitting validation
## Claim-evidence matrix
## Alternative explanations
## Proposed scientific outcome
## Proposed next action
## Unresolved assumptions

## Codex quality gate
- Review ID:
- Outcome:
- Scientific outcome:
- Good-enough reason:
- Next action:

## Codex findings
| ID | Severity | Location | Problem | Scientific consequence | Minimum fix | Status |
|---|---|---|---|---|---|---|

## Unverified Codex assumptions
```

Write the deterministic sections to `ITERATION_REVIEW.md`, keep the stage
`running`, and invoke Codex. Preserve its raw response separately, then append the
parsed gate, outcome, findings, assumptions, and next action to the same artifact.
Mark the artifact `accepted` only on `PASS`; otherwise close it as `done` and
apply the returned transition.

### Review timing and packet

Start one fresh `mcp__codex__codex` call after deterministic evidence synthesis
in each modeling iteration. The proxy injects the authoritative system prompt.
Provide concrete file paths for:

- locked `NeuroSpecification.md`;
- relevant model and experiment code;
- relevant tests and optional acceleration parity evidence;
- optional training/fitting plan and diagnostics;
- production run specs, raw metrics, logs, and artifact manifest;
- the deterministic sections of `ITERATION_REVIEW.md`;
- prior open finding IDs and changes made since the prior iteration.

Do not call Codex before evidence synthesis is complete unless the loop is
blocked and the researcher explicitly requests diagnostic review.

### Gate behavior

| Outcome | Loop action |
|---|---|
| `PASS` | Accept the reviewed scope, append the review checkpoint, and advance to visualization. |
| `REFUSE` | Preserve every finding, increment the iteration, return to step 2, and repeat steps 2-5. |

### Review invocation and finding lifecycle

1. Reserve one review ID for the iteration and record the request before invoking
   the MCP tool.
2. Save the full raw response immediately in `reviews/<review_id>_raw.md`.
3. On resume, parse an existing raw response; never issue a second call for the
   same iteration.
4. Parse gate, outcome, findings, assumptions, and next action into
   `ITERATION_REVIEW.md` without rewriting reviewer language.
5. Give findings stable IDs across iterations.
6. For `REFUSE`, record the code/test/run evidence expected to close each required
   finding.
7. On the next iteration, pass prior findings and actual changed artifacts.
8. Close a finding only when Codex accepts its evidence.
9. Preserve every prior raw review.
10. If the MCP tool is unavailable, mark the gate `blocked`; do not silently
   self-review and call it independent.

### Iterative review rule

Use one fresh Codex review per completed iteration. The review itself does not debate or
spawn another review loop. `REFUSE` causes the modeling loop to revise,
rerun, rebuild `ITERATION_REVIEW.md`, and call Codex again in the next iteration.

This preserves generator-reviewer separation without creating two orchestration
systems.

## 10. Iteration transitions and stopping

### Iteration record

Write one immutable record per iteration:

```markdown
# Modeling iteration <N>

- Specification: ID, version, hash
- Iteration goal:
- New information expected:
- Package-skill route:
- Optional workflow: none | training | fitting | hybrid
- Implementation changes:
- Acceleration and parity:
- Production run IDs:
- Integrated review artifact:
- Codex review ID, gate, and scientific outcome:
- Open findings:
- Next transition and reason:
- Budget used and remaining:
```

Do not overwrite prior iteration records.

### Legal revision targets

| Finding owner | Return stage |
|---|---|
| BrainX API, equation mapping, state, units, solver, input, observation, metric code | Implementation. |
| Task loss, split, reset, gradient, optimizer, checkpoint, tuning | Training workflow reference. |
| Objective, likelihood, bounds/priors, recovery, identifiability, predictive checks | Parameter-fitting workflow reference. |
| Acceleration parity | Reject optimization or return to implementation. |
| Process, environment, checkpoint, artifact collection | Experiment runner. |
| Analysis aggregation, exclusion, uncertainty, claim-evidence mapping | Integrated review stage and, if required, a new run. |
| Hypothesis, represented scale, data inclusion, primary outcome, success criterion, claim scope | Specification revision plus researcher. |

### Circuit breakers

Stop automatic iteration when:

- maximum iterations, runs, compute, or wall-clock budget is exhausted;
- two valid iterations add no new discriminating evidence;
- the same major Codex finding remains after two attempted revisions;
- the same execution failure exceeds its retry budget;
- training keeps tuning without validation improvement under its stop rule;
- fitting remains non-identifiable for the intended interpretation;
- the next useful action crosses the locked specification boundary;
- the required Codex reviewer remains unavailable;
- accepted artifacts contradict each other and existing evidence cannot resolve
  the conflict.

Do not relax evidence criteria, remove controls, widen parity tolerance, or hide
failed runs to keep iterating.

### Successful closure

Close when:

- the active specification is locked and traceable;
- required runs and controls are valid;
- `ITERATION_REVIEW.md` has one scoped result category;
- the Codex review returns `PASS` for that category;
- every accepted claim has a claim-evidence row;
- memory and machine state agree on closure;
- final figures, if requested, link to accepted evidence;
- remaining uncertainty is explicit.

## 11. Persistence and project artifacts

### Recommended project layout

```text
project-root/
|-- NeuroSpecification.md
|-- brainmodeling-memory.md
|-- data/
|   |-- DATA_MANIFEST.md
|   |-- raw/
|   `-- processed/
|-- model/
|-- experiments/
|   |-- EXPERIMENT_PLAN.md
|   |-- iterations/
|   |   `-- iteration-<N>/
|   |       |-- ITERATION.md
|   |       `-- ITERATION_REVIEW.md
|   `-- runs/
|       `-- <run_id>/
|           |-- RUN_SPEC.md
|           |-- config.json
|           |-- provenance.json
|           |-- stdout.log
|           |-- metrics.json
|           `-- artifacts.json
|-- reviews/
|   `-- <review_id>_raw.md
|-- figures/
|   `-- FIGURE_MANIFEST.md
`-- .brainx-loop/
    |-- state.json
    |-- events.jsonl
    |-- decisions.jsonl
    `-- locks/
```

Create only the directories activated by the project.

### Artifact authority

| Artifact | Owner | Mutation rule |
|---|---|---|
| `NeuroSpecification.md` | Modeling loop | Versioned and researcher-locked. |
| `brainmodeling-memory.md` | Modeling loop | Append one iteration/step checkpoint after every step, including blockers; never overwrite earlier checkpoints. |
| `.brainx-loop/state.json` | Modeling loop | Atomic machine-readable transitions. |
| `events.jsonl` and `decisions.jsonl` | Modeling loop | Append-only; corrections supersede. |
| `ITERATION.md` | Modeling loop | Immutable after the integrated review stage. |
| Run specs/config/provenance/logs/metrics | Experiment runner | Immutable after launch/completion. |
| `ITERATION_REVIEW.md` | Modeling loop | `running` through evidence synthesis and Codex review; `accepted` only on `PASS`. |
| Raw Codex review | Modeling loop | Store verbatim and never rewrite. |
| Parsed gate and findings | Modeling loop | Append to `ITERATION_REVIEW.md`; preserve stable finding IDs. |
| `FIGURE_MANIFEST.md` | Visualization | Update when figure or source evidence changes. |

### Machine-readable state

```json
{
  "schema_version": 1,
  "active_spec": {"id": "NS-001", "version": 2, "hash": "..."},
  "iteration": 3,
  "stage": "integrated-review",
  "status": "running",
  "iteration_goal": "distinguish parameter tradeoff using protocol B",
  "expected_new_information": "break the sensitivity ridge",
  "open_finding_ids": ["QR-012"],
  "active_run_ids": ["run-0007"],
  "budgets": {
    "max_iterations": 6,
    "iterations_used": 3,
    "max_runs": 12,
    "runs_used": 5,
    "max_compute_hours": 24,
    "compute_hours_used": 8.5
  },
  "next_obligation": "invoke Codex and complete ITERATION_REVIEW.md",
  "updated_at": "..."
}
```

Use only:

| Status | Meaning |
|---|---|
| `pending` | Not started. |
| `running` | Started but required artifact is incomplete. |
| `done` | Artifact exists but its gate has not accepted it. |
| `accepted` | Codex or owning deterministic gate passed with evidence. |
| `failed` | Execution failed and bounded recovery may remain. |
| `blocked` | Researcher input or required capability is needed. |
| `skipped` | The specification makes the stage inapplicable. |
| `superseded` | A newer spec, iteration, run, or review replaced it. |

Write state atomically. On resume, verify referenced paths and hashes before
trusting status.

### Control settings

```yaml
AUTO_ENGINEERING_FIXES: true
AUTO_ITERATE_AFTER_REVISE: true
AUTO_EXECUTION_RETRY: true
REQUIRE_RESEARCHER_SPEC_LOCK: true
REQUIRE_RESEARCHER_SPEC_REVISIONS: true
REQUIRE_CODEX_QUALITY_GATE: true
MAX_ITERATIONS: 6
MAX_RUNS: 12
MAX_EXECUTION_RETRIES_PER_RUN: 2
MAX_WALL_CLOCK_HOURS: 24
MAX_COMPUTE_HOURS: 24
```

Do not use one global `AUTO_PROCEED` flag. Engineering autonomy and scientific
authority are different permissions.

## 12. Visualization

### `brainx-visualization` boundary

Keep visualization separate because figure mechanics and domain plot types are
large enough for progressive disclosure. It never decides acceptance.

Require it to consume the active BrainX route so scale-specific states, events,
quantities, and observation mappings retain their biological meaning.

| Mode | Use | Source rule |
|---|---|---|
| `diagnostic` | State traces, raster, phase plane, loss/gradient, residual, recovery, posterior predictive, sensitivity | May use unaccepted runs when clearly labeled diagnostic. |
| `final` | Condition, perturbation, fitted observable, uncertainty, and multiscale figures | Use Codex-accepted run IDs unless failed/null evidence is intentionally shown and labeled. |

The visualization workflow must state its question, source runs, transformations,
smoothing, aggregation, exclusions, units, uncertainty, sample size, and control
references. Render and inspect for blank output, clipping, misleading scale,
overplotting, and inconsistency with raw values. Record all source hashes in
`FIGURE_MANIFEST.md`.

## 13. Proposed skill-bundle structure

```text
skills/
|-- brainx-modeling-loop/
|   |-- SKILL.md                    # spec + integrated result/Codex gate
|   |-- references/
|   |   |-- training-workflow.md
|   |   |-- parameter-fitting-workflow.md
|   |   |-- parameter-fitting-workflow/
|   |   |   `-- scripts/fitting_hh_neuron.py
|   |   |-- run-experiment.md
|   |   |-- monitor-experiment.md
|   |   `-- visualization-routing.md
|   `-- scripts/
|       |-- loop_state.py
|       `-- artifact_lint.py
|-- brainx-visualization/
|   |-- SKILL.md
|   `-- references/
|       |-- cellular-and-spiking-figures.md
|       |-- mass-and-network-figures.md
|       |-- training-and-fitting-diagnostics.md
|       `-- figure-integrity.md
|-- brainx-general-guard/            # existing
|-- brainx-install/                  # existing
|-- brainunit/                       # existing
|-- brainstate/                      # existing
|-- braincell/                       # existing
|-- brainevent/                      # existing
|-- brainmass/                       # existing
|-- brainpy-state/                   # existing
|-- braintrace/                      # existing
`-- brainx-acceleration/       # planned in plan.md; implement or restore
```

Do not add separate skills for the experiment loop, specification, training,
result review, or neuroscience review. Do not duplicate the inline
specification or integrated review contract in references.

The root `SKILL.md` is intentionally larger because it owns the three essential
contracts. Keep optional variants and API detail in routed references.

## 14. Implementation roadmap

### Phase 0: Freeze the single-loop contract

Deliver:

- final root workflow and legal transitions;
- final inline specification format;
- final Codex MCP invocation packet and verified system-prompt injection;
- training/fitting reference boundaries;
- machine state and append-only event/decision schemas;
- three end-to-end fixture projects.

Exit when every artifact has one authoritative writer, every iteration has one
Codex gate, and a negative result can close successfully.

### Phase 1: Build a forward-simulation vertical slice

Implement:

1. `brainx-modeling-loop` with specification, package routing, implementation,
   integrated result/Codex review, iteration, memory, and resume;
2. `references/run-experiment.md` plus `references/monitor-experiment.md` for local immutable runs;
3. `brainx-visualization` for diagnostic and final traces.

Use spike-frequency adaptation, neural compass, or binocular rivalry. Simulate
Codex MCP responses through fixtures until the real review-agent tool exists.

Exit when `REFUSE` produces a new full iteration and the next Codex call can close
stable findings without a second loop or reviewer skill.

### Phase 2: Add training reference

Write `references/training-workflow.md` using temporal-order learning and
sleep-memory replay. Add tiny-batch checks, state-reset tests, bounded tuning,
checkpoint selection, multi-seed confirmation, and monitoring separation. Build
both fixtures around routed BrainX models and BrainState/BrainUnit semantics; do
not use framework-neutral training examples.

Exit when held-out results cannot influence tuning and invalid gradients or state
leakage are caught before the quality gate.

### Phase 3: Add parameter-fitting reference

Write `references/parameter-fitting-workflow.md` with the differentiable path,
bounded derivative-free path, exact-pipeline recovery gate, and the routed
BrainCell HH fitting source script nested under its reference directory. Keep the
simulator, parameter quantities, State lifecycle, and observation mapping
BrainX-native on both fitting paths.

Exit when fitted parameters cannot be interpreted before recovery and
non-identifiability returns a bounded conclusion.

### Phase 4: Integrate acceleration and real Codex MCP review

Implement or restore `brainx-acceleration`. Connect the real Codex MCP
review call to its injected system prompt and the root skill's artifact-path packet. Preserve raw response,
stable findings, closure evidence, and unavailable-tool behavior.

Exit when every completed iteration has exactly one Codex review and acceleration
cannot pass without parity.

### Phase 5: Harden persistence and installation

Add atomic state updates, stale-lock recovery, artifact hashing, memory
compaction, manifest entries, adapter installation, validation, and upgrade
behavior.

Exit when `done`-but-unaccepted work is revalidated, superseded specs/runs remain
traceable, and installer updates preserve project state.

## 15. Evaluation plan

### Scenarios

| Scenario | Primary behavior |
|---|---|
| Spike-frequency adaptation | Inline spec, cellular routing, production run, integrated result/Codex gate, and figure. |
| Sound localization | Point-neuron delays, event routing, acceleration parity, and review findings. |
| Cortical wave obstacle | Performance work and preservation of wave dynamics. |
| Temporal-order learning | Training reference activation, target alignment, state reset, and bounded tuning. |
| Sleep-memory replay | Long temporal training and conditional BrainTrace routing. |
| Neural compass or grid-cell sweep | Aggregate dynamics, sweep discipline, and claim-evidence review. |
| Single-cell perturbation failure | Valid negative/inconclusive result and `PASS` without positive-result pressure. |
| Interrupted run | Stage state, checkpoint compatibility, immutable run identity, and resume. |
| Codex MCP unavailable | Quality gate remains blocked without losing completed work. |
| Specification drift attempt | Review returns `REFUSE` with an out-of-contract finding; the locked specification is not changed silently. |

### Structural tests

- Only `brainx-modeling-loop` owns scientific iteration.
- The specification schema has one copy in the root skill; the review contract has
  one copy in `mcp-servers/codex/system-prompt.md`.
- Training and fitting exist as references, not skills.
- Every subflow declares and composes its routed BrainX package knowledge; no
  subflow is a generic framework workflow around a simulator callback.
- No separate experiment-loop, result-review, or neuroscience-review skill exists.
- Every reference route states activation condition and owned decisions.
- No workflow file duplicates package API catalogs.
- `manifest.json`, package files, and installer adapters include intended skills.
- State and artifact schemas parse and validate.

### Behavioral tests

1. Turn an idea into a locked specification without another specification skill.
2. Reject execution when the spec is draft or its hash is stale.
3. Resume from every stage status, including deterministic evidence awaiting the
   Codex call within the integrated review stage.
4. Detect missing or corrupt artifacts rather than trusting state.
5. Inject unit, reset, update-order, observation, split, and metric defects; verify
   the integrated review stage blocks acceptance.
6. Verify exactly one Codex call occurs per completed iteration.
7. Verify `REFUSE` starts a full new iteration and carries stable finding IDs.
8. Verify held-out data cannot direct training tuning.
9. Verify poor recovery blocks mechanistic fitting interpretation.
10. Reject training, fitting, acceleration, execution, and visualization fixtures
    that bypass routed BrainX state, unit, or model semantics.
11. Verify a valid refutation or bounded inconclusive result can receive `PASS`.
12. Verify every repeat names the new information expected.
13. Exhaust a budget and stop without relaxing the contract.

### Quality measures

- completeness of locked specifications before implementation;
- accepted claims with valid run/artifact links;
- run provenance completeness;
- injected scientific/code fault detection rate;
- resume fidelity;
- unrelated package skills opened per iteration;
- invalid-run rejection rate;
- negative-result acceptance without specification drift;
- Codex finding closure with evidence;
- exactly one Codex review per completed iteration;
- speedup only among parity-passing acceleration changes;
- figure-to-run traceability.

Do not optimize evaluation for supported hypotheses or numeric reviewer scores.

## 16. Lessons from reference systems

### Adopt

- From ARIS, adopt resumable `done` versus `accepted` state, raw reviewer traces,
  stable findings, bounded iteration, and the rule that monitoring does not judge
  scientific support.
- From NORA, adopt a focused active contract, explicit run tracking, stage-owned
  artifacts, and separated automation flags.
- From training-check, adopt process-health versus training-quality separation.
- From automatic hyperparameter tuning, adopt baseline-first execution, bounded
  run history, phase-aware search, and checkpoint compatibility.
- From parameter-recovery guidance, adopt exact-pipeline recovery, realistic data
  volume/noise, bias/tradeoff analysis, and limited interpretation when recovery
  fails.

### Do not adopt

- literature search, novelty scoring, paper writing, venue scoring, or submission;
- a universal research-quality score;
- nested experiment or review loops;
- separate specification, training, result-review, or reviewer skills;
- duplicated contracts in references;
- an unbounded auto-proceed workflow;
- dependence on W&B, SSH, tmux, one GPU provider, or one operating system;
- reviewer instructions that reward positive results;
- a framework that bypasses existing BrainX package skills.

### Reference sources

- [Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)
- [Night Owl Research Agent](https://github.com/GRIND-Lab-Core/night_owl_research_agent)
- [Training check](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/training-check/SKILL.md)
- [Automatic hyperparameter tuning](https://github.com/zxh0916/auto-hparam-tuning/blob/main/skills/auto-hparam-tuning/SKILL.md)
- [Parameter recovery checker](https://github.com/NeuroAIHub/awesome_cognitive_and_neuroscience_skills/blob/master/skills/parameter-recovery-checker/SKILL.md)

## 17. Recommended first implementation

Build the forward-simulation vertical slice first:

```text
brainx-modeling-loop
  -> inline specification
  -> package routing and implementation
  -> optional acceleration
  -> experiment runner
  -> integrated result review + one Codex neuroscience quality gate
       -> PASS or full revised iteration
  -> memory and visualization
```

This proves the single-loop contract before training and fitting add complexity.
Once it survives interruption, negative results, review revisions, and
specification changes, add the two optional references without changing the root
workflow.
