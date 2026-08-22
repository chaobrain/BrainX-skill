# BrainX closed-loop brain modeling workflow plan

## 1. Purpose and boundary

### Mission

Build one first-level BrainX modeling skill that turns a researcher's idea into a
specified, implemented, executed, reviewed, iterated, and visualized brain model.
Make the loop resumable and make every accepted scientific statement traceable to
immutable run evidence.

Use one loop and one independent review gate:

```text
brainx-modeling-loop

idea
  -> specification
  -> BrainX implementation
  -> optional training or fitting workflow
  -> optional acceleration
  -> production experiment
  -> integrated result review + Codex neuroscience quality gate
  -> accept, revise, revise specification, or stop
  -> memory and visualization
```

The Codex review is iterative because every revised experiment returns to the
same quality gate. Do not add a separate review loop, experiment loop, or result
review skill.

### In scope

- Convert a researcher-provided brain modeling idea into a concise scientific
  specification.
- Inspect user-provided data and define its mapping to a BrainX model.
- Build BrainX-native models across cellular, point-neuron, event-driven,
  population, regional, and multiscale representations.
- Run forward simulation, perturbation, task training, and parameter fitting.
- Route selectively to the existing BrainX package skills in `plan.md`.
- Audit performance while preserving scientific behavior.
- Run one Codex MCP neuroscience quality review after each completed experiment
  iteration.
- Iterate on implementation or experiment design within a locked specification.
- Persist decisions, failed attempts, accepted evidence, and unresolved risks.
- Produce diagnostic and final scientific figures.

### Out of scope

- Literature discovery, novelty search, citation management, and paper ranking.
- Autonomous idea generation from literature.
- Manuscript, grant, rebuttal, or submission-package writing.
- General machine-learning experimentation unrelated to brain modeling.
- A standalone neuroscience-review skill or generic reviewer score.
- A second scientific loop nested inside the modeling loop.
- Replacing researcher judgment when the hypothesis, biological abstraction,
  data contract, or evidence criterion must change.

The workflow may preserve citations or provenance supplied by the researcher,
but it must not start a literature-search branch by itself.

### Relationship to `plan.md`

Keep `plan.md` as the source of truth for package skills and their progressive
disclosure. Use this document as the source of truth for the end-to-end modeling
loop, project artifacts, quality gate, and iteration policy.

Do not duplicate package APIs here. The loop decides which package skills and
references to open; those files remain authoritative for BrainX mechanics.

## 2. Simplified architecture

### Main decisions

1. Use `brainx-modeling-loop` as the only first-level workflow skill.
2. Use one scientific iteration loop from specification through Codex quality
   review.
3. Put the compact researcher-request and inspected-data contract for
   `NeuroSpecification.md` directly in `brainx-modeling-loop/SKILL.md`.
4. Put result-review rules and the condensed Codex neuroscience quality-review
   contract directly in the same root skill as one integrated gate.
5. Make the Codex call the single end-of-iteration quality gate. It reviews the
   specification, code, execution evidence, and proposed result verdict together.
6. Put training and parameter-fitting workflows in optional Markdown references
   under `brainx-modeling-loop/references/`.
7. Make every routed subflow compose the routed BrainX package knowledge. Do
   not permit generic training, fitting, acceleration, execution, or visualization
   workflows that merely wrap a BrainX model.
8. Keep only the activation boundary and critical training/fitting warnings in the
   root skill.
9. Keep `brainx-experiment-runner` as a focused execution skill. It may retry
   processes but does not own scientific iteration.
10. Keep `brainx-visualization` as a focused optional skill.
11. Reuse existing package skills and `brainx-acceleration-audit` selectively.
12. Do not create `brainx-experiment-loop`, `brainx-training`,
    `brainx-parameter-fitting`, `brainx-result-review`,
    `brainx-neuro-specification`, or `brainx-neuroscience-review` as separate
    skills.

### Ownership

```text
brainx-modeling-loop
  owns:
    scientific specification
    iteration sequence
    optional workflow routing
    integrated result-review and Codex neuroscience quality gate
    revision decision
    memory and stopping policy

  opens on demand:
    references/training-workflow.md
    references/parameter-fitting-workflow.md
    package-specific BrainX skills and references
    brainx-acceleration-audit
    brainx-experiment-runner
    brainx-visualization
```

The loop is the only scientific acceptance owner. The experiment runner only
reports execution completion. The Codex reviewer supplies an independent gate
verdict; it does not mutate project state or edit code.

## 3. Single-loop workflow

```text
Researcher idea or existing project
        |
        v
[S0] Initialize or resume
        |  validate state, specification hash, memory, runs, open findings
        v
[S1] Parse the request, inspect available data, and write NeuroSpecification.md
        |
        +---- researcher locks scientific contract? ---- no --> clarify or stop
        v
[S2] Route modeling scales and implement BrainX model/protocol
        |
        +---- task learning? ---------> open training-workflow.md
        +---- parameter inference? ---> open parameter-fitting-workflow.md
        `---- forward simulation? ----> open neither
        v
[S3] Accelerate only when justified and prove parity
        |
        +---- parity failure ------------------------------> reject optimization
        v
[S4] Run immutable production experiment(s)
        |
        +---- execution failure ---------------------------> bounded same-config retry
        v
[S5] Build result assessment and invoke Codex MCP as one integrated quality gate
        |
        +---- ACCEPT --------------------------------------> S6
        +---- REVISE --------------------------------------> earliest affected S2-S5
        +---- SPEC_REVISION_REQUIRED ----------------------> S1 + new version + researcher
        `---- BLOCK ---------------------------------------> stop for researcher
        v
[S6] Update memory, render accepted/diagnostic figures, and close
     or define another iteration with explicit new evidence expected
```

One pass from S1/S2 through S5 is one modeling iteration. A `REVISE` verdict
starts another iteration under the same locked specification. A scientific
contract change starts a new specification version, not a hidden continuation.

### What "good enough" means

The Codex gate asks whether the completed iteration is good enough to support its
proposed scientific outcome. "Good enough" does not mean positive, impressive,
or publishable. It means:

- the code faithfully implements the locked specification;
- the run and analysis are valid and reproducible enough for the declared scope;
- required controls and uncertainty are present;
- the result verdict follows from raw evidence;
- alternative explanations are bounded honestly;
- no critical or major defect remains that could change the conclusion.

A rigorous refutation or bounded inconclusive result may receive `ACCEPT`.

## 4. `brainx-modeling-loop` root skill design

### Purpose and boundary

Use `brainx-modeling-loop` for a new, resumed, or revised BrainX modeling project.
It owns the whole scientific lifecycle from specification to acceptance.

It does not own exhaustive package APIs, platform-specific execution mechanics,
or detailed training/fitting variants. Route those details precisely.

### Mental model

Treat each iteration as a falsifiable implementation of one locked scientific
contract. The loop accepts an iteration only when the integrated result and Codex
quality gate accepts the proposed outcome at its stated scope.

### Canonical workflow

1. Locate project root and current loop artifacts.
2. Choose `new`, `resume`, `repair-state`, or `revise-specification` entry mode.
3. Check BrainX package presence without inspecting installed package source;
   route installation or compatibility work to `brainx-install` when needed.
4. Parse the researcher request, inspect available data, and create or update
   `NeuroSpecification.md` using the inline format below.
5. Resolve blocking ambiguities and obtain researcher lock.
6. Define the iteration goal and the new information expected.
7. Invoke `brainx-general-guard`, classify represented scales, and open only
   owning package skills.
8. Implement the model, protocol, controls, observation, and required tests.
9. Open the training or fitting reference only when its activation condition
   matches.
10. Invoke acceleration only when profiling or expected cost justifies it.
11. Invoke `brainx-experiment-runner` for immutable production runs.
12. Build the result assessment and invoke the future Codex MCP review-agent as
    one integrated gate using the inline contract below.
13. Store raw review and parse its gate verdict into the same iteration review.
14. On `REVISE`, return to the earliest affected stage and start a new iteration.
15. On `SPEC_REVISION_REQUIRED`, propose a new specification version and ask the
    researcher.
16. On `ACCEPT`, update memory and invoke visualization as needed.
17. Close or define another iteration only when it expects new discriminating
    evidence.

## 5. Inline `NeuroSpecification.md` format

Use this file as a compact handoff from researcher intent and inspected data to
BrainX implementation. Do not turn it into a complete methods plan or duplicate
decisions owned by routed workflows. Keep the format directly in
`brainx-modeling-loop/SKILL.md`; do not create a separate specification skill,
template, or authoritative reference.

```text
researcher request
  -> concise intent summary
  -> inspection of supplied data
  -> observed data contract and model-data mapping
  -> confirmation of unresolved decisions
  -> locked NeuroSpecification.md
```

```markdown
# NeuroSpecification

- ID:
- Version:
- Status: draft | locked | superseded
- Parent version:
- Researcher approval:

## Researcher request
- Goal in the researcher's words:
- Question, hypothesis, or behavior to model:
- Requested outputs or decisions:
- Execution mode: forward-simulation | task-training | parameter-fitting | hybrid
- Scope limits and user constraints:

## Inspected data contract
- Mode: none | synthetic | observed | mixed
- Source paths and immutable raw/processed identities:
- Inspected files, variables, dtypes, shapes, and axes:
- Sampling or time base, physical units, and independence unit:
- Missingness, artifacts, exclusions, and quality limits:
- Existing or required preprocessing and the subset that fits each transform:
- Train/validation/test or fit/check boundaries, when applicable:
- Mapping from source variables to model inputs, targets, and observables:
- Unresolved schema, unit, leakage, or model-data mismatches:

## BrainX modeling handoff
- Represented biological scales and routed BrainX package skills:
- Modeling abstraction or model family:
- Inputs, interventions, baseline, and controls:
- Latent-state-to-observation mapping:
- Fixed, trainable, or fitted parameters and their units:
- Essential run constraints: duration, sampling, seeds, compute, or precision:

## Acceptance boundary
- Primary result and minimum evidence:
- Invalid-result conditions:
- Allowed interpretation and explicit non-claims:
- Iteration, run, compute, and wall-clock limits:

## Open decisions
| ID | Missing decision or assumption | Consequence if wrong | Blocking stage | Owner |
|---|---|---|---|---|

## Revision history
| Version | Change | Reason | Approved by | Supersedes |
|---|---|---|---|---|
```

### Specification rules

- Parse the researcher's words into a testable request without inventing a
  hypothesis, biological mechanism, or success threshold.
- Inspect available data before writing the data contract. Record observed schema
  and provenance; do not infer units, axes, parameter meaning, or independence
  from names.
- Keep raw data read-only, identify processed data separately, and fit
  preprocessing only on permitted subsets.
- When no data is supplied, declare `none` or `synthetic` and record the required
  synthetic-data or observation assumptions.
- Keep detailed equations, optimizer settings, fitting methods, and package APIs
  out of the specification; route them to the owning BrainX workflow and skills.
- Define the observation mapping and the minimum acceptance boundary before
  production.
- Mark unresolved decisions as implementation-blocking, production-blocking, or
  interpretation-limiting.
- Lock the specification by content hash and version.
- Any post-result change to a locked scientific field creates a new version.

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
| `brainx-acceleration-audit` | Own the representative baseline, profiling, performance changes, and parity proof. |

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

### Optional workflow references

Keep the decision boundary in the root and the full workflow in one reference.

| Execution mode | Root action | Exact reference |
|---|---|---|
| `forward-simulation` | Implement simulation, controls, observation, and metrics directly; open neither optional reference | None. |
| `task-training` | Preserve parameter/state/reset/split/checkpoint invariants, then open the training workflow | `references/training-workflow.md`. |
| `parameter-fitting` | Preserve units/observation/recovery/identifiability invariants, then open the fitting workflow | `references/parameter-fitting-workflow.md`. |
| `hybrid` | Open both in the phase order locked by the specification; keep objectives, splits, checkpoints, and gates separate | Both references. |

Do not make `brainx-training` or `brainx-parameter-fitting` standalone skills.
They are branches of one modeling loop.

### `references/training-workflow.md` plan

#### Purpose

Compose task learning with the routed BrainX model, BrainState parameter and state
lifecycle, BrainUnit quantities, the owning model package, and conditional
BrainTrace behavior. Do not write a generic training guide or duplicate package
APIs.

#### Required contents

1. Select trainable parameter state and exclude hidden/runtime state.
2. Declare temporal, batch, sample, and target axes with units.
3. Define train, validation, and held-out partitions by the correct independence
   unit.
4. Define state initialization and reset for trials, batches, subjects, sequences,
   and evaluation.
5. Route loss, surrogate, optimizer, schedule, and checkpoint mechanics to owning
   local references.
6. Verify finite gradients reach only intended parameters.
7. Overfit a tiny batch or short sequence.
8. Run an unchanged baseline before tuning.
9. Tune only within a declared search space and budget using validation metrics.
10. Select checkpoints on validation data and confirm with multiple seeds.
11. Evaluate held-out data only after selection is complete.

#### Monitoring split

| Layer | Checks | Allowed action |
|---|---|---|
| Process health | Process, device, disk, checkpoints, log progress | Retry, resume, or stop broken execution. |
| Training quality | NaN/Inf, gradients, loss/validation trend, state leakage, metric correctness | Continue, wait, declare invalid, or start a revised iteration. |

Do not use monitoring as the final quality gate. The integrated review stage
judges the completed iteration after production evidence exists.

#### Hyperparameter rules

- Record run 0 as the unchanged baseline.
- Give each tunable value a type, range, scale, default, and rationale.
- Change the smallest coherent set needed to test one tuning hypothesis.
- Record every configuration, including failed and pruned runs.
- Resume only when architecture, state shapes, optimizer state, and data semantics
  remain compatible.
- Never use held-out metrics to direct tuning.
- Stop on budget, repeated non-improvement, invalid gradients, or leakage.

#### Critical failures retained in the root route

- differentiating hidden/runtime state;
- state leakage across independent samples;
- target or temporal-axis misalignment;
- held-out leakage;
- best-seed reporting instead of declared aggregation;
- successful loss reduction mistaken for biological validity.

### `references/parameter-fitting-workflow.md` plan

#### Purpose

Compose the inverse problem with a BrainX-native simulator, BrainUnit parameter
and observation quantities, BrainState lifecycle and transformations, and the
routed model package. Do not write a generic fitting guide, duplicate package
APIs, or turn the root skill into an inference catalog.

#### Method selection

| Condition | Preferred family | Required validation |
|---|---|---|
| Differentiable simulator and objective | BrainState/BrainX gradient optimization | Gradient checks, multiple starts, recovery, held-out protocols. |
| Nondifferentiable objective or simulator branch | Bounded derivative-free/search optimization | Multiple starts, budget trace, sensitivity/loss landscape, recovery. |
| Intractable likelihood and required uncertainty | Simulation-based inference through an explicit external boundary | Prior predictive, recovery/calibration, posterior predictive checks. |
| Tractable likelihood | Likelihood-based estimation | Likelihood tests, optimizer/sampler diagnostics, predictive checks. |

#### Required contents

1. Classify fitted, fixed, nuisance, hierarchical, and observation parameters.
2. Preserve units and apply explicit constraint transforms.
3. Record researcher-provided bounds or priors without inventing precision.
4. Implement the observation model from simulator output to measured data.
5. Define objective, likelihood, or distance with weights and reduction axes.
6. Test nominal, boundary, and invalid parameter values.
7. Run sensitivity or landscape diagnostics.
8. Perform parameter recovery using the exact real-data fitting pipeline and
   realistic duration, sampling, noise, missingness, and trial count.
9. Check scale-aware error, bias, tradeoffs, boundary hits, coverage/calibration
   when applicable, and failure by parameter region.
10. Fit observed data only after recovery passes or the researcher accepts a
    documented interpretation limit.
11. Run held-out or posterior predictive checks and report uncertainty.

#### Interpretation gate

Classify each parameter as `interpretable`, `weakly-identified`, or
`non-identifiable-under-this-protocol`. Good aggregate fit with poor recovery may
permit prediction but not strong mechanistic interpretation.

#### External inference boundary

Keep the simulator BrainX-native. Convert units and stateful outputs only at a
declared external-library boundary. Test target units, parameter order, output
shape, batch semantics, and round-trip behavior.

#### Critical failures retained in the root route

- fitting latent state directly to measured data without an observation model;
- trusting one optimizer start;
- interpreting boundary-hitting or non-identifiable parameters;
- recovery with more or cleaner data than the real protocol;
- changing bounds or summaries after seeing the observed result;
- stripping units at the inference boundary.

## 7. Acceleration

Invoke `brainx-acceleration-audit` only when profiling or expected production
cost shows a material need. The audit owns the representative baseline execution
and the parity proof; the modeling loop does not run a separate pre-acceleration
correctness stage.

Require the audit to consume the active package route and preserve the BrainX
model's state, unit, event, solver, randomness, and transformation semantics. Do
not treat generic JAX speedup as sufficient evidence.

Use this contract:

1. Define a representative baseline workload and the scientifically relevant
   outputs, metrics, gradients, stochastic summaries, and parity tolerances.
2. Profile before changing code.
3. Record baseline outputs, compile time, runtime, peak memory, precision, device,
   seeds, and environment.
4. Apply one coherent optimization at a time.
5. Execute the same workload and compare the declared outputs under the locked
   tolerances.
6. Separate compile time from steady-state runtime.
7. Reject changes that fail parity.
8. Treat any scientific-behavior change as implementation revision rather than
   acceleration.

The current repository plans `brainx-acceleration-audit` in `plan.md` but does not
ship it in `manifest.json`. Until implemented or restored, mark acceleration
`skipped` or `blocked`; never assume it passed.

## 8. Experiment execution

### `brainx-experiment-runner` boundary

Keep `brainx-experiment-runner` as a focused skill because process execution,
checkpointing, environment handling, and artifact collection vary independently
from the scientific loop. It must not contain a scientific iteration or result
acceptance policy.

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

Keep the deterministic assessment and the complete condensed Codex neuroscience
quality-review contract together in `brainx-modeling-loop/SKILL.md`. They are one
end-of-iteration stage with one output artifact. Do not create a
`brainx-result-review` skill, reviewer skill, review prompt reference, or second
copy of either contract.

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
| `spec-revision-required` | The useful next action crosses a locked field | Return to researcher after Codex confirms the boundary. |

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
- Gate:
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
Mark the artifact `accepted` only on `ACCEPT`; otherwise close it as `done` and
apply the returned transition.

### Review timing and packet

Call the future Codex MCP review-agent tool exactly once after deterministic
evidence synthesis in each modeling iteration. Provide paths for:

- locked `NeuroSpecification.md`;
- relevant model and experiment code;
- relevant tests and optional acceleration parity evidence;
- optional training/fitting plan and diagnostics;
- production run specs, raw metrics, logs, and artifact manifest;
- the deterministic sections of `ITERATION_REVIEW.md`;
- prior open finding IDs and changes made since the prior iteration.

Do not call Codex before evidence synthesis is complete unless the loop is
blocked and the researcher explicitly requests diagnostic review.

### Contract

```text
Act as an independent computational-neuroscience quality reviewer.
Do not edit files. Do not search literature. Read the supplied specification,
code, tests, raw run artifacts, and deterministic result assessment.

Question: Is this completed iteration good enough to accept its proposed
scientific outcome under NeuroSpecification.md?

Check:
1. model equations/components and code match the declared biological abstraction;
2. units, signs, axes, shapes, state lifecycle, reset, update order, delays, dt,
   solver, initial conditions, and noise are correct;
3. inputs, perturbation dose, controls, and observation mapping implement the
   declared protocol;
4. training parameter selection, target alignment, splits, reset, gradients,
   checkpoint selection, seed aggregation, and held-out isolation are valid when
   training is active;
5. fitting parameter order/units, bounds or priors, objective, recovery,
   identifiability, uncertainty, and predictive checks are valid when fitting is
   active;
6. acceleration preserves scientifically relevant behavior;
7. reported metrics and exclusions match raw artifacts and the locked rules;
8. controls, uncertainty, reproducibility, and provenance are sufficient for the
   proposed result category;
9. the claim-evidence matrix supports the exact claim scope and does not hide
   alternative explanations or unfavorable runs;
10. the proposed next action stays within the locked specification.

Good enough means valid and sufficient for the scoped outcome. It does not mean
positive, novel, or publishable. A rigorous refutation or bounded inconclusive
result may be accepted.

Return exactly:
GATE: ACCEPT | REVISE | SPEC_REVISION_REQUIRED | BLOCK
SCIENTIFIC_OUTCOME: SUPPORTED | PARTIALLY_SUPPORTED | REFUTED | INCONCLUSIVE | INVALID
GOOD_ENOUGH_REASON: one concise evidence-based explanation
FINDINGS:
- ID: stable ID
  SEVERITY: critical | major | minor
  LOCATION: file and line or artifact path
  PROBLEM: concrete defect
  SCIENTIFIC_CONSEQUENCE: how it affects validity or interpretation
  MINIMUM_FIX: smallest sufficient correction or experiment
UNVERIFIED_ASSUMPTIONS:
- assumption and why available artifacts cannot resolve it
NEXT_ACTION: ACCEPT_AND_CLOSE | REVISE_IN_SPEC | ASK_RESEARCHER | STOP
```

Do not request or use a numeric quality score.

### Gate behavior

| Gate | Loop action |
|---|---|
| `ACCEPT` | Accept the Codex scientific outcome at its exact scope, update memory, and visualize or close. |
| `REVISE` | Start a new iteration at the earliest affected stage; preserve the locked specification and prior evidence. |
| `SPEC_REVISION_REQUIRED` | Stop automatic work, propose a versioned spec change, and ask the researcher. |
| `BLOCK` | Stop and report the unresolved scientific or capability boundary. |

### Review invocation and finding lifecycle

1. Reserve one review ID for the iteration and record the request before invoking
   the MCP tool.
2. Save the full raw response immediately in `reviews/<review_id>_raw.md`.
3. On resume, parse an existing raw response; never issue a second call for the
   same iteration.
4. Parse gate, outcome, findings, assumptions, and next action into
   `ITERATION_REVIEW.md` without rewriting reviewer language.
5. Give findings stable IDs across iterations.
6. For `REVISE`, record the code/test/run evidence expected to close each required
   finding.
7. On the next iteration, pass prior findings and actual changed artifacts.
8. Close a finding only when Codex accepts its evidence.
9. Preserve every prior raw review.
10. If the MCP tool is unavailable, mark the gate `blocked`; do not silently
   self-review and call it independent.

### Iterative review rule

Use one Codex review per completed iteration. The review itself does not debate or
spawn another review loop. A failed gate causes the modeling loop to revise,
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
- the Codex gate returns `ACCEPT` for that category;
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
| `brainmodeling-memory.md` | Modeling loop | Curated only from accepted or durable evidence. |
| `.brainx-loop/state.json` | Modeling loop | Atomic machine-readable transitions. |
| `events.jsonl` and `decisions.jsonl` | Modeling loop | Append-only; corrections supersede. |
| `ITERATION.md` | Modeling loop | Immutable after the integrated review stage. |
| Run specs/config/provenance/logs/metrics | Experiment runner | Immutable after launch/completion. |
| `ITERATION_REVIEW.md` | Modeling loop | `running` through evidence synthesis and Codex review; `accepted` only on `ACCEPT`. |
| Raw Codex review | Modeling loop | Store verbatim and never rewrite. |
| Parsed gate and findings | Modeling loop | Append to `ITERATION_REVIEW.md`; preserve stable finding IDs. |
| `FIGURE_MANIFEST.md` | Visualization | Update when figure or source evidence changes. |

### Persistent memory format

```markdown
# Brain modeling memory

## Current state
- Active specification and hash:
- Current iteration and stage:
- Next obligation:
- Remaining budgets:

## Locked decisions
| ID | Decision | Rationale | Specification | Evidence |
|---|---|---|---|---|

## Established findings
| Finding | Scope | Evidence runs/reviews | Confidence | Superseded by |
|---|---|---|---|---|

## Failed or partial attempts
| Attempt | Conditions | Outcome | Reusable lesson | Evidence |
|---|---|---|---|---|

## Open Codex findings and assumptions
| ID | Issue | Required fix/discriminator | Source review | Status |
|---|---|---|---|---|

## Accepted artifacts
| Artifact | Purpose | Specification | Provenance |
|---|---|---|---|

## Last checkpoint
- Completed action:
- Safe resume point:
```

Persist a conclusion only after Codex `ACCEPT`. Persist a failure when it prevents
repetition or changes a future choice. Do not store raw logs, transient tool
failures, unsupported guesses, or metrics without run IDs.

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
|   |   |-- persistence-and-resume.md
|   |   `-- visualization-routing.md
|   `-- scripts/
|       |-- loop_state.py
|       `-- artifact_lint.py
|-- brainx-experiment-runner/
|   |-- SKILL.md
|   `-- references/
|       |-- run-provenance.md
|       |-- monitoring-and-recovery.md
|       `-- checkpoint-compatibility.md
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
`-- brainx-acceleration-audit/       # planned in plan.md; implement or restore
```

Do not add separate skills for the experiment loop, specification, training,
fitting, result review, or neuroscience review. Do not duplicate the inline
specification or integrated review contract in references.

The root `SKILL.md` is intentionally larger because it owns the three essential
contracts. Keep optional variants and API detail in routed references.

## 14. Implementation roadmap

### Phase 0: Freeze the single-loop contract

Deliver:

- final root workflow and legal transitions;
- final inline specification format;
- final integrated result-review and Codex quality-gate contract;
- training/fitting reference boundaries;
- machine state and append-only event/decision schemas;
- three end-to-end fixture projects.

Exit when every artifact has one authoritative writer, every iteration has one
Codex gate, and a negative result can close successfully.

### Phase 1: Build a forward-simulation vertical slice

Implement:

1. `brainx-modeling-loop` with specification, package routing, implementation,
   integrated result/Codex review, iteration, memory, and resume;
2. `brainx-experiment-runner` for local immutable runs;
3. `brainx-visualization` for diagnostic and final traces.

Use spike-frequency adaptation, neural compass, or binocular rivalry. Simulate
Codex MCP responses through fixtures until the real review-agent tool exists.

Exit when `REVISE` produces a new full iteration and the next Codex call can close
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

Write `references/parameter-fitting-workflow.md` with one differentiable
trace-fitting case. Add a likelihood-free branch only when it demonstrates a
necessary distinct workflow. Keep the simulator, parameter quantities, state
lifecycle, and observation mapping BrainX-native on both branches.

Exit when fitted parameters cannot be interpreted before recovery and
non-identifiability returns a bounded conclusion.

### Phase 4: Integrate acceleration and real Codex MCP review

Implement or restore `brainx-acceleration-audit`. Connect the real Codex MCP
review-agent call to the root skill's inline contract. Preserve raw response,
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
| Single-cell perturbation failure | Valid negative/inconclusive result and `ACCEPT` without positive-result pressure. |
| Interrupted run | Stage state, checkpoint compatibility, immutable run identity, and resume. |
| Codex MCP unavailable | Quality gate remains blocked without losing completed work. |
| Specification drift attempt | Gate returns `SPEC_REVISION_REQUIRED`; loop creates a version and asks the researcher. |

### Structural tests

- Only `brainx-modeling-loop` owns scientific iteration.
- The specification schema and integrated result/Codex contract each have exactly
  one copy in the root `SKILL.md`.
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
7. Verify `REVISE` starts a full new iteration and carries stable finding IDs.
8. Verify held-out data cannot direct training tuning.
9. Verify poor recovery blocks mechanistic fitting interpretation.
10. Reject training, fitting, acceleration, execution, and visualization fixtures
    that bypass routed BrainX state, unit, or model semantics.
11. Verify a valid refutation or bounded inconclusive result can receive `ACCEPT`.
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
- From simulation-based inference guidance, adopt predictive checks, calibration,
  explicit simulator boundaries, and uncertainty reporting when selected.

### Do not adopt

- literature search, novelty scoring, paper writing, venue scoring, or submission;
- a universal research-quality score;
- nested experiment or review loops;
- separate specification, training, fitting, result-review, or reviewer skills;
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
- [Simulation-based inference skill](https://github.com/smestern/sciagent/blob/main/docs/domains/computational-neuro/skills/sbi/SKILL.md)

## 17. Recommended first implementation

Build the forward-simulation vertical slice first:

```text
brainx-modeling-loop
  -> inline specification
  -> package routing and implementation
  -> optional acceleration
  -> experiment runner
  -> integrated result review + one Codex neuroscience quality gate
       -> ACCEPT or full revised iteration
  -> memory and visualization
```

This proves the single-loop contract before training and fitting add complexity.
Once it survives interruption, negative results, review revisions, and
specification changes, add the two optional references without changing the root
workflow.
