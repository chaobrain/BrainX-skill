---
name: brainx-modeling-loop
description: Use first for an end-to-end BrainX modeling project. Start fresh or resume from brainmodeling-memory.md, write a compact NeuroSpecification.md, study the relevant BrainX skills deeply, implement and accelerate the model, run experiments, send code and results to Codex through MCP, repeat from implementation when review refuses, and visualize only after review passes.
---

# BrainX modeling loop

## Get started

Read `brainmodeling-memory.md` first when it exists, then choose exactly one entry case.

| Entry case | Use when | Action |
|---|---|---|
| `fresh-new` | `brainmodeling-memory.md` does not exist and no prior loop work must be preserved. | Create the memory file, set the current position to step 0, and start the specification. |
| `resume` | `brainmodeling-memory.md` or prior loop artifacts exist. | Read the recorded step, verify its referenced artifacts, and continue from the first unfinished action in that step. |    


Do not create another entry case. Recover missing or inconsistent state inside `resume` by using existing artifacts to locate the earliest unfinished step. Do not repeat a completed experiment or Codex review merely because terminal context was lost.

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

For `fresh-new`, create the file with its title, start iteration 1 at step 0, and
append the first checkpoint after step 0 finishes or blocks. After every step,
including a blocked step, append a new `## Checkpoint` block. `Iteration` and `Step` are metadata identifying
the step whose artifacts and milestones the block records; `Artifacts` and
`Important milestones` are the checkpoint content.

Never overwrite an earlier checkpoint. Append corrections as a new checkpoint
with the same iteration and step and state which earlier record it supersedes.
For `resume`, read the checkpoints in order, verify the latest artifact pointers,
and continue from the first unfinished action after the latest valid checkpoint.

```text
fresh-new -> step 0 -> step 1 -> step 2 -> step 3 -> step 4 -> step 5
                              ^                          |
                              |-------- REFUSE ----------|

step 5 PASS -> step 6 -> complete
```

Each checkpoint contains the artifacts created or updated by that step and the
important milestones reached during it. Keep full code, logs, results, and
reviews in their owning artifacts and point to them from the checkpoint.

## Step 0: Write the specification

Turn the researcher's request and inspected data into a short `NeuroSpecification.md`. Ask the researcher only for missing decisions that would change the model, data interpretation, or acceptance boundary. Do not infer them.

Use only this core structure:

```markdown
# NeuroSpecification

- Status: draft | locked
- Researcher approval:

## Researcher request
- Brain-modeling question or behavior:
- Requested model, experiment, or comparison:
- Execution mode: forward-simulation | task-training | parameter-fitting | hybrid
- Required outputs:
- Constraints:

## Inspected data contract
- Data sources and inspected contents:
- Shapes, axes, sampling/time base, and physical units:
- Required preprocessing and the subset used to fit each transform:
- Mapping from data to model inputs, targets, and observables:
- Known data limitations or unresolved mismatches:

## Acceptance boundary
- Evidence required for success, failure, or an inconclusive result:
- Required baselines and controls:
- Invalid-result conditions:
- Allowed claims and explicit non-claims:
```

Inspect supplied data before completing its contract. Keep raw data read-only, preserve units and axis meaning, and prevent preprocessing leakage. Lock the specification with researcher approval before step 1.

**Step result:** a locked `NeuroSpecification.md` and a memory checkpoint pointing to step 1.

## Step 1: Study the relevant BrainX modeling skills

Study before implementing. Understand the selected BrainX abstractions, exact APIs, execution lifecycle, invariants, and canonical scripts deeply enough to design the model without guessing.

1. Open `brainx-general-guard` and identify every biological scale explicitly represented by `NeuroSpecification.md`.
2. Read each selected modeling skill completely:
   - `braincell` for ions, channels, compartments, or morphology;
   - `brainpy-state` for point neurons, synapses, or spiking networks;
   - `brainmass` for aggregate populations, regions, or whole-brain dynamics.
3. Open supporting package skills only when the model requires them:
   - `brainunit` for physical quantities and unit-aware boundaries;
   - `brainstate` for State, Modules, environments, initialization, randomness, and transforms;
   - `brainevent` for event data, connectivity, and event-driven plasticity;
   - `braintrace` only for temporal training limited by BPTT memory.
4. Follow every selected skill's routing instructions. Read all references and canonical scripts likely to affect this model.
5. Trace construction, initialization, State and data flow, update order, execution, outputs, and validation through the relevant examples.
6. Verify unfamiliar API names and signatures in the owning skill or its routed documentation. Do not inspect installed package source for modeling knowledge.
7. Save the study record as an artifact and record its completion and selected training/fitting coverage as important milestones.

Do not write the implementation during step 1. Finish the study record first.

**Step result:** a complete BrainX study record and an implementation design grounded in the selected skills, then a memory checkpoint pointing to step 2.

## Optional training and fitting coverage for steps 2-5

Training and fitting extend the same loop; they do not create separate stages or entry cases.

| Specification mode | Coverage for steps 2-5 |
|---|---|
| `forward-simulation` | Open neither optional reference. |
| `task-training` | Open `references/training-workflow.md` at the start of step 2 and keep its requirements active through implementation, acceleration, experiment execution, and Codex review. |
| `parameter-fitting` | Open `references/parameter-fitting-workflow.md` at the start of step 2 and keep its requirements active through implementation, acceleration, experiment execution, and Codex review. |
| `hybrid` | Open `references/training-workflow.md` and `references/parameter-fitting-workflow.md` at step 2, then keep their objectives, State lifecycles, validation evidence, and review coverage distinct through step 5. |

Record the active coverage as an important milestone. A step is incomplete until its modeling requirements and every active training/fitting requirement are satisfied.

## Step 2: Implement the model and experiment code

Implement from the step-1 study record and the locked specification. Open the selected training or parameter-fitting reference before coding and apply it throughout this step.

1. Build the model through the selected BrainX package abstractions and highest-level owning APIs.
2. Implement required data processing, initialization, inputs, interventions, controls, observation mapping, training or fitting workflow, metrics, and artifact outputs.
3. Preserve BrainUnit quantities, BrainState lifecycle, package update order, randomness, and represented biological scale.
4. Add focused checks for equations or components, shapes, units, State initialization/reset, update order, inputs, controls, observables, and metrics.
5. Run only small implementation checks needed to prove the code is ready for acceleration and the experiment runner. Leave production execution to step 4.
6. Record the code and test artifacts plus any important implementation milestone.

When step 5 returns `REFUSE`, begin a new iteration here. Read every reviewer finding, preserve the locked specification, make the smallest sufficient implementation or experiment-design correction, update memory, and then repeat steps 3, 4, and 5.

**Step result:** BrainX-native model and experiment code with focused checks, then a memory checkpoint pointing to step 3.

## Step 3: Accelerate the code

Open `brainx-acceleration` and study the parts relevant to the implemented workload. Apply its workflow while keeping any active training/fitting coverage in force.

1. Establish the representative baseline and relevant outputs before changing performance code.
2. Apply only acceleration changes supported by the acceleration skill.
3. Compare model outputs, State, units, randomness, and numerical tolerances against the baseline.
4. When training is active, also preserve losses, gradients, parameter updates, reset behavior, and checkpoint meaning. When fitting is active, preserve objectives, parameter order/units, recovery behavior, and inference outputs.
5. Reject an optimization that changes the scientific result.
6. Record the acceleration and parity artifacts plus the important outcome milestone.

**Step result:** accelerated code with parity evidence, or an explicit unchanged decision, then a memory checkpoint pointing to step 4.

## Step 4: Run the experiment

Hand the locked specification, implementation, acceleration evidence, run configuration, seeds, expected outputs, artifact locations, and active training/fitting requirements to `references/run-experiment.md`. Follow it through target-device smoke and immutable production or replication launch, then open `references/monitor-experiment.md` for monitoring and artifact collection.

Step 4 must eventually return inspectable run artifacts: the exact configuration, code/data/environment identity, logs, raw results, metrics, training/fitting evidence when active, and artifact paths. Process completion does not imply review acceptance.

**Step result:** completed experiment artifacts and a memory checkpoint pointing to step 5.

## Step 5: Ask Codex to review code and results

Use the configured BrainX Codex MCP server. Its
`mcp-servers/codex/system-prompt.md` is the authoritative review contract and is
injected as `base-instructions` into every fresh `mcp__codex__codex` call. Do not
duplicate that contract in this skill or paste it into the review request.

Start one fresh `mcp__codex__codex` call for each completed iteration. Send the
locked specification, step-1 study record, implementation and tests, acceleration
evidence, experiment configuration, raw run artifacts, metrics, deterministic
result assessment, claim-evidence matrix, proposed next action, and all evidence
required by active training/fitting coverage. Preserve the returned raw response
and `threadId` as review artifacts. Treat the returned `content` as the reviewer
response in the calling agent's current turn: classify it immediately, update
the loop checkpoint, and report or act on the verdict without asking the
researcher to open another context. Preserve `threadId` for reviewer follow-up;
it is not needed for the calling agent to read the initial response.

Before calling, verify that every listed path exists. Set the MCP call's working
directory to the project root when the tool exposes a working-directory argument,
then call `mcp__codex__codex` with a `prompt` in this form:

```text
Review completed BrainX iteration <N>.

PROJECT_ROOT: <absolute project-root path>
ACTIVE_COVERAGE: none | training | fitting | training+fitting

SPECIFICATION:
- <path>/NeuroSpecification.md

LOOP_MEMORY:
- <path>/brainmodeling-memory.md

BRAINX_STUDY_RECORD:
- <path to the step-1 study artifact>

IMPLEMENTATION:
- <path to each model, data-processing, and experiment source file>

TESTS:
- <path to each relevant test and its result artifact>

ACCELERATION_AND_PARITY:
- <path to baseline, parity, runtime, and memory evidence>

EXPERIMENT_RUN:
- <path to exact run configuration>
- <path to code, data, and environment provenance>
- <path to logs>
- <path to raw results>
- <path to metrics>
- <path to artifact manifest>

RESULT_ASSESSMENT:
- <path to deterministic assessment and claim-evidence matrix>

TRAINING_OR_FITTING_EVIDENCE:
- <path to every active coverage artifact, or none>

PRIOR_REVIEW_EVIDENCE:
- <path to prior raw reviews and addressed findings, or none>

Read every listed artifact and review this iteration using the injected reviewer
contract. Do not edit files. Return the required structured outcome.
```

List concrete files, not only directories. Keep large data in place and provide
its manifest, provenance, and the exact result files needed for verification.
Do not omit an unfavorable, failed, or excluded run when it affects the proposed
scientific outcome.

Do not start an iteration review with `mcp__codex__codex-reply`; only a fresh
`codex` call receives the injected system prompt. If the configured MCP server is
unavailable, record step 5 as blocked in `brainmodeling-memory.md` and preserve
all completed experiment artifacts.

Configure the Codex MCP registration with `tool_timeout_sec = 1800`, or another
explicit budget longer than the largest expected review. A host timeout may be
reported as `user cancelled MCP tool call`; when no `threadId` or response is
returned, diagnose the MCP budget before treating it as researcher cancellation.

Record the full Codex response and one of two outcomes:

| Outcome | Transition |
|---|---|
| `REFUSE` | Record every finding and required correction, increment the iteration, set the current step to 2, and repeat steps 2-5 with the same optional coverage. |
| `PASS` | Record the accepted code/result scope and set the current step to 6. |

Do not argue with a refusal inside step 5 and do not substitute self-review for the MCP Codex call. The next Codex call occurs only after the revised implementation has passed again through acceleration and experiment execution.

**Step result:** a preserved Codex response and either a return to step 2 or a memory checkpoint pointing to step 6.

## Step 6: Visualize the accepted result

Hand the review-passed specification, code, run artifacts, results, and accepted scope to the BrainX visualization workflow.

The planned route is `references/visualization-workflow.md`. That instruction will be authored later; do not embed or invent it here. Until it exists, preserve the accepted artifacts and record step 6 as blocked in `brainmodeling-memory.md`.

After visualization completes, record the figure paths as artifacts and loop completion as an important milestone, then report the accepted result and remaining limitations.

**Step result:** visualization linked to review-passed evidence and a completed memory checkpoint.
