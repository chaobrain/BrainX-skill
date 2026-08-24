# BrainX task-training workflow

Use this reference when `NeuroSpecification.md` selects `task-training` or the
training phase of `hybrid`. It owns the experimental contract around task data,
BrainX State, optimization, monitoring, tuning, checkpoint selection, and
held-out evaluation. The routed package skills still own model equations,
construction, simulation, and package-specific APIs.

Use task training to optimize model or readout weights over many input-target
examples. Use `parameter-fitting-workflow.md` instead to estimate
scientific model parameters against observations. In `hybrid`, keep the two
objectives, trainable State collections, data partitions, checkpoints, and
reported evidence distinct.

## Underlying training principle

Training reuses the scientific forward model: reset runtime State at the
declared independence boundary, execute the same BrainX rollout used for
evaluation, reduce aligned outputs and targets to one scalar loss,
differentiate only the declared parameter State, and apply one optimizer update
at the declared cadence.

`ParamState` represents learned values that persist across batches. Hidden,
short-term, random, delay, eligibility, monitor, membrane, conductance, and
other runtime State represent the condition of one rollout; reset or preserve
each role according to the scientific protocol, never because an optimizer or
batch loop makes reset convenient.

Do not put a framework-neutral learner around a BrainX simulator callback. Keep
the forward model, trainable State, runtime State, reset lifecycle, temporal
execution, physical units, and represented biological scale BrainX-native.

## Lock the training contract

Declare the experiment before writing the train step. A later change to the
task, target, independence unit, data inclusion, primary metric, or accepted
claim changes the locked specification; do not treat it as hyperparameter
tuning.

| Contract field | Required decision |
|---|---|
| Samples | Identify the independence unit: trial, sequence, subject, animal, session, recording, or another declared unit. |
| Axes and units | Name batch, time, feature, neuron, region, and target axes; record every transpose, temporal alignment, reduction, and unit boundary. |
| Partitions | Split immutable sample IDs by the independence unit before fitting preprocessing; prevent subject, session, temporal-neighbor, augmentation, and preprocessing leakage. |
| State lifecycle | Declare which State resets at sample, batch, sequence, phase, training/evaluation, and seed boundaries and which learned State persists. |
| Trainables | Select only intended parameter State; enumerate every excluded runtime State and any frozen parameter State. |
| Objective | Lock loss inputs, target alignment, reduction axes, units, regularization, primary metric, metric direction, and chance or null baseline. |
| Update cadence | Declare whether one optimizer update follows a timestep, window, complete sequence, batch, or accumulated set of batches. |
| Selection | Lock validation metric, checkpoint rule, seed aggregation, search space, budget, minimum evidence, patience, and stop rule before tuning. |
| Held-out use | Reserve held-out data until configuration and checkpoint selection are complete; never use it to continue, stop, tune, or choose a seed. |

Record the contract as an iteration artifact:

```yaml
training_contract:
  independence_unit: subject
  axes:
    input: [batch, time, feature]
    target: [batch, class]
    rollout_input: [time, batch, feature]
  reset_boundary: independent_sequence
  trainable_state: declared ParamState collection only
  split_fit_scope: training subjects
  update_cadence: one update per minibatch of complete sequences
  primary_metric: validation loss
  metric_direction: minimize
  checkpoint_rule: minimum validation loss within each run
  config_selection_rule: minimum mean validation loss across declared seeds
  seed_aggregation: mean and declared uncertainty interval
  tuning_budget: 12 completed or invalidated configurations
  held_out_access: after all model and checkpoint selection
```

Replace the example values with the locked scientific design. Store exact
sample IDs, preprocessing state, seeds, code/data/environment identity, and
configuration outside this compact contract and point to those artifacts.

## Pass the mechanical gates

Do not start a production baseline until every gate passes on a small case.

| Gate | Required evidence |
|---|---|
| Data and leakage | Disjoint split IDs, training-only preprocessing fit, expected shapes/units, target range, class or target distribution, and explicit time alignment. |
| Forward parity | Training and evaluation call the same scientific rollout and observation path; only declared fit/evaluation behavior differs. |
| State reset | Replaying one sample from the same declared initial State reproduces its output; reordering independent samples does not change their predictions. |
| Gradient reach | Loss and every intended gradient are finite; expected trainables receive nonzero signal; excluded State is absent from the gradient and optimizer trees. |
| Update isolation | One update changes intended trainables and optimizer State only; it does not silently change frozen parameters, data transforms, or persistent scientific configuration. |
| Temporal reduction | A synthetic target at a known timestep changes only the intended scored output or window; masks exclude padding and ignored phases. |
| Tiny-case overfit | The unchanged pipeline can overfit a tiny batch or short sequence and move the declared metric in the expected direction. |
| Replay and compilation | Declared seeds replay exactly where promised, and repeated shape-compatible calls reuse the model graph without State-shape drift. |

Treat failure as diagnostic evidence. Fix data, State, alignment, gradient, or
objective mechanics before changing learning rate, optimizer, surrogate,
architecture, or model dynamics.

## Establish the baseline and run ledger

Run 0 is the unchanged declared baseline. Give every run an immutable identity
and append it to one ledger before interpreting results.

| Run field | Record |
|---|---|
| Identity | Run ID, parent or warm-start ancestor, status, start/end time, and artifact paths. |
| Provenance | Exact code revision, data/split manifest, preprocessing state, package/environment identity, device, seed, and command or entry point. |
| Configuration | Complete resolved model, training, optimizer, schedule, reset, batching, and stopping configuration; do not store only overrides. |
| Outcomes | Raw training/validation curves, gradient and State diagnostics, checkpoints, runtime/resource data, and the primary plus guardrail metrics. |
| Decision | `CONTINUE`, `WAIT`, `STOP_INVALID`, `STOP_RULE`, `STOP_BUDGET`, or `COMPLETE`, with the evidence and locked rule that justified it. |

Do not compare a tuned run only with another tuned run. Preserve run 0 even
when it fails, because its failure constrains the next hypothesis.

## Monitor process health and training quality

Check process liveness, device activity, checkpoint/log freshness, loss,
validation metric, learning rate, and gradient norm at the declared cadence. A
live process can train invalidly; unavailable metrics mean unknown, not failed.

| Evidence over the declared window | Decision | Action |
|---|---|---|
| Metrics are temporarily unreachable but the process is observable | `WAIT` | Retry observation; preserve the last evidence and make no quality claim. |
| Process exit, unexpected idle device, stale outputs, storage failure, or no checkpoint progress is confirmed | `STOP_INVALID` | Preserve logs and the last valid checkpoint; retry only the identical compatible run. |
| NaN/Inf, invalid gradients, State leakage, or target/reset misalignment appears | `STOP_INVALID` | Preserve the first failing step and diagnostics; return to implementation. |
| Training loss and the locked validation metric improve | `CONTINUE` | Continue; widen the check interval only after repeated healthy checks. |
| Metrics are flat, noisy, or slightly worse before minimum evidence or patience | `WAIT` | Keep or shorten the interval and gather the next planned checkpoint. |
| Loss diverges across the window, validation degradation exceeds patience, or gradients repeatedly explode or vanish | `STOP_RULE` | Stop under the locked rule and preserve the metric window and reason. |
| Maximum epochs or target criterion is reached with valid artifacts | `COMPLETE` | Close the run and compare it through the locked selection rule. |
| The tuning budget is exhausted | `STOP_BUDGET` | Launch no new configuration; compare every completed or invalidated run. |

Do not stop on one noisy point. For every stop, record the run ID, metric window,
checkpoint and log paths, exact rule, and reason. Monitoring ends or invalidates
a run; it never validates a scientific claim.

## Tune hyperparameters with bounded evidence

Use this bounded hyperparameter-tuning loop: `run -> analyze -> record ->
choose`. Do not launch tuning until run 0 and the mechanical gates establish
what needs improvement.

1. Diagnose the limiting signal from the complete run history: optimization,
   generalization, temporal credit assignment, gradient scale, State leakage,
   capacity, or compute/memory.
2. State one falsifiable tuning hypothesis and the validation result that would
   support or reject it.
3. Choose the smallest coherent parameter set that tests that hypothesis. Give
   each value a type, range, scale, default, and scientific or optimization
   rationale.
4. Create a complete new run record before execution. Change no unrecorded
   model, split, preprocessing, reset, metric, seed aggregation, or stop rule.
5. Analyze the primary validation metric with its guardrails, gradient/State
   diagnostics, resource cost, and variability. Append failed, stopped, and
   pruned runs to the same ledger.
6. Stop when the locked budget or patience is exhausted, the hypothesis is
   rejected, gradients or data become invalid, or no remaining candidate can
   answer the declared question.

Establish `Adam` or another routed canonical baseline before a specialized
optimizer. Do not change optimizer, scheduler, surrogate, temporal reduction,
reset policy, and architecture together. When parallel resources are available,
run independent configurations concurrently only if each has an allocated
budget entry, isolated BrainState/RNG/checkpoint artifacts, and the same locked
comparison contract.

## Resume and reuse checkpoints safely

Checkpoint compatibility is a State-graph and experiment-contract decision,
not a matching filename.

| Case | Rule |
|---|---|
| Resume an interrupted run | Restore model parameters, optimizer and scheduler State, RNG State, counters, preprocessing identity, and runtime configuration only when the complete configuration and State tree are identical. |
| Compare independent hyperparameter candidates | Start every candidate from the same declared initialization or base checkpoint. Do not give one candidate inherited training progress and compare it as an independent run. |
| Warm-start a later phase | Permit only when the plan declares a dependent phase and parameter structure, shapes, dtypes, units, task, targets, preprocessing, partitions, model dynamics, and runtime-State contract remain compatible. Record the ancestor checkpoint. |
| Change optimizer or schedule | Parameter-only restore may be valid, but do not restore incompatible optimizer or scheduler State; treat the run as a warm start, not an identical resume. |
| Change architecture, parameter shapes, solver, `dt`, surrogate, target semantics, preprocessing, data partitions, or State/reset meaning | Restart from the declared initialization. Do not coerce or partially load State to preserve progress. |

Save checkpoint metadata with the full State-tree signature and resolved
configuration. Validate a restored checkpoint with one deterministic evaluation
before training continues.

## Select without touching held-out data

Apply the locked checkpoint rule to validation evidence only. Confirm the
selected configuration across every declared seed or replication and report
seed-wise results plus the locked aggregate; never select or present the best
seed alone.

After every modeling, preprocessing, tuning, checkpoint, seed, and stop decision
is frozen, restore the selected checkpoint for each declared replica, reset
evaluation State, and evaluate the untouched held-out partition. Preserve raw
predictions, sample IDs, per-sample or per-group metrics, aggregation, and
uncertainty. Do not tune, choose another checkpoint, or suppress a seed after
seeing held-out results.

Successful optimization proves that the declared training pipeline can learn
the task under its tested conditions. It does not by itself establish
biological validity, mechanism recovery, identifiability, or generalization
beyond the locked held-out population.

## Return training evidence to the modeling loop

Return these artifacts through steps 2-5 of `brainx-modeling-loop`:

- the locked training contract, split IDs, and fitted preprocessing state;
- axis, unit, target-alignment, State-selection, reset, gradient, and tiny-case
  tests;
- run 0 and the immutable ledger for every completed, stopped, failed, or pruned
  run;
- raw training and validation curves plus process, gradient, State, runtime, and
  resource diagnostics;
- every checkpoint, its selection trace, State-tree signature, and compatibility
  decision;
- seed-wise and aggregate validation results;
- the untouched held-out predictions, metrics, aggregation, and uncertainty;
- unresolved training limitations and every claim this evidence does not
  support.

Report an out-of-contract request to change the task, target, independence unit,
data inclusion, primary metric, or accepted claim. Do not edit the locked
specification inside this coverage. Report implementation, optimizer, schedule,
reset, checkpoint, or bounded-tuning defects as exact in-contract corrections.
Step 5 still emits only `REFUSE` or `PASS` and owns the loop transition.

## Common failures

- Random row splitting when subjects, sessions, sequences, or neighboring time
  windows are the independence unit.
- Learned preprocessing fitted outside the training partition.
- BrainCell scientific-parameter estimation mislabeled as task training.
- Hidden or runtime State included in the optimizer collection.
- Dynamical State carried across independent samples or reset across a boundary
  that should preserve it.
- Raw `jax.grad`, `jax.jit`, or a non-BrainX trainer wrapped around
  State-mutating BrainX code.
- Target indices shifted relative to the temporal output being scored, or padded
  timesteps included in the loss.
- Optimizer updates placed inside a temporal loop when the contract specifies
  one update per complete sequence or minibatch.
- Tuning started before finite gradients and tiny-case overfitting pass.
- Multiple optimizer, surrogate, architecture, reset, and reduction decisions
  changed in one run without a testable hypothesis.
- Checkpoint selection, early stopping, seed choice, or tuning influenced by
  held-out metrics.
- A warm-started run compared as if it were an independent candidate.
- Best-seed reporting, silent failed-run exclusion, or aggregation changed after
  results.
- BrainTrace activated for generic speed rather than a demonstrated temporal
  memory bottleneck and an accepted estimator boundary.

## Package and reference routing

Read every active modeling skill before implementing training, then open only
the smallest routed reference that owns the next decision. The package skills
own model construction and train-step mechanics; this reference owns the
training contract, gates, experiment discipline, and evidence.

| Need | Required route |
|---|---|
| Large temporal workloads limited by sequence-length-dependent BPTT activation memory | First open the active package's checkpointed-control-flow reference. When ordinary BPTT remains memory-limited and an online eligibility-trace estimator is scientifically acceptable, open `skills/package-skills/braintrace/SKILL.md` and use BrainTrace as the memory-efficient alternative. Its memory is constant only in sequence length and still scales with batch size, participating parameters, hidden dimensions, and estimator choice. |

### Choose Braintools training components

| Decision | BrainPy-State route | BrainMass route |
|---|---|---|
| Loss or neuroscience metric | `skills/package-skills/brainpy-state/references/braintools/metric.md` | `skills/package-skills/brainmass/references/braintools/metric.md` |
| Optimizer, scheduler, clipping, or update lifecycle | `skills/package-skills/brainpy-state/references/braintools/optimizer.md` | `skills/package-skills/brainmass/references/braintools/optimizer.md` |
| Parameter initialization | `skills/package-skills/brainpy-state/references/braintools/parameter-initializer.md` | `skills/package-skills/brainmass/references/braintools/parameter-initializer.md` |
| Input encoding | `skills/package-skills/brainpy-state/references/braintools/data-preprocessing.md` | `skills/package-skills/brainmass/references/braintools/data-preprocessing.md` |
| Hard-spike backward derivative | `skills/package-skills/brainpy-state/references/braintools/surrogate.md` | `skills/package-skills/brainmass/references/braintools/surrogate.md` |
| Phase-structured cognitive trials | Open the shared `skills/package-skills/brainmass/references/braintools/cogtask.md`, then preserve the BrainPy-State model's time-major input contract. | `skills/package-skills/brainmass/references/braintools/cogtask.md` |
