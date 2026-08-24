# BrainX diagnosis: learning temporal order

## Evidence studied

Generated evidence:

- `prompt.md`, including its explicit package and transform requirements.
- `run0/temporal_order_learning.py`, `test_temporal_order_learning.py`, and
  `README.md` line by line.
- `run0/codex-events.jsonl`, `codex-stderr.log`, `agent-final.md`, and
  `harness-metadata.txt`, including the agent's failed and corrected attempts.
- `run0/temporal_order_relearning.png`, opened at its native 1980 by 1350
  resolution and checked for readable labels, clipping, overlap, and whether the
  plotted observables support the captions.
- The final entry point and both `unittest` tests, executed unchanged with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`.

Execution evidence:

```text
Teacher-free output spikes [A-first, B-first]
  after A->B training:
    A then B: [1, 0]
    B then A: [0, 0]
  after B->A training:
    A then B: [1, 0]
    B then A: [0, 1]
```

Both final tests pass. Reviewer probes also ran each learned condition from
fresh dynamical State and reproduced the recorded contiguous-probe counts. An
impulse check located the intended AB delayed/direct coincidence at step 30.
The current silent interval therefore prevents numerical carryover in this
parameterization, but the implementation does not establish independence by
State lifecycle.

Review standard:

- `skills/brainx-general-guard/SKILL.md`.
- `skills/brainpy-state/SKILL.md`, `references/component-selection.md`,
  `references/projection-patterns.md`, and
  `references/brain-dynamics-delay-protocol.md`.
- `skills/brainevent/SKILL.md`, `references/connectivity-variants.md`,
  `references/sparse-formats.md`, and `references/synaptic-plasticity.md`.
- `skills/brainstate/SKILL.md`,
  `references/brainstate/brainstate-control-flow-patterns.md`,
  `references/brainstate/transformation-vmap-expansion.md`,
  `references/collective_model_operations.md`, and
  `references/simulation-environment.md`.
- `skills/brainunit/SKILL.md`, `references/array-creation.md`,
  `references/array-mechanics.md`, and
  `references/quantity-inspection-and-conversion.md`.

Relevant local examples:

- `skills/brainpy-state/references/scripts/109_fast_global_oscillation.py` for
  manual `Delay` retrieval, point-neuron State, and transformed time execution.
- `skills/brainpy-state/references/scripts/103_COBA_2005.py` and
  `107_gamma_oscillation_1996.py` for `LIFRef`, explicit voltage initialization,
  network initialization, and `for_loop` monitoring.
- `skills/brainpy-state/references/scripts/training-snn.py` and
  `201_surrogate_grad_lif_fashion_mnist.py` for batched initialization,
  independent sequence State, and transformed rollouts.
- `skills/brainevent/references/scripts/coba_ei_teaching.py` for
  `BinaryArray @ connectivity` inside a BrainPy-State module and compiled loop.
- `skills/brainstate/scripts/lif_neuron_model.py` and `integrator_rnn.py` for
  State-owned recurrent dynamics and complete-loop transformation.

Authoritative pages checked:

- `brainpy.state.LIFRef` generated API, especially the separate `V_rest` and
  `V_initializer` parameters.
- BrainState generated APIs for `Delay`, `Delay.update`,
  `Delay.retrieve_at_step`, `for_loop`, `vmap`, `init_all_states`, and
  `reset_all_states`, plus the official Delay Protocol and Vectorization
  tutorial.
- BrainEvent Event Array, Matrix Operations, and Synaptic Plasticity pages for
  `BinaryArray`, dense event products, and
  `update_dense_on_binary_pre`.
- BrainUnit generated quantity conversion and unit-aware math pages used by the
  time axis and trace decay.

The globally resolved Run 0 BrainX skill directories were verified
byte-for-byte against this repository. The baseline is therefore valid. The
separate `pre-refinement-isolated-control/` result used the same skill snapshot
and is control evidence, not a refinement checkpoint.

## Executive diagnosis

Run 0 is runnable and its core delayed-coincidence mechanism is scientifically
interpretable. It uses unit-bearing physical parameters, correct BrainPy-State
neurons, BrainState State and transforms, and the appropriate dense BrainEvent
communication and plasticity operators. Teacher-free probes confirm that AB
learns the A-first output and BA later learns the B-first output.

The main defect is trial semantics. `vmap` batches only pure stimulus
construction and idealized offline scoring; all stateful trials are flattened
into one continuous `for_loop`. Neuron, delay, and trace State therefore persist
between nominally independent trials. The chosen silence makes this run
numerically stable, but independence is accidental rather than encoded in the
workflow.

The scientific claim is also stronger than the evidence. Phase 2 learns a new,
orthogonal BA feature pair while retaining the AB association. That is valid
opposite-order acquisition, but it is not overwriting a previous preference.
The plotted acquisition curve is computed from ideal feature vectors and
weights, not from teacher-free circuit responses.

Finally, the transcript exposes two preventable skill gaps even though the
final source corrected them: default LIF voltage initialization caused
immediate spikes until `V_initializer` was supplied, and manual delay retrieval
required a `DELAY_STEPS - 1` compensation because retrieval occurred before
insertion. The root and routed references do not state these decisions sharply
enough.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `temporal_order_learning.py:284-304` | Training and probe trials are concatenated into one continuous stateful trajectory; only the teaching trace is reset at trial starts. | Neuron, refractory, and delay State can contaminate the next trial when timing or parameters change. The current result depends on a sufficiently long silent interval. | Make sequence boundaries explicit. Reset or reinitialize dynamical State for every independent trial while preserving learned `LongTermState` weights. Batch only trials whose State and weight updates are independent. |
| P1 | `temporal_order_learning.py:323-325`, `346-369` | The acquisition plot uses ideal event features multiplied by weights, not actual circuit responses. | The figure can report an improving preference even if membrane, threshold, refractory, or residual State prevents the output population from expressing it. | Plot teacher-free output responses or label the quantity strictly as an idealized weight-derived drive and verify it against actual probes. |
| P2 | `temporal_order_learning.py:284-343`, `368`, `425-428` | Phase 2 learns BA on feature rows disjoint from AB and retains the AB mapping. | “Preference adapts after reversal” suggests replacement, while the experiment demonstrates acquisition of a second association. | State the demonstrated claim as opposite-order acquisition with retention, or design a reversal protocol that changes the target for the same cue and measures forgetting plus reacquisition. |
| P2 | `temporal_order_learning.py:195-201` | Manual delay retrieval precedes insertion and compensates with `DELAY_STEPS - 1`. | The correct result depends on an implicit completed-step convention and is vulnerable to an off-by-one change. | Prefer update-then-retrieve with step `d`, where step 0 is the inserted current value, or document the alternate order beside the code and verify it with an impulse test. |
| P2 | `test_temporal_order_learning.py:33-43` | The end-to-end test does not reset probes independently, assert delay arrival, or assert all phase-2 behavior including AB retention. | It cannot distinguish true independent recognition from favorable carryover and does not lock down the actual learning claim. | Add an impulse-level delay check, fresh-State teacher-free probes, and assertions for both orders after each phase. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, current, voltage, resistance, and decay constants | BrainUnit quantities | BrainUnit quantities | Correct | Keep units through simulation and convert only for host output. |
| Exact conversion from durations to step counts | `Quantity.to_decimal(u.ms)` plus an integer-grid check | BrainUnit conversion plus host validation | Correct host boundary | Keep the explicit integer-grid failure because manual step retrieval requires it. |
| Tone and target construction | Pure JAX arrays | JAX pure-array boundary; BrainState transform is optional | Correct mechanics | Do not present this as batching the stateful trial. |
| Batch encoding | `brainstate.transform.vmap(_encode_one_trial)` | `brainstate.transform.vmap` | Correct pure mapping but semantically superficial | Reserve the trial-batching claim for a complete independent trial callable. |
| Result and input records | Frozen dataclasses | Python host boundary | Correct | Keep if they improve reporting; BrainX has no record-container API requirement. |
| Circuit composition | `brainstate.nn.Module` | BrainState `Module` graph | Correct | Keep neural, delay, trace, and plastic weight ownership on the module. |
| Sensory and output neurons | `brainpy.state.LIFRef` | BrainPy-State neuron dynamics | Correct | Keep explicit membrane and refractory parameters. |
| Initial membrane voltage | `braintools.init.Constant(V_rest)` | `V_initializer` plus BrainTools initializer | Correct final code and essential | Teach this choice in the root skill so agents do not discover it through spurious initial spikes. |
| Temporal memory | `brainstate.nn.Delay` | BrainState general delay protocol | Correct abstraction | State the update/retrieval order and test an impulse. A BrainPy projection delay is not a direct replacement because the model needs the delayed event as an explicit feature alongside the current event. |
| Persistent learned weights | `brainstate.LongTermState` | BrainState State roles | Correct | Preserve this State across independent trial resets. |
| Per-trial teaching trace | `brainstate.ShortTermState` | BrainState State roles | Correct role, incomplete lifecycle | Reset it together with all other dynamical trial State. |
| Event-driven dense communication | `brainevent.BinaryArray(features) @ weights` | BrainEvent binary event product | Correct | Dense storage is appropriate for the fixed 4 by 2 layer. |
| Online bounded plasticity | `update_dense_on_binary_pre` | BrainEvent dense plasticity | Correct | Keep weights in State and the update inside the transformed learning step. |
| Unit-aware trace decay | `u.math.exp(-dt / tau)` | BrainUnit math | Correct | Keep the exponent dimensionless. |
| Per-step environment | `brainstate.environ.context(t=time)` | BrainState environment | Correct | Keep `dt` outside and `t` inside the step. |
| Initial graph State allocation | `brainstate.nn.init_all_states(circuit)` | BrainState collective lifecycle | Correct once, insufficient for independent trials | Reinitialize or reset the intended dynamical states at sequence boundaries without discarding learned weights. |
| Stateful time execution | `brainstate.transform.for_loop(circuit.update, ...)` | BrainState `for_loop` | Correct | Keep timestep effects in State and the complete time loop transformed. |
| Compilation | `brainstate.transform.jit(run)` | BrainState `jit` | Correct | Keep the stable complete-loop boundary. |
| Sequential online learning trials | Flattened into the time loop | BrainState State plus sequential `for_loop` execution | Semantically valid dependency ordering | Do not `vmap` trials when trial N updates weights consumed by trial N+1. Make their boundary resets explicit. |
| Independent probe trials | Appended to the same flattened trajectory | BrainState batched State, `vmap`, and collective init/reset | Incomplete | Run probes from independent dynamical State with shared read-only learned weights, using native batch State or a state-aware mapped trial callable. |
| Trial-level score mapping | `vmap` over ideal features and per-trial weights | BrainState `vmap` on a pure function | Mechanically correct, weak scientific observable | Replace or corroborate with actual teacher-free response metrics. |
| Device-to-host result conversion | `np.asarray(...)` after execution | NumPy host boundary | Correct | Keep host conversion after the transformed run. |
| Trial indexing and report shaping | NumPy indexing and reshape | Host analysis boundary | Correct for this small report | Avoid recording every timestep of weight history when only trial endpoints are needed in larger runs. |
| Figure construction and saving | High-level `matplotlib.pyplot` | Host visualization boundary | Correct | Keep the readable 2 by 2 summary, but make observable labels match what was measured. |
| CLI and filesystem output | `argparse` and `pathlib.Path` | Python host boundary | Correct | No BrainX replacement applies. |
| Focused tests | Standard-library `unittest` | Python test boundary | Correct runner | Expand scientific invariants rather than adding a test dependency. |
| Randomness | Deterministic stimulus and initialization | No random API needed | Correct | Do not add seeding to a deterministic experiment. |

## Missing, bypassed, or misused BrainX APIs

### Stateful `vmap` or native batched State

`vmap` should wrap a complete trial only when each mapped lane has independent
dynamical State and does not perform order-dependent shared weight writes.
Declare mapped State through the documented `in_states` and `out_states`
contract, or initialize a leading batch dimension with the collective
lifecycle APIs and run the batched module natively. Use this for teacher-free
AB/BA probes with shared read-only learned weights. Do not map the online
learning trials because later trials consume weights updated by earlier trials.

### `init_all_states`, `reset_all_states`, and vmapped lifecycle helpers

The code calls `init_all_states` only once. Independent trials require an
explicit lifecycle boundary for neuron, refractory, trace, and delay State.
Use collective initialization/reset when the participating modules implement
that lifecycle, and deliberately preserve the learned `LongTermState` weights.
Silence inside one continuous trajectory is a model input, not a State reset.

### Canonical `Delay.update` and retrieval ordering

The selected general `Delay` abstraction is appropriate, but the call order is
noncanonical. After `delay.update(current)` (or `delay(current)`),
`retrieve_at_step(0)` denotes the newly inserted current sample and step `d`
denotes the sample `d` completed updates earlier. Retrieving before insertion
changes the index required for the same physical latency. The skill should show
one order and require an impulse check rather than forcing agents to derive the
offset during debugging.

### Explicit `LIFRef.V_initializer`

`V_rest` defines the dynamics' resting voltage; it does not by itself promise
that initial membrane State equals that value. When startup activity matters,
pass `V_initializer=braintools.init.Constant(V_rest)` or an intentional
distribution. The final code does this correctly, but only after the transcript
records spurious initial spikes from the omitted initializer.

### Transform-native plasticity State

Run 0 correctly keeps weights and traces in BrainState State and calls the
BrainEvent update inside the transformed timestep. The current BrainEvent
plasticity reference instead teaches Python loops and ordinary attributes as
its canonical patterns. Revise that reference so online examples own plastic
weights and traces as State and lower the repeated update through
`brainstate.transform.for_loop` plus `jit`.

No BrainPy projection, BrainTrace estimator, BrainTools input generator, or
BrainX visualization API is missing. The custom explicit feature construction,
deterministic tone input, local online rule, and high-level Matplotlib report
are legitimate boundaries for this task.

## Performance and code simplicity

The complete 2,170-step baseline trajectory is lowered through one BrainState
`for_loop` and `jit`, avoiding Python timestep overhead. BrainEvent touches only
active feature rows during communication and plasticity. Dense storage is the
simplest representation for eight weights, and BrainUnit operations remain on
device until host reporting. These are good choices.

The `vmap` calls do not accelerate the expensive stateful work: they vectorize
small pure encoding and scoring functions. A genuine independent probe batch
would vectorize neural State and the full probe rollout. Sequential learning
should remain sequential because weights carry across trials.

The loop returns the entire weight matrix at every timestep, then keeps only
trial endpoints. That is harmless here but scales as time steps times weight
count. Return or collect trial-end weights at a coarser boundary when the
network grows. Host-side reshaping, small statistics, CLI handling, and plotting
are otherwise appropriate and simpler than forcing them into BrainX.

## Skill improvements

1. Refine `skills/brainx-general-guard/SKILL.md` to distinguish independent
   mapped trials from sequential learning trials whose shared State creates a
   dependency. State that input encoding alone does not satisfy a request to
   batch trials.
2. Refine the `LIFRef` row and canonical examples in
   `skills/brainpy-state/SKILL.md` to require an explicit `V_initializer` when
   startup voltage affects spikes, transients, or comparisons.
3. Extend
   `skills/brainpy-state/references/brain-dynamics-delay-protocol.md` with one
   authoritative manual `brainstate.nn.Delay` workflow. Define insertion before
   retrieval, step 0, the off-by-one consequence of reversing call order, and an
   impulse-level verification.
4. Replace the Python-loop canonical plasticity pattern in
   `skills/brainevent/references/synaptic-plasticity.md` with a compact
   transform-native Module whose weights and traces are BrainState State. Keep
   storage choice and exact update-operator routing in the reference.
5. Do not change `plan.md`: these edits sharpen existing ownership and routing;
   they do not change package scope or progressive-disclosure architecture.

## Checks for the next run

- Preserve the exact 666 prompt bytes and SHA-256
  `cadc2cd3bdf8be86e4744a271c5d3ff23f893985d8bf84afe3e3f363465883e5`.
- Install the refined repository snapshot before launching `run1`; do not count
  another same-snapshot sample as a numbered run.
- Confirm that the generated LIF populations specify an intentional
  `V_initializer` before runtime debugging.
- Confirm delay semantics with a direct impulse/feature timing assertion and no
  unexplained `d - 1` compensation.
- Keep online learning trials sequential when weights carry from one trial to
  the next; reset dynamical State at each independent trial boundary.
- Require `vmap` or native batched State to cover complete independent probe
  trials, not only stimulus construction or offline scoring.
- Compare fresh-State and batched teacher-free AB/BA probes after both phases.
- Make the report distinguish opposite-order acquisition with AB retention from
  true overwriting reversal, and derive any learning curve from actual circuit
  responses or label an idealized drive precisely.
- Run the unchanged entry point and focused tests with the BrainX virtualenv,
  inspect the generated figure, and compare correctness, implementation
  simplicity, transform scope, and debugging retries with Run 0.
