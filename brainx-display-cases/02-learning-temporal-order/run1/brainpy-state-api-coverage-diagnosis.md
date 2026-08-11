# BrainX diagnosis: learning temporal order, Run 1

## Evidence studied

Generated evidence:

- `run1/temporal_order_learning.py`, `README.md`, `agent-final.md`,
  `codex-events.jsonl`, `codex-stderr.log`, and `harness-metadata.txt`.
- `run1/temporal_order_learning.png`, visually inspected at its native 1440 by
  1600 resolution.
- The exact Run 1 source executed unchanged from a separate verification
  directory with the required BrainX virtualenv.
- The agent's compile check, detector-selectivity assertion, smoke run, default
  run, and failed-then-corrected unit-boundary attempt from the event log.

Run 1 used the same 666 prompt bytes, prompt hash, model, reasoning effort, CLI
version, virtualenv, and isolated harness as the baseline. Its skill checkpoint
was the repository snapshot after the Run 0 diagnosis and first refinement.

Verification output:

```text
phase 1 A->B accuracy: 88.9%
phase 2 B->A accuracy: 88.9%
held-out mixed-order accuracy: 100.0%
final weights [AB->A, AB->B, BA->A, BA->B]:
[1.         0.05       0.04592895 1.        ]
```

The figure is readable and unclipped. It plots actual pre-feedback circuit
correctness, trial-end weights, and the final teacher-free mapped response
matrix. The detector-selectivity probe reports `[[1, 0], [0, 1]]` for one AB
and one BA trial.

The Run 0 review standard remains applicable: BrainX general guard,
BrainPy-State, BrainEvent, BrainState, BrainUnit, all routed references and
examples listed in the Run 0 diagnosis, and the official generated APIs for
the concrete calls used here. Run 1 additionally uses the filter-based
`vmap2` State-axis contract from
`skills/brainstate/references/brainstate/transformation-vmap-expansion.md`.

## Executive diagnosis

Run 1 materially improves on Run 0. It chooses `V_initializer` before the first
execution, stores plastic weights and traces in BrainState State, performs the
BrainEvent update inside the transformed timestep, keeps causally dependent
learning trials sequential, and maps complete independent evaluation
rollouts. The learning curve now measures circuit decisions before feedback,
and the captions correctly describe storage of both temporal orders instead of
implying that the first association was overwritten.

One lifecycle defect remains. The training trials are sequential only because
their learned weights depend on earlier trials, but the implementation also
carries membrane, refractory, tone-trace, eligibility, and post-trace State
across trial boundaries. Reviewer probes measured nonzero State at later trial
starts, including a tone-trace value of `0.3856`. The current silent interval
and delayed first onset decay that residue below the detector threshold before
it can create a false event, but trial independence still depends on timing.

The mapped evaluation is scientifically independent and correct. It repeats
the learned weights per lane and maps every State, however, rather than sharing
one read-only learned weight State while mapping only dynamical State. That is
behaviorally equivalent in this small nonplastic probe but uses more memory and
makes the stated shared-weight contract imprecise.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `temporal_order_learning.py:90-142`, `258-286` | Nominal training trials are flattened into one trajectory and do not reset nonpersistent State at trial starts. | Tone, eligibility, post-spike, membrane, and refractory history can alter the next trial. Correctness currently depends on the chosen silence and onset. | Preserve learned weights across trials but reset neural and trace State at each logical trial boundary unless intertrial carryover is an explicit part of the scientific protocol. |
| P2 | `temporal_order_learning.py:313-357` | Evaluation repeats the learned weights across the batch and maps all State. | Results are correct, but read-only weights are copies rather than genuinely shared State; memory scales with trials times weights. | Keep one unbatched read-only weight State and map only per-trial dynamical State, or state clearly that weights are replicated. |
| P2 | Run 1 folder and event log | No reusable focused test artifact was generated; the only scientific assertion is an inline detector probe in the transcript. | Delay/order selectivity, mapped independence, bounds, and final response can regress without a checked test entry point. | Add focused tests for order selectivity, trial-boundary lifecycle, bounded weights, and mapped teacher-free decisions. |
| P3 | `temporal_order_learning.py:1-12`, `400-426` | Phase 2 stores the orthogonal BA association while retaining AB. | This demonstrates opposite-order acquisition, not overwriting of one cue-target mapping. | Keep the current precise “stores each temporal order” wording; use a same-cue target reversal only when true unlearning is required. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical timing and electrical parameters | BrainUnit quantities | BrainUnit | Correct | Keep unit-bearing model parameters. |
| Duration-to-step conversion | `Quantity.to_decimal` plus host rounding | BrainUnit plus host validation | Correct for current grid-aligned values | Reject nonintegral `dt` ratios instead of silently rounding if parameters become configurable. |
| Trial schedule construction | NumPy host loop, then BrainUnit/JAX arrays | Host protocol boundary | Correct | The loop constructs inputs; it is not simulation execution. |
| Circuit graph | `brainstate.nn.Module` | BrainState Module | Correct | Keep ownership centralized. |
| Three point-neuron populations | `brainpy.state.LIFRef` | BrainPy-State | Correct | Keep explicit startup initialization. |
| Initial voltage | `V_initializer=braintools.init.Constant(V_REST)` | BrainPy-State plus BrainTools | Correct and improved from Run 0 | No further change. |
| Temporal-order mechanism | unit-aware tone schedule plus decaying `ShortTermState` trace | BrainState State and BrainUnit math | Correct alternative to a delay line | State that this is a trace-defined coincidence window, not an exact axonal delay. |
| Detector communication | boolean order event into LIF detector | BrainPy-State custom model behavior | Correct | No higher-level order-detector API applies. |
| Plastic readout storage | CSR data in `LongTermState` | BrainEvent CSR plus BrainState | Correct | Preserve across trial resets. |
| Event communication | `BinaryArray(detector_spike) @ CSR` | BrainEvent | Correct | No change. |
| Eligibility and post traces | `ShortTermState` with BrainUnit decay | BrainState plus BrainUnit | Correct roles, incomplete training lifecycle | Reset at logical independent trial starts. |
| Bounded online update | `update_csr_on_binary_pre` inside `update` | BrainEvent plasticity | Correct | Keep topology arrays fixed. |
| Unit-aware branch | final `u.math.where` | BrainUnit math | Correct final source | The first raw `jnp.where` failure was corrected without a new skill rule. |
| Per-step environment | `dt` outer context and `t` inner context | BrainState environment | Correct | No change. |
| Training execution | BrainState `jit` around `for_loop` | BrainState transforms | Correct causal ordering | Add trial-boundary resets without mapping dependent weight updates. |
| Trial-end weights | full timestep history reshaped to endpoints | BrainState monitor output plus host analysis | Correct but memory-heavy | Return or sample weights only at needed boundaries in larger models. |
| Actual pre-feedback decisions | response-window output spikes | BrainPy-State output plus host metric | Correct and improved from Run 0 | Keep. |
| Independent held-out trials | complete nested `for_loop` under `vmap2` | BrainState State-aware vectorization | Correct and improved from Run 0 | Map only dynamical State if learned weights are intended to be shared. |
| Batched initialization | explicit trace batch shape plus `init_all_states(..., batch_size=...)` | BrainState collective initialization | Correct | Avoid duplicating batch policy in multiple constructors when a single lifecycle method can own it. |
| Evaluation score | mapped teacher-free spike counts | BrainState/BrainPy-State output | Correct | Keep actual output rather than idealized weight scores. |
| Host summaries | NumPy conversion, accuracy, and predictions | Host analysis boundary | Correct | No BrainX replacement applies. |
| Visualization | high-level Matplotlib | Host visualization boundary | Correct | Figure is readable and scientifically labeled. |
| CLI and output path | `argparse`, `pathlib` | Python host boundary | Correct | No change. |
| Verification | compile, runs, inline assertion | Python test boundary | Partial | Save focused tests as artifacts. |

## Missing, bypassed, or misused BrainX APIs

### Selective State lifecycle at trial boundaries

The missing operation is not a different learning transform. Learning weights
must stay sequential, but this does not require every State role to leak across
trials. Reset or reinitialize hidden and short-term State at trial boundaries
while preserving `LongTermState` weights. Use collective lifecycle filters or
an explicit module reset method that encodes the same role boundary. Treat a
silent intertrial interval as scientific model time only when carryover is
intentional.

### Shared State under `vmap2`

Run 1 correctly maps the complete held-out rollout, but maps an already repeated
weight array through `OfType(brainstate.State)`. For truly shared read-only
weights, keep their State unbatched and map only neuron and trace State through
`state_in_axes` and `state_out_axes`. A repeated mapped array is replicated
State, even when every copy begins with the same values.

No delay API is missing. This model chooses a decaying trace window rather than
an exact delayed event and gives its time constant explicit units. No
BrainTrace, projection, input-generator, metric, or visualization replacement
is required.

## Performance and code simplicity

The expensive training and evaluation work is compiled and contains no Python
timestep loop. Run 1 now uses `vmap2` around the complete independent rollout,
so vectorization covers neural dynamics rather than only preprocessing. This is
the most consequential performance and semantic improvement over Run 0.

Replicating the four weights across 16 evaluation lanes is negligible here but
does not scale. Shared read-only State avoids trials times weight storage. The
training loop also returns four weights at every step and discards all but
trial endpoints; a larger model should record more selectively. Host protocol
construction, decoding, and plotting remain appropriate boundaries.

## Skill improvements

1. Refine `skills/brainx-general-guard/SKILL.md` to separate State dependency
   from State lifecycle: keep trials sequential when learned State carries, but
   reset unrelated dynamical State at logical independent trial boundaries
   unless carryover is scientifically intended.
2. Refine `skills/brainevent/references/synaptic-plasticity.md` beside its
   weight/trace lifecycle guidance: sequential weight learning must preserve
   `LongTermState` while resetting per-trial traces and neural State.
3. Refine
   `skills/brainstate/references/brainstate/transformation-vmap-expansion.md` to
   distinguish one shared read-only State from repeated identical mapped State.
4. Do not repeat the existing BrainUnit `u.math` rule. The agent found and fixed
   the raw `jnp.where` misuse in one smoke run, and the root skill already owns
   that decision.

## Checks for the next run

- Install the second refined snapshot before creating `run2/`.
- Keep learned weights causally sequential across training trials.
- Reset membrane, refractory, detector, eligibility, post-trace, and other
  nonpersistent State at each logical independent trial boundary, or explicitly
  justify intertrial carryover as part of the model.
- Map complete independent evaluation rollouts.
- If the report says weights are shared, verify that read-only weight State is
  unbatched rather than repeated and mapped.
- Use intentional `V_initializer` values from the first implementation.
- Plot actual circuit responses before feedback and describe dual association
  retention accurately.
- Save reusable focused tests and inspect the generated figure.
- Reuse the exact prompt fingerprint, model, effort, virtualenv, CLI, and
  isolated harness from Runs 0 and 1.
