# BrainX diagnosis: learning temporal order, Run 3

## Evidence studied

Generated evidence:

- `run3/temporal_order.py`, `README.md`, `agent-final.md`,
  `codex-events.jsonl`, `codex-stderr.log`, and `harness-metadata.txt`.
- `run3/temporal_order_relearning.png`, visually inspected at its native 1800
  by 1350 resolution.
- The unchanged entry point executed from a temporary output directory with
  the required BrainX virtualenv.
- A reviewer lifecycle probe that ran one taught trial, called the artifact's
  reset boundary, and checked trace, voltage, and weight State.

Run 3 preserved the exact 666 prompt bytes, SHA-256, model, reasoning effort,
CLI version, virtualenv, and empty workspace. Codex resolved the freshly
reinstalled global skills at `/Users/nijiachen/.agents/skills`; the user
explicitly accepted this installation model, and the consumed skill trees were
verified byte-for-byte identical to the current repository snapshot. The first
post-refinement launch is preserved separately as `run3-invalid-auth/` because
isolating `HOME` made the login keychain unavailable before model execution.

Independent execution reproduced:

```text
Acquisition scores [AB, BA] x [output 0, output 1]:
[[1.103 0.071]
 [0.071 1.103]]
Reversal scores [AB, BA] x [output 0, output 1]:
[[0.098 1.103]
 [2.105 0.097]]
Accuracy: acquired=100%, immediate reversal=0%, relearned=100%,
          vmapped jittered batch=100%
```

The lifecycle probe produced:

```text
before reset trace max: 0.31663697957992554
after reset trace max: 0.0
sensory V reset: [-60. -60.] mV
output V reset: [-60. -60.] mV
weight preserved: True
```

The review standard was established primarily from the same concrete Python
corpus used for the earlier diagnoses:

- `skills/brainpy-state/references/scripts/103_COBA_2005.py`,
  `107_gamma_oscillation_1996.py`, and `109_fast_global_oscillation.py` for
  point-neuron initialization, delay, transformed execution, and monitoring.
- `skills/brainpy-state/references/scripts/training-snn.py` and
  `201_surrogate_grad_lif_fashion_mnist.py` for sequence lifecycle and batched
  State.
- `skills/brainevent/references/scripts/coba_ei_teaching.py` for event
  communication inside a compact BrainPy-State Module.
- `skills/brainstate/scripts/lif_neuron_model.py` and `integrator_rnn.py` for
  State ownership and complete-rollout transformation.

The matching `source_html_references/` inventories routed verification to the
official APIs for `LIFRef`, `Expon`, `Delay`, `init_all_states`,
`reset_all_states`, `for_loop`, `vmap2`, `BinaryArray`,
`update_dense_on_binary_post`, BrainUnit quantity math, and BrainTools
initializers. The examples establish composition; the API pages establish the
exact contracts.

## Executive diagnosis

Run 3 resolves every important Run 2 defect. It demonstrates a true same-cue
label reversal, resets trial-scale State while preserving learned weights, and
maps complete independent stateful evaluations with `vmap2`. Its implementation
claims match the source, and the figure uses actual teacher-free circuit scores.

The simplicity refinement also has a visible effect. The main source falls from
605 to 429 lines and removes `ExperimentConfig`, all three result classes, the
custom delay class, JSON serialization, CLI configuration, and the separate
test framework. One Module owns the real neural and plastic State; one compiled
rollout owns a trial; a small host loop owns causally sequential learning; and
one mapped operation owns independent evaluation.

The remaining complexity is mostly scientific reporting and explicit State-axis
setup, not framework scaffolding. The source can still be tightened: `Path` is
unused, deterministic seeding has no effect, and the four-panel plot is larger
than the minimum demonstration. Those are P3 artifact issues already covered by
the current simplicity rule, not evidence for another skill refinement cycle.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P3 | `temporal_order.py:193-205`; `README.md:19-37` | The decision score combines spike count with a small peak-voltage tie-break, but the README describes only the learning rule. | The reported margin is scientifically valid but not fully defined in the narrative. | State in one sentence that ties in spike count are resolved by peak subthreshold voltage, or use spike count alone if parameters can avoid ties. |
| P3 | `temporal_order.py:10`, `207` | `Path` is unused and `brainstate.random.seed(7)` has no effect in a deterministic experiment. | Two lines imply filesystem or stochastic behavior that does not exist. | Remove both unless randomness or path handling is introduced. |
| P3 | `temporal_order.py:347-362` | Stateful evaluation replicates every State, including identical learned weights, across 12 lanes. | Memory scales with batch size times weight size, though the cost is negligible for four weights and the README accurately calls them copies. | Keep for this small writable-State rollout; share read-only weights only if evaluation is split into a no-write step and the added structure remains simpler. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical timing, voltage, current, and time constants | Module-level BrainUnit quantities | BrainUnit | Correct and direct | Keep constants concrete for this one-off example. |
| Tone protocol | Unit-aware masks in `make_trial` | BrainUnit/JAX pure-data boundary | Correct | Keep. |
| Point-neuron dynamics | Two sensory and two output `LIFRef` neurons | BrainPy-State | Correct minimal model | Keep explicit `V_initializer`. |
| Synaptic current | `brainpy.state.Expon` | BrainPy-State | Correct | Keep unit-bearing `g_initializer`. |
| Axonal history | `brainstate.nn.Delay` | BrainState delay | Correct owning API | Keep update-before-integer-step retrieval. |
| Delay conversion | `int(round(AXONAL_DELAY / DT))` | BrainUnit ratio plus host integer | Correct for fixed grid | Add an integer-grid assertion only if delay becomes configurable. |
| Eligibility trace | `ShortTermState` plus unit-aware decay | BrainState and BrainUnit | Correct | Keep. |
| Learned efficacy | `LongTermState` 2 by 2 matrix | BrainState | Correct | Preserve across resets. |
| Binary communication | `BinaryArray(arrived_spikes) @ weight` | BrainEvent | Correct | Dense storage is simplest for four weights. |
| Potentiation and depression | Two `update_dense_on_binary_post` calls | BrainEvent plasticity | Correct true-reversal rule | Keep target and competitor events explicit. |
| Environment | Outer `dt`, per-step `t` context | BrainState environment | Correct | Keep. |
| Timestep execution | One `for_loop` wrapped by one `jit` | BrainState transforms | Correct stable boundary | Keep. |
| Sequential learning | Small Python loops over epochs and two orders | Host orchestration around one compiled trial | Correct dependency and lifecycle boundary | Keep; do not nest trials into a transform that prevents reset. |
| Trial reset | `reset_all_states` on neural, synaptic, and delay modules plus trace reset | BrainState collective lifecycle | Correct and independently verified | Keep learned weight outside this reset. |
| Independent evaluation | `vmap2` over the complete `rollout` | BrainState state-aware mapping | Correct and satisfies the prompt | Keep State axes explicit. |
| Mapped State preparation | Broadcast every State to a leading trial axis | BrainState State plus JAX PyTree boundary | Correct for separate writable lane copies | Avoid claiming shared weights. |
| Host scoring | NumPy spike counts and voltage tie-break | Host analysis boundary | Correct | Document the tie-break. |
| Figure | High-level `matplotlib.pyplot` | Host visualization boundary | Correct, readable, and unclipped | A smaller figure would also suffice but is not required. |
| Verification | Inline assertions plus independent rerun | Python verification boundary | Focused and passing | No separate test framework is necessary for this self-contained tutorial. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing, bypassed, or misused in the final Run 3
source.

The host epoch loops are intentional: each learning trial consumes weights from
the previous trial and must reset unrelated State at its boundary. The expensive
timestep work still runs through one compiled `for_loop`. This is the lifecycle
boundary the Run 2 implementation failed to represent.

The `vmap2` call maps the complete teacher-free rollout and declares mapped
State axes. Input construction is separate and is not presented as the mapped
simulation. Replicating learned weights is accurate because the update method
still writes weight State with a zero learning rate; the artifact does not claim
that those values are shared.

## Performance and code simplicity

Run 3 compiles one fixed-shape trial and reuses it across sequential learning.
It resets State outside that compiled callable, which makes the lifecycle
visible without introducing a second transformed trial loop. Independent
evaluation maps 12 complete rollouts and then compiles that mapping. Dense
event communication and four stored weights are appropriate for this scale.

Compared with Run 2, the source removes 176 lines and the most expensive
abstraction categories: configuration framework, result records, custom delay,
dual batch implementation, CLI surface, JSON artifact, and test file. The
remaining one class represents real shared State and behavior rather than
organizational scaffolding. Two output files plus a concise README are
proportionate to a request that asks to show learning and reversal.

## Skill improvements

No further skill edit is justified by Run 3.

The post-Run 2 changes worked as intended:

1. The general guard's absolute-simplicity rule removed speculative framework
   structure while preserving units, State lifecycle, compilation, mapping,
   verification, and figure quality.
2. The BrainState control-flow refinement produced one compiled trial plus a
   small sequential host loop, so reset semantics are explicit.
3. The transform-claim rule produced a genuine complete-rollout `vmap2` instead
   of vmapped preprocessing or mislabeled native batching.
4. The example-first study rule led the agent to open the relevant Python
   scripts and routed API contracts before implementation.

Do not add skill text for the unused import, deterministic seed, or score
narrative. The current editing test and simplicity rule already cover those
local cleanup decisions.

## Checks for completion

- Preserve Runs 0 through 3 and both invalid/control folders as evidence.
- Confirm the exact prompt fingerprint and skill checkpoint in each metadata
  file.
- Confirm Run 3 executes unchanged and reproduces all four reported accuracy
  checks.
- Confirm trial reset clears trace and neural State while preserving learned
  weights.
- Confirm `vmap2` wraps the complete independent rollout.
- Confirm the figure remains readable and unclipped.
- Validate changed skill folders, run the installer suite, and verify the
  reinstalled global skills are byte-identical to the repository.
