# BrainX diagnosis: learning temporal order, Run 2

## Evidence studied

Generated evidence:

- `run2/temporal_order_learning.py`, `test_temporal_order_learning.py`,
  `README.md`, `agent-final.md`, `codex-events.jsonl`, `codex-stderr.log`, and
  `harness-metadata.txt`.
- `run2/artifacts/temporal_order_reversal.png`, visually inspected at its
  native 1620 by 1530 resolution.
- The entry point and `unittest` suite, executed unchanged from a temporary
  output directory with the required BrainX virtualenv.
- Reviewer probes comparing a sequential second training trial with the same
  trial from fresh State and measuring the residual trace State after the
  sequential phase.

Run 2 used the same 666 prompt bytes, prompt hash, model, reasoning effort, CLI
version, virtualenv, and isolated harness as Runs 0 and 1. Its skill checkpoint
was the repository snapshot after the Run 1 diagnosis and second refinement.
The absolute-simplicity rule was written only after Run 2 and was not present
in this evaluated snapshot.

Unchanged execution produced:

```text
Untrained accuracy:       50%
After acquisition:        100%
At task reversal:         0%
After reversal learning:  100%

Ran 1 test in 4.783s
OK
```

The reviewer lifecycle probe measured:

```text
end sensory_trace max: 0.196199432015419
end readout_trace max: 3.2947609724942595e-05
end teacher_trace max: 0.04256347566843033
second-trial matches isolated: True
```

The current timing prevents those residual values from changing the probed
second-trial detector output, but the logical trial boundary still does not
reset State.

Primary Python examples used to establish the expected composition:

- `skills/brainpy-state/references/scripts/103_COBA_2005.py` and
  `107_gamma_oscillation_1996.py` for `LIFRef`, explicit initialization,
  Module composition, and transformed time execution.
- `skills/brainpy-state/references/scripts/109_fast_global_oscillation.py` for
  explicit BrainState delay use and monitored point-neuron execution.
- `skills/brainpy-state/references/scripts/training-snn.py` and
  `201_surrogate_grad_lif_fashion_mnist.py` for per-sequence State lifecycle,
  batched initialization, and complete transformed rollouts.
- `skills/brainevent/references/scripts/coba_ei_teaching.py` for
  `BinaryArray @ connectivity` inside a BrainPy-State Module and one compiled
  State-aware time loop.
- `skills/brainstate/scripts/lif_neuron_model.py` and `integrator_rnn.py` for
  State-owned recurrent dynamics and compact execution boundaries.

The matching inventories in `source_html_references/` were used to locate the
official BrainPy-State neuron, BrainState Delay and transform, BrainEvent event
array and plasticity, BrainUnit quantity, and BrainTools initializer APIs. The
exact contracts checked were `brainpy.state.LIFRef`,
`brainstate.nn.init_all_states`, `brainstate.nn.Delay`,
`brainstate.transform.for_loop`, `brainstate.transform.vmap`,
`brainevent.BinaryArray`, `brainevent.update_dense_on_binary_pre`, BrainUnit
quantity conversion/math, and `braintools.init.Constant`.

The Run 0 and Run 1 review corpora remain applicable. Their diagnoses record
the complete routed skill, reference, example, and API-page inventory.

## Executive diagnosis

Run 2 is the strongest scientific result so far. It reverses the label for the
same two detector cues, measures the immediate loss without teaching, and then
shows reacquisition. The source runs, the focused test passes, the weights move
in the expected directions, and the figure clearly shows 50% -> 100% -> 0% ->
100%.

Its implementation quality regresses sharply. The main file grows from 452
lines in Run 0 and 529 lines in Run 1 to 605 lines. A one-off six-neuron
demonstration becomes a configurable mini-framework: `ExperimentConfig` owns
22 fixed fields, three `NamedTuple` result types wrap local values, a custom
`FixedEventDelay` replaces the package delay, evaluation has a second batch
mode, and CLI/JSON/reporting layers expand around the model. This structure was
not required by the prompt and does not reveal the scientific mechanism more
clearly.

The event log explains the growth. The agent tried to map all State, found that
the package `Delay` includes a scalar ring index that did not fit that mapped
axis contract, switched to native batch State, and then wrote a custom delay to
make the batched layout work. It solved the immediate runtime problem by adding
abstractions instead of simplifying the evaluation design. Run 2 predates the
new absolute-simplicity rule, so the evaluated guard constrained BrainX
ownership and transform choice but did not make minimal delivered code an
independent acceptance criterion.

Two earlier semantic problems also remain. Training trials are nested in one
compiled `for_loop` with no per-trial reset, so short-term and neural State
cross logical boundaries. The only `vmap` constructs pure input arrays;
stateful evaluation uses a native batch axis. The README and figure nevertheless
describe “vmap trials,” which does not satisfy the prompt's explicit request to
batch trials with `vmap` and conflicts with the guard's own rule that input
construction does not batch the simulation.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `temporal_order_learning.py:285-306` | The outer `for_loop` carries all circuit State across nominally independent training trials. Only the acquisition/reversal phase boundary is reset. | Membrane, refractory, delay, sensory, readout, and teacher history can affect the next trial. Current correctness depends on the long quiet tail. | Compile one complete trial rollout, execute learning trials sequentially because weights depend on prior trials, and reset only trial-scale State at every boundary while preserving learned weights. |
| P1 | `temporal_order_learning.py:276-278`, `309-345`; `README.md:11-13`; figure panel 3 | `vmap` covers only pure trial encoding; stateful evaluation uses native batched State, yet the report calls the trials vmapped. | The implementation does not meet the prompt's explicit stateful-trial `vmap` requirement and misstates the execution evidence. | Map a complete independent evaluation trial, or state plainly that the requirement is unmet. Do not use vmapped preprocessing as evidence of vmapped simulation. |
| P2 | `temporal_order_learning.py:94-126` | `FixedEventDelay` reimplements a fixed-grid history buffer after the mapped package `Delay` path failed. | A second delay protocol adds manual shape, batch, reset, indexing, and validation logic. It is correct for the tested grid but enlarges the failure surface. | First simplify the mapped design so the owning delay API can remain. Keep custom delay logic only when the required batch semantics are verified as an API gap and cannot be expressed by a simpler complete-trial mapping. |
| P2 | `temporal_order_learning.py:28-78` | `ExperimentConfig` and three result types generalize a fixed demonstration without a second real configuration or reusable consumer. | Readers must follow indirection through 26 fields before seeing the six-neuron model; unit defaults also caused an avoidable dataclass debugging cycle. | Keep fixed scientific parameters as nearby unit-bearing constants. Introduce a configuration object only when the prompt requires parameter variation or it removes more code than it adds. |
| P2 | `temporal_order_learning.py:553`; `README.md:11-13` | The figure and README use “vmap” as a presentation claim rather than describing the actual native batch implementation. | A correct scientific result is supported by inaccurate implementation metadata. | Label the observable, not the desired API checklist, and verify implementation claims against source. |
| P3 | `temporal_order_learning.py:473-573` | Three plot panels, a JSON summary, and flexible output-path plumbing repeat the same four stage results. | Reporting consumes roughly one sixth of the main file and obscures the model without adding a distinct scientific conclusion. | Keep one learning/reversal figure and concise printed checks unless additional artifacts are requested. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical parameters | Unit-bearing fields in `ExperimentConfig` | BrainUnit quantities | Correct mechanics, excessive container | Keep the units; remove the one-use configuration framework. |
| Duration-to-step conversion | `Quantity.to_decimal` and host rounding | BrainUnit plus host validation | Correct for the grid | Validate only the delay ratio that requires integral steps. |
| Tone schedule | Unit-aware comparisons in `encode_one` | BrainUnit/JAX pure-data boundary | Correct | Keep one compact encoder. |
| Input batching | `brainstate.transform.vmap(encode_one)` | Pure-function mapping | Mechanically correct, not the requested trial batching | Do not count it as stateful simulation batching. |
| Circuit graph | `TemporalOrderCircuit(Module)` | BrainState Module | Correct and useful | Keep the single model class because it owns real State and behavior. |
| Fixed experiment settings | `ExperimentConfig` | Python host boundary | Unnecessary abstraction | Use concrete constants for this one-off example. |
| Input/result records | Three `NamedTuple` classes | Python host boundary | Mostly unnecessary | Return ordinary tuples or one small result mapping at the host boundary. |
| Point-neuron populations | Three `brainpy.state.LIFRef` groups | BrainPy-State | Correct | Keep explicit scientific parameters and `V_initializer`. |
| Initial membrane voltage | `braintools.init.Constant(resting_voltage)` | BrainTools initializer | Correct | Keep. |
| Event delay | Custom `FixedEventDelay` with `ShortTermState` | `brainstate.nn.Delay`, or a verified custom State boundary | BrainX-native but lower level than desired | Prefer the package delay in a simpler single-trial model. |
| Temporal-order feature | Delayed event crossed with a decaying sensory trace | BrainEvent plus model-specific State | Scientifically valid custom behavior | Keep, but express it directly. |
| Learned weights | `LongTermState` | BrainState State role | Correct | Preserve across trial resets. |
| Per-trial traces | Three `ShortTermState` values | BrainState State roles | Correct roles, incorrect lifecycle | Reset at every independent trial boundary. |
| Event communication | `BinaryArray @ dense weights` | BrainEvent | Correct | Keep dense storage for four weights. |
| Online update | `update_dense_on_binary_pre` inside the step | BrainEvent plasticity | Correct | Keep the signed teacher rule and bounds. |
| Trace decay | `u.math.exp(-dt / tau)` | BrainUnit math | Correct | Keep the exponent dimensionless. |
| Stateful timesteps | `for_loop` inside `jit` | BrainState transforms | Correct inner boundary | Compile one stable trial callable. |
| Sequential learning | A second nested `for_loop` | BrainState control flow | Dependency order is correct; lifecycle is not | Use an explicit coarse trial boundary that can reset short-term State. |
| Independent evaluation | Native leading batch State | BrainState native batch axis | Scientifically independent, not vmapped | Use a complete state-aware mapped trial when `vmap` is explicitly required. |
| State initialization | `init_all_states` at construction and phase reversal | BrainState collective lifecycle | Incomplete | Reset neural and short-term State per trial, not only per phase. |
| Accuracy and predictions | NumPy host calculations | Host analysis boundary | Correct | Keep concise host-side scoring. |
| Randomness | `brainstate.random.seed(seed)` with deterministic inputs and initialization | No RNG operation is used | Redundant | Remove seed plumbing unless stochastic behavior is introduced. |
| Figure | High-level `matplotlib.pyplot` | Host visualization boundary | Readable and unclipped | Reduce duplicate panels and correct the batching label. |
| JSON, CLI, filesystem | `argparse`, `json`, `pathlib` | Python host boundary | Valid APIs, not required by the prompt | Keep only the output path needed to show the result. |
| Regression test | Standard-library `unittest` | Python test boundary | Focused and passing | Add a lifecycle assertion if this design remains. |

## Missing, bypassed, or misused BrainX APIs

### Complete-trial `brainstate.transform.vmap`

Run 2 does not map a stateful trial. It maps only `encode_one`, then evaluates a
single module whose State already has a leading batch axis. Native batching can
be a good implementation choice, but it is not `vmap`. When the user explicitly
requires `vmap`, wrap the complete independent trial callable and declare its
State axes according to the documented contract. If a selected Module prevents
that composition, disclose the limitation instead of relabeling preprocessing
or native batching.

### Per-trial collective lifecycle

`init_all_states` is called at construction and once at the reversal boundary.
The learned weight dependency does not require short-term State carryover. Use
the documented collective lifecycle on the neural/delay subgraph or an explicit
small reset method at each trial boundary while leaving `LongTermState` weights
unchanged. A host loop across a small number of sequential learning trials is a
valid orchestration boundary when it permits the required reset and calls one
compiled per-trial rollout.

### `brainstate.nn.Delay`

The event log records a real difficulty combining the general delay's scalar
ring index with the attempted mapped State axes. That justifies investigating a
different composition, not automatically shipping a new delay class. A custom
fixed buffer is acceptable only after the task still requires semantics the
owning API cannot express under a simpler single-trial mapping, and its impulse,
reset, unit-grid, and batch behavior must remain tested.

No BrainTrace algorithm, sparse connectivity, BrainTools input generator, or
package visualization API is required. The deterministic tone encoder, tiny
dense readout, host accuracy calculation, and high-level Matplotlib figure are
legitimate boundaries.

## Performance and code simplicity

The inner 100-step trial executes through BrainState control flow, and online
plasticity remains inside State-aware execution. Dense event communication is
appropriate for the four readout weights. Independent evaluation runs in one
native batch. These choices are efficient for the small model.

The outer nested transform trades lifecycle clarity for compilation. It returns
sensory spikes, detector spikes, output spikes, and the full 2 by 2 weight
matrix at every step of every trial, even though reporting retains only trial
end weights and response-window counts. A compiled per-trial rollout with a
small sequential host loop is simpler and encodes the reset boundary directly.

The dominant problem is structural, not raw line count: 22 configuration
fields, three record types, two batch modes, a custom delay, seven reporting
functions, and four output surfaces surround one fixed experiment. The skill
should require substantial example/API study but judge the delivered code by
the minimum concepts and files needed to preserve science, correctness,
performance, verification, and requested output quality.

## Skill improvements

1. Refine `skills/brainx-general-guard/SKILL.md` with a separate absolute
   coding-simplicity rule: intensive study should discover the BrainX APIs that
   remove code; it must not result in a generic framework around a one-off
   scientific demonstration.
2. State explicitly that configuration objects, result classes, helper layers,
   CLI options, and extra artifacts require demonstrated variation, reuse, or a
   net reduction in complexity.
3. Refine the transform boundary: never use Python for timesteps, but allow a
   small host loop across causally sequential trials when each iteration calls
   one compiled rollout and must reset selected State.
4. State that native batch State and vmapped input construction must not be
   described as stateful `vmap`. When the prompt explicitly requires `vmap`,
   map the complete independent operation or report the unmet constraint.
5. Synchronize the distinct simplicity principle into `plan.md`; keep it
   separate from BrainX-native boundaries and highest-level API selection.

## Checks for the next run

- Preserve the exact prompt fingerprint and isolated harness.
- Require a materially smaller source whose structure exposes the six-neuron
  mechanism directly; a fixed one-off example should not need
  `ExperimentConfig` or multiple result classes.
- Keep all physical parameters unit-bearing and retain explicit
  `V_initializer`.
- Keep learning trials sequential, but reset every neural, delay, and trace
  State at each logical trial boundary while preserving learned weights.
- Map a complete independent evaluation trial with `vmap`; do not count input
  construction or native batching.
- Prefer the owning delay API. If custom delay State remains necessary, require
  a documented API gap and an impulse/reset test.
- Measure actual teacher-free circuit behavior before learning, after
  acquisition, immediately after label reversal, and after relearning.
- Keep the true same-cue reversal result and a clear, unclipped figure while
  removing redundant reporting layers.
- Run the unchanged entry point and focused tests, inspect the figure, and
  compare source size, abstraction count, lifecycle correctness, and transform
  claims with Run 2.
