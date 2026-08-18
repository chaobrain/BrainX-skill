# BrainX diagnosis: prior bias under ambiguous evidence

## Evidence studied

Generated and archived artifacts:

- `prior_bias_decision.py`
- `README.md`
- `results/prior_bias_decision.png`
- `results/summary.json`
- `__pycache__/prior_bias_decision.cpython-312.pyc`
- `agent-final.md`, `codex-events.jsonl`, `codex-stderr.log`, and
  `harness-metadata.txt`

Execution and output checks:

- Ran an unchanged copy with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`.
- Reproduced the probability curves and every scientific summary value:
  zero-evidence shift `0.28125`, mean weak-evidence shift
  `0.23958333333333334`, and mean strong-evidence shift `0.0`.
- Confirmed `py_compile` passes. The optional `ruff` check is unavailable in
  the frozen environment.
- Inspected the archived `2396 x 1460` PNG at original resolution. The circuit,
  sample trajectories, probability intervals, labels, and speed measurements
  are readable and unclipped.
- Reproduced the run's lifecycle failure: after
  `vmap_init_all_states(..., axis_size=7)`, all eight BrainPy dynamical States
  had shape `(7, 24)`, while `vmap_reset_all_states(..., axis_size=7)` changed
  them to `(24,)`.
- Verified that a direct absolute-path snapshot passed to
  `assign_state_values` restored every dynamical shape and returned no
  unexpected or missing paths when all 12 model States were captured.

Owning skills and routed references:

- `skills/brainx-general-guard/SKILL.md`
- `skills/brainpy-state/SKILL.md`
- `skills/brainpy-state/references/component-selection.md`
- `skills/brainpy-state/references/projection-patterns.md`
- `skills/brainstate/SKILL.md`
- `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`
- `skills/brainstate/references/brainstate/transformation-jit-expansion.md`
- `skills/brainstate/references/collective_model_operations.md`
- `skills/brainunit/SKILL.md`

Closest executable examples:

- `skills/brainpy-state/references/scripts/103_COBA_2005.py`
- `skills/brainpy-state/references/scripts/sound_localization.py`

Authoritative API pages:

- `brainpy.state.LIFRef`
- `brainpy.state.AlignPostProj`
- `brainpy.state.Expon`
- `brainpy.state.COBA`
- `brainstate.nn.EventFixedProb`
- `brainstate.transform.vmap2`
- `brainstate.nn.vmap_init_all_states`
- `brainstate.nn.vmap_reset_all_states`
- `brainstate.nn.assign_state_values`
- BrainState Collective Operations

## Executive diagnosis

The final artifact answers the question and fixes the preceding checkpoint's
main scientific/API defect. It models two point-neuron populations with four
explicit BrainPy projections: two same-choice excitatory projections and two
cross-choice inhibitory projections. Event communication, exponential
synaptic dynamics, conductance-based output, and postsynaptic ownership remain
separate, and every projection runs before the neuron update.

The experiment keeps physical units through neuronal and synaptic execution,
maps 1,152 independent condition/trial lanes with semantic State filters, uses
one transformed time loop, and benchmarks one stable compiled callable after
synchronization. The unchanged-copy rerun exactly reproduced the scientific
results. Wilson intervals make probability uncertainty visible, and the figure
shows a resolved weak-evidence transition rather than only one ambiguous point.

One material BrainState lifecycle problem occurred during generation. The
agent followed the current collective-operations guidance and tried
`vmap_reset_all_states(..., axis_size=...)`; the call collapsed each mapped LIF
State from `(lane, neuron)` to `(neuron,)`, so the first compiled projection
received a scalar event and failed. The final artifact works around this by
capturing dynamical State objects and assigning their initial values before
each rollout. That recovery is operationally correct, but
`assign_state_values` is the named BrainState API for path-keyed restoration.

The final script is 453 lines, compared with 286 in the preceding checkpoint.
Explicit projections account for necessary growth. First-passage decoding,
Wilson intervals, JSON reporting, a circuit schematic, and detailed plotting
account for most of the remaining increase. Those additions improve the
artifact, but they are not all required by the model correction and do not
justify more root-skill simplicity language; the general guard already states
that rule directly.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P3 | `prior_bias_decision.py:173` through `prior_bias_decision.py:188` and `results/summary.json` | One declared seed and one fixed connectivity realization support the reported curves; Wilson intervals quantify finite-trial binomial uncertainty but not seed or network-realization sensitivity. | The result is a strong reproducible demonstration, not a population-level robustness claim over random circuits. | Keep the conclusion descriptive, or repeat the scientific measurement over declared seeds/connectivity realizations when robustness is required. This remains host-side experimental design. |
| P3 | `prior_bias_decision.py:313` through `prior_bias_decision.py:328` | The trajectory panel selects the two smallest and two largest final decision variables in each prior condition. | It intentionally shows both unfolding directions but is not a random or paired sample of trials. | Label the examples as selected extremes, or use declared fixed trial indices when representative sampling matters. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Represent point-neuron choices | Two `brainpy.state.LIFRef` populations | `brainpy.state.LIFRef` | Correct. The task explicitly requests a noisy point-neuron circuit. | Keep. |
| Represent recurrent excitation | A-to-A and B-to-B `AlignPostProj` objects | BrainPy projections | Correct and materially improved. | Keep explicit projections. |
| Represent mutual inhibition | A-to-B and B-to-A `AlignPostProj` objects | BrainPy projections | Correct and materially improved. | Keep explicit projections. |
| Communicate spikes | Four `brainstate.nn.EventFixedProb` modules | BrainState event communication | Correct for probabilistic sparse connectivity. | Keep unless the scientific graph requires another wiring rule. |
| Apply synaptic kinetics | `brainpy.state.Expon.desc` | BrainPy synapse dynamics | Correct for a linear decaying postsynaptic trace and AlignPost. | Keep. |
| Convert conductance to current | Excitatory and inhibitory `brainpy.state.COBA.desc` outputs | BrainPy synaptic output | Correct. Reversal potentials define the current sign from postsynaptic voltage. | Keep. |
| Preserve projection update order | Read previous spikes, call four projections, then update both populations | BrainPy projection lifecycle | Correct. Inputs are deposited before postsynaptic integration. | Keep. |
| Represent evidence, prior, voltage, current, conductance, and time | BrainUnit quantities | BrainUnit | Correct. Unit-bearing values persist through model execution. | Keep. |
| Initialize independent lanes | `vmap_init_all_states(..., axis_size=1152)` | BrainState lifecycle | Correct. Dynamical States receive a leading lane axis. | Assert one representative State shape when reset behavior is performance-critical. |
| Map complete independent steps | `vmap2` with semantic dynamical-State filters | BrainState State-aware mapping | Correct. Writable dynamical State is mapped while parameters remain shared. | Keep. |
| Generate independent stochastic inputs | `brainstate.random.normal` inside the mapped step | BrainState randomness | Correct. Seeding reproduces the complete scientific run. | Add multi-seed evaluation only for a robustness claim. |
| Advance simulation time | One `brainstate.transform.for_loop` | BrainState control flow | Correct. No Python timestep loop is used. | Keep. |
| Compile execution | One `brainstate.transform.jit(rollout)` | BrainState JIT | Correct. The callable and input shapes remain stable. | Keep. |
| Reset independent benchmark rollouts | Captured `(State, initial_value)` pairs reassigned before each call | `brainstate.nn.assign_state_values` or verified `vmap_reset_all_states` | The workaround preserves shape and execution, but it follows a failed documented vmapped reset and bypasses the named restoration API. | Capture all States by absolute path and restore with `assign_state_values`; require empty unexpected/missing results. Use `vmap_reset_all_states` only after a representative shape check for the selected modules. |
| Decode choices and decision times | Host NumPy first-passage calculation | Host scientific rule | Correct. It uses the same cumulative variable shown in the trajectory panel. | State or count final-sign fallbacks if non-crossing trials become material. |
| Estimate choice probabilities and uncertainty | NumPy reductions and Wilson intervals | Host statistics | Correct boundary; no BrainX psychometric API owns this calculation. | Keep. |
| Measure speed | `perf_counter`, blocked first call, and five blocked steady calls | Host timing plus JAX synchronization | Correct. The plot clearly reports aggregate simulated seconds per wall second and excludes compilation. | Keep. |
| Plot requested outputs | Matplotlib circuit, trajectory, probability, and speed panels | Host visualization | Correct and readable. | The schematic and JSON are optional if a shorter artifact is preferred. |
| Serialize measurements | JSON | Host reporting | Correct. | Keep only when machine-readable results are useful. |

## Missing, bypassed, or misused BrainX APIs

### `brainstate.nn.vmap_reset_all_states`

The generated agent initially used this API exactly as the current official
guide and local reference show. For the mapped BrainPy LIF circuit, it changed
every dynamical State from `(lane, neuron)` to `(neuron,)` and caused a scalar
event-communication failure. The official API page promises independent
batched reset, but its examples do not assert a dynamical-State shape for a
BrainPy module.

Do not present the API as unconditionally shape-preserving across all module
families. After vmapped initialization, verify a representative State shape
before and after reset. If the lane axis changes, restore a path-keyed snapshot
instead of compiling or timing the collapsed graph.

### `brainstate.nn.assign_state_values`

Use this API instead of a manual `(State, value)` reassignment loop when exact
initial vmapped values must be restored. Capture all model States by their
existing absolute paths, restore them, and require both returned mismatch
collections to be empty. The verified run3 review restored all eight
dynamical-State shapes from `(24,)` to `(7, 24)` with no mismatches when all 12
model States were included.

No BrainX API should replace first-passage decoding, Wilson intervals, JSON
serialization, host timing, or custom multi-panel presentation.

## Performance and code simplicity

The execution structure is strong:

- 1,152 independent lanes are mapped in one operation;
- one transformed loop owns all 600 time steps;
- one stable JIT callable is reused for compilation and steady measurements;
- device work is blocked before every timer stops;
- five steady measurements are reported individually and reduced by a host
  median.

The archived median steady call is `0.879 s`, or `786x` aggregate real time.
The unchanged rerun measured `0.877 s` and reproduced the complete probability
table. Timing variation changes the speed bars and PNG hash but not scientific
values.

The four explicit projections add real scientific structure that must remain.
The circuit schematic, JSON output, generalized plotting helper, and separate
snapshot helpers contribute code that is useful but not essential to answering
the prompt. This is an artifact-level concision opportunity, not evidence for
another general-guard rule.

## Skill improvements

### `brainstate/references/collective_model_operations.md`

Qualify the vmapped reset workflow with a shape-preservation invariant. State
that the official API intends to reset each lane independently, but the selected
module's `reset_state` behavior must preserve the leading lane axis. Require a
representative before/after shape check for BrainPy dynamics or other modules
whose reset method may recreate State at feature shape.

Add the verified path-keyed snapshot workflow with `assign_state_values` as the
fallback for exact repeated-rollout restoration. Replace the current final
workflow summary, which repeats the unverified vmapped reset and a known
mismatched flattened snapshot.

### Other skills and `plan.md`

No root BrainState, BrainPy-State, BrainUnit, general-guard, or plan change is
justified. The projection, scale, unit, mapping, loop, compilation, randomness,
and simplicity guidance all led to the correct final architecture.

## Checks for the next run

1. Preserve the explicit same-choice excitatory and cross-choice inhibitory
   projections.
2. Preserve projection-before-neuron update order and unit-bearing synaptic
   weights, currents, voltages, evidence, prior, and time.
3. Preserve semantic mapped-State filters, one transformed time loop, and one
   stable compiled callable.
4. Do not use `vmap_reset_all_states` without proving that one representative
   dynamical State keeps `(lane, feature...)` shape.
5. Prefer `assign_state_values` with a direct all-State path snapshot when exact
   vmapped initial values must be restored; reject unexpected or missing paths.
6. Reproduce the declared seed and confirm weak-evidence probability shift
   remains materially larger than strong-evidence shift.
7. Keep the figure readable and unclipped while avoiding optional scaffolding
   that does not improve the requested scientific comparison.
