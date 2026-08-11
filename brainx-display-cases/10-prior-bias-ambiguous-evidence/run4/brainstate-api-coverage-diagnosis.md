# BrainX diagnosis: prior bias under ambiguous evidence

## Evidence studied

Generated and archived artifacts:

- `prior_bias_decision.py`
- `README.md`
- `results/prior_bias_decision.png`
- `results/summary.json`
- `agent-final.md`, `codex-events.jsonl`, `codex-stderr.log`, and
  `harness-metadata.txt`

Execution and output checks:

- Ran an unchanged copy with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`.
- Reproduced every scientific value in `results/summary.json`, including the
  full probability table, conditional decision-time medians, and threshold
  crossing fractions.
- Confirmed `py_compile` passes.
- Inspected the archived `2501 x 788` PNG at original resolution. Trajectories,
  probability curves, labels, error bars, and speed bars are readable and
  unclipped.
- Confirmed the script is 381 lines, down from 453 in the preceding checkpoint,
  while retaining explicit recurrent projections and complete requested output.

Owning skills and routed references:

- `skills/brainx-general-guard/SKILL.md`
- `skills/brainpy-state/SKILL.md`
- `skills/brainpy-state/references/component-selection.md`
- `skills/brainpy-state/references/projection-patterns.md`
- `skills/brainstate/SKILL.md`
- `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`
- `skills/brainstate/references/brainstate/transformation-jit-expansion.md`
- `skills/brainstate/references/brainstate/randomness-and-reproducibility.md`
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
- `brainstate.nn.assign_state_values`
- BrainState Collective Operations

## Executive diagnosis

The artifact is scientifically coherent, BrainX-native, reproducible, and
complete. It represents one point-neuron scale with two `LIFRef` populations,
two same-choice excitatory projections, and two cross-choice inhibitory
projections. It does not introduce aggregate rate State or require BrainMass.

The prior is now a genuine pre-evidence pulse: baseline runs for `50 ms`, the
prior acts until evidence begins at `130 ms`, and evidence then drives the two
populations oppositely for `300 ms`. The decision variable accumulates the
population spike-rate difference only after evidence begins. The declared seed
reproduces all scientific results exactly.

The probability shift is concentrated near ambiguity. Across evidence
`[-0.075, 0.0, 0.075] mA`, the mean absolute shift is `0.1171875`; across
`[-0.3, -0.15, 0.15, 0.3] mA`, it is `0.0234375`. At zero evidence,
`P(choice A)` changes from `0.515625` to `0.609375`. The strongest endpoints
are nearly saturated.

The corrected lifecycle guidance was followed directly. The artifact snapshots
all model States by absolute path and restores them with
`brainstate.nn.assign_state_values`, rejecting unexpected or missing paths. It
does not call the vmapped reset path that collapsed lane axes in the preceding
checkpoint. The global random key is also restored outside the timed region,
so steady measurements execute the same compiled stochastic rollout.

No further skill edit is justified. The remaining limitations are host-side
statistical and reporting choices already outside BrainX API ownership.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P2 | `prior_bias_decision.py:212` through `prior_bias_decision.py:227` | The plotted 95% intervals use a clipped normal approximation. At probabilities `0` and `1`, it produces zero-width intervals. | Endpoint uncertainty is understated even with only 64 trials. The probability estimates remain valid, but the bars are not reliable near the boundaries. | Use Wilson or exact binomial intervals. Keep this calculation host-side; no BrainX API owns it. |
| P3 | `results/summary.json` | The result uses one declared seed and one fixed connectivity realization. | It is a reproducible model demonstration, not a robustness estimate over random circuits. | Repeat the scientific measurement over declared seeds or connectivity realizations only when a robustness claim is required. |
| P3 | `prior_bias_decision.py:152` through `prior_bias_decision.py:155` and `README.md:32` | Median decision time is conditional on threshold crossing; at zero evidence only `12.5%` of unbiased and `42.1875%` of biased trials cross. Remaining choices use the final sign. | A reader could mistake the reported central decision-time medians for all-trial medians. | Label them as crossing-conditional wherever displayed, or report a censored/time-to-bound statistic. The JSON already exposes crossing fractions. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Select modeling scale | Point-neuron spiking circuit only | BrainPy-State | Correct. No aggregate population variable is represented. | Keep. |
| Represent choice populations | Two `brainpy.state.LIFRef` populations | `brainpy.state.LIFRef` | Correct. | Keep. |
| Represent recurrent excitation | A-to-A and B-to-B `AlignPostProj` objects | BrainPy projections | Correct. | Keep explicit. |
| Represent mutual inhibition | A-to-B and B-to-A `AlignPostProj` objects | BrainPy projections | Correct. | Keep explicit. |
| Communicate spikes | Four `brainstate.nn.EventFixedProb` modules | BrainState event communication | Correct for the selected probabilistic wiring. | Keep unless another connectivity rule is scientifically required. |
| Apply synaptic kinetics | Excitatory and inhibitory `Expon.desc` objects | BrainPy synapse dynamics | Correct for linear exponential decay and AlignPost. | Keep. |
| Convert conductance to current | Excitatory and inhibitory `COBA.desc` outputs | BrainPy synaptic output | Correct. Reversal potentials own current sign and voltage dependence. | Keep. |
| Preserve update order | Previous spikes feed projections before both neurons update | BrainPy projection lifecycle | Correct. | Keep. |
| Represent prior protocol | Unit-aware current gated before evidence onset | BrainUnit plus model logic | Correct. The prior changes pre-evidence State rather than remaining a hidden evidence offset. | Keep. |
| Represent signed evidence | Equal and opposite unit-aware currents after evidence onset | BrainUnit plus model logic | Correct. | Keep. |
| Add stochastic drive | `brainstate.random.normal` inside the mapped step | BrainState randomness | Correct. Each lane receives transformed stochastic execution. | Keep. |
| Allocate independent State lanes | `vmap_init_all_states(..., axis_size=896)` | BrainState lifecycle | Correct. | Keep. |
| Map independent conditions and trials | `vmap2` with semantic dynamical-State filters | BrainState State-aware mapping | Correct. Parameters remain shared and unexpected writes raise. | Keep. |
| Advance time | One `brainstate.transform.for_loop` | BrainState control flow | Correct. | Keep. |
| Compile execution | One stable `brainstate.transform.jit` callable | BrainState JIT | Correct. | Keep. |
| Restore exact rollout State | Direct all-State path snapshot plus `assign_state_values` | BrainState collective lifecycle | Correct and materially improved. Both mismatch collections are enforced. | Keep. |
| Restore stochastic replay | Saved global key plus `brainstate.random.set_key` | BrainState randomness | Correct for timing the same stochastic workload. | Keep outside the timed region, as implemented. |
| Decode choices | Host first-passage/final-sign rule | Host scientific rule | Correct and documented. | Clarify conditional decision-time reporting. |
| Estimate probabilities | Host NumPy reduction with half credit for exact ties | Host statistics | Correct. | Keep. |
| Estimate uncertainty | Clipped normal interval | Host statistics | Valid only away from boundaries. | Replace with Wilson or exact binomial intervals. |
| Measure speed | Blocked first call and five blocked steady calls | Host timing plus JAX synchronization | Correct. Restoration is excluded consistently. | Keep. |
| Plot requested outputs | Matplotlib trajectories, probabilities, and speed | Host visualization | Correct, concise, and readable. | Keep. |
| Save measurements and report | JSON and Markdown | Host reporting | Correct. | Keep when machine-readable results are useful. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing, bypassed, or misused in the final artifact.

`brainstate.nn.assign_state_values` correctly replaces manual State-object
assignment for exact repeated-rollout restoration. `vmap_reset_all_states` is
correctly avoided because the selected BrainPy dynamical graph has not proven
that reset preserves the mapped lane axis.

No BrainX API should replace the experiment-specific first-passage rule,
probability calculation, uncertainty interval, timing, JSON serialization, or
custom presentation.

## Performance and code simplicity

The execution path is compact and stable:

- 896 independent bias/evidence/trial lanes are mapped together;
- one transformed loop owns all 430 time steps;
- one compiled callable is reused for the first and five steady runs;
- model State and the random key are restored outside timing;
- device work is blocked before each timer stops.

The archived median compiled run is `0.723 s`, corresponding to `532.6`
condition-equivalent simulated seconds per wall second. The unchanged review
copy measured `0.749 s`, or `514.4` simulated seconds per wall second, while
reproducing every scientific value.

The script decreased from 453 to 381 lines. Necessary explicit projections,
the temporal prior protocol, first-passage analysis, JSON output, and the
requested three-panel figure explain its remaining size. It no longer contains
the failed vmapped-reset attempt or a manual `(State, value)` restoration loop.
Further code reduction is possible in plotting style, but not enough to justify
more skill guidance beyond the existing absolute-simplicity rule.

## Skill improvements

No further skill improvement is justified by run4.

- `brainx-general-guard` correctly kept the model at one point-neuron scale.
- `brainpy-state` correctly forced explicit projections for recurrence and
  routed the agent to the projection workflow.
- `brainstate` correctly supplied mapped State, transformed control flow, JIT,
  randomness, and exact path-keyed restoration.
- `brainunit` correctly preserved physical quantities through model execution.
- `plan.md` remains synchronized with the implemented skill boundaries.

Per the refinement stopping rule, do not mine earlier checkpoints for another
edit after the latest run produces no material skill defect.

## Checks for completion

1. Explicit same-choice excitation and cross-choice inhibition are present.
2. Projections run before neuron integration.
3. Prior and evidence have distinct unit-aware temporal phases.
4. Mapped dynamical State remains lane-owned; exact restoration uses
   `assign_state_values` and rejects path mismatches.
5. One transformed time loop and one stable JIT callable own execution.
6. The declared seed reproduces every scientific value.
7. Weak-evidence probability shift materially exceeds strong-evidence shift.
8. The figure is readable, unclipped, and contains trajectories, choice
   probabilities, and measured speed.
9. The unchanged script runs successfully and passes `py_compile`.
