# BrainX diagnosis: prior bias under ambiguous evidence

## Evidence studied

Generated and archived artifacts:

- `prior_bias_decision.py`
- `README.md`
- `prior_bias_results.png`
- `__pycache__/prior_bias_decision.cpython-312.pyc`
- `agent-final.md`, `codex-events.jsonl`, `codex-stderr.log`, and
  `harness-metadata.txt`

Execution and output checks:

- Ran an unchanged copy with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`.
- Reproduced both psychometric curves, an ambiguous-evidence mean absolute
  shift of `0.245`, a strong-evidence shift of `0.002`, and a valid PNG.
- Confirmed `py_compile` passes.
- Inspected the archived `2259 x 1153` PNG at original resolution. Labels,
  trajectories, probabilities, and timing annotations are visible and
  unclipped.

Owning skills and routed references:

- `skills/brainx-general-guard/SKILL.md`
- `skills/brainpy-state/SKILL.md`
- `skills/brainpy-state/references/component-selection.md`
- `skills/brainpy-state/references/projection-patterns.md`
- `skills/brainstate/SKILL.md`
- `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`
- `skills/brainstate/references/brainstate/transformation-jit-expansion.md`
- `skills/brainstate/references/brainstate/randomness-and-reproducibility.md`
- `skills/brainunit/SKILL.md`

Closest executable examples:

- `skills/brainpy-state/references/scripts/103_COBA_2005.py`
- `skills/brainpy-state/references/scripts/sound_localization.py`

Authoritative API pages:

- `brainpy.state.LIFRef`
- `brainpy.state.AlignPostProj`
- `brainpy.state.align_post_projection`
- `brainpy.state.Expon`
- `brainpy.state.CUBA`
- `brainstate.transform.vmap2`

The generated event log confirms that the run read the four root skills plus
the BrainPy component-selection and BrainState transform references. It did
not open `projection-patterns.md` or a recurrent-network script before
implementing recurrent excitation and mutual inhibition.

## Executive diagnosis

The artifact runs, is reproducible, uses the prompt-requested BrainPy-State
point-neuron path, and supports the qualitative conclusion. BrainUnit
quantities remain attached through membrane, current, rate, and time
calculations. BrainState correctly owns per-lane State, independent mapped
randomness, the transformed time loop, and one stable compiled rollout. The
speed panel measures first-call and steady execution of that same callable and
blocks asynchronous work before stopping each timer.

The main scientific and API problem is the recurrent circuit itself. The two
LIF populations are not connected through BrainPy synapses or projections.
Instead, the code reduces each population's spikes to one smoothed scalar rate
and feeds hand-written self-excitation and cross-inhibition currents back to
every neuron. This is a hybrid point-neuron/mean-field model, not the explicit
recurrent spiking circuit claimed by the module and README.

Choose one scale deliberately. For a point-neuron circuit, represent recurrent
excitation and mutual inhibition with BrainPy projections and explicit
communication, synaptic dynamics, output, and postsynaptic roles. If scalar
population rates and rate-to-rate coupling are the intended mechanism, use a
BrainMass decision or population model, or declare a genuine multiscale model
and open both package skills.

The remaining scientific limitation is uncertainty. The figure presents one
fixed-seed estimate from 128 trials per condition without confidence intervals
or a multi-seed robustness check. The observed effect is large and the result
is valid as a descriptive simulation, but the plot does not quantify sampling
uncertainty for the probability curves or their difference.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `prior_bias_decision.py:39` through `prior_bias_decision.py:99` | The artifact claims recurrent excitation and mutual inhibition between point-neuron populations but implements only scalar population-rate feedback in custom `HiddenState`. | There is no point-neuron connectivity, synaptic State, transmission rule, or projection update order. The model's represented scale is ambiguous and bypasses the main BrainPy-State network abstractions. | For a spiking circuit, construct A-to-A and B-to-B excitatory projections plus A-to-B and B-to-A inhibitory projections through `comm`, `syn`, `out`, and `post`. If mean-field coupling is intentional, route the aggregate mechanism to BrainMass or explicitly compose both scales. |
| P2 | `prior_bias_decision.py:112` through `prior_bias_decision.py:136` and the probability panel | One fixed seed and 128 trials produce point estimates without intervals or a repeated-seed robustness check. | The page cannot distinguish a stable bias effect from binomial or seed-specific variation, even though the observed contrast is large. | Add Wilson/binomial intervals for each probability and a paired or independent interval consistent with the noise design, or verify the qualitative contrast across several declared seeds. Keep this host-side because no BrainX API owns these statistics. |
| P3 | `prior_bias_decision.py:151` through `prior_bias_decision.py:166` | The first eight zero-evidence lanes are shown without a stated selection policy and the two bias panels are not paired noise realizations. | The trajectories satisfy "show several choices unfolding" but should not be interpreted as matched before/after examples. | Label them as independent sample trials, or deliberately share random draws and document a paired design. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Choose a point-neuron decision model | Two `brainpy.state.LIFRef` populations | `brainpy.state.LIFRef` | Correct for an explicitly point-neuron circuit; refractory LIF behavior is enough for this demonstration. | Keep if the circuit remains spiking; do not add more detailed neurons without a scientific reason. |
| Initialize membrane State | `braintools.init.Constant` plus `vmap_init_all_states` | Braintools initializer and BrainState lifecycle | Correct. Each independent lane receives initialized dynamical State. | Keep. |
| Represent recurrent excitation and mutual inhibition | Custom scalar rates and rate-to-current equations | BrainPy projections and synaptic components, or BrainMass at aggregate scale | Bypassed and scientifically ambiguous. | Use explicit projections for the point-neuron path, or select BrainMass for a rate model. |
| Communicate spikes between populations | No connectivity object | `brainstate.nn.EventFixedProb`, a dense communication Module, or BrainEvent connectivity | Missing for a claimed recurrent spiking network. | Select communication from the intended all-to-all, sparse, or fixed-degree wiring. |
| Apply synaptic dynamics | Manual low-pass scalar rate | `brainpy.state.Expon` or another selected synapse | Missing for explicit spike-driven recurrence. | Use an exponential-family synapse when that temporal response matches the model. |
| Convert synaptic activity to current | Hand-written `rate * nA / Hz` | `brainpy.state.CUBA` or `COBA` | The units are correct, but the package abstraction is bypassed. | Choose CUBA for voltage-independent current or COBA for reversal-potential-dependent conductance and calibrate weights accordingly. |
| Represent evidence and prior | Host arrays converted once to `nA` quantities | BrainUnit quantity boundary | Correct. Evidence and prior are physical currents and remain unit-bearing in the model. | Keep. |
| Add stochastic current | `brainstate.random.randn` inside the mapped step | BrainState operational randomness | Correct. It supplies per-population and per-neuron current noise under State-aware transforms. | Keep; name the independent-condition policy explicitly. |
| Reproduce the experiment | `brainstate.random.seed(SEED)` | BrainState seeding | Correct. Independent complete runs reproduce the archived probabilities. | Add a multi-seed robustness check if the conclusion is inferential. |
| Map evidence, prior, and trial lanes | `vmap_init_all_states` plus `vmap2` with semantic State filters | BrainState State-aware mapping | Correct. Writable `HiddenState` and `ShortTermState` are mapped; parameters remain shared; unexpected writes raise. | Keep. |
| Advance simulation time | One `for_loop` over a unit-aware time axis with `t` context | BrainState environment and control flow | Correct. No Python timestep loop is used. | Keep. |
| Compile and benchmark | One `compiled_rollout = brainstate.transform.jit(rollout)` reused for first and steady calls | BrainState JIT | Correct and materially improved. Every measured result is blocked. | Keep the exact callable and shape stable. |
| Reset independent rollouts | `vmap_init_all_states` inside `rollout` | BrainState lifecycle | Correct for identical initialization cost in first and steady measurements. | Keep, and state whether reset time is intentionally included in throughput. |
| Decode choices | Mean A/B rates over the final 50 ms and a strict comparison | Host scientific rule | Valid if forced binary final-window dominance is the declared rule. | State that ties fall to B or add an undecided margin if scientifically required. |
| Estimate probabilities and effect sizes | NumPy reductions | Host statistics | Correct boundary; no dedicated BrainX psychometric API applies. | Add uncertainty without inventing a BrainTools API. |
| Validate output | Shape, finiteness, probability bounds, effect ordering, and strong-evidence checks | Host validation | Useful and deterministic. | Add another seed or a reduced deterministic mechanism check so tuning and validation do not rely on the same realization alone. |
| Plot trajectories, probabilities, and speed | Matplotlib multi-panel figure | Host visualization boundary | Appropriate; the custom composition is not replaced by a single BrainX plot API. | Add uncertainty marks and clarify whether trajectory panels are paired. |
| Document and save the artifact | README and PNG | Host filesystem/reporting boundary | Correct. | Keep. |

## Missing, bypassed, or misused BrainX APIs

### `brainpy.state.AlignPostProj` or `align_post_projection`

Use these to replace the hand-written recurrent-current path when the model is
intended to be a point-neuron network. The official API separates
communication, postsynaptic synaptic State, current conversion, and receiving
population. The function form also owns spike generation. Four projections
express same-choice excitation and cross-choice inhibition without custom
population-rate State.

Do not add projections if the scientific model is intentionally a scalar
population-rate model. In that case, select BrainMass instead of wrapping a
mean-field mechanism around LIF populations.

### `brainstate.nn.EventFixedProb` or another communication Module

Use a communication Module inside each projection when spikes travel through a
defined recurrent graph. Choose dense, probabilistic sparse, or fixed-degree
connectivity from the scientific wiring and performance requirement. Do not
invent connectivity merely to satisfy an API checklist.

### `brainpy.state.Expon`

Use an exponential synapse when recurrent spikes should create a decaying
postsynaptic signal. The official API stores conductance in `HiddenState` and
supports AlignPost. It replaces the artifact's scalar rate smoother only when
an exponential spike-response mechanism is intended.

### `brainpy.state.CUBA` or `COBA`

Use CUBA for voltage-independent current and COBA when reversal-potential and
postsynaptic-voltage dependence matter. This decision fixes weight units,
signs, and operating scale. It is not a cosmetic substitution.

### BrainMass decision models

`brainmass.WongWangStep` is the higher-level alternative when the modeled
variables are competing aggregate population gates or rates. It is not an
automatic replacement for an explicitly requested point-neuron circuit.

No BrainX API should replace the experiment-specific probability difference,
uncertainty calculation, timing with `perf_counter`, PNG saving, or README.

## Performance and code simplicity

The execution structure is sound:

- one complete step is mapped across 2,304 independent lanes;
- one transformed loop owns all 900 time steps;
- one stable JIT callable owns reset plus rollout;
- asynchronous outputs are blocked for every timed call;
- three steady measurements are reduced with a host median.

The reported throughput includes State reset and simulation but excludes model
construction, plot rendering, and file output. That boundary is reasonable and
should be stated next to the chart if throughput is compared with another
implementation.

The custom rate feedback adds two State objects and several equations while
bypassing the package's named recurrent-network composition. Replacing it with
native projections improves semantic clarity and API coverage, though it may
change performance and the psychometric operating point. Recalibrate weights
and validate behavior after that scientific change; do not preserve the old
numbers as an artificial regression target.

The plotting code is longer than the model orchestration because the requested
output combines trajectories, psychometrics, and timing. Its layout is readable
and this is a legitimate custom-presentation boundary.

## Skill improvements

### `brainx-general-guard`

Add an implementation-scale invariant: do not silently introduce aggregate
rate State into a point-neuron model or point neurons into an aggregate model
for convenience. When both mechanisms are scientifically represented, treat
the task as multiscale and open both owning skills.

### `brainpy-state`

Strengthen the recurrent-network decision boundary beside the projection
workflow. When recurrent excitation, mutual inhibition, or spike-driven
connectivity is named in a point-neuron task, require explicit BrainPy
projections and route to `references/projection-patterns.md` before coding.
Allow a hand-written population-rate feedback State only when the user
explicitly requests a hybrid/mean-field coupling and the BrainMass boundary is
handled.

### `component-selection.md`

Add the same boundary immediately after the selection order: a recurrent
point-neuron mechanism must complete the synapse, output, communication, and
projection decisions. A scalar mean-spike feedback variable is a different
modeling scale, not a shorthand for those decisions.

### `brainstate` and `brainunit`

No change is justified. Their current mapped-State, randomness, whole-rollout
JIT, environment, unit-aware reduction, range, and conversion guidance was
followed correctly.

### `plan.md`

Synchronize the general guard and BrainPy sections with the scale-consistency
and recurrent-projection rules. Add the hybrid mean-field bypass to BrainPy's
common failures.

## Checks for the next run

1. Confirm the agent opens `projection-patterns.md` and a closely related
   recurrent-network script before implementing point-neuron recurrence.
2. Require one of two explicit paths:
   - a point-neuron circuit with same-choice excitatory and cross-choice
     inhibitory projections; or
   - an aggregate population model owned by BrainMass, with any multiscale
     composition stated explicitly.
3. Reject a point-neuron model whose only recurrent mechanism is custom scalar
   population-rate feedback.
4. Preserve unit-bearing evidence, prior, membrane, time, synaptic weights, and
   currents.
5. Preserve per-lane State mapping, independent random draws, one transformed
   time loop, one stable JIT callable, and blocked timing.
6. Assert time, lane, and choice-output shapes; reproduce one declared seed and
   show a different seed changes stochastic trajectories.
7. Show uncertainty or a multi-seed robustness check for the probability shift.
8. Inspect the final figure for readable labels, visible weak-to-strong
   evidence behavior, several trajectories, and an honest timing boundary.
