# Braintools cognitive tasks

Use this reference to choose a pre-built cognitive paradigm or construct a
phase-based task that generates time-major training trials. It covers task
composition, features, value encoders, labels, variable-duration trials, and
sampling contracts.

## Choose the task path

| Decision | Use | Important constraint |
|---|---|---|
| Standard paradigm | Instantiate a pre-built `Task` subclass | Override parameters, not its phase machinery, unless the paradigm itself changes. |
| One-off custom task | Construct `Task(phases=..., input_features=..., output_features=..., trial_init=...)` | Put all trial-specific random values in `trial_init`. |
| Reusable task family | Subclass `Task` and implement `define_features()`, `define_phases()`, and `trial_init()` | Keep feature definitions and phase construction deterministic. |
| Categorical target | Keep `output_mode='categorical'` and write `outputs={'label': ...}` | Targets have shape `(T,)` per trial. |
| Continuous or multi-channel target | Set `output_mode='vector'` and write output features by name | Targets have shape `(T, num_outputs)` per trial. |
| Variable-duration trial | Use `VariableDuration`, `If`, `Switch`, or `While` | Request `return_mask=True` and size computation with `max_trial_duration()`. |

## Core task framework

`Task` executes a phase tree against one mutable trial-level `Context`, then
exposes deterministic indexed sampling and vectorized batch sampling.

| API | Description |
|---|---|
| `Task(phases=None, input_features=None, output_features=None, trial_init=None, name=None, output_mode='categorical', seed=None, num_classes=None, dt=None, **kwargs)` | Use as the dataset-like task owner. It binds the phase tree, feature layout, output mode, seed, and time step. |
| `Context` | Use as the shared per-trial state. It carries RNG state, input/output buffers, timing, phase history, and values written by `trial_init`. |
| `Task.sample_trial(index=0, key=None)` | Use when metadata and the uncompiled trial path are needed; it returns `(X, Y, info)`. |
| `Task.sample(index)` / `task[index]` | Use for one JIT-compiled trial; it returns `(X, Y)`. |
| `Task.batch_sample(size, time_first=True, return_meta=False, start_index=0, return_mask=False)` | Use for training batches. It folds the trial index into the task seed and returns time-major arrays by default. |
| `Task.max_trial_duration(ctx=None)` | Use to obtain the static timestep upper bound for variable-length buffers. |
| `Task.get_trial_meta()` | Override when `batch_sample(..., return_meta=True)` should return task-specific metadata. |

```python
import brainunit as u
from braintools.cogtask import (
    Delay,
    Feature,
    Fixation,
    Response,
    Stimulus,
    Task,
    circular,
    concat,
    label,
)

fixation = Feature(1, "fixation")
stimulus = Feature(8, "stimulus")
choice = Feature(2, "choice")

task = Task(
    phases=concat([
        Fixation(100 * u.ms, inputs={"fixation": 1.0}),
        Stimulus(
            500 * u.ms,
            inputs={"stimulus": circular("direction")},
        ),
        Delay(500 * u.ms, inputs={"fixation": 1.0}),
        Response(
            100 * u.ms,
            outputs={"label": label("ground_truth")},
        ),
    ]),
    input_features=fixation + stimulus,
    output_features=fixation + choice,
    trial_init=lambda ctx: ctx.update(
        ground_truth=ctx.rng.choice(2),
        direction=ctx.rng.uniform(0.0, 2.0 * 3.14159),
    ),
    seed=0,
)

X, Y = task.batch_sample(32)
assert X.shape[1] == 32
assert Y.shape[:2] == X.shape[:2]
```

**Invariant:** wrap a context-key label with `label("key")`. A bare string is
treated as a literal value and fails instead of reading from `Context`.

## Phases and composition

Phases own one trial epoch. Declarative phases fill named feature slices from
constants or callables with the contract `f(ctx, feature) -> array`.

| API | Description |
|---|---|
| `Phase` | Subclass only when declarative input/output dictionaries cannot express the epoch. |
| `DeclarativePhase(duration, inputs=None, outputs=None, noise=None, on_enter=None, on_exit=None, name=None)` | Use for custom named epochs. Input values may be scalar, `(feature.num,)`, or `(duration, feature.num)`. |
| `Fixation(...)`, `Stimulus(...)`, `Delay(...)`, `Response(...)`, `Cue(...)`, `Blank(...)` | Use these semantic names for ordinary epochs; they share the declarative interface. |
| `Sample(...)`, `Test(...)`, `Recall(...)`, `Match(...)`, `Comparison(...)` | Use these semantic names for working-memory epochs. |
| `Sequence(...)` / `a >> b` | Execute phases sequentially. |
| `Repeat(...)` / `a * n` | Repeat one phase or compound phase a fixed number of times. |
| `Parallel(...)` / `a \| b` | Execute phase contributions over the same interval. |
| `concat(phases)` | Build a sequence from a list. |
| `If(condition, then, else_=None, name='If')` | Select one of two branches from trial state. Packed mode uses `jax.lax.cond`. |
| `Switch(...)` | Dispatch among several phase branches. The packed selector must return a Python-hashable key, not a tracer. |
| `While(...)` | Repeat while a condition is true, bounded by `max_iterations`. The packed condition must return a Python `bool`. |
| `VariableDuration(min_duration, max_duration, ctx_key, ...)` | Reserve `max_duration` but use the per-trial duration in `ctx[ctx_key]`; `min_duration` and `max_duration` must have matching units. |

Conditional choices must be derivable from `trial_init` or an earlier phase
hook. Keep tensor outputs authoritative in compiled code; phase-history metadata
is best effort when branches are traced.

## Features, encoders, and labels

`FeatureSet` concatenates fixed-width logical channels into one flat vector and
assigns each feature a slice through `.i`.

| API | Description |
|---|---|
| `Feature(num, name=None)` | Define one fixed-width input or output channel. |
| `FeatureSet` | Collect features and manage their shifted slices. |
| `CircleFeature` | Define an angular or directional feature with a value range. |
| `feature_a + feature_b` / `feature_a \| feature_b` | Concatenate copied features without mutating the operands. |
| `one_hot(...)` | Encode a discrete class as a one-hot feature value. |
| `identity(...)` | Pass a context value through directly. |
| `circular(...)` | Encode a direction with cosine tuning. |
| `von_mises(...)` | Encode a direction with a circular-normal population profile. |
| `cos_sin(...)` | Encode a direction index as repeated cosine/sine pairs. |
| `scalar(...)` | Broadcast one scalar across a feature. |
| `gaussian(...)` | Encode a scalar with a Gaussian bump. |
| `ctx_value(...)` | Read a dynamic value directly from `Context`. |
| `label(value)` | Produce a categorical label specification from a literal, callable, or context key. |
| `match_label(...)` | Produce match/non-match target codes. |
| `comparison_label(...)` | Produce comparison target codes. |

Noise values in a declarative phase map feature names to sigma quantities in
`ms**0.5`. The runtime samples noise per phase and scales it by
`1 / sqrt(dt)` so variance is stable when `dt` changes.

## Pre-built tasks

Use the paradigm whose trial semantics match the experiment.

| Family | API | Purpose |
|---|---|---|
| Decision making | `PerceptualDecisionMaking(...)` | Perceptual decision making with motion coherence. |
| Decision making | `PerceptualDecisionMakingDelayResponse(...)` | Perceptual decision making with a response delay. |
| Decision making | `ContextDecisionMaking(...)` | Context-dependent multimodal decisions. |
| Decision making | `SingleContextDecisionMaking(...)` | A single-context decision task. |
| Decision making | `PulseDecisionMaking(...)` | Decisions from discrete evidence pulses. |
| Working memory | `DelayMatchSample(...)` | Delayed match-to-sample. |
| Working memory | `DualDelayMatchSample(...)` | Dual delayed match-to-sample. |
| Working memory | `DelayComparison(...)` | Compare values across a delay. |
| Working memory | `DelayMatchCategory(...)` | Match stimulus category across a delay. |
| Working memory | `DelayPairedAssociation(...)` | Delayed paired association. |
| Working memory | `GoNoGo(...)` | Go/no-go response selection. |
| Working memory | `IntervalDiscrimination(...)` | Compare or classify time intervals. |
| Working memory | `PostDecisionWager(...)` | Decision confidence with a wager option. |
| Working memory | `ReadySetGo(...)` | Timing reproduction from ready/set cues. |
| Working memory | `DelayDirectionReproduction(...)` | Reproduce a direction after a delay. |
| Working memory | `ImmediateDirectionReproduction(...)` | Reproduce a direction without a memory delay. |
| Working memory | `DelayDirectionClassification(...)` | Classify direction after a delay. |
| Working memory | `ImmediateDirectionClassification(...)` | Classify direction immediately. |
| Reasoning | `HierarchicalReasoning(...)` | Integrate hierarchical cues or rules. |
| Reasoning | `ProbabilisticReasoning(...)` | Make decisions under probabilistic evidence. |
| Motor | `AntiReach(...)` | Produce an anti-reach or anti-saccade response. |
| Motor | `Reaching1D(...)` | Produce a one-dimensional reaching output. |
| Motor | `EvidenceAccumulation(...)` | Produce a motor response from accumulated evidence. |

## Sampling, masks, and reproducibility

For one fixed-length trial, `X` has shape `(T, num_inputs)` and categorical `Y`
has shape `(T,)`. `batch_sample(B)` returns `(T, B, ...)` unless
`time_first=False`.

For a variable-length task:

- `max_trial_duration()` returns a Python `int` suitable for a static buffer
  dimension.
- `batch_sample(..., return_mask=True)` returns a Boolean mask in the same
  time/batch layout as `X` and `Y`.
- Trailing buffer positions stay zero and have a false mask.
- Every traced `step_count(ctx)` must satisfy
  `0 <= step_count(ctx) <= max_steps(ctx)`.

A task constructed with `seed=N` derives trial `i` with
`jax.random.fold_in(PRNGKey(N), i)`. Reusing the same index is deterministic;
advance `start_index` between batches to avoid overlapping trials. An explicit
`dt=` overrides the environment; otherwise the task reads
`brainstate.environ.get_dt()`.

## Official source

- `https://brainx.chaobrain.com/braintools/apis/cogtask.html`
