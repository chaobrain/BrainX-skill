# BrainMass API coverage diagnosis

## Scope

This diagnosis compares the pasted Wong-Wang prior-bias experiment with the
current BrainX skills, task-relevant repository examples, and official BrainX
API documentation.

The program explicitly models aggregate competing populations, so BrainMass is
the owning modeling skill. BrainState owns State, randomness, environment, and
transformed execution. BrainUnit owns time/current quantities and raw-value
extraction. BrainTools is relevant only where it has a named analysis,
visualization, input, or task API; it does not own this experiment's Wilson
interval, matched-condition comparison, example selection, or JSON assembly.

Sources studied:

- `skills/brainmass/SKILL.md`
- `skills/brainmass/references/scripts/wong-wang-decision-making.py`
- `skills/brainmass/references/scripts/wilson-cowan-ei-dynamics.py`
- `skills/brainmass/references/noiseprocesses.md`
- `skills/brainmass/references/batch-transform-acceleration.md`
- `skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`
- `skills/brainmass/references/visualization-analysis-api.md`
- `skills/brainmass/references/modellibrary.md`
- `skills/brainstate/SKILL.md` and its randomness, control-flow, and vectorization
  routes
- `skills/brainunit/SKILL.md` and its conversion route
- Official BrainMass `Simulator`, `WongWangStep`, orchestration, batch/accelerate,
  parameter-sweep, and decision-making pages

The pasted source is
`/Users/nijiachen/.codex/attachments/a42720c8-9152-4991-8239-6ca8cb0544b1/pasted-text.txt`.

## Executive diagnosis

The scientific model selection and stochastic setup are BrainX-native and
correctly use `WongWangStep`, `GaussianNoise`, BrainUnit quantities,
`brainstate.random.seed`, State-aware control flow, and asynchronous-result
blocking. The main missed abstraction is `brainmass.Simulator.run()`: the code
manually reconstructs initialization, duration-to-step conversion, environment
scoping, transformed looping, monitoring, and downsampling even though
`Simulator.run()` owns all six operations and accepts the required constant
coherence input.

The performance benchmark has a separate, more serious problem. It creates the
loop body inside every timed call and invokes `for_loop` without first creating
one stable outer `brainstate.transform.jit` callable. Local checks showed that
the reported "steady" measurements include repeated transformation/compilation
overhead. The benchmark should keep a custom loop, because it measures
compilation versus steady execution, but it must construct and JIT that rollout
once outside the timed repetitions.

No BrainX API replaces the Wilson interval, paired scientific contrasts,
example-selection policy, JSON serialization, template substitution, or custom
interactive HTML. Those parts should remain ordinary Python/NumPy host logic.

## BrainX ownership map

| Current responsibility | Current implementation | BrainX coverage | Diagnosis |
|---|---|---|---|
| Select a two-choice population model | `brainmass.WongWangStep` | `WongWangStep` | Correct. Do not use `WongWangExcInhStep`; that model owns resting-state E/I dynamics, not the reduced two-choice circuit. |
| Add independent current noise | Two `brainmass.GaussianNoise` objects with `sigma=0.02 * u.nA` | `GaussianNoise` / `WhiteNoise` | Correct. The sigma unit matches the current input. |
| Reproduce matched noise across bias conditions | Reseed each `run_condition()` with the same seed | `brainstate.random.seed` | Correct common-random-number design. Preserve the same shape and seed policy when exact pairing matters. |
| Represent coherence and prior bias | Dimensionless NumPy/JAX arrays | Documented `WongWangStep.update(coherence=...)` input | Correct. Coherence is dimensionless and constrained to `[-1, 1]`; do not attach a physical unit or replace simple model-input arrays merely to avoid JAX. |
| Represent duration | Bare `DURATION_MS = 800.0` plus manual conversion | BrainUnit `Quantity` | Replace with `DURATION = 800.0 * u.ms`. |
| Convert duration to steps | `N_STEPS = int(DURATION_MS / ...)` | `Simulator.run(duration)` | Remove. `Simulator` computes `int(duration / dt)` and warns when duration is not an integer multiple. |
| Initialize model State | `brainstate.nn.init_all_states(model)` | `Simulator.run(init_states=True)` | Remove on the standard run path. Keep explicit initialization only in a genuinely custom transformed rollout. |
| Set active `dt` | `brainstate.environ.context(dt=DT)` | `Simulator(model, dt=DT)` | Remove from the standard run path. `Simulator` scopes `dt`, `i`, and `t` during the rollout. |
| Set persistent global `dt` | `brainstate.environ.set(dt=DT)` in `build_data()` | Explicit `Simulator(..., dt=DT)` and benchmark context | Remove for this standalone model once both paths scope `dt` themselves. Keep a persistent value when an API such as a delayed `Network` reads it during construction. |
| Supply constant coherence | Closure calls `model.update(coherence=drive)` | `Simulator.run(inputs=...)` | Use `inputs=lambda i, t: drive`. Do not pass the one-dimensional drive directly: an array input is interpreted as time-major `(n_steps, ...)`. |
| Execute time steps | `brainstate.transform.for_loop` | `Simulator.run(jit=True)` | Replace in `run_condition`. `Simulator` composes State-aware JIT and `for_loop`. |
| Record selected decision-variable trajectories | Return `difference[example_pool]` from the step | Callable/dict `monitors` | Use `monitors=lambda m: ...`; this keeps the custom scientific observable without owning the loop. |
| Downsample trajectories | `trajectory[::sample_every]` after the run | `sample_every=` | Use `Simulator.run(sample_every=50)`. Its sampled `ts` matches post-update samples at 0.1, 5.1, 10.1 ms, and so on. |
| Represent evidence conditions and trials | Flatten both into `in_size=11 * 256` | `in_size` plus `batch_size` | Prefer `in_size=11` decision units and `batch_size=256` independent realizations. Keep the flat layout only when preserving an existing random-stream/output baseline is more important than semantic axes. |
| Obtain sample times | Reconstruct with `np.arange` and `DT.mantissa` | `result["ts"]` | Use the unit-aware time axis returned by `Simulator`; convert only for JSON. |
| Force a final choice | `(S1 - S2) > 0` | No equivalent with identical semantics | Keep if the scientific rule is final gating dominance. `get_decision()` is related but not equivalent. |
| Threshold a decision and retain undecided trials | Not used | `WongWangStep.get_decision(threshold=15 * u.Hz)` | Use only when firing-rate threshold crossing is the intended decision definition. It returns `1`, `-1`, or `0`. |
| Compute psychometric probabilities | Boolean means and sums | No dedicated BrainX psychometric API | Keep. BrainTools classification losses and cognitive-task generators are not psychometric estimators. |
| Compute Wilson intervals | Custom `wilson_interval()` | No BrainX replacement found | Keep. This is small, auditable domain statistics. |
| Select representative paired traces | `select_examples()` | No BrainX replacement | Keep. This is presentation policy, not simulation infrastructure. |
| Benchmark asynchronous JAX work | `perf_counter` plus `jax.block_until_ready` | Documented BrainMass/JAX timing pattern | Keep both, but create one stable compiled rollout before timing. |
| Report device/platform | `jax.devices()` | JAX interoperability boundary | Keep. BrainX does not need to wrap backend reporting. |
| Convert quantities for JSON | `.to(unit).mantissa` | `Quantity.to_decimal(unit)` | Replace with `to_decimal`; it combines validated conversion and unit removal at the explicit serialization boundary. |
| Serialize JSON and fill an HTML template | `json`, `pathlib`, string replacement | No BrainX replacement needed | Keep. `braintools.file` is for MATLAB/checkpoint I/O, not this web payload. |
| Render a custom interactive page | External HTML template | Optional `brainmass.viz` / BrainTools visualization | Keep custom HTML if interactivity and the existing page design are requirements. Use `brainmass.viz` only when a Matplotlib scientific figure is sufficient. |

## Findings

### P1: `run_condition` duplicates the standard BrainMass runner

The current function correctly uses BrainState primitives, but it owns machinery
that the owning package intentionally consolidates:

- manual step-count derivation;
- manual `init_all_states`;
- manual `environ.context(dt=...)`;
- manual State-aware `for_loop`;
- manual monitoring of a derived observable;
- manual output decimation and time reconstruction.

The official `Simulator.run` signature is:

```python
run(
    duration,
    *,
    inputs=None,
    monitors=None,
    transient=None,
    sample_every=None,
    batch_size=None,
    init_states=True,
    jit=True,
)
```

Its callable input is forwarded to `model.update`: a tuple is splatted and any
other result is passed as the single argument. Its callable or dict monitor can
record the selected `S1 - S2` values. The pasted experiment therefore does not
need a custom time loop.

Use this semantically shaped rewrite:

```python
DT = 0.1 * u.ms
DURATION = 800.0 * u.ms
N_TRIALS = 256
EXAMPLE_POOL_SIZE = 32
SAMPLE_EVERY = 50


def run_condition(bias: float, seed: int = 91):
    """Run matched-noise trials across every evidence level."""
    brainstate.random.seed(seed)
    drive = jnp.asarray(EVIDENCE + bias)
    zero_index = int(np.flatnonzero(EVIDENCE == 0.0)[0])

    model = brainmass.WongWangStep(
        in_size=EVIDENCE.size,
        noise_s1=brainmass.GaussianNoise(
            EVIDENCE.size,
            sigma=NOISE_SIGMA,
        ),
        noise_s2=brainmass.GaussianNoise(
            EVIDENCE.size,
            sigma=NOISE_SIGMA,
        ),
    )

    result = brainmass.Simulator(model, dt=DT).run(
        DURATION,
        inputs=lambda _i, _t: drive,
        monitors=lambda m: (
            m.S1.value - m.S2.value
        )[:EXAMPLE_POOL_SIZE, zero_index],
        sample_every=SAMPLE_EVERY,
        batch_size=N_TRIALS,
    )
    jax.block_until_ready(result["output"])

    # Batched State is (trial, evidence); the reporting layout is
    # (evidence, trial).
    final_difference = np.asarray(model.S1.value - model.S2.value).T
    choices = final_difference > 0.0
    return choices, np.asarray(result["output"]), result["ts"]
```

This exact pattern was executed locally at the requested 800 ms, 11-condition,
256-trial scale. It completed in 1.19 seconds on this environment, returned
monitor output `(160, 32)`, returned sampled times from 0.1 through 795.1 ms,
stored final State as `(256, 11)`, and produced choices `(11, 256)` after the
transpose. The runtime is machine-specific; the shapes and post-update sampling
semantics are API invariants.

If existing generated JSON must remain numerically aligned with a historical
baseline, first keep the flattened `in_size=evidence_by_trial.size` and replace
only the runner machinery. Moving trials to `batch_size` changes array layout and
can change the exact PRNG-to-trial assignment even though the distribution and
matched-seed design remain correct.

### P1: the "steady" benchmark does not reuse one compiled rollout

`benchmark_batch()` defines `step` inside `run_once()` and directly calls
`brainstate.transform.for_loop` on every repetition. The timed repetitions do
not call one stable transformed function. This violates the repository's own
acceleration rule: warm up the exact function and shape once, then time later
calls to that same function.

A local diagnostic on this machine showed the effect:

| Width | Current first | Current reported steady | Stable-JIT steady |
|---:|---:|---:|---:|
| 1 | 0.256 s | 0.190 s | 0.026 s |
| 16 | 0.312 s | 0.286 s | 0.083 s |

These numbers are environment-specific, but the conclusion is not: the current
"steady" path includes transformation overhead and materially understates
steady throughput.

For this benchmark, do not replace the loop with ordinary `Simulator.run()` and
claim compilation reuse. The benchmark specifically needs an explicit stable
callable whose first and later calls can be timed separately. Construct it once:

```python
brainstate.random.seed(700 + batch_size)
model = brainmass.WongWangStep(
    in_size=1,
    noise_s1=brainmass.GaussianNoise(1, sigma=NOISE_SIGMA),
    noise_s2=brainmass.GaussianNoise(1, sigma=NOISE_SIGMA),
)
brainstate.nn.init_all_states(model, batch_size=batch_size)
drive = jnp.zeros((1,), dtype=jnp.float32)
steps = jnp.arange(int(DURATION / DT))


def step(_):
    model.update(coherence=drive)
    return model.S1.value[0, 0] - model.S2.value[0, 0]


def rollout():
    with brainstate.environ.context(dt=DT):
        return brainstate.transform.for_loop(step, steps)


run_once = brainstate.transform.jit(rollout)
```

Time the first `run_once()` separately, block its returned array, then call that
same `run_once` object for every steady repetition. Use `batch_size` as the
reported independent-trial count if the chart claims to measure trial batching.
If it intentionally measures one flat vector of decision units instead, retain
`in_size=batch_size` but name the axis and throughput accordingly.

### P2: the current axes work numerically but obscure the experiment

`np.repeat(EVIDENCE, N_TRIALS)` followed by
`WongWangStep(in_size=evidence_by_trial.size)` is legal: the official model calls
`in_size` the number of independent decision units. It is not a BrainX API error.
However, it merges two scientifically distinct axes:

- evidence condition;
- stochastic trial.

BrainMass already gives these axes different meanings. Use `in_size` for the
eleven independently driven decision units and `batch_size` for the 256
realizations. A fully monitored trajectory then has shape
`(time, trial, evidence)`. This removes manual `repeat`/`reshape` bookkeeping and
makes reductions over trials explicit.

Use `brainstate.transform.vmap` only when each evidence value requires a model
constructed with different parameters or another workflow that cannot be
expressed by broadcasting `Simulator.run(inputs=...)`. A constant coherence
vector does not require a mapped model constructor.

### P2: raw unit extraction uses the wrong boundary idiom

The following patterns are numerically valid but bypass the direct BrainUnit
boundary API:

```python
float(DT.to(u.ms).mantissa)
float(NOISE_SIGMA.to(u.nA).mantissa)
```

Use:

```python
dt_ms = float(DT.to_decimal(u.ms))
duration_ms = float(DURATION.to_decimal(u.ms))
noise_sigma_na = float(NOISE_SIGMA.to_decimal(u.nA))
times_ms = np.asarray(sampled_ts.to_decimal(u.ms))
```

`to_decimal(target_unit)` validates compatibility, converts to the named scale,
and removes the wrapper in one explicit operation. Do not use `.mantissa` as a
conversion operation. Once data has crossed into a JSON payload, NumPy rounding,
lists, and Python scalars are appropriate.

### P2: decision semantics must be made explicit

The current rule:

```python
choice_1 = (model.S1.value - model.S2.value) > 0.0
```

forces every trial into choice 1 or choice 2 at the final time, including trials
whose firing rates never reach a decision threshold. This matches the repository's
mirrored psychometric example, so it is not an accidental BrainX bypass.

`WongWangStep.get_decision(threshold=15 * u.Hz)` is the relevant unused API when
the intended scientific definition is thresholded firing-rate choice. It returns
`1` for population 1, `-1` for population 2, and `0` for undecided. Switching to
it changes probabilities, denominators, confidence intervals, and possibly the
example-selection groups. The skill must teach this decision boundary instead
of asserting that one rule always replaces the other.

### P2: matched-condition effects lack paired uncertainty

The same seed in both `run_condition` calls deliberately pairs noise realization
`j` in the unbiased and biased conditions. That is useful for isolating the bias
effect. The individual psychometric probabilities receive Wilson intervals, but
`zeroDelta` and `strongDelta` receive no interval and the separate Wilson
intervals do not quantify a paired difference.

Add a paired bootstrap interval or another paired binary-outcome interval when
the page interprets these deltas inferentially. This is domain statistics with no
identified BrainX wrapper; do not fabricate a BrainTools API for it. Also state
that the common-random-number design estimates a conditional paired contrast,
not two independent binomial samples.

### P3: custom host-side logic is appropriate

Keep the following code outside BrainX:

- `wilson_interval()`;
- `select_examples()` and its group priorities;
- loops that turn fixed arrays into JSON records;
- `statistics.median` over host timing floats;
- `time.perf_counter`;
- `jax.devices()` reporting;
- `json.dumps`, `Path.read_text`, `Path.write_text`, placeholder validation, and
  template replacement.

Replacing these small, explicit boundaries with generic BrainUnit/JAX operations
would not improve scientific correctness or execution. The general guard should
not pressure an agent to rewrite all NumPy or Python merely because the program
uses BrainX elsewhere.

## APIs that are relevant but not automatic replacements

| API | Use here when | Do not use here when |
|---|---|---|
| `WongWangStep.get_decision()` | A firing-rate threshold and explicit undecided state define behavior. | Final gating dominance is the declared choice rule. |
| `brainstate.transform.vmap()` | A condition changes model construction/parameters or another whole computation must be mapped. | One model can accept a broadcast vector of coherence inputs and `batch_size` owns trials. |
| `braintools.cogtask.PerceptualDecisionMaking` | Generating phase-structured stimuli and labels for training/evaluating a task-performing model. | Estimating a psychometric curve from a fixed-coherence `WongWangStep` experiment. It is a task generator, not an analysis API. |
| `brainmass.viz` | A standard Matplotlib time series, phase portrait, connectivity plot, or spectrum is the output. | The required artifact is the existing custom interactive HTML page. |
| `braintools.metric` | A documented neuroscience metric such as FC, FCD, PSD, or a supervised loss is required. | Computing Wilson binomial intervals or the experiment-specific paired bias summaries. |

## Skill improvements

### `brainx-general-guard`

1. Add the missing purpose/boundary and cross-cutting routing described in
   `plan.md`. The repository skill currently selects modeling scale but does not
   explicitly route unit-bearing time/current work to BrainUnit or State,
   randomness, initialization, and transformations to BrainState.
2. Put owning-package orchestration before raw BrainState primitives:
   `BrainMass Simulator/Network/Fitter -> BrainState transforms only when the
   package orchestrator cannot express the workflow -> raw JAX only for pure
   array/PyTree work`. The current guard's transformation section can lead an
   agent to rebuild a runner with `for_loop` even when `Simulator` owns it.
3. Change the high-level API table so the selected modeling package comes first.
   The current wording jumps from BrainUnit to BrainTools and omits APIs such as
   `brainmass.Simulator`, `Network`, `Fitter`, and `brainmass.viz` that own their
   package workflows.
4. Define legitimate generic-code boundaries: dimensionless documented JAX
   inputs, host-side statistics, serialization, device inspection, timing, and
   custom presentation logic. "Avoid raw NumPy or JAX" without these boundaries
   is too broad and encourages unnecessary wrappers.
5. After the instruction to study related scripts, require reconciliation with
   the current root canonical path. A mirrored or older example can remain
   scientifically useful while containing low-level loops superseded by a newer
   package orchestrator.
6. Fix the non-sentence-case heading `Never inspect BrainX packages in the venv.
   for knowledge, ONLY check presence` and align the repository skill with its
   more complete `plan.md` section.

Suggested guard decision rule:

> Use the highest-level API in the selected owning package that preserves the
> scientific operation. Open BrainState control flow only when that package's
> runner cannot express required inputs, monitors, State effects, or benchmark
> boundaries. Keep pure host reporting and explicit interoperability boundaries
> in ordinary Python, NumPy, or JAX.

### `brainmass`

1. Expand the root `Simulator.run` row to expose the decision-relevant controls:
   `inputs`, callable/dict `monitors`, `sample_every`, `batch_size`,
   `init_states`, and `jit`. The current row lists only monitors and transient,
   so it does not teach the shortest complete path for driven experiments.
2. Add a focused `references/simulator-input-monitor-api.md` sourced from the
   official orchestration API. Teach array versus callable input semantics,
   monitor forms, output keys, post-update time semantics, sampling, initialization,
   and the `(time, batch, in_size...)` result layout.
3. Route driven models and custom observables to that reference from the root
   simulation section. Route custom transformed loops to
   `batch-transform-acceleration.md` only after `Simulator` is shown insufficient.
4. Add the axis invariant beside stochastic batching: `in_size` is the per-trial
   model/decision-unit shape; `batch_size` is independent State realizations.
   Do not flatten conditions and trials unless a preserved external layout
   requires it.
5. Teach `WongWangStep.get_decision()` in the model-library decision route and
   contrast thresholded rate decisions with final `S1 - S2` dominance.
6. Update or supplement `references/scripts/wong-wang-decision-making.py` with a
   batched psychometric example using `Simulator(inputs=..., batch_size=...)`.
   Its current nested Python trial loops and custom `for_loop` conflict with the
   skill's declared high-level canonical path.
7. Add a stable benchmark example to `batch-transform-acceleration.md`: define
   `step` once, define the full rollout once, wrap it once with
   `brainstate.transform.jit`, warm up that exact callable, and block every timed
   result. The current bullet list states this rule but does not show enough
   mechanics to prevent the pasted bug.

### `brainunit` and `brainstate`

No new core concepts are required.

- BrainUnit already says to use `to_decimal(target_unit)` at raw boundaries and
  not substitute `.mantissa`; improve routing from the guard and BrainMass
  rather than duplicating that explanation.
- BrainState already teaches State-aware `jit`, `for_loop`, randomness, and
  environment context. The missing link is BrainMass's decision boundary between
  `Simulator` and custom transformed execution.

### `plan.md`

Keep the design source of truth synchronized with the skill changes:

1. Change the BrainMass simulation rules that currently prescribe
   `environ.context()` plus `for_loop` as the general path. State `Simulator`
   first and route context/control flow only for custom rollouts.
2. Register the proposed simulator input/monitor reference and its official
   orchestration source.
3. Add callable inputs, monitor forms, sampling, and trial-axis semantics to the
   BrainMass essential concepts.
4. Add the owning-package-orchestrator-first rule and legitimate host boundaries
   to the general guard plan.

## Verification to add with the skill changes

Use small, fast examples rather than a full 800 ms by 256-trial artifact build:

1. Run three evidence values with four trials and assert the monitored output is
   `(sampled_time, selected_trials)`.
2. Assert final `S1 - S2` State is `(trial, evidence)` and the reporting transpose
   is `(evidence, trial)`.
3. Assert `sample_every=5` returns post-update times `0.1, 0.6, 1.1, ... ms` for
   `dt=0.1 ms`.
4. Rerun the same stochastic condition after reseeding and assert replay; rerun
   with a different seed and assert it differs.
5. Test both choice definitions: gating sign always produces a binary choice,
   while a sufficiently high `get_decision` threshold can return zero.
6. For benchmark examples, inspect that one compiled callable is created outside
   the timed loop. Do not assert wall-clock speed in CI.

## Recommended implementation order

1. Fix the benchmark so currently published throughput claims become valid.
2. Replace `run_condition` infrastructure with `Simulator.run(inputs=...,
   monitors=..., sample_every=...)` while retaining the flat layout for one
   numerical-regression comparison.
3. Move trials to `batch_size` and approve the regenerated stochastic artifact.
4. Replace `.to(...).mantissa` JSON boundaries with `to_decimal(...)` and source
   sample times from `result["ts"]`.
5. Decide and document whether the page reports forced final choices or
   thresholded decisions with undecided outcomes.
6. Update the guard, BrainMass skill/reference/example, and `plan.md` together so
   the next agent selects the high-level path without losing the legitimate
   custom-statistics and custom-HTML boundaries.

## Official evidence

- BrainMass Simulator and orchestration:
  https://brainx.chaobrain.com/brainmass/reference/orchestration.html
- Generated `Simulator` API:
  https://brainx.chaobrain.com/brainmass/reference/generated/brainmass.Simulator.html
- Generated `WongWangStep` API:
  https://brainx.chaobrain.com/brainmass/reference/generated/brainmass.WongWangStep.html
- Official Wong-Wang decision example:
  https://brainx.chaobrain.com/brainmass/gallery/case_studies/decision_making.html
- Batch and acceleration workflow:
  https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
- Parameter sweeps:
  https://brainx.chaobrain.com/brainmass/howto/parameter_sweeps.html
- BrainTools cognitive-task API:
  https://brainx.chaobrain.com/braintools/apis/cogtask.html
- BrainTools metric API:
  https://brainx.chaobrain.com/braintools/apis/metric.html
