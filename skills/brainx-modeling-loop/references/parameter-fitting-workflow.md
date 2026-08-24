# BrainX parameter-fitting workflow

## Purpose and boundary

Use this reference after `brainx-general-guard` selects the represented biological scales and their owning model skills. Keep model construction, simulation, physical units, dynamical State, observation, and biological meaning inside the active BrainX route while estimating unknown parameters from observed data.

Use this reference for the `parameter-fitting` part of `brainx-modeling-loop` steps 2-5. Keep task-training objectives, data splits, and State lifecycles separate during a `hybrid` workflow. Route parameter sweeps without observed targets to the active model skill instead.

## Underlying mental model

A parameter map represents the scientific unknowns. It binds every optimizer coordinate to one named BrainX value, physical unit, valid domain, transform, and reporting meaning.

The BrainX simulator represents the generative model. Fitted parameter State persists while membrane, channel, synaptic, delay, noise, monitor, and other runtime State follow an explicit reset policy for every independent candidate and protocol.

The observation model represents measurement. It maps latent BrainX output into the same signal, axes, time base, preprocessing, missingness, and units as the observed data before the objective is evaluated.

The objective represents the fitting claim. It defines which discrepancy is minimized, how components are weighted and reduced, and which observations are reserved for validation.

Parameter recovery represents the interpretation gate. A fitted value may support a mechanistic claim only when the exact fitting pipeline can recover that parameter under the actual observation protocol.

## API structure overview

| Need | Required route |
|---|---|
| Ion, channel, compartment, morphology, or conductance fitting | Open `braincell` for the model and `brainstate` for custom State-aware differentiation. Recreate or reset all cell runtime State for every independent candidate. |
| Point-neuron, synapse, or spiking-network fitting | Open `brainpy-state`. Route custom gradients through `brainstate`; open the BrainPy-State surrogate reference only when gradients must cross a discrete spike operation. |
| Aggregate population, region, or whole-brain fitting | Open `brainmass`. Prefer `brainmass.Fitter` and its routed fitting reference before building a custom optimizer loop. |
| Physical parameters or observations | Open `brainunit`. Preserve quantities through parameter application, simulation, observation, and unit-compatible comparison. |
| Constrained parameters or regularization | Open the active package's BrainState parameter reference. Record both the unconstrained optimizer value and constrained physical value. |
| BrainCell input, objective, or optimizer selection | Open `braincell/references/braintools/input-current.md`, `metric.md`, or `optimizer.md` for the operation being selected. |
| BrainPy-State or BrainMass objective or optimizer selection | Open the active package's routed Braintools metric or optimizer reference. Prefer package-owned BrainMass objectives when they express the scientific comparison. |

Study the selected package skills and their fitting-related examples before implementing. Trace construction, initialization, parameter application, runtime State reset, protocol execution, observation, and reduction end to end.

## Lock the fitting contract

Lock these decisions before inspecting observed-data fit results.

| Contract field | Required decision |
|---|---|
| Scientific target | Classify fitted, fixed, nuisance, observation, hierarchical, and initial-state parameters. State which fitted values require mechanistic interpretation. |
| Parameter domain | Record name, BrainX destination, unit, valid bounds, transform, initialization, and provenance. Do not invent biological precision. |
| Data contract | Record independence unit, axes, sampling or time base, physical units, missingness, and immutable fit, validation, and held-out partitions. |
| Protocol | Match input currents, stimuli, duration, trial count, conditions, noise, initial conditions, and reset boundaries to the observed experiment. |
| Observation model | Define filtering, mixing, forward modeling, delays, downsampling, feature extraction, normalization, and measurement noise from latent State to data. |
| Objective | Define every component, direction, weight, reduction axis, normalization, regularizer, and invalid-domain result. |
| Estimator | Lock backend, starts, seeds, budget, convergence or stopping rule, and candidate-selection rule. |
| Validation | Lock domain tests, gradient or candidate-batch checks, recovery design, predictive checks, and parameter-specific interpretation criteria. |

Use one explicit map whenever parameters cross a flat-array or optimizer boundary:

| Position | Name | BrainX destination | Role | Unit | Transform | Bounds | Reported value |
|---:|---|---|---|---|---|---|---|

Never rely on dictionary order or an unlabeled vector to preserve scientific meaning. Test encode -> decode -> encode round trips at nominal and boundary values.

## Select the fitting backend

Choose the simplest BrainX backend supported by the complete simulator-to-objective path. Check required optional dependencies before locking the estimator.

| Condition | Use | Required checks |
|---|---|---|
| The complete simulator-to-objective path is differentiable and numerically stable | `brainmass.Fitter(..., backend="grad")` or `brainstate.transform.grad` with a Braintools gradient optimizer | Finite and nonzero gradients, a small finite-difference comparison, multiple starts, exact-pipeline recovery, and held-out prediction. |
| A package-owned fitter supports the required gradient-free objective | Its bounded gradient-free backend | Finite bounds, full budget, multiple starts, landscape or sensitivity evidence, recovery, and held-out prediction. |
| One scalar bounded or constrained objective matches a documented SciPy method | `braintools.optim.ScipyOptimizer` | Parameter order and unit reconstruction, method compatibility, finite bounds, stopping behavior, multiple starts, recovery, and held-out prediction. |
| The objective is discontinuous, discrete, black-box, or benefits from population-batched candidates | `braintools.optim.NevergradOptimizer` when its optional dependency is available | One loss per candidate, candidate independence, unit-bearing bounds, total evaluation count, multiple starts, recovery, and held-out prediction. |

Prefer gradients when they are valid. Do not select derivative-free fitting merely because the BrainCell example uses Nevergrad; that example establishes a custom candidate-evaluation pattern, not a universal backend choice. If Nevergrad is unavailable, test whether `braintools.optim.ScipyOptimizer` satisfies the locked objective before changing libraries or requesting dependency installation.

Use raw SciPy or another generic optimizer only when no routed BrainTools backend provides a required estimator strategy, vectorized candidate contract, callback, constraint, or stopping rule. Before implementation, save the `brainx-general-guard` API-gap artifact naming the checked BrainTools APIs, exact missing capability, smallest external boundary, and parity evidence. A missing optional Nevergrad dependency alone is not a BrainTools capability gap.

## Apply the scale-specific pattern

| Route | Pattern to preserve |
|---|---|
| BrainCell | The HH fitting example constructs and initializes one cell rollout for each unit-bearing conductance and capacitance candidate, evaluates time-major current protocols, scores voltage with Braintools metrics, and batches derivative-free candidates. Preserve candidate independence and unit-bounded search; open the BrainCell Braintools input, metric, and optimizer references and reconcile exact channel and State APIs before adapting it. |
| BrainPy-State | The training examples initialize runtime State at each independent sequence, differentiate only the selected `ParamState` collection, transform the complete loss step, and update parameters outside the temporal loop. Reuse that lifecycle for an inverse objective; task loss itself is not parameter fitting. |
| BrainMass | The fitting examples prefer `Fitter`, mark fitted values with `brainstate.nn.Param(..., fit=True)`, and fit phase-appropriate observations such as settled amplitude rather than an unaligned oscillatory waveform. Use gradient fitting first and change backends only when the objective requires it. |
| Braintools | Gradient optimizers register trainable State and consume matching gradient trees. `ScipyOptimizer` and `NevergradOptimizer` instead own a complete objective and do not use the register/update lifecycle; Nevergrad requires one loss per candidate. |

## Construct one candidate evaluation

One candidate evaluation must apply parameters, reset independent runtime State, execute the exact protocol, build the declared observation, and return one declared scalar loss.

| API | Description |
|---|---|
| `brainstate.nn.Param(..., fit=True)` | Use for a fitted model value that needs BrainState discovery, a valid-domain transform, or regularization. Read its constrained physical value in the model. |
| `brainstate.nn.init_all_states(model, ...)` | Use at an independent rollout boundary when the selected model route requires model-wide runtime State initialization. Do not reset fitted parameter State. |
| `brainstate.transform.grad(loss_fn, grad_states=..., return_value=True)` | Use for a custom stateful differentiable objective. Select only intended parameter State and transform the complete loss evaluation. |
| `brainmass.Fitter(...)` | Use when BrainMass owns the simulator and its loss, prediction, objective, and backend interfaces express the inverse problem. |
| `braintools.input.Constant(...)` | Use when the fitted cellular protocol contains timed constant sections; generate the unit-aware time-major current under the rollout `dt`. |
| `braintools.metric.squared_error(...)` | Use for a standard waveform discrepancy; preserve time and candidate axes until the declared reduction and keep custom observation features separate. |
| `braintools.optim.ScipyOptimizer(...)` | Use when one scalar objective and a documented SciPy method satisfy the fitting contract; it owns the standalone optimization and returns the SciPy result. |
| `braintools.optim.NevergradOptimizer(...)` | Use for a custom black-box objective that accepts candidate-stacked parameters and returns one loss per candidate. Preserve unit-bearing bounds and independent runtime State. |

For a BrainCell current-clamp fit, keep custom file parsing at the host boundary but generate the inferred protocol and standard waveform loss through BrainTools:

```python
import brainstate
import braintools
import brainunit as u


with brainstate.environ.context(dt=0.1 * u.ms):
    current = braintools.input.Constant([
        (0.0 * u.pA, 50.0 * u.ms),
        (25.0 * u.pA, 250.0 * u.ms),
        (0.0 * u.pA, 200.0 * u.ms),
    ])()


def waveform_rmse_mV(predicted, observed):
    mse = braintools.metric.squared_error(
        predicted.to_decimal(u.mV),
        observed.to_decimal(u.mV),
        reduction="mean",
    )
    return u.math.sqrt(mse)
```

Keep a custom voltage-peak detector, observation transform, or domain penalty only when no routed BrainTools metric expresses the declared feature. Do not reimplement standard waveform losses merely because one objective component is custom.

Use the BrainMass path as the canonical high-level workflow:

```python
import brainmass
import brainstate
import braintools
import brainunit as u


def settled_amplitude(model):
    result = brainmass.Simulator(model, dt=0.1 * u.ms).run(
        300.0 * u.ms,
        monitors=["x"],
        transient=150.0 * u.ms,
    )
    x = u.get_magnitude(result["x"])
    x = x - u.math.mean(x)
    return u.math.sqrt(u.math.mean(x ** 2))


target = settled_amplitude(
    brainmass.HopfStep(
        in_size=1,
        a=1.5,
        w=0.3,
        beta=1.0,
        init_x=braintools.init.Constant(0.5),
        init_y=braintools.init.Constant(0.0),
    )
)
model = brainmass.HopfStep(
    in_size=1,
    a=brainstate.nn.Param(
        0.1,
        t=brainstate.nn.SigmoidT(0.05, 3.0),
        fit=True,
    ),
    w=0.3,
    beta=1.0,
    init_x=braintools.init.Constant(0.5),
    init_y=braintools.init.Constant(0.0),
)


def loss_fn(candidate):
    prediction = settled_amplitude(candidate)
    return (prediction - target) ** 2, prediction


fit = brainmass.Fitter(
    model,
    braintools.optim.Adam(lr=0.05),
    loss_fn=loss_fn,
    backend="grad",
).fit(n_steps=100)

assert fit.best_loss <= fit.history[0]
assert u.math.isfinite(fit.best_params["a"])
assert u.math.abs(fit.best_params["a"] - 1.5) < 0.1
```

This proves only that one synthetic target is mechanically recoverable. It does not establish identifiability across the parameter domain or under the observed protocol.

## Validate fitting mechanics

1. Evaluate nominal, boundary, and invalid parameter values. Verify shapes, units, finite outputs, expected failures, and constraint behavior.
2. Verify deterministic replay or characterize stochastic variation under declared seeds. Reset every State whose carryover is not part of the protocol.
3. Inspect every objective component before reduction. Confirm its direction, unit behavior, weight, and contribution at nominal values.
4. For oscillatory data, use scientifically justified phase-insensitive observations such as amplitude, spectrum, FC, or FCD when waveform phase is not aligned.
5. For gradient fitting, compare autodiff with finite differences on a small stable case and inspect zero, exploding, `NaN`, or parameter-missing gradients.
6. For derivative-free fitting, verify candidate-axis shape, one loss per candidate, unit reconstruction, bounds, invalid-candidate behavior, and total evaluation count.
7. Inspect sensitivity, tradeoffs, flat directions, discontinuities, and numerical boundaries before running the full fit.
8. When a generic optimizer, input generator, metric, or integration utility replaces BrainTools, verify the saved API-gap artifact and its unit, State, shape, and numerical parity evidence.

## Run exact-pipeline parameter recovery

Recovery must use the same simulator, protocol, observation model, preprocessing, bounds, starts, seeds, objective, backend, budget, stopping rule, and candidate-selection rule as the observed-data fit.

| Step | Action | Required evidence |
|---|---|---|
| 1. Draw truth | Sample labeled physical parameter sets across the full plausible joint domain, including nuisance variation. | Generating table with names, units, provenance, and seeds. |
| 2. Generate observations | Run the BrainX model with the real duration, stimuli, sampling, noise, missingness, and State reset policy. | Latent and observed outputs plus simulation diagnostics. |
| 3. Fit blindly | Run the unchanged fitting pipeline without exposing truth to initialization, stopping, or selection. | Every start, trace, failure, boundary hit, and selected candidate. |
| 4. Compare | Join recovered and generating values by parameter name and unit. | Bias, MAE or RMSE, recovered-versus-true association, boundary rate, and fit-failure rate. |
| 5. Inspect tradeoffs | Examine paired errors, sensitivity, profile losses, or objective contours. | Ridges, compensating parameters, flat regions, and failure regions. |
| 6. Gate interpretation | Apply criteria locked before observed fitting to each parameter separately. | `interpretable`, `weakly-identified`, or `non-identifiable-under-this-protocol`. |

Do not copy universal sample counts or correlation thresholds from another model. Choose enough generating datasets to resolve the error, tradeoff, and failure behavior required by the scientific claim, then justify and lock that count.

High recovered-versus-true correlation can coexist with large bias. Low aggregate loss can coexist with parameter tradeoffs. Report multiple diagnostics and preserve failed fits rather than dropping them.

Poor parameter recovery may still permit a predictive result. In that case, report prediction under the tested protocol and explicitly withhold parameter-level interpretation.

## Fit observed data and check predictions

Fit observed data only after recovery passes the locked parameter-specific criteria or the researcher approves a narrower predictive claim.

- Preserve every start, seed, trace, failed evaluation, boundary hit, and candidate-selection decision.
- Select candidates with the locked rule; do not report only the most favorable start or seed.
- Evaluate held-out data or protocols through the same observation and reset path used during fitting.
- Report raw predictions, residuals, parameter uncertainty or start-to-start variability, and protocol-wise metrics.
- Treat good prediction as evidence about prediction under the tested protocol, not parameter identifiability or biological validity.

## Return fitting evidence

| Artifact | Required contents |
|---|---|
| Fitting contract | Parameter map, data partitions, protocol, observation model, objective, estimator, budgets, seeds, and locked criteria. |
| Mechanics checks | Parameter round trips, State reset, unit and shape tests, nominal and boundary runs, gradient or candidate-batch checks. |
| Recovery evidence | Generating parameters, every recovered estimate, bias and error summaries, tradeoffs, boundary and failure rates, and parameter classifications. |
| Observed fit | Every run, trace, failure, selected result, constrained physical parameters, and reproducibility metadata. |
| Predictive evidence | Held-out simulations, raw observations, residuals, variability or uncertainty, and protocol-wise metrics. |
| Claim boundary | Supported parameter interpretations, prediction-only results, non-identifiable targets, model limitations, and unresolved alternatives. |

## Package routing

| Route | Open when |
|---|---|
| `parameter-fitting-workflow/scripts/fitting_hh_neuron.py` | Studying the official complete BrainCell HH derivative-free fitting composition: custom channels, unit-bearing bounds, per-candidate cell initialization, time-major rollout, voltage loss, candidate `vmap`, and Nevergrad. Also open the three BrainCell Braintools references below; supply the upstream trace CSVs and reconcile source-version APIs before adapting it. |
| `braincell` -> `references/braintools/input-current.md` | Generating timed current sections, pulses, waveforms, or stochastic stimulation for a cellular fit. |
| `braincell` -> `references/braintools/metric.md` | Selecting standard waveform losses or neuroscience metrics while separating custom observation features. |
| `braincell` -> `references/braintools/optimizer.md` | Selecting gradient optimizers, `ScipyOptimizer`, or `NevergradOptimizer` and checking their lifecycle, unit, and dependency boundaries. |
| `brainmass` -> `references/fitting-with-objectives-api.md` | Selecting `Fitter` interfaces, objectives, callbacks, `FitResult` fields, or gradient versus gradient-free backends. |
| `brainmass` -> `references/scripts/eeg-fitting-with-gradients.py` | Studying a complete phase-insensitive EEG fitting and synthetic-recovery example. |
| `brainmass` -> `references/scripts/gradient-free-fitting.py` | A bounded non-differentiable or black-box BrainMass objective requires Nevergrad or SciPy. |
| `brainstate` -> gradient and parameter references | Differentiating custom BrainCell or BrainPy-State rollouts, selecting parameter State, or applying constraints and regularization. |
| Active model package -> Braintools optimizer reference | Selecting gradient optimizers, schedules, SciPy, Nevergrad, or their distinct lifecycle and unit contracts. |
| Active model package -> Braintools metric reference | Selecting regression, spike-train, spectral, synchronization, FC, or other observation-space metrics. |

## Boundaries and common failures

- Fitting latent State directly to measured data without the declared observation model.
- Flattening unit-bearing parameters without a tested name, order, unit, and transform map.
- Resetting fitted parameters between candidates or carrying runtime State across independent candidates.
- Treating a surrogate gradient as evidence that a fitted spiking parameter is identifiable.
- Fitting raw oscillatory waveforms when phase is not experimentally aligned.
- Selecting derivative-free fitting before checking whether gradients are valid.
- Treating an unavailable optional Nevergrad dependency as permission to bypass another suitable BrainTools optimizer.
- Replacing a BrainTools-owned optimizer, metric, input, or integrator without a recorded capability gap and parity evidence.
- Changing bounds, summaries, exclusions, weights, or recovery criteria after inspecting observed-data results.
- Running recovery with longer, cleaner, denser, or easier data than the real protocol.
- Interpreting aggregate fit quality as parameter recovery.
- Hiding failed starts, invalid simulations, or boundary solutions.

## BrainX source patterns

- BrainCell HH fitting: `parameter-fitting-workflow/scripts/fitting_hh_neuron.py`, mirrored from `https://github.com/chaobrain/braincell/blob/main/examples/single_compartment/SC01_fitting_a_hh_neuron.py`
- BrainPy-State SNN training lifecycle: `https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html`
- BrainMass gradient fitting: `https://brainx.chaobrain.com/brainmass/tutorials/06_fitting_with_gradients.html`
- BrainMass gradient-free fitting: `https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html`
