# Fitting and objectives

Use this reference when choosing the `Fitter` interface, defining a scalar target, selecting a built-in objective, configuring a backend or search space, interpreting `FitResult`, or diagnosing a fit that optimizes the wrong quantity.

## Choose the fitting interface

| Interface | Use when | Loss ownership |
|---|---|---|
| `Fitter(..., loss_fn=loss_fn)` | The model must be reduced to a derived scalar such as amplitude, spectral peak, or a custom statistic. | `loss_fn(model) -> (scalar_loss, aux)` owns the entire loss, including regularization. |
| `Fitter(..., predict=predict, objective=objective)` | A trajectory or observation is compared directly with a target through a reusable objective. | `Fitter` evaluates `objective(predict(model)[transient:], target) + model.reg_loss()`. |

Do not supply both `loss_fn` and `predict`. Use the derived-scalar route for phase-degenerate oscillators rather than pointwise time-series RMSE.

## Configure `Fitter`

| API | Description |
|---|---|
| `brainstate.nn.Param(value, t=..., fit=True)` | Mark one parameter trainable; a transform keeps the physical value constrained while optimization occurs in its stored space. |
| `brainmass.Fitter(model, optimizer=None, *, loss_fn=None, objective=None, predict=None, backend="grad", callbacks=None, transient=None, search_space=None)` | Construct one fitting problem and discover the model's trainable `ParamState` values. |
| `Fitter.fit(target=None, n_steps=100, verbose=False)` | Execute optimizer steps, generations, or restarts according to the backend and return the best-seen `FitResult`. |
| `Fitter.optimizer` | Inspect the active optimizer; derivative-free backends create it lazily during `fit()`. |

`transient` on `Fitter` is a leading sample count applied to the prediction path. It is distinct from the unit-aware duration or step-count transient accepted by `Simulator.run()`.

## Use a reusable objective

Every objective builder returns a JIT-, gradient-, and vmap-compatible callable over `(prediction, target)`.

| API | Description |
|---|---|
| `brainmass.objectives.timeseries_rmse()` | Use for aligned time series; subtraction remains unit-checked before reduction. |
| `brainmass.objectives.fc_corr(as_loss=True)` | Use to minimize `1 - correlation` between static FC matrices derived from trajectories. |
| `brainmass.objectives.fc_rmse()` | Use to minimize RMSE between static FC matrices. |
| `brainmass.objectives.cosine_sim(as_loss=True, epsilon=...)` | Use for scale-insensitive flattened trajectory comparison. |
| `brainmass.objectives.fcd(window_size=..., step_size=..., as_loss=True)` | Use to compare FCD matrices when matrix correlation is the intended target. |
| `brainmass.objectives.fcd_wasserstein(...)` | Use for a smooth FCD-distribution distance in gradient-based fitting. |
| `brainmass.objectives.fcd_ks(...)` | Use for the literature-standard non-smooth KS FCD-distribution distance in evaluation or gradient-free fitting. |
| `brainmass.objectives.combine((weight, objective), ...)` | Use to sum explicitly weighted objective callables into one scalar loss. |

Scores such as `fc_corr`, `cosine_sim`, and `fcd` maximize by default. Pass `as_loss=True` before handing them to a minimizer.

For FCD distribution fitting, compare the off-diagonal distribution rather than the raw FCD matrix unless matrix correspondence is scientifically intended. Constant zero-variance signals make KDE-based distribution objectives singular and can return `nan`.

## Canonical `predict` plus `objective` workflow

This minimal workflow isolates the interface: `predict` runs the model, the objective compares its result with the target, and `Fitter` adds registered regularization.

```python
import brainmass
import brainstate
import brainunit as u
import braintools
import jax.numpy as jnp
from brainstate.nn import Param

class Gain(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = Param(0.0, fit=True)

    def update(self):
        return self.gain.value()

def predict(model):
    output = brainmass.Simulator(model, dt=0.1 * u.ms).run(
        1.0 * u.ms,
        monitors=None,
    )["output"]
    return jnp.mean(output)

fitter = brainmass.Fitter(
    Gain(),
    braintools.optim.Adam(lr=0.2),
    predict=predict,
    objective=brainmass.objectives.timeseries_rmse(),
)
result = fitter.fit(target=jnp.asarray(2.0), n_steps=60)

assert result.backend == "grad"
assert result.best_loss <= result.history[0]
assert "gain" in result.best_params
```

For scientific use, make `predict` return the observation space that matches the target and validate it before optimization.

## Choose a backend

| Backend | Optimizer argument | Use when | `n_steps` means |
|---|---|---|---|
| `"grad"` | A `braintools.optim` optimizer, normally `Adam` first | The model and objective are differentiable; use this default whenever gradients are usable. | Gradient updates |
| `"nevergrad"` | Options dict or method string | The objective is discrete, black-box, noisy, rugged, or otherwise lacks a reliable gradient. | Generations |
| `"scipy"` | Options dict or method string | A few parameters suit a classical local, bounded, or derivative-free SciPy method. | Random restarts |

Gradient-free backends need a finite box for every fitted parameter. Derive it from a finite parameter transform such as `SigmoidT(low, high)` or pass `search_space={name: (low, high)}`.

Open `scripts/gradient-free-fitting.py` for the complete shared-objective workflow with Nevergrad differential evolution and SciPy Nelder-Mead. Do not implement gradient-free fitting as only a `backend=` substitution because bounds and optimizer arguments also change.

Open `scripts/eeg-fitting-with-gradients.py` when the task needs a complete Jansen-Rit workflow that constructs a phase-invariant EEG target, fits the connectivity parameter with gradients, and compares the recovered trace.

Open `brainstate/parameter-constraints-regularization.md` when a fitted
parameter needs a valid-domain transform, penalty, or modeling prior; that
reference routes to the exhaustive transform and regularizer catalog. Open
`braintools/parameter-initializer.md` when the starting value needs a reusable
initialization policy. Open `braintools/metric.md` only when a custom statistic
is not already owned by `brainmass.objectives`. Open
`braintools/optimizer.md` when choosing a gradient optimizer, schedule, or
standalone optimizer family beyond the canonical `Adam` example.

## Interpret `FitResult`

| Field | Meaning |
|---|---|
| `backend` | Backend that produced the result. |
| `best_loss` | Lowest scalar loss observed. |
| `best_params` | Mapping of fitted parameter names to best constrained physical values. |
| `history` | Per-step loss for gradient fitting, per-candidate loss for Nevergrad, or best loss per SciPy restart. |
| `n_steps` | Iterations actually run, including early stopping. |
| `prediction` | Best prediction for the `predict` plus `objective` path; `None` for a custom `loss_fn`. |
| `optimizer` | Underlying optimizer object. |
| `raw` | Backend-specific result, such as SciPy's result object. |
| `model` | Fitted model restored to the best-seen parameters. |

Use `best_loss <= history[0]` only as a minimal optimization check. Also validate recovered parameter plausibility, held-out observables, units, stability, and identifiability.

## Callbacks and regularization

Callbacks receive `{"step", "loss", "best_loss", "model"}` once per gradient step. Return `True` to stop early; this documented early-stop behavior belongs to the gradient backend.

Use the `predict` plus `objective` path when model `reg_loss()` should be included automatically. When using `loss_fn`, add any intended regularization yourself.

## Diagnose fitting failures

- No trainables: confirm every intended parameter is `Param(..., fit=True)` and fixed parameters remain plain.
- Flat or zero gradient: validate the observable, scalar reduction, State reset, and differentiable boundaries before changing optimizers.
- Oscillatory loss: replace pointwise time-series error with amplitude, spectrum, FC, FCD, or another phase-appropriate summary.
- Unit mismatch: keep units through physical subtraction; strip magnitude only for scale-free metrics.
- `nan` FCD loss: inspect constant signals, window length, sample count, and KDE degeneracy.
- Derivative-free construction error: provide finite transform-derived or explicit bounds.
- Good training loss but poor science: evaluate held-out observables and parameter identifiability; a low scalar loss does not validate the model.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/orchestration.html`
- `https://brainx.chaobrain.com/brainmass/tutorials/06_fitting_with_gradients.html`
- `https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html`
- `https://brainx.chaobrain.com/brainmass/howto/custom_objective.html`
