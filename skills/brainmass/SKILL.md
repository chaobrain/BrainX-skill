---
name: brainmass
description: Use for neural-mass modeling of aggregate population activity at the local-circuit, brain-region, multi-region, or whole-brain scale. Use alone for mass-model studies or together with BrainPy-State and/or BrainCell when aggregate dynamics participate in a multiscale model.
---

# BrainMass

## Purpose and boundary

Use BrainMass to translate a regional or whole-brain modeling question into:

`select a model -> construct a *Step -> add noise or coupling -> simulate -> observe -> validate or fit`

## Underlying principle of BrainMass

A neural-mass model represents the average activity of a neuronal population. It defines population-level state variables and differential equations that govern how they evolve over time.

A `*Step` implementation represents the executable one-step dynamics of that model for one or more brain regions. It stores the dynamical variables and advances their differential equations by one integration step.

A noise process represents unresolved fluctuations or stochastic drive. It is attached to a model variable and sampled inside the model update.

`Network` represents whole-brain wiring. It maps structural connectivity, transmission distance, speed, and a coupling rule into delayed inter-region input.

### API structure

| API family | Use |
|---|---|
| Neural Mass Models | Choose regional phenomenological, physiological, seizure, decision, oscillator, or whole-brain dynamics. |
| Noise Processes | Add Gaussian, white, Ornstein-Uhlenbeck, Brownian, or colored stochastic drive to model variables. |
| Coupling Mechanisms | Choose how structural connections transform regional activity into instantaneous or delayed network input. |
| Forward Models | Map latent neural activity to modality-specific BOLD, EEG, MEG, or lead-field outputs. |
| Observation Models | Apply HRF kernels, hemodynamic BOLD dynamics, or temporal averaging to simulated activity. |
| HORN Models | Build harmonic-oscillator recurrent networks for cognitive and sequence tasks. |
| Orchestration | Run and fit models through `Network`, `Simulator`, `Fitter`, `FitResult`, and objective functions. |
| Datasets | Load or register connectomes, signals, and task datasets through typed data containers. |
| Visualization | Plot trajectories, phase portraits, connectivity, functional connectivity, and spectra. |
| Utilities & Types | Discover models, inspect `ModelInfo`, compute delay indices, and use shared helpers. |

## Discover and simulate a model

Every public regional model follows the same `*Step` contract, so choose by scientific observable and dynamical regime, then let `Simulator` own initialization and the transformed time loop.

| API | Description |
|---|---|
| `brainmass.list_models()` | Use before choosing a model; it returns `ModelInfo` records with name, category, state-variable count, and use case. |
| `brainmass.<Model>Step(in_size, **params)` | Use after selecting a model family; `in_size` sets the number of regions or parallel units and parameters broadcast to that shape. |
| `brainmass.Simulator(model, dt=...)` | Use for the default, standard run path; it stores the model and integration step without changing the model equations. |
| `Simulator.run(duration, ..., monitors=..., transient=...)` | Use to initialize State, execute the compiled loop, discard a warm-up interval, and return a dict of time-major trajectories plus unit-aware `ts`. |

```python
import brainmass
import brainunit as u

models = brainmass.list_models()
hopf = next(model for model in models if model.name == "HopfStep")
assert hopf.category == "phenomenological"

node = brainmass.HopfStep(in_size=1, a=0.25, w=0.3)
result = brainmass.Simulator(node, dt=0.1 * u.ms).run(
    200.0 * u.ms,
    monitors=["x", "y"],
    transient=20.0 * u.ms,
)

assert set(result) == {"x", "y", "ts"}
assert result["x"].shape == (1800, 1)
assert result["ts"].shape == (1800,)
```

Open `references/modellibrary.md` when choosing among model families, distinguishing similarly named Wong-Wang variants, comparing state cost, or locating a model's observable. Open `references/datasets-api.md` when the workflow needs bundled or registered input data.

##  Noise and stochastic run

Noise belongs to the model, while `batch_size` asks `Simulator` to initialize and run independent State realizations in one time-major result.

| API | Description |
|---|---|
| `brainmass.OUProcess(in_size, sigma=..., tau=...)` | Use for mean-reverting temporally correlated drive; `sigma` sets output scale and `tau` sets correlation time. |
| `brainstate.random.seed(seed)` | Use before constructing or running every stochastic result that must be reproducible; it resets the BrainState random stream. |
| `Simulator.run(..., batch_size=N)` | Use for `N` independent trials; it initializes batched model and noise State and returns trajectories shaped `(time, batch, regions)`. |

```python
import brainmass
import brainstate
import brainunit as u

brainstate.random.seed(7)
node = brainmass.HopfStep(
    in_size=1,
    a=-0.05,
    w=0.3,
    noise_x=brainmass.OUProcess(
        in_size=1,
        sigma=0.1,
        tau=20.0 * u.ms,
    ),
)
trials = brainmass.Simulator(node, dt=0.1 * u.ms).run(
    100.0 * u.ms,
    monitors=["x"],
    batch_size=8,
)

assert trials["x"].shape == (1000, 8, 1)
```

Seed again before a repeated run when exact replay matters.

Open `references/noiseprocesses.md` when choosing white, OU, Brownian, or colored noise, using a noise process directly, or checking units and State behavior. Open `references/batch-transform-acceleration.md` when batching parameters, writing a custom transformed loop, or checkpointing a long differentiated rollout.

## Build a delay-coupled network

`Network` wraps an `N`-region `*Step` model with structural connectivity, optional conduction delays, and one coupling current while preserving the `Simulator` contract.

| API | Description |
|---|---|
| `brainstate.environ.set(dt=...)` | Use to establish the persistent integration step from which a delay-coupled `Network` sizes its history buffer. |
| `brainmass.datasets.load_dataset("example_connectome")` | Use for the bundled typed `Connectome`; it returns weights, unit-aware distances, and region labels. |
| `brainmass.Network(node, conn=..., distance=..., speed=..., coupling=..., coupled_var=..., k=...)` | Use to couple an `N`-region node bank; it derives `distance / speed` delays, validates the coupled State, and returns a Module driven like the original node. |
| `Simulator.run(..., monitors=lambda m: m.node.<state>.value)` | Use to record coupled node State; a callable traverses the `Network.node` boundary and stores the result under `"output"`. |

```python
import brainmass
import brainstate
import brainunit as u

dt = 0.1 * u.ms
brainstate.environ.set(dt=dt)

connectome = brainmass.datasets.load_dataset("example_connectome")
n_region = connectome.weights.shape[0]
nodes = brainmass.HopfStep(in_size=n_region, a=0.2, w=0.3)
network = brainmass.Network(
    nodes,
    conn=connectome.weights,
    distance=connectome.distances,
    speed=10.0 * u.mm / u.ms,
    coupling="diffusive",
    coupled_var="x",
    k=0.5,
)
result = brainmass.Simulator(network, dt=dt).run(
    200.0 * u.ms,
    monitors=lambda model: model.node.x.value,
    transient=20.0 * u.ms,
)

assert result["output"].shape == (1800, n_region)
```


Open `references/coupling-network-api.md` when selecting diffusive, additive, Laplacian, sigmoidal, tanh, or Jansen-Rit coupling; configuring instantaneous coupling; fitting coupling parameters; or building a direct coupling object.

## Map activity to observables

An observation model converts hidden activity to the modality actually measured, so choose its temporal and physical assumptions before defining the comparison target.

| API | Description |
|---|---|
| `brainmass.HRFBold(period=..., downsample_period=..., kernel=...)` | Use for fast differentiable BOLD fitting; it temporally averages, convolves with an HRF kernel, and decimates to the requested repetition time. |
| `brainmass.BOLDSignal(in_size=...)` | Use when four-state Balloon-Windkessel hemodynamics matter more than the simpler convolution path. |
| `brainmass.EEGLeadFieldModel(...)` | Use with a physically calibrated EEG lead field; it projects regional sources to sensor potentials while preserving units. |
| `brainmass.LeadFieldModel(...)` | Use for a generic physical lead-field projection with explicit dipole and sensor units. |
| `brainmass.MEGLeadFieldModel(...)` | Use for the MEG specialization; match lead-field, dipole, and magnetic-field units. |
| `brainmass.LeadfieldReadout(...)` | Use when a unitless lead-field matrix should be learned end to end rather than supplied as a calibrated physical operator. |

```python
import brainmass
import brainunit as u

node = brainmass.WilsonCowanStep(in_size=1)
neural = brainmass.Simulator(node, dt=1.0 * u.ms).run(
    6000.0 * u.ms,
    monitors=["rE"],
    transient=500.0 * u.ms,
)["rE"]

observer = brainmass.HRFBold(
    period=720.0 * u.ms,
    downsample_period=20.0 * u.ms,
    kernel=brainmass.GammaHRFKernel(),
)
bold = observer(u.get_magnitude(neural), dt=1.0 * u.ms)

assert bold.ndim == 2
assert bold.shape[1] == 1
```

Strip units only at an explicitly documented raw-array boundary. Use `HRFBold` for the canonical differentiable BOLD path.

Open `references/forward-observation-api.md` when choosing an HRF kernel, Balloon-Windkessel BOLD, temporal averaging, EEG/MEG lead fields, a trainable readout, or modality-specific unit handling.

## Fit trainable parameters

`Fitter` discovers `Param(fit=True)` values, evaluates one scalar objective through the model workflow, and returns the best constrained parameters without changing the scientific simulation path.

| API | Description |
|---|---|
| `brainstate.nn.Param(value, ..., fit=True)` | Use to mark exactly the model parameters that may change; an optional transform constrains their physical values. |
| `brainmass.objectives.<builder>()` | Use with `predict=` for reusable time-series, FC, FCD, cosine, or combined objectives; each builder returns a scalar callable. |
| `brainmass.Fitter(model, optimizer, loss_fn=..., backend="grad")` | Use for a custom `loss_fn(model) -> (scalar_loss, aux)`; the default gradient backend differentiates through the simulation and optimizer steps. |
| `brainmass.Fitter(model, optimizer, predict=..., objective=...)` | Use when a trajectory prediction is compared directly with a target by a reusable objective. |
| `Fitter.fit(target=None, n_steps=...)` | Use to run backend-specific iterations and return `FitResult` with best loss, parameters, history, prediction, and fitted model. |

```python
import brainmass
import brainunit as u
import braintools
import jax.numpy as jnp
import numpy as np
from brainstate.nn import Param

signal = brainmass.datasets.load_dataset("example_signal").signal[:, 0]
target_amp = jnp.asarray(
    float(np.sqrt(np.mean((signal - signal.mean()) ** 2)))
)

model = brainmass.HopfStep(
    in_size=1,
    a=Param(0.05, fit=True),
    w=0.3,
    beta=1.0,
    init_x=braintools.init.Constant(0.5),
)

def loss_fn(candidate):
    x = brainmass.Simulator(candidate, dt=0.1 * u.ms).run(
        300.0 * u.ms,
        monitors=["x"],
        transient=150.0 * u.ms,
    )["x"][:, 0]
    amplitude = jnp.sqrt(jnp.mean((x - jnp.mean(x)) ** 2))
    return (amplitude - target_amp) ** 2, amplitude

fitter = brainmass.Fitter(
    model,
    braintools.optim.Adam(lr=0.05),
    loss_fn=loss_fn,
)
fit = fitter.fit(n_steps=50)

assert fit.best_loss <= fit.history[0]
assert "a" in fit.best_params
```

Fit phase-insensitive summaries such as amplitude, spectrum, FC, or FCD when raw oscillatory traces can differ only by phase. Use `backend="grad"` whenever the workflow is differentiable.

Open `references/fitting-with-objectives-api.md` when choosing `loss_fn` versus `predict` plus `objective`, selecting built-in objectives, configuring callbacks, interpreting `FitResult`, or changing backends. Open `references/brainstate/parameter-constraints-regularization.md` when a fitted parameter needs a valid-domain transform, penalty, or prior. Open `references/braintools/optimizer.md` when choosing an optimizer or learning-rate scheduler beyond the canonical `Adam` path. Open `references/scripts/gradient-free-fitting.py` only when the objective is discrete, jagged, black-box, or has unavailable or unreliable gradients; its bounded search-space workflow is required.

## Reference routing

Choose the workflow category first, then open only the smallest reference that owns the next decision.

### Model selection and simulation

| Reference | Open when |
|---|---|
| `references/modellibrary.md` | Choosing a model family, observable, state cost, or similarly named variant. |
| `references/noiseprocesses.md` | Selecting a noise spectrum, correlation structure, State behavior, units, or direct noise workflow. |
| `references/datasets-api.md` | Loading, inspecting, generating, or registering BrainMass data containers. |

### Networks, observation, and analysis

| Reference | Open when |
|---|---|
| `references/coupling-network-api.md` | Selecting or configuring coupling, delays, connectivity conventions, direct coupling objects, or trainable network parameters. |
| `references/forward-observation-api.md` | Selecting BOLD, HRF kernels, temporal averaging, EEG/MEG lead fields, or trainable readouts. |
| `references/visualization-analysis-api.md` | Plotting or computing time-series, FC, FCD, and spectral summaries. |

### Fitting, exploration, and scaling

| Reference | Open when |
|---|---|
| `references/fitting-with-objectives-api.md` | Selecting a fitter interface, objective, backend, search space, callback, or result field. |
| `references/parameter-sweeps-and-regime-analysis.md` | Mapping regimes or sensitivities over parameter grids instead of fitting one target. |
| `references/batch-transform-acceleration.md` | Writing custom transformed execution, batching parameters, timing JAX, or checkpointing long gradients. |

### HORN task training

| Reference | Open when |
|---|---|
| `references/horn-task-training.md` | Training HORN networks on minibatched sequence tasks with held-out metrics. |
| `references/braintools/cogtask.md` | Generating phase-structured cognitive trials for direct HORN task training. |
| `references/braintools/data-preprocessing.md` | Encoding custom experimental or task inputs before training; ordinary HORN features do not require it. |

### Shared parameter and optimization support

| Reference | Open when |
|---|---|
| `references/brainstate/parameter-constraints-regularization.md` | Constraining or regularizing fitted parameters and directly trained HORN parameters. |
| `references/brainstate/parameter-transforms-regularizers-catalog.md` | Selecting an exact transform or regularizer after the parameter workflow reference routes here. |
| `references/braintools/metric.md` | Selecting supervised HORN losses, held-out metrics, or custom fitting statistics outside `brainmass.objectives`. |
| `references/braintools/optimizer.md` | Selecting gradient optimizers, schedules, standalone search wrappers, and their update lifecycle. |
| `references/braintools/parameter-initializer.md` | Initializing model State, fitted starting values, HORN weights, or distance-modulated parameters. |
| `references/braintools/surrogate.md` | Training a custom path with hard thresholds or spikes; canonical BrainMass and HORN workflows do not need surrogates. |
