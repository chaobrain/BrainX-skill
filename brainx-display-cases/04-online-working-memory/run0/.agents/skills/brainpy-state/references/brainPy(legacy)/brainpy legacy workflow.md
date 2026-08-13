# Legacy BrainPy workflow

## Purpose and boundary

Use this workflow for legacy BrainPy 2.x simulation, training, analysis, and object-oriented transformation.

Use only legacy `brainpy` APIs. Route modern `brainpy.state` work elsewhere.

Legacy BrainPy is not fully compatible with BrainX: do not combine it with BrainUnit or BrainTrace; BrainState, BrainEvent, and Braintools are compatible with BrainPy legacy. For any cellular modeling, use BrainPy-State with BrainCell; never use legacy BrainPy ion dynamics or ion-channel dynamics.

## Underlying principle of legacy BrainPy

`DynamicalSystem` represents an evolving neuron, synapse, population, or network. Its `update()` method advances the biological dynamics one step; nesting composes cellular, synaptic, circuit, and network scales.

`brainpy.math.Variable` stores time-varying neural state such as membrane potential, channel gates, conductance, synaptic current, spikes, firing rate, or adaptation.

`brainpy.connect` defines neural wiring; `brainpy.dyn` projections carry presynaptic activity through synaptic dynamics into postsynaptic input.

`DSRunner` represents an in-silico protocol: apply stimuli, advance the model, and record neural time series.

### API structure

Choose the namespace that owns the operation:

| API | Use |
|---|---|
| `brainpy` module | Define core objects and expose legacy runners, trainers, visualization, and top-level APIs. |
| `brainpy.math` module | Create arrays and Variables; configure execution; apply object-aware transformations and control flow. |
| `brainpy.dnn` module | Build trainable neural-network and communication layers. |
| `brainpy.dyn` module | Select neuron, synapse, projection, reservoir, and recurrent dynamics. |
| `brainpy.integrators` module | Construct and configure ODE, SDE, FDE, and joint-equation integrators. |
| `brainpy.analysis` module | Analyze phase planes, bifurcations, fast-slow systems, and fixed or slow points. |
| `brainpy.connect` module | Construct and inspect neural connectivity. |
| `brainpy.encoding` module | Encode continuous or categorical data into neural activity and spike representations. |
| `brainpy.initialize` module | Initialize model state, parameters, and trainable weights. |
| `brainpy.inputs` module | Generate time-varying currents, stimuli, and spike inputs. |
| `brainpy.losses` module | Compute training objectives and regularization penalties. |
| `brainpy.measure` module | Measure neural activity and relationships from simulation outputs. |
| `brainpy.optim` module | Optimize trainable Variables and schedule learning rates. |
| `brainpy.running` module | Execute independent experiments and parameter sweeps. |
| `brainpy.mixin` module | Provide behavioral contracts shared by legacy dynamics and projections. |

### 1. Simulate dynamics with `DSRunner`

`DSRunner` advances one legacy `DynamicalSystem` through a structural time loop while applying inputs and recording time-major monitor histories.

| API | Description |
|---|---|
| `bp.DSRunner(target, inputs=..., monitors=..., jit=..., dt=...)` | Use for the canonical simulation; it binds one `DynamicalSystem` to its input, monitor, timestep, and compilation policy. |
| `runner.run(duration=..., inputs=..., reset_state=...)` | Use to execute a duration or a time-indexed input; set `reset_state` explicitly when the run must be independent. |
| `runner.mon.ts` | Use after execution for the generated time axis. |
| `runner.mon[name]` | Use after execution for the time-major trajectory registered under `name`. |
| `bm.for_loop(step, operands)` | Use instead when a custom step function and its returned values express the rollout more directly than runner inputs and monitors. |

```python
import brainpy as bp
import brainpy.math as bm


bm.set_dt(0.1)
neurons = bp.dyn.LifRef(
    10,
    V_rest=-60.0,
    V_reset=-60.0,
    V_th=-50.0,
    tau=20.0,
    tau_ref=5.0,
    V_initializer=bp.init.Constant(-60.0),
)

runner = bp.DSRunner(
    target=neurons,
    monitors=['V', 'spike'],
    jit=True,
    progress_bar=False,
)
stimulus = bm.ones(1000) * 20.0
runner.run(inputs=stimulus, reset_state=True)

assert runner.mon['V'].shape == (1000, 10)
assert runner.mon['spike'].shape == (1000, 10)
assert runner.mon.ts.shape == (1000,)
```

Monitor only the Variables, names, and neuron indices required by the analysis; reset explicitly before an independent rollout.

Open `infrastructure/More about simulation.md` when configuring static, iterable, or functional inputs; indexed or callable monitors; repeated rollouts; prediction axes; or state reset. Open `infrastructure/Input generation.md` when constructing stimulus arrays.

### 2. Fit a reservoir readout

`RidgeTrainer` keeps reservoir dynamics fixed and solves a trainable linear readout from the complete state-target sequence, so warm-up must precede fitting without an intervening reset.

| API | Description |
|---|---|
| `bm.environment(bm.batching_mode)` | Use while constructing the canonical reservoir model so its trainable readout receives batched execution semantics. |
| `bd.chaos.LorenzEq(duration, dt=...)` | Use for the official quickstart's Lorenz forecasting data; it requires the optional `brainpy_datasets` package and returns time-indexed `xs`, `ys`, and `zs`. |
| `bp.dyn.NVAR(...)` | Use for the source-backed nonlinear vector autoregression reservoir with delayed polynomial features. |
| `bp.RidgeTrainer(model, alpha=...)` | Use when the complete reservoir-state matrix can be collected before solving the readout; `alpha` controls ridge regularization. |
| `trainer.predict(warmup_x)` | Use before fitting to establish delayed reservoir state without updating the readout. |
| `trainer.fit([train_x, train_y])` | Use to solve the readout from batch-major, time-aligned input and target sequences. |
| `trainer.predict(test_x)` | Use after fitting for held-out prediction. |

```python
import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd


def get_subset(data, start, end):
    values = bm.hstack([
        data.xs[start:end],
        data.ys[start:end],
        data.zs[start:end],
    ])
    return values.reshape((1,) + values.shape)


class NGRC(bp.DynamicalSystem):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.reservoir = bp.dyn.NVAR(
            num_in,
            delay=4,
            order=2,
            stride=5,
        )
        self.readout = bp.dnn.Dense(
            self.reservoir.num_out,
            num_out,
            mode=bm.training_mode,
        )

    def update(self, x):
        return self.readout(self.reservoir(x))


dt = 0.01
data = bd.chaos.LorenzEq(100.0, dt=dt)
warmup_x = get_subset(data, 0, int(20.0 / dt))
train_x = get_subset(data, int(20.0 / dt), int(80.0 / dt))
train_y = get_subset(data, int(20.0 / dt) + 1, int(80.0 / dt) + 1)
test_x = get_subset(data, int(80.0 / dt), int(100.0 / dt) - 1)
test_y = get_subset(data, int(80.0 / dt) + 1, int(100.0 / dt))

with bm.environment(bm.batching_mode):
    model = NGRC(num_in=3, num_out=3)

model.reset(1)
trainer = bp.RidgeTrainer(model, alpha=1e-6)
_ = trainer.predict(warmup_x)
_ = trainer.fit([train_x, train_y])
predictions = trainer.predict(test_x)
test_loss = bp.losses.mean_squared_error(test_y, predictions)

assert predictions.shape == test_y.shape
assert test_loss.ndim == 0
```

Keep data batch-major as `(batch, time, feature)` and align shifted forecast targets to the input length. Do not reset between warm-up and fitting.

Open `training/trainingworkflows.md` when choosing offline ridge, online RLS or FORCE, BPTT, or a custom gradient loop. Open `training/trainer library.md` when selecting among `RidgeTrainer`, `OfflineTrainer`, `OnlineTrainer`, `ForceTrainer`, `BPTT`, and `BPFF`.

### 3. Train recurrent dynamics through time

`BPTT` differentiates a temporal loss through recurrent dynamics, updates only trainable Variables, and preserves ordinary dynamical state as rollout state rather than optimizer parameters.

| API | Description |
|---|---|
| `bm.training_environment()` | Use while constructing the model so recurrent cells, readouts, and trainable state receive legacy training semantics. |
| `bp.BPTT(model, loss_fun=..., optimizer=...)` | Use when the loss must propagate through a recurrent or spiking trajectory; it binds rollout, differentiation, loss, and updates. |
| `trainer.fit(train_data, num_epoch=...)` | Use with an iterable or callable yielding `(inputs, targets)` batches. |
| `trainer.get_hist_metric(phase='fit', metric='loss')` | Use after fitting to inspect recorded training loss. |
| `model.reset(batch_size)` | Use before an independent evaluation trajectory so predictions do not inherit training state. |
| `trainer.predict(inputs)` | Use after fitting to execute the trained temporal model. |

```python
import brainpy as bp
import brainpy.math as bm


dt = 0.04
num_step = int(1.0 / dt)
num_batch = 32


@bm.jit
def make_batch(mean=0.025, scale=0.01):
    sample = bm.random.normal(size=(num_batch, 1, 1))
    bias = mean * 2.0 * (sample - 0.5)
    noise = bm.random.normal(size=(num_batch, num_step, 1))
    inputs = bias + scale / dt**0.5 * noise
    targets = bm.cumsum(inputs, axis=1)
    return inputs, targets


def train_data():
    for _ in range(10):
        yield make_batch()


class RNN(bp.DynamicalSystem):
    def __init__(self, num_in, num_hidden):
        super().__init__()
        self.recurrent = bp.dyn.RNNCell(
            num_in,
            num_hidden,
            train_state=True,
        )
        self.readout = bp.dnn.Dense(num_hidden, 1)

    def update(self, x):
        return self.readout(self.recurrent(x))


with bm.training_environment():
    model = RNN(num_in=1, num_hidden=100)


def loss_fun(predictions, targets):
    return bp.losses.mean_squared_error(predictions, targets)


trainer = bp.BPTT(
    model,
    loss_fun=loss_fun,
    optimizer=bp.optim.Adam(lr=1e-3),
)
trainer.fit(train_data, num_epoch=2)
loss_history = trainer.get_hist_metric(phase='fit', metric='loss')

test_x, test_y = make_batch()
model.reset(num_batch)
predictions = trainer.predict(test_x)

assert predictions.shape == test_y.shape
assert len(loss_history) > 0
```

Reset recurrent state only at independent sequence boundaries; keep rollout Variables out of optimizer updates.

Open `training/trainingworkflows.md` for online, offline, BPTT, custom-gradient, and reset workflows. Open `training/loss library.md`, `training/optimizers.md`, or `training/surrogate gradients.md` after fixing output reduction and the gradient boundary.

### 4. Analyze model dynamics

Legacy analyzers reuse a constructed model but require explicit state ranges, swept or fixed parameters, and resolution so the resulting dynamical structure answers the intended question.

| API | Description |
|---|---|
| `bm.enable_x64()` | Use before constructing a numerically sensitive analyzer so root finding and stability calculations run at higher precision. |
| `bp.analysis.Bifurcation1D(...)` | Use to continue fixed points of a one-variable model while one or more parameters vary. |
| `bp.analysis.PhasePlane2D(...)` | Use to compute nullclines, vector fields, fixed points, and trajectories for two state variables. |
| `bp.analysis.SlowPointFinder(...)` | Use when a high-dimensional model requires candidate-based fixed- or slow-point optimization and optional Jacobians. |

```python
import brainpy as bp
import brainpy.math as bm


bm.enable_x64()
model = bp.dyn.ExpIF(1, delta_T=1.0)

bifurcation = bp.analysis.Bifurcation1D(
    model=model,
    target_vars={'V': [-70.0, -55.0]},
    target_pars={'I': [0.0, 6.0]},
    resolutions={'I': 0.01},
)
bifurcation.plot_bifurcation(show=True)
```

Match Variable and parameter names exactly; validate ranges and resolution against simulation before interpreting missing structure.

Open `analysis.md` when choosing phase-plane, bifurcation, fast-slow, fixed-point, slow-point, or linearization analysis, or when configuring candidates and post-processing.

### 5. Make mutable state visible to transformations

Legacy object-aware transformations separate discoverable `Variable` state from static Python structure, preserving mutation across JIT compilation, differentiation, vectorization, and control flow.

| API | Description |
|---|---|
| `bp.BrainPyObject` | Subclass when an object owns Variables or nested BrainPy objects that transformations must discover. |
| `bm.Variable(value)` | Store mutable dynamical state and update its array through `.value`. |
| `bm.TrainVar(value)` | Mark only optimizer-controlled state as trainable. |
| `bm.NodeList`, `bm.NodeDict` | Store nested BrainPy objects in traversal-aware containers. |
| `bm.VarList`, `bm.VarDict` | Store Variables in traversal-aware containers. |
| `bm.jit(...)`, `bm.cls_jit` | Compile functions or bound methods while preserving discovered Variable mutation. |
| `bm.grad(...)` | Differentiate with respect to explicit arguments or discovered trainable Variables. |
| `bm.for_loop(...)` | Compile a fixed-length loop and stack step outputs without a Python timestep loop. |

```python
import brainpy as bp
import brainpy.math as bm


class Accumulator(bp.BrainPyObject):
    def __init__(self):
        super().__init__()
        self.total = bm.Variable(bm.zeros(1))

    @bm.cls_jit
    def __call__(self, value):
        self.total.value += value
        return self.total.value


accumulator = Accumulator()
assert bm.allclose(accumulator(2.0), bm.array([2.0]))
assert bm.allclose(accumulator(3.0), bm.array([5.0]))
```

Update discovered Variables through `.value`, indexed assignment, or augmented assignment; rebinding them or changing ordinary attributes after tracing is invisible to compiled code. Store nested objects and Variables on attributes or in `NodeList`, `NodeDict`, `VarList`, or `VarDict`.

Open `infrastructure/object oriented transformations and control flows.md` when choosing JIT, gradients, Jacobians, vectorization, structural branches or loops, or transformation-aware containers.

## Reference routing

Open only the smallest reference that owns the decision.

### Model construction and dynamics

| Reference | Open when |
|---|---|
| `built-in dynamic neuron model.md` | Selecting a built-in legacy neuron family, parameterization, or initialization. |
| `connecting neurons.md` | Selecting explicit, probabilistic, distance-based, or just-in-time connectivity. |
| `route activity through connectivity.md` | Choosing dense, sparse, event-driven, or just-in-time communication over an established connection structure. |
| `synaptic projections.md` | Composing presynaptic activity, communication, synaptic dynamics, output current, and postsynaptic targets. |
| `synpase properties.md` | Selecting synaptic kinetics, output type, alignment, delay, or short-term behavior. |
| `customize neuron and synpase.md` | Implementing a custom neuron, synapse, output, communication operator, or projection after built-ins are exhausted. |
| `integrators.md` | Selecting or defining ODE, SDE, fractional, or joint-equation integration for a custom model. |
| `infrastructure/delays.md` | Adding delayed Variables, accesses, or projection activity. |

### Simulation and infrastructure

| Reference | Open when |
|---|---|
| `infrastructure/array creation and mechanics.md` | Creating, indexing, reshaping, combining, converting, or functionally updating legacy `brainpy.math` arrays. |
| `infrastructure/brainpy math environment setting.md` | Configuring `dt`, dtype, execution mode, precision, or platform. |
| `infrastructure/object oriented transformations and control flows.md` | Choosing object-aware JIT, gradients, Jacobians, vectorization, loops, scans, branches, or transformation-aware containers. |
| `infrastructure/Input generation.md` | Constructing sectioned, ramped, oscillatory, noisy, or spike-train stimuli. |
| `infrastructure/More about simulation.md` | Configuring `DSRunner` inputs, monitors, reset, repeated rollouts, prediction axes, or parameter-exploration execution. |
| `infrastructure/Parallel experiment execution.md` | Running independent simulations with CPU workers or JAX mapping and controlling batch memory. |
| `infrastructure/Multi-device array sharding.md` | Sharding one legacy array computation across multiple devices rather than mapping independent experiments. |

### Training

| Reference | Open when |
|---|---|
| `training/trainingworkflows.md` | Choosing and executing offline ridge, online RLS or FORCE, BPTT, or a custom gradient loop. |
| `training/trainer library.md` | Selecting a trainer family or exact constructor and fit/predict lifecycle. |
| `training/prebuilt neural network layers.md` | Selecting a legacy trainable dense, recurrent, convolutional, normalization, pooling, or activation layer. |
| `training/parameter initializers.md` | Selecting an initializer for trainable parameters or model state. |
| `training/loss library.md` | Selecting a loss after output orientation and reduction are established. |
| `training/optimizers.md` | Selecting a legacy optimizer, scheduler, registration pattern, or update lifecycle. |
| `training/surrogate gradients.md` | Training a hard-threshold spiking model that needs a legacy surrogate derivative. |

### Analysis

| Reference | Open when |
|---|---|
| `analysis.md` | Selecting and executing low-dimensional phase-plane, bifurcation, fast-slow, or high-dimensional fixed- and slow-point analysis. |

## Boundaries and common failures

- Do not mix this workflow with `brainpy.state`, BrainState, BrainUnit, or Braintools APIs.

## Official sources

- https://brainpy.readthedocs.io/quickstart/simulation.html
- https://brainpy.readthedocs.io/tutorial_simulation/simulation_dsrunner.html
- https://brainpy.readthedocs.io/quickstart/training.html
- https://brainpy.readthedocs.io/tutorial_math/variables.html
- https://brainpy.readthedocs.io/core_concept/brainpy_transform_concept.html
- https://brainpy.readthedocs.io/quickstart/analysis.html
