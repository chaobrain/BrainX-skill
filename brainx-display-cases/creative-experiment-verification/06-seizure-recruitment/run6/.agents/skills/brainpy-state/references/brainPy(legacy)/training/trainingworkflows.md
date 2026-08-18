# Training workflows

Use this reference after selecting a legacy BrainPy trainer. It covers the
canonical data and execution flow for offline reservoir fitting, online recurrent
fitting, backpropagation through time, and a custom gradient loop. Open
`trainer library.md` when the trainer family is still undecided.

## Workflow selection

| Workflow | Use when | Essential order |
|---|---|---|
| Offline ridge | A readout can be solved from the complete state-target sequence. | Reset model -> warm up recurrent state -> fit `(X, Y)` -> predict held-out input. |
| Online RLS or FORCE | Readout parameters must update as time steps arrive. | Reset model -> warm up -> traverse paired training sequence while updating -> predict. |
| BPTT | Loss must differentiate through recurrent dynamics and trainable state. | Enter training mode -> construct model -> define loss and optimizer -> fit batches -> inspect metrics -> predict. |
| Custom backpropagation | The built-in trainer cannot express the required loss, gradient target, or update schedule. | Predict with `DSTrainer` -> differentiate selected train variables -> optimizer update inside one compiled step. |

## Shared reservoir model and data contract

Online and offline reservoir trainers consume batch-major arrays. Keep time on
axis 1 and features on the final axis.

```python
import brainpy as bp
import brainpy.math as bm


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


# Expected data shapes:
# warmup_x: (batch, warmup_time, features)
# train_x:  (batch, train_time, features)
# train_y:  (batch, train_time, outputs)
# test_x:   (batch, test_time, features)
# test_y:   (batch, test_time, outputs)
```

Create the trainable readout in training mode. Keep `X` and `Y` aligned along
batch and time axes; a forecast-ahead task shifts the target time window without
changing those axes.

## Offline ridge workflow

Use offline ridge when all reservoir states and targets can be collected before
solving the readout.

| API | Description |
|---|---|
| `RidgeTrainer(model, alpha=...)` | Use for the canonical closed-form ridge path; `alpha` controls Tikhonov regularization. |
| `trainer.predict(warmup_x)` | Use before fitting to advance delayed or recurrent state without changing fitted parameters. |
| `trainer.fit([train_x, train_y])` | Use to solve the readout from the complete paired sequence. |
| `trainer.predict(test_x)` | Use after fitting for held-out prediction. |

```python
with bm.environment(bm.batching_mode):
    offline_model = NGRC(num_in=3, num_out=3)

offline_model.reset(1)
offline = bp.RidgeTrainer(offline_model, alpha=1e-6)

_ = offline.predict(warmup_x)
_ = offline.fit([train_x, train_y])
offline_predictions = offline.predict(test_x)

assert offline_predictions.shape == test_y.shape
```

Do not reset between warm-up and `fit()`: the warm-up exists to establish the
state from which the training sequence should begin. Reset explicitly before a
scientifically independent trial.

## Online recurrent workflow

Use online fitting when the readout should update while the training sequence is
processed.

| API | Description |
|---|---|
| `OnlineTrainer(model, fit_method=bp.algorithms.RLS(), dt=...)` | Use for recursive least squares with explicit algorithm state and simulation step. |
| `trainer.predict(warmup_x)` | Use to establish recurrent history before learning. |
| `trainer.fit([train_x, train_y])` | Use to update parameters incrementally across paired time steps. |
| `trainer.predict(test_x)` | Use to evaluate the fitted recurrent model without further fitting. |

```python
with bm.environment(bm.batching_mode):
    online_model = NGRC(num_in=3, num_out=3)

online_model.reset(1)
online = bp.OnlineTrainer(
    online_model,
    fit_method=bp.algorithms.RLS(),
    dt=0.01,
)

_ = online.predict(warmup_x)
_ = online.fit([train_x, train_y])
online_predictions = online.predict(test_x)

assert online_predictions.shape == test_y.shape
```

Use `ForceTrainer` when the experiment specifically requires FORCE learning.
Keep the same warm-up, paired-fit, and held-out-prediction lifecycle.

## BPTT workflow

Use `BPTT` when gradients must pass through the recurrent trajectory rather than
fit only a readout.

| API | Description |
|---|---|
| `bm.training_environment()` | Use while constructing the model so trainable layers and states receive training semantics. |
| `model.train_vars().unique()` | Use when the loss or optimizer needs the unique trainable-variable collection. |
| `BPTT(model, loss_fun=..., optimizer=...)` | Use to bind temporal simulation, differentiation, loss, and parameter updates. |
| `trainer.fit(train_data, num_epoch=...)` | Use with a callable or iterable yielding `(inputs, targets)` batches. |
| `trainer.get_hist_metric(phase='fit', metric='loss')` | Use to verify convergence after fitting. |

```python
import brainpy as bp
import brainpy.math as bm


dt = 0.04
num_step = int(1.0 / dt)
num_batch = 128


@bm.jit
def make_batch(mean=0.025, scale=0.01):
    sample = bm.random.normal(size=(num_batch, 1, 1))
    bias = mean * 2.0 * (sample - 0.5)
    noise = bm.random.normal(size=(num_batch, num_step, 1))
    inputs = bias + scale / dt**0.5 * noise
    targets = bm.cumsum(inputs, axis=1)
    return inputs, targets


def train_data():
    for _ in range(100):
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


schedule = bp.optim.ExponentialDecay(
    lr=0.025,
    decay_steps=1,
    decay_rate=0.99975,
)
optimizer = bp.optim.Adam(lr=schedule, eps=1e-1)
trainer = bp.BPTT(model, loss_fun=loss_fun, optimizer=optimizer)

trainer.fit(train_data, num_epoch=30)
loss_history = trainer.get_hist_metric(phase='fit', metric='loss')

test_x, test_y = make_batch()
model.reset(num_batch)
predictions = trainer.predict(test_x)
assert predictions.shape == test_y.shape
assert loss_history is not None
```

For a spiking model, keep the same BPTT lifecycle. Encode input as a batched time
series, pass required spike variables through `monitors`, and set
`loss_has_aux=True` when the loss returns metrics such as accuracy. Add spike
count penalties only when they express an intended training objective.

## Custom backpropagation workflow

Use a custom loop only when `BPTT` cannot express the required gradient boundary
or update schedule.

| API | Description |
|---|---|
| `DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)` | Use to obtain the model's temporal predictions inside a custom loss. |
| `bm.grad(loss_fun, grad_vars=model.train_vars().unique(), has_aux=..., return_value=True)` | Use to differentiate exactly the selected legacy BrainPy train variables and return loss or auxiliary metrics. |
| `bp.optim.Optimizer(..., train_vars=...)` | Use to bind the same variable collection to the update rule. |
| `@bm.jit` | Use around the complete gradient-and-update step, not around fragmented scalar operations. |

Reset the model inside the loss when each batch represents an independent
trajectory. Preserve state when batches are consecutive segments of one
trajectory. This choice changes the modeled history and must be explicit.

## Common failures

- Do not construct a trainable readout outside the legacy training or batching
  environment required by the model.
- Do not transpose trainer data to time-major form unless the configured runner
  explicitly requires it; the canonical online and offline contract is
  `(batch, time, feature)`.
- Do not skip warm-up for delayed or recurrent reservoirs.
- Do not reset between warm-up and fitting unless the fit sequence is intended
  to start from a fresh state.
- Do not pass independent `X` and `Y` arguments to online or offline `fit()`;
  pass one two-item list or tuple.
- Do not optimize every mutable variable by default. Select the unique trainable
  variables and verify that state reset and parameter update affect the intended
  objects.
- Do not replace these legacy training APIs with similarly named APIs from a
  different BrainPy implementation.

## Routing

- Open `trainer library.md` for the complete trainer-family selection table and
  constructor contracts.
- Open `loss library.md` for built-in loss selection.
- Open `optimizers.md` for optimizer and learning-rate-schedule selection.
- Open `surrogate gradients.md` when a custom hard-threshold spiking model needs
  an explicit surrogate derivative.
- Open `../infrastructure/object oriented transformations and control flows.md` when a custom loop
  needs transformed scans, loops, or object-aware gradients.

## Sources

- Training a Brain Dynamics Model: https://brainpy.readthedocs.io/quickstart/training.html
- Training with Online Algorithms: https://brainpy.readthedocs.io/tutorial_training/online_training.html
- Training with Back-propagation Algorithms: https://brainpy.readthedocs.io/tutorial_training/bp_training.html
