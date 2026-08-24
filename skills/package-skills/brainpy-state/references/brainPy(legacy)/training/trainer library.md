# Trainer library

Use this reference to select a trainer from the top-level legacy `brainpy`
module. Choose from temporal credit assignment, feedforward backpropagation,
incremental recurrent fitting, and whole-sequence regression. Open
`trainingworkflows.md` after selecting the trainer for canonical fit, warm-up,
prediction, and validation patterns.

## Trainer selection

| Training requirement | Use | Key constraint |
|---|---|---|
| Reuse simulation and prediction mechanics in a custom trainer | `DSTrainer` | It is the structural trainer base; it does not select a learning algorithm. |
| Differentiate through recurrent time steps | `BPTT` | Supply a loss and a legacy BrainPy optimizer; training data is an iterable or callable that yields input-target batches. |
| Backpropagate through a feedforward model | `BPFF` | Use for feedforward networks; do not pay the recurrent unrolling cost of BPTT. |
| Update a recurrent readout as samples arrive | `OnlineTrainer` | Supply an online algorithm such as RLS; fit data must be an `(X, Y)` pair. |
| Use the FORCE convenience interface | `ForceTrainer` | It is the specialized online FORCE path. |
| Fit a recurrent readout from a collected sequence | `OfflineTrainer` | Supply an offline algorithm; fit data must be an `(X, Y)` pair. |
| Use ridge regression without configuring an offline algorithm | `RidgeTrainer` | It is the specialized offline ridge path. |

## Trainer constructors

| API | Description |
|---|---|
| `DSTrainer(target, **kwargs)` | Use as the base for a custom dynamical-system trainer; it inherits runner configuration and provides prediction over the target. |
| `BPTT(target, loss_fun, optimizer=None, loss_has_aux=False, loss_auto_run=True, seed=None, shuffle_data=None, **kwargs)` | Use for backpropagation through time; `loss_fun` may name a `brainpy.losses` function or accept `(predictions, targets)`. |
| `BPFF(target, loss_fun, optimizer=None, loss_has_aux=False, loss_auto_run=True, seed=None, shuffle_data=None, **kwargs)` | Use for ordinary feedforward backpropagation with the same loss and optimizer contract. |
| `OnlineTrainer(target, fit_method=None, **kwargs)` | Use for recurrent online fitting; `fit_method` may be a supported name, a configuration dictionary, an online-algorithm instance, or a callable. |
| `ForceTrainer(target, alpha=1.0, **kwargs)` | Use for FORCE learning with regularization parameter `alpha`. |
| `OfflineTrainer(target, fit_method=None, **kwargs)` | Use for whole-sequence recurrent fitting; `fit_method` may be a supported name, a configuration dictionary, an offline-algorithm instance, or a callable. |
| `RidgeTrainer(target, alpha=1e-7, **kwargs)` | Use for ridge regression with Tikhonov regularization coefficient `alpha`. |

Use explicit algorithm objects when the method has parameters that affect the
experiment:

```python
online = bp.OnlineTrainer(model, fit_method=bp.algorithms.RLS(alpha=1e-5))
offline = bp.OfflineTrainer(
    model,
    fit_method=bp.algorithms.RidgeRegression(alpha=0.1),
)
```

String shortcuts such as `fit_method='rls'` and `fit_method='ridge'` are useful
only when their defaults are intentional. A dictionary such as
`{'name': 'ridge', 'alpha': 0.1}` selects and configures the method together.

## Fit and prediction lifecycle

| API | Description |
|---|---|
| `DSTrainer.predict(inputs, reset_state=False, shared_args=None, eval_time=False)` | Use for prediction with runner semantics; it returns model output and optionally evaluation time. |
| `OnlineTrainer.fit((X, Y), reset_state=False, shared_args=None)` | Use to update the fitted parameters incrementally while traversing a paired sequence. It rejects data that is not a two-item list or tuple. |
| `OfflineTrainer.fit((X, Y), reset_state=False, shared_args=None)` | Use to collect the sequence and solve the selected offline fit; it returns the modeled output. |
| `BPTT.fit(train_data, test_data=None, num_epoch=100, num_report=-1, reset_state=True, shared_args=None, ...)` | Use to run epoch-based gradient training from an iterable or callable batch source; it resets state by default. |
| `BPTT.predict(inputs, reset_state=False, shared_args=None, eval_time=False)` | Use after fitting to run a time series; reset explicitly when the prediction is an independent trajectory. |
| `BPTT.get_hist_metric(phase='fit', metric='loss', which='report')` | Use to retrieve reported or detailed fit/test metrics after training. |

Online and offline trainers treat inputs as batched by default. Their canonical
input has shape `(num_sample, num_time, num_feature)`. Preserve this batch-major
layout unless the trainer is explicitly configured otherwise.

## Boundaries and common failures

- Do not select `BPTT` for a feedforward-only model; use `BPFF`.
- Do not select `OnlineTrainer` when the algorithm needs the complete feature
  matrix before solving; use `OfflineTrainer`.
- Do not pass a bare `X` array to online or offline `fit()`; pass `(X, Y)` as a
  list or tuple.
- Do not omit a reservoir warm-up when delayed or recurrent state must settle
  before fitting.
- Do not assume prediction starts from a fresh trajectory. The base, BPTT,
  online, and offline prediction paths preserve model state by default.
- Do not import these trainers from another BrainPy implementation. This is the
  legacy top-level `brainpy` trainer library.

## Routing

- Open `trainingworkflows.md` for offline ridge, online RLS, BPTT, and custom
  gradient-loop execution order.
- Open `loss library.md` when selecting built-in loss functions.
- Open `optimizers.md` when selecting learning-rate schedules or gradient
  optimizers for `BPTT` and `BPFF`.
- Open `prebuilt neural network layers.md` when the trainable model requires
  legacy `brainpy.dnn` layers.

## Sources

- `brainpy` module API: https://brainpy.readthedocs.io/apis/brainpy.html
