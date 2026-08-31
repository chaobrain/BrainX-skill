# Braintools optimizer selection

Use this reference after the BrainCell objective, rollout, State reset, and gradient target are correct. Select an optimizer interface and learning-rate policy without changing the simulation graph or confusing gradient-step optimizers with standalone search wrappers.

## Choose a gradient optimizer

Choose the optimizer family by training regime and resource constraint. Ordinary reverse-mode optimizers register the selected `ParamState` tree once and accept the matching gradient tree through `update(grads)`.

### General minibatch optimizers

Use these optimizers for ordinary repeated gradient steps; establish `Adam` or `SGD` as the baseline before selecting a specialized update rule.

| API | Description |
|---|---|
| `SGD(...)` | Use for stochastic gradient descent with optional momentum and weight decay when a simple non-adaptive baseline is required. |
| `Momentum(...)` | Use for momentum updates without selecting the broader `SGD` interface. |
| `MomentumNesterov(...)` | Use when the optimization method requires Nesterov-accelerated momentum. |
| `Adam(...)` | Use as the canonical adaptive default after validating finite, nonzero gradients. |
| `AdamW(...)` | Use when the training policy requires decoupled weight decay; set `weight_decay` deliberately. |
| `Adagrad(...)` | Use when per-parameter learning rates should accumulate squared-gradient history. |
| `Adadelta(...)` | Use when the Adadelta extension of Adagrad is explicitly required. |
| `RMSprop(...)` | Use when the method should scale updates by a moving average of squared gradients. |
| `Adamax(...)` | Use when the infinity-norm Adam variant is required. |
| `Nadam(...)` | Use when combining Adam with Nesterov acceleration. |
| `RAdam(...)` | Use when rectifying adaptive learning-rate variance addresses observed early-step instability. |
| `Yogi(...)` | Use when the Yogi additive second-moment update is part of the intended method. |
| `Rprop(...)` | Use when resilient backpropagation should adapt per-parameter step sizes from gradient signs. |
| `AdaBelief(...)` | Use when adapting step size from the optimizer's belief in the gradient direction is required. |
| `Lion(...)` | Use when a sign-momentum optimizer is chosen to reduce optimizer-State memory. |
| `Novograd(...)` | Use when layer-wise gradient normalization with momentum is required. |
| `Fromage(...)` | Use only when the Free-scale Optimal Method for Adaptive Gradient method is explicitly selected. |

### Large-scale and memory-efficient optimizers

Use these optimizers only when large-batch scaling or optimizer-State memory is an explicit constraint.

| API | Description |
|---|---|
| `Lamb(...)` | Use for layer-wise adaptive moments in an explicitly large-batch training regime. |
| `Lars(...)` | Use for layer-wise adaptive rate scaling in an explicitly large-batch training regime. |
| `Adafactor(...)` | Use when factorized second-moment estimates are needed to reduce optimizer-State memory. |
| `SM3(...)` | Use for memory-efficient adaptive updates in a large or sparse model. |

### Wrapper and full-batch optimizers

Use these variants when the update requires an optimizer composition or a smooth full-batch objective rather than an ordinary standalone minibatch rule.

| API | Description |
|---|---|
| `Lookahead(...)` | Use when wrapping an inner Optax gradient transformation with periodic slow-weight interpolation. |
| `LBFGS(...)` | Use for a small, smooth, full-batch problem suited to limited-memory quasi-Newton updates; verify its step contract before replacing a minibatch optimizer. |

### Custom Optax transformations

Use the bridge when no named Braintools optimizer expresses the required Optax transformation.

| API | Description |
|---|---|
| `OptaxOptimizer(tx=..., lr=...)` | Use when supplying a custom Optax transformation while preserving Braintools parameter registration, optimizer State, gradient clipping, weight decay, and scheduler integration. |

Do not switch optimizer, scheduler, surrogate, and temporal reduction simultaneously. Establish an `Adam` loss and gradient baseline, then vary one optimization decision.

## Gradient-step lifecycle

Every ordinary gradient optimizer owns persistent optimizer State and mutates exactly the registered trainable State:

| API | Use and result |
|---|---|
| `model.states(brainstate.ParamState)` | Select trainable model State; dynamical voltage, conductance, spike, and delay State remain outside the optimizer. |
| `optimizer.register_trainable_weights(params)` | Register the parameter tree and initialize optimizer State. |
| `brainstate.transform.grad(loss_fn, params, return_value=True)` | Produce a gradient tree matching `params` and the scalar loss. |
| `optimizer.update(grads)` | Apply one update to the registered parameters. |

```python
params = net.states(brainstate.ParamState)
optimizer = braintools.optim.Adam(lr=3e-3)
optimizer.register_trainable_weights(params)

@brainstate.transform.jit
def train_step():
    brainstate.nn.init_all_states(net, batch_size=batch_size)
    grads, loss = brainstate.transform.grad(
        loss_fn,
        params,
        return_value=True,
    )()
    optimizer.update(grads)
    return loss
```

Register the same tree passed to `grad`. Do not rebuild or re-register that tree inside every compiled step.

Use `braintools.optim.UniqueStateManager(...)` only when an arbitrary nested PyTree may contain repeated references to the same `State`; it deduplicates State identity before optimization. An ordinary `model.states(brainstate.ParamState)` collection already expresses the canonical model-owned target.

## Learning-rate schedulers

Pass an `LRScheduler` instance as `lr=` to any gradient-based `OptaxOptimizer`. A float is treated as a constant learning rate.

| API | Description |
|---|---|
| `LRScheduler(...)` | Subclass when implementing a custom schedule; it defines attachment, stepping, current-rate access, and State serialization. |
| `StepLR(...)` | Use for staircase decay by `gamma` every `step_size` scheduler steps. |
| `MultiStepLR(...)` | Use for multiplicative decay at explicitly chosen milestones. |
| `ConstantLR(...)` | Use to multiply the base learning rate by a constant factor for a configured interval. |
| `LinearLR(...)` | Use to scale the learning rate linearly between configured factors. |
| `ExponentialLR(...)` | Use for smooth multiplicative decay by `gamma` at every scheduler step. |
| `PolynomialLR(...)` | Use when the learning rate must follow a polynomial decay. |
| `ExponentialDecayLR(...)` | Use for exponential decay with explicit step-based control. |
| `CosineAnnealingLR(...)` | Use for smooth cosine annealing from the base rate toward `eta_min`; match `T_max` to the scheduler-call cadence. |
| `CosineAnnealingWarmRestarts(...)` | Use for cosine annealing that periodically restarts its cycle. |
| `WarmupCosineSchedule(...)` | Use when one schedule must warm up first and then anneal by cosine. |
| `CyclicLR(...)` | Use to oscillate the learning rate between configured bounds. |
| `OneCycleLR(...)` | Use for a single super-convergence cycle; derive `total_steps` from the actual number of scheduler calls. |
| `ReduceLROnPlateau(...)` | Use when a validation metric should trigger decay after stalled improvement; call `step(metric=...)`. |
| `WarmupScheduler(...)` | Use to increase the learning rate linearly during a warmup phase. |
| `PiecewiseConstantSchedule(...)` | Use for step-wise transitions among explicitly configured constant rates. |
| `ChainedScheduler(...)` | Use to apply multiple scheduler transformations together. |
| `SequentialLR(...)` | Use to switch among schedulers at configured epoch milestones. |

```python
scheduler = braintools.optim.StepLR(
    base_lr=3e-3,
    step_size=20,
    gamma=0.5,
)
optimizer = braintools.optim.Adam(lr=scheduler)
optimizer.register_trainable_weights(params)

# After each epoch, outside the neural time-step transform:
optimizer.lr.step()
current_lr = optimizer.lr.get_last_lr()
```

For validation-driven scheduling, use the different call contract:

```python
scheduler = braintools.optim.ReduceLROnPlateau(
    base_lr=3e-3,
    mode="min",
    factor=0.5,
    patience=5,
)
optimizer = braintools.optim.Adam(lr=scheduler)

# After validation:
scheduler.step(metric=float(validation_loss))
```

Count scheduler calls explicitly. Stepping an epoch-configured schedule once per batch silently changes its effective timescale.

## Standalone and forward-mode optimizers

These interfaces own more of the optimization operation than `update(grads)`:

| API | Description |
|---|---|
| `ScipyOptimizer(loss_fun, bounds, method=...)` | Use for a smooth bounded or constrained objective solved as one optimization problem; match sequence bounds to `loss_fun(*params)` or dict bounds to `loss_fun(**params)`, then call `minimize(n_iter=...)` to obtain a SciPy result whose `x` matches the bounds structure. |
| `NevergradOptimizer(batched_loss_fun, bounds, n_sample, ...)` | Use when gradients are unavailable or unreliable; accept candidate-stacked parameters, return one loss per candidate, and call `minimize(n_iter=...)` to obtain the best parameter structure. |
| `SOFO(model, loss_fn, ...)` | Use for a feed-forward model when sampled forward-mode second-order directions are required; register its `ParamState`s, then call `step(inputs, targets)` to compute and apply the direction and return the pre-update loss. |
| `SOFOScan(rnn_cell, loss_fn, ...)` | Use for a stateful one-step recurrent cell with `(latent, inputs) -> (new_latent, output)`; `step(inputs, targets)` propagates forward-mode tangents through the sequence and applies one update. |

`ScipyOptimizer` and `NevergradOptimizer` deliberately raise for the State-based register/update methods. Do not place either inside the canonical compiled minibatch train step.

When SciPy bounds carry BrainUnit quantities, the wrapper optimizes numeric mantissas and calls `loss_fun` without units; reconstruct any required physical quantities inside the objective. Nevergrad reattaches units to returned parameters when the corresponding bounds carry units. Apply either optimizer's returned parameters to model State explicitly.

For Nevergrad, total objective evaluations are approximately `n_iter * n_sample`. `budget=` configures Nevergrad's internal strategy but does not cap that outer evaluation count.

## Routing and official sources

Open `metric.md` for loss and reduction choice and `input-current.md` for generated current-clamp protocols. Open `../scripts/fitting_hh_neuron.py` for the complete BrainCell candidate-evaluation and Nevergrad composition; reconcile its source-version APIs with the BrainCell root skill before adapting it.

Official sources:

- `https://brainx.chaobrain.com/braintools/apis/optim.html`
- `https://brainx.chaobrain.com/braintools/optim/01_nevergrad_optimizer.html`
- `https://brainx.chaobrain.com/braintools/optim/02_scipy_optimizer.html`
- `https://brainx.chaobrain.com/braintools/optim/03_optax_getting_started.html`
- `https://brainx.chaobrain.com/braintools/optim/04_learning_rate_scheduling.html`
- `https://brainx.chaobrain.com/braintools/optim/05_advanced_optimizers.html`
