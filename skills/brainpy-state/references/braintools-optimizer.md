# Braintools Optimizer Reference

Source mirrored: https://brainx.chaobrain.com/braintools/optim/index.html

Use this BrainPy training reference when optimizer or learning-rate-scheduler selection goes beyond the default `braintools.optim.Adam` pattern.

## Used by

- `skills/brainpy/SKILL.md` through the `references/brainpy-training.md` parent route

Official source phrase: "Optimization guides highlight practical solvers for tuning models and experiments."

## Scope

- Use `braintools.optim` for optimizer objects and learning-rate schedulers in BrainPy training loops.
- Keep optimizer state registered against the model trainable states, usually `model.states(brainstate.ParamState)`.
- Keep gradient computation in `brainstate.transform.grad(...)` and apply updates through the optimizer update path already used by the nested training reference.
- Use this reference for selecting optimizer families, scheduler families, or external optimization routes; keep BrainPy loss, gradient, JIT, and rollout structure in the `references/brainpy-training.md` parent.

## Tutorial Routing

- NevergradOptimizer tutorial: use for gradient-free parameter tuning and experiment search.
- ScipyOptimizer tutorial: use for SciPy-based routines rather than BrainState gradient-step loops.
- Getting Started with optax Optimizers: use when selecting Optax-backed optimizers through Braintools.
- Learning Rate Scheduling Strategies: use when the training loop needs schedules such as decay, warmup, cosine, cyclic, or plateau behavior.
- Advanced Optimizers and Techniques: use when a task asks for optimizer variants beyond standard SGD/Adam-style choices.

## Optimizer And Scheduler Families Listed By The Index

- Optimizers include `SGD`, `Momentum`, `MomentumNesterov`, `Adam`, `AdamW`, `Adagrad`, `Adadelta`, `RMSprop`, `Adamax`, `Nadam`, `RAdam`, `Lamb`, `Lars`, `Lookahead`, `Yogi`, `LBFGS`, `Rprop`, `Adafactor`, `AdaBelief`, `Lion`, `SM3`, `Novograd`, `Fromage`, `SOFO`, and `SOFOScan`.
- Scheduler utilities include `LRScheduler`, `StepLR`, `MultiStepLR`, `ConstantLR`, `LinearLR`, `ExponentialLR`, `PolynomialLR`, `ExponentialDecayLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `WarmupCosineSchedule`, `CyclicLR`, `OneCycleLR`, `ReduceLROnPlateau`, `WarmupScheduler`, `PiecewiseConstantSchedule`, `ChainedScheduler`, and `SequentialLR`.
- External optimizer wrappers include `ScipyOptimizer` and `NevergradOptimizer`.

## BrainState Training Pattern Reminder

```python
params = model.states(brainstate.ParamState)
optimizer = braintools.optim.Adam(lr=1e-2)
optimizer.register_trainable_weights(params)

@brainstate.transform.jit
def train_step(x, y):
    def loss_fn():
        ...
    grads, loss = brainstate.transform.grad(loss_fn, params, return_value=True)()
    optimizer.update(grads)
    return loss
```

Keep this as the default loop shape unless the selected optimizer tutorial requires a different optimization interface.
