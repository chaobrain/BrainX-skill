# Legacy BrainPy optimizers

Use this reference after a legacy BrainPy loss and gradient target are correct.
It covers `brainpy.optim` optimizers and schedulers that mutate named
`bm.Variable` objects; do not substitute Braintools optimizers or BrainState
`ParamState` trees in this workflow.

## Apply a gradient update

A legacy optimizer owns a named dictionary of trainable variables, scheduler
State, and any algorithm-specific moment or velocity variables. The gradient
dictionary passed to `update()` must have the same keys and compatible shapes.

| API | Description |
|---|---|
| `bp.optim.Optimizer(lr, train_vars=None, name=None)` | Subclass only for a custom update rule; it normalizes `lr` to a scheduler and owns the registered trainable variables. |
| `bm.grad(func, grad_vars=train_vars, return_value=True)` | Differentiate a scalar loss with respect to the same named variable dictionary. |
| `optimizer.update(grads)` | Mutate the registered variables and advance optimizer-owned State from a matching gradient dictionary. |
| `optimizer.vars()` | Inspect scheduler, step, velocity, and moment variables that must remain visible to object-aware transformations. |

```python
import brainpy as bp
import brainpy.math as bm


class Objective(bp.BrainPyObject):
    def __init__(self):
        super().__init__()
        self.weight = bm.TrainVar(bm.array([1.0]))

    def __call__(self):
        return bm.mean((self.weight - 3.0) ** 2)


objective = Objective()
train_vars = {'weight': objective.weight}
optimizer = bp.optim.Adam(lr=0.05, train_vars=train_vars)
value_and_grad = bm.grad(
    objective,
    grad_vars=train_vars,
    return_value=True,
)

grads, loss = value_and_grad()
optimizer.update(grads)

assert grads.keys() == train_vars.keys()
assert loss.ndim == 0
```

Process gradients between differentiation and `update()` when clipping or
another explicit transform is required. Do not rename, drop, or add keys while
doing so.

## Choose an optimizer

Establish `Adam` or `SGD` as the baseline, then select a specialized rule only
when its behavior matches the training regime.

| API | Description |
|---|---|
| `bp.optim.SGD(lr, train_vars=None, weight_decay=None, name=None)` | Use for plain stochastic gradient descent and as the simplest update baseline. |
| `bp.optim.Momentum(lr, train_vars=None, momentum=0.9, weight_decay=None, name=None)` | Use when velocity-based momentum is required. |
| `bp.optim.MomentumNesterov(lr, train_vars=None, weight_decay=None, momentum=0.9, name=None)` | Use for Nesterov accelerated momentum. |
| `bp.optim.Adagrad(lr, train_vars=None, weight_decay=None, epsilon=1e-6, name=None)` | Use when accumulated squared gradients should adapt each parameter's rate. |
| `bp.optim.Adadelta(lr=1.0, train_vars=None, weight_decay=None, epsilon=1e-6, rho=0.95, name=None)` | Use for the Adadelta extension with decaying accumulated statistics. |
| `bp.optim.RMSProp(lr, train_vars=None, weight_decay=None, epsilon=1e-6, rho=0.9, name=None)` | Use for a moving average of squared gradients. Preserve the legacy capitalization `RMSProp`. |
| `bp.optim.Adam(lr, train_vars=None, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=None, name=None)` | Use as the canonical adaptive baseline after verifying finite, nonzero gradients. |
| `bp.optim.AdamW(lr, train_vars=None, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, amsgrad=False, name=None)` | Use when decoupled weight decay is required. |
| `bp.optim.LARS(lr, train_vars=None, momentum=0.9, weight_decay=0.0001, tc=0.001, eps=1e-5, name=None)` | Use for layer-wise adaptive rate scaling in an explicitly large-batch regime. |
| `bp.optim.Adan(lr=0.001, train_vars=None, betas=(0.02, 0.08, 0.01), eps=1e-8, weight_decay=0.02, no_prox=False, name=None)` | Use only when the Adan adaptive Nesterov method is intentionally selected. |

Optimizer classes create different auxiliary State. `SGD` may only expose
scheduler step State; `Momentum` adds velocity variables; `Adam` adds step and
first- and second-moment variables. Preserve all of them across checkpoints and
compiled updates.

## Choose a learning-rate schedule

Every optimizer `lr` becomes a `bp.optim.Scheduler`. A scalar or `bm.Variable`
is normalized to `bp.optim.Constant` by `make_schedule()`.

| API | Description |
|---|---|
| `bp.optim.make_schedule(scalar_or_schedule)` | Normalize a scalar, variable, or existing scheduler to the scheduler interface. |
| `bp.optim.Constant(lr, last_epoch=-1)` | Use for an unchanging learning rate. |
| `bp.optim.StepLR(lr, step_size, gamma=0.1, last_epoch=-1)` | Multiply the rate by `gamma` every `step_size` epochs. |
| `bp.optim.MultiStepLR(lr, milestones, gamma=0.1, last_epoch=-1)` | Decay at explicitly listed epoch milestones. |
| `bp.optim.CosineAnnealingLR(lr, T_max, eta_min=0.0, last_epoch=-1)` | Anneal from the initial rate to `eta_min` over the configured cosine horizon. |
| `bp.optim.CosineAnnealingWarmRestarts(lr, num_call_per_epoch, T_0, T_mult=1, eta_min=0.0, last_epoch=-1, last_call=-1)` | Use cosine annealing with expanding or fixed restart periods; match calls per epoch to the actual update cadence. |
| `bp.optim.ExponentialLR(lr, gamma, last_epoch=-1)` | Multiply the rate by `gamma` every epoch. |
| `bp.optim.ExponentialDecayLR(lr, decay_steps, decay_rate, last_epoch=-1, last_call=-1)` | Apply call-count-based exponential decay over `decay_steps`. |
| `bp.optim.InverseTimeDecayLR(lr, decay_steps, decay_rate, staircase=False, last_epoch=-1, last_call=-1)` | Apply inverse-time decay, optionally in staircase form. |
| `bp.optim.PolynomialDecayLR(lr, decay_steps, final_lr, power=1.0, last_epoch=-1, last_call=-1)` | Decay polynomially toward `final_lr`. |
| `bp.optim.PiecewiseConstantLR(boundaries, values, last_epoch=-1, last_call=-1)` | Use explicit piecewise-constant rates over call-count boundaries. |

```python
scheduler = bp.optim.ExponentialDecayLR(
    lr=0.1,
    decay_steps=100,
    decay_rate=0.99,
)
optimizer = bp.optim.Adam(
    lr=scheduler,
    train_vars=train_vars,
)

initial_rate = optimizer.lr(0)
later_rate = optimizer.lr(100)
assert later_rate < initial_rate
```

Calling a scheduler with an explicit index evaluates that step. Calling without
an index evaluates its built-in training step. Count optimizer and scheduler
calls against the intended epoch or batch timescale; a schedule configured in
epochs decays too quickly if advanced once per batch.

To change a constant schedule intentionally, use `optimizer.lr.lr = value` or
`optimizer.lr.set_value(value)`. Do not replace the scheduler object during a
compiled training run.

## JIT and checkpoint optimizer State

If an update is wrapped in `bm.jit`, include the variables returned by
`optimizer.vars()` in the transformation's dynamic-variable set when they are
not already discoverable through the transformed object. Omitting them can
freeze or lose step, velocity, and moment updates.

Checkpoint all of the following together:

| State | Why it matters |
|---|---|
| `train_vars` | Contains the parameters being optimized. |
| `optimizer.vars()` | Contains schedule position and algorithm-specific accumulated State. |
| Model dynamical State when resuming mid-rollout | Determines the next simulation and gradient result. |

For a new independent training run, reset model State and construct a new
optimizer when moment and scheduler histories must also restart. Resetting only
neuronal State does not restart Adam moments or scheduler time.

## Customize only the missing mechanism

Subclass `bp.optim.Optimizer` when a required update rule is absent. Accept
`lr` and `train_vars`, register every dynamically changing auxiliary variable,
and implement `update(grads)` with the same named-gradient contract.

Subclass `bp.optim.Scheduler` when the learning-rate policy is absent. Implement
`__call__(i=None)` so explicit indices and the built-in step have clear,
consistent semantics.

## Source-backed failures

- Pass a dictionary of legacy `bm.Variable` or `bm.TrainVar` objects as
  `train_vars`; do not pass BrainState `ParamState` collections.
- Make the differentiated objective scalar and validate gradient finiteness
  before changing optimizer families.
- Keep gradient keys and shapes identical to the registered trainable-variable
  dictionary.
- Preserve optimizer-owned variables through JIT and checkpointing.
- Change one of loss, optimizer, schedule, or gradient processing at a time so a
  training regression remains diagnosable.

## Official sources

- `https://brainpy.readthedocs.io/apis/optim.html`
- `https://brainpy.readthedocs.io/tutorial_toolbox/optimizers.html`
- Generated optimizer and scheduler pages linked from the official API index.
