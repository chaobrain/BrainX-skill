# BrainState Constrained and Regularized Parameters

Use this first-layer reference after `skills/brainstate/SKILL.md` has selected `brainstate.nn.Param` for a value that needs a valid-domain transform, a regularization penalty, a modeling prior, or fixed parameter-like storage. It owns the operational workflow and common patterns. Open `references/brainstate/parameter-transforms-regularizers-catalog.md` only when selecting an exact built-in transform or regularizer beyond the common choices shown here.

Keep `brainstate.nn` parameter transforms separate from execution transforms such as `brainstate.transform.jit`, `grad`, and `vmap`.

## Parameter contract

A `ParamState` is a bare trainable array. `brainstate.nn.Param` layers two orthogonal concerns on top of it: a constraint transform and a regularization term.

- `param.val` is the underlying `ParamState`; optimizers update its unconstrained array.
- `param.value()` applies the forward transform and returns the constrained model-space value. With the default `IdentityT`, constrained and unconstrained values coincide.
- `param.set_value(value)` applies the inverse transform when storing a constrained value back.
- `param.reg_loss()` evaluates the attached `reg=` object on the current constrained value. The loss function must explicitly add this penalty to its objective.
- `nn.Const` is equivalent to `nn.Param(value, fit=False)`. It is excluded from `model.states(brainstate.ParamState)`, so `grad` and optimizers leave it unchanged.

The constraint how-to returns dimensionless examples as `brainunit.Quantity` values. Use `brainunit.get_magnitude(...)` only when a plain JAX value is required for printing, comparison, or a dimensionless objective.

Official sources:

- https://brainx.chaobrain.com/brainstate/concepts/the_parameter_model.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

## Canonical constrained training workflow

This workflow keeps optimizer targets in unconstrained `ParamState`s, reads transformed values in the forward pass, and adds every attached regularization penalty to the data loss:

```python
import brainstate
from brainstate import nn
import brainunit as u
import jax.numpy as jnp


class ConstrainedLinear(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Param(
            brainstate.random.randn(din, dout) * 0.1,
            reg=nn.L2Reg(1e-3),
        )
        self.gain = nn.Param(
            jnp.array(1.0),
            t=nn.SoftplusT(lower=0.0),
        )

    def __call__(self, x):
        return (x @ self.w.value()) * self.gain.value()

    def reg_penalty(self):
        return sum(p.reg_loss() for p in self.nodes(nn.Param).values())


model = ConstrainedLinear(4, 2)
params = model.states(brainstate.ParamState)
x = brainstate.random.randn(16, 4)
y = brainstate.random.randn(16, 2)


def loss_fn():
    mse = jnp.mean((model(x) - y) ** 2)
    return mse + u.get_magnitude(model.reg_penalty())


@brainstate.transform.jit
def train_step():
    grads, loss = brainstate.transform.grad(
        loss_fn,
        params,
        return_value=True,
    )()
    for key in params:
        params[key].value -= 0.1 * grads[key]
    return loss
```

`brainstate.transform.grad` differentiates the unconstrained `ParamState` collection. The parameter transform constrains what `model(...)` sees; a regularizer changes training only because `loss_fn()` adds `model.reg_penalty()`.

Official sources:

- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

## Common constraint patterns

Use `.value()` in model computation. Mutate `.val.value` only as optimizer-side, unconstrained storage.

```python
rate = nn.Param(
    jnp.array(0.5),
    t=nn.SoftplusT(lower=0.0),
)
rate.val.value = jnp.array(-10.0)
assert float(u.get_magnitude(rate.value())) > 0.0

mix = nn.Param(
    jnp.array(0.5),
    t=nn.SigmoidT(lower=0.0, upper=1.0),
)

probs = nn.Param(
    jnp.zeros(3),
    t=nn.SimplexT(),
)
p = u.get_magnitude(probs.value())
assert jnp.all(p >= 0.0)
assert jnp.isclose(p.sum(), 1.0)
```

- `SoftplusT(lower)` smoothly maps the real line to `(lower, infinity)`.
- `SigmoidT(lower, upper)` smoothly maps to the open interval `(lower, upper)`.
- `SimplexT()` produces non-negative entries that sum to one. Its output has one more final-axis entry than its unconstrained input.

Open `parameter-transforms-regularizers-catalog.md` for negative, symmetric bounded, ordered, unit-vector, masked, composite, or projection-like transforms.

Official source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

## Regularization patterns

Call `reg.loss(value)` directly for a standalone penalty, or attach the regularizer with `reg=` and call `param.reg_loss()`:

```python
weights = jnp.array([3.0, -4.0])

l1_penalty = nn.L1Reg(0.1).loss(weights)
l2_penalty = nn.L2Reg(0.1).loss(weights)

p = nn.Param(
    weights,
    reg=nn.ElasticNetReg(
        l1_weight=1.0,
        l2_weight=1.0,
        alpha=0.5,
    ),
)
elastic_net_penalty = p.reg_loss()
```

For these values, the official tutorial defines L1 as `0.1 * (abs(3) + abs(-4))`, L2 as `0.1 * (3**2 + 4**2)`, and reports `16.0` for the shown elastic-net configuration.

Use `nn.ChainedReg(*regularizations, weight=1.0)` when one parameter needs several penalties. For a prior-based regularizer, `param.reset_to_prior()` stores the regularizer's `reset_value()` through the parameter transform; it does not call `sample_init()`.

Open `parameter-transforms-regularizers-catalog.md` to select classical, structural, or prior-distribution regularizers.

Official sources:

- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html
- https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.ChainedReg.html

## Fixed values with `nn.Const`

Use `nn.Const` when a value belongs in the module graph and forward computation but must not be collected as trainable state:

```python
class Scaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Param(jnp.ones(3))
        self.gain = nn.Const(jnp.array(2.0))

    def __call__(self, x):
        return x * self.weight.value() * self.gain.value()


model = Scaler()
trainable = model.states(brainstate.ParamState)
```

`trainable` contains the underlying State for `weight`; `gain` is absent. `Const` means non-trainable, not immutable: explicit `set_value()` calls can still change it.

Official sources:

- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html
- https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.Const.html

## Decision and failure rules

- Need only a valid domain: attach `t=` without inventing a regularizer.
- Need only a penalty or modeling prior: attach `reg=` without inventing a transform.
- Need both: attach `t=` and `reg=` to the same `nn.Param`, read `.value()` in the model, and add `.reg_loss()` to the objective.
- Need a fixed graph value: use `nn.Const`; do not expect it in a `ParamState` collection.
- Need an exact built-in transform or regularizer: open `parameter-transforms-regularizers-catalog.md`.
- Never read `param.val` as the constrained forward value or update `param.value()` as optimizer storage.
- Do not treat `SigmoidT` bounds as inclusive or `SoftplusT(lower)` as able to attain `lower`.
- Do not attach regularization and assume the training objective includes it automatically.
- Match a prior's support to the parameter domain; use `t=` when the domain must be guaranteed by construction.
- Do not assume every `nn.Transform` subclass is a smooth bijection; the catalog distinguishes projection-like transforms.

## Mirror source URLs

- https://brainx.chaobrain.com/brainstate/concepts/the_parameter_model.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html
- https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html
- https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html
- https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.Param.html
- https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.Const.html
