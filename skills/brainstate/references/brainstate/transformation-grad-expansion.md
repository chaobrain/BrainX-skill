# BrainState Gradient and Autodiff Expansion

Use this reference after `skills/brainstate/SKILL.md` when a task needs gradient targets beyond the main skill's minimal parameter update, exact `return_value`/`has_aux` destructuring, state discovery in an arbitrary function, vector- or matrix-valued sensitivities, second derivatives, or an optimizer-integrated fitting step. The main skill owns the canonical `ParamState` collection and manual update; the parameter reference owns the choice between `ParamState` and `nn.Param`.

These three sources do not establish BrainCell solver, reset, or conductance-fitting semantics. Specialized dynamics belongs to the matching BrainCell or BrainPy-State skill; use this reference only for the source-supported autodiff contract around that computation.

Official sources:

- https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

## Select the derivative target

BrainState's gradient system has two independent selectors: `argnums` selects positional function arguments, as in JAX, and `grad_states` selects `State` objects. Both selectors can be active in one transform.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

### Function arguments with `argnums`

Use this path for sensitivities of explicit inputs or scalar hyperparameters. A scalar `argnums` returns one argument gradient; a list returns gradients in the selected argument order.

```python
import jax.numpy as jnp

from brainstate.transform import grad


def loss_fn(x, y, scale):
    """Simple loss function with multiple arguments."""
    return scale * jnp.sum((x - y) ** 2)


x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([0.5, 1.5, 2.5])
scale = 2.0

grad_fn_x = grad(loss_fn, argnums=0)
grad_x = grad_fn_x(x, y, scale)

grad_fn_xy = grad(loss_fn, argnums=[0, 1])
grad_x, grad_y = grad_fn_xy(x, y, scale)
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

### Arguments and State in one pass

Use both selectors when one loss depends on explicit differentiable arguments and mutable model State. In the official example, `grad_states` targets the model collection while `argnums=0` also differentiates the regularization coefficient; therefore the gradient result is grouped as `(state_grads, coeff_grad)`.

```python
import jax
import brainstate


class LinearRegressor(brainstate.nn.Module):
    """Simple linear regression model."""

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.weight = brainstate.ParamState(
            jnp.zeros((in_features, out_features))
        )
        self.bias = brainstate.ParamState(jnp.zeros((out_features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.weight.value + self.bias.value


reg_model = LinearRegressor(1)
xs = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
y_true = 3.0 * xs + 1.0


def penalized_loss(
    l2_coeff: float,
    inputs: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Loss with L2 regularization."""
    pred = reg_model(inputs)
    mse = jnp.mean((pred - target) ** 2)
    l2 = (
        jnp.sum(reg_model.weight.value ** 2)
        + jnp.sum(reg_model.bias.value ** 2)
    )
    return mse + l2_coeff * l2


grad_penalized = grad(
    penalized_loss,
    grad_states=reg_model.states(brainstate.ParamState),
    argnums=0,
    return_value=True,
)

(state_grads, coeff_grad), loss_val = grad_penalized(
    0.5,
    xs,
    y_true,
)
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

### Any `State` may be a gradient target

Gradient computation is not limited to `ParamState`. The official example targets an ordinary `State` directly:

```python
regular_state = brainstate.State(
    jnp.array(2.0),
    name="regular_state",
)


def compute_with_state(x):
    return jnp.sum((x * regular_state.value) ** 2)


grad_fn = grad(
    compute_with_state,
    grad_states=[regular_state],
)
gradient = grad_fn(jnp.array([1.0, 2.0, 3.0]))
```

Here `gradient[0]` corresponds to the first State in the supplied list. This capability does not redefine which States should be trained; it only widens the set of valid sensitivity targets.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

## Discover State in an arbitrary function

Use `StateFinder` when the differentiable computation is not an `nn.Module` and its State dependencies must be discovered. `filter` selects State types, `usage='all'` finds read and write State, and `return_type='dict'` returns a dictionary that can be passed as `grad_states`.

```python
from brainstate.transform import StateFinder


scale = brainstate.ParamState(jnp.array(1.5), name="scale")
offset = brainstate.ParamState(jnp.array(-0.2), name="offset")
cache = brainstate.State(jnp.array(0.0), name="cache")


def energy(x: jax.Array) -> jax.Array:
    """Energy function using external states."""
    shifted = x * scale.value + offset.value
    scale.value = scale.value + 0.0
    cache.value = jnp.sum(shifted)
    return jnp.sum(jnp.square(shifted))


finder = StateFinder(
    energy,
    filter=brainstate.ParamState,
    usage="all",
    return_type="dict",
)
all_param_states = finder(jnp.ones((2,)))

energy_grad = grad(
    energy,
    grad_states=all_param_states,
    return_value=True,
)
state_grads, energy_value = energy_grad(
    jnp.array([1.0, 3.0])
)
```

The finder excludes `cache` because the filter selects only `ParamState`, even though `energy` writes that State.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

## Destructure gradient returns exactly

All BrainState gradient transformations share a return structure controlled by `grad_states`, `argnums`, `has_aux`, and `return_value`. `has_aux=True` means the differentiated function returns `(loss, aux)`; `return_value=True` additionally returns the differentiated loss value from the same pass.

| `grad_states` | `argnums` | `has_aux` | `return_value` | Result |
|---|---|---:|---:|---|
| `None` | selected | `False` | `False` | `arg_grads` |
| `None` | selected | `True` | `False` | `(arg_grads, aux)` |
| `None` | selected | `False` | `True` | `(arg_grads, loss)` |
| `None` | selected | `True` | `True` | `(arg_grads, loss, aux)` |
| selected | `None` | `False` | `False` | `state_grads` |
| selected | `None` | `True` | `False` | `(state_grads, aux)` |
| selected | `None` | `False` | `True` | `(state_grads, loss)` |
| selected | `None` | `True` | `True` | `(state_grads, loss, aux)` |
| selected | selected | `False` | `False` | `(state_grads, arg_grads)` |
| selected | selected | `True` | `False` | `((state_grads, arg_grads), aux)` |
| selected | selected | `False` | `True` | `((state_grads, arg_grads), loss)` |
| selected | selected | `True` | `True` | `((state_grads, arg_grads), loss, aux)` |

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

Bundle the most information-rich case as one loss and one destructuring pattern:

```python
example_model = LinearRegressor(1)


def loss_with_metrics(
    l2_coeff: float,
    x: jax.Array,
    target: jax.Array,
):
    """Loss function that returns auxiliary metrics."""
    pred = example_model(x)
    mse = jnp.mean((pred - target) ** 2)
    l2 = jnp.sum(example_model.weight.value ** 2)
    loss = mse + l2_coeff * l2
    metrics = {
        "mae": jnp.mean(jnp.abs(pred - target)),
        "mse": mse,
        "l2": l2,
    }
    return loss, metrics


grad_complete = grad(
    loss_with_metrics,
    grad_states=example_model.states(brainstate.ParamState),
    argnums=0,
    has_aux=True,
    return_value=True,
)

(
    (state_grads, coeff_grad),
    loss_val,
    aux_metrics,
) = grad_complete(0.3, xs, y_true)
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

## Select vector, Jacobian, or second-order derivatives

BrainState provides `vector_grad`, `jacrev`, `jacfwd`, `jacobian`, and `hessian` with the same State-aware signature pattern as `grad`.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

### Sum gradients of vector outputs with `vector_grad`

`vector_grad` is for a vector-valued function and computes the sum of gradients across all output dimensions. It is not the full Jacobian.

```python
from brainstate.transform import vector_grad


def vector_fun(x):
    """Vector-valued function."""
    return jnp.array(
        [
            x[0] * x[1],
            jnp.sin(x[0]),
            x[0] ** 2 + x[1] ** 2,
        ]
    )


x0 = jnp.array([1.0, 2.0])
vgrad = vector_grad(vector_fun)
result = vgrad(x0)
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

### Compute full Jacobians

The tutorial defines `jacrev` as reverse-mode Jacobian, efficient for many inputs and few outputs; `jacfwd` is forward-mode, efficient for few inputs and many outputs; `jacobian` is an alias for `jacrev`.

```python
from brainstate.transform import jacrev, jacfwd, jacobian


def multi_output(x):
    """Function with multiple outputs."""
    return jnp.array(
        [
            x[0] * x[1],
            jnp.sin(x[0]),
            jnp.exp(x[1]),
        ]
    )


x0 = jnp.array([1.0, 2.0])
result_rev = jacrev(multi_output)(x0)
result_fwd = jacfwd(multi_output)(x0)
result_alias = jacobian(multi_output)(x0)

assert jnp.allclose(result_rev, result_fwd)
assert jnp.allclose(result_rev, result_alias)
```

The same transform can target State. In the official model example, the output-to-parameter Jacobian preserves State paths:

```python
jac_model = LinearRegressor(2)


def model_output(x):
    """Multiple outputs from a model."""
    return jac_model(x)


jac_states = jacrev(
    model_output,
    grad_states=jac_model.states(brainstate.ParamState),
)

x_input = jnp.array([1.0, 2.0])
param_jacobian = jac_states(x_input)
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

### Compute second derivatives with `hessian`

`hessian` computes second-order derivatives.

```python
from brainstate.transform import hessian


def quadratic(x):
    """Quadratic function."""
    return jnp.dot(x, x)


x0 = jnp.array([1.0, 2.0])
hess_fn = hessian(quadratic)
result = hess_fn(x0)

expected = 2 * jnp.eye(2)
assert jnp.allclose(result, expected)
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

## Bundle fitting loss, auxiliary metrics, and differentiation

The autodiff tutorial's fitting example compiles an MSE-plus-L2 objective, returns the MSE and L2 terms as auxiliary data, and builds one State-targeted gradient callable. Keep the loss scalar first and auxiliary diagnostics second when `has_aux=True`.

```python
training_model = LinearRegressor(1)

true_weight = 3.0
true_bias = 1.0
x_train = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 1)
y_train = (
    true_weight * x_train
    + true_bias
    + 0.1 * brainstate.random.normal(size=x_train.shape)
)


@brainstate.transform.jit
def training_loss(x, y):
    """MSE loss with L2 regularization."""
    pred = training_model(x)
    mse = jnp.mean((pred - y) ** 2)
    l2 = 0.01 * (
        jnp.sum(training_model.weight.value ** 2)
        + jnp.sum(training_model.bias.value ** 2)
    )
    return mse + l2, {"mse": mse, "l2": l2}


loss_grad_fn = grad(
    training_loss,
    grad_states=training_model.states(brainstate.ParamState),
    has_aux=True,
    return_value=True,
)

grads, loss_val, aux = loss_grad_fn(x_train, y_train)
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html

For `nn.Param` regularization, the parameters tutorial places the penalty inside the differentiated scalar objective. Attaching a regularizer without adding its result to the loss does not reproduce this training path:

```python
def loss_fn():
    data_loss = jnp.mean((model(x) - y) ** 2)
    return data_loss + model.reg_penalty()
```

Official source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

## Integrate with the compiled training step

The training tutorial's advanced step differentiates once, receives the loss with `return_value=True`, clips the gradient PyTree, and passes the clipped gradients to a registered optimizer. The whole step is wrapped in `brainstate.transform.jit`, so State reads and writes are tracked across the transform boundary.

```python
import braintools
from brainstate.nn import clip_grad_norm
from braintools.metric import (
    softmax_cross_entropy_with_integer_labels,
)


optimizer = braintools.optim.Adam(lr=1e-2)
optimizer.register_trainable_weights(
    model.states(brainstate.ParamState)
)
params = model.states(brainstate.ParamState)


@brainstate.transform.jit
def train_step(x, y):
    def loss_fn():
        logits = model(x)
        return softmax_cross_entropy_with_integer_labels(
            logits,
            y,
        ).mean()

    grads, loss = brainstate.transform.grad(
        loss_fn,
        params,
        return_value=True,
    )()
    grads = clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(grads)
    return loss
```

Return to `skills/brainstate/SKILL.md` for the general training structure. Open `references/braintools-optimizer-reference.md` only for optimizer or scheduler selection.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html

## Failure rules

- Match destructuring to all four controls: `grad_states`, `argnums`, `has_aux`, and `return_value`.
- When both `grad_states` and `argnums` are selected, do not flatten their gradients together; the result groups them as `(state_grads, arg_grads)`.
- With `has_aux=True`, return `(scalar_loss, aux)` from the differentiated function; do not make auxiliary metrics the differentiated result.
- Do not use `vector_grad` when the full output-by-input Jacobian is required; it sums gradients across output dimensions.
- Choose `jacrev` versus `jacfwd` from the documented input/output-size tradeoff, and remember that `jacobian` aliases `jacrev`.
- Do not assume only `ParamState` can receive gradients; any explicit `State` can be a sensitivity target.
- Do not assume an attached regularizer changes fitting unless its penalty is added to the scalar loss.
- Do not infer BrainCell integration, reset, unit, or solver semantics from these sources; route specialized dynamics to the matching BrainCell or BrainPy-State skill.

## Mirror source URLs

- https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
