# Customizing primitive transforms

Use this reference when an existing ETP operation must consume a shape-preserving transformation of a stored parameter while eligibility traces and optimizer gradients remain attached to the raw parameter. Open `custom ETP primitives.md` first when adding transform hooks to a newly registered operation.

## Choose the transform hook

Use the hook owned by the operation and trainable factor; do not transform the stored `ParamState` in place.

| API | Transform hook |
|---|---|
| `braintrace.matmul(x, weight, bias=None, *, weight_fn=None, bias_fn=None)` | Use `weight_fn` for the matrix and `bias_fn` for bias. |
| `braintrace.element_wise(weight, *, weight_fn=None)` | Use `weight_fn` for the element-wise parameter. |
| `braintrace.conv(x, kernel, bias=None, *, ..., kernel_fn=None, bias_fn=None)` | Use `kernel_fn`, not `weight_fn`, for the convolution kernel; use `bias_fn` for bias. |
| `braintrace.sparse_matmul(x, weight, *, sparse_mat, bias=None, weight_fn=None, bias_fn=None)` | Use `weight_fn` for stored sparse values and `bias_fn` for bias. |
| `braintrace.lora_matmul(x, B, A, *, alpha=1.0, bias=None, b_fn=None, a_fn=None, bias_fn=None)` | Use `b_fn` and `a_fn` for the two LoRA factors and `bias_fn` for bias. |

All hooks default to `None`, which preserves the untransformed operation.

```python
import jax.numpy as jnp
import braintrace

x = jnp.ones((4, 3))
weight = jnp.arange(15.0).reshape(3, 5) / 10.0
bias = jnp.linspace(-1.0, 1.0, 5)

output = braintrace.matmul(
    x,
    weight,
    bias=bias,
    weight_fn=jnp.tanh,
    bias_fn=jnp.abs,
)
expected = x @ jnp.tanh(weight) + jnp.abs(bias)
assert jnp.allclose(output, expected)
```

Use hooks for masks, standardization, sign constraints, squashing, and other differentiable shape-preserving reparameterizations.

## Chain-rule placement

The forward operation uses `V = f(W_raw)`, but the trace and optimizer gradient target `W_raw`; apply the transform Jacobian exactly once inside `xy_to_dw` by differentiating the same transformed implementation with `jax.vjp`.

| ETP rule | Transform behavior |
|---|---|
| `xy_to_dw` | Differentiate through the implementation that applies the hook so `f'(W_raw)` enters once. |
| `dt_to_t` | Keep transform-free; it propagates an already raw-parameter trace through time. |
| `init_drtrl` | Keep transform-free; it allocates parameter-dimensional trace storage. |
| `init_pp` | Keep transform-free; it allocates output-dimensional df-trace storage. |

**Invariant:** Do not multiply by `f'(W_raw)` in both `xy_to_dw` and `dt_to_t`. That double-counts the transform Jacobian and produces incorrect online gradients.

## Add hooks to a registered primitive

Apply each hook in the implementation, expose it as a static primitive parameter, and make `xy_to_dw` differentiate that same forward computation. Do not change the other three ETP rules.

```python
import jax


def _scaled_matmul_impl(
    *args,
    scale=1.0,
    has_bias=False,
    weight_fn=None,
    bias_fn=None,
):
    x, weight = args[0], args[1]
    if weight_fn is not None:
        weight = weight_fn(weight)

    output = scale * (x @ weight)
    if has_bias:
        bias = args[2]
        if bias_fn is not None:
            bias = bias_fn(bias)
        output = output + bias
    return output


def _scaled_xy_to_dw(
    x,
    hidden_dim,
    weights,
    *,
    scale=1.0,
    has_bias=False,
    weight_fn=None,
    bias_fn=None,
):
    def forward(trainable):
        args = [x, trainable["weight"]]
        if has_bias:
            args.append(trainable["bias"])
        return _scaled_matmul_impl(
            *args,
            scale=scale,
            has_bias=has_bias,
            weight_fn=weight_fn,
            bias_fn=bias_fn,
        )

    _, pullback = jax.vjp(forward, weights)
    return pullback(hidden_dim)[0]
```

Register this transform-aware `xy_to_dw` with the unchanged `dt_to_t`, `init_drtrl`, and `init_pp` rules from the primitive's original contract.

**Invariant:** The implementation used by the forward primitive and the function differentiated by `xy_to_dw` must apply identical transforms and static parameters.

## Quantity behavior

User-facing wrappers split a `brainunit.Quantity` into mantissa and unit, apply the transform to the unitless mantissa, and then recombine the unit. Keep transforms shape-preserving and write them for mantissas.

```python
import brainunit as u

x_quantity = jnp.ones((4, 3)) * u.volt
weight_quantity = jnp.ones((3, 5)) * u.siemens
current = braintrace.matmul(
    x_quantity,
    weight_quantity,
    weight_fn=lambda value: value**2,
)
print(u.get_unit(current))
# A
```

## Fast-path gating

Closed-form D-RTRL fast paths compute derivatives with respect to the transformed value `V` and omit `f'(W_raw)`. Disable those paths whenever a transform hook is active so execution falls back to the correct VJP rule path.

| API or primitive family | Behavior |
|---|---|
| `FastPathRules(instant, recurrent, solve, applicable)` | Defines the closed-form parameter-dimensional kernel bundle and its applicability gate. |
| `get_fast_path_rules(primitive)` | Returns the registered fast-path bundle or `None`. |
| `etp_mm`, `etp_mv`, `etp_elemwise` | May provide a fast path; `applicable(eqn_params)` must return `False` when any transform hook is non-`None`. |
| Sparse, convolution, and LoRA primitives | Have no fast path and always use the general rule path. |

```python
from braintrace._op import etp_mm_p, get_fast_path_rules

fast_path = get_fast_path_rules(etp_mm_p)
assert fast_path is not None
assert fast_path.applicable(
    {"weight_fn": None, "bias_fn": None}
)
assert not fast_path.applicable(
    {"weight_fn": jnp.tanh, "bias_fn": None}
)
```

**Invariant:** A fast path is an optimization only. Its gate must fall back to the transform-aware rule path whenever the closed form omits part of the chain rule.

## Exactness validation

Validate both the forward transform and its online gradient. The development oracle compares a sequence gradient from the online algorithm with the BPTT gradient for the same model factory and loss.

```python
from braintrace._algorithm.oracle import (
    assert_param_gradients_close,
    bptt_param_gradients,
    online_param_gradients,
)

bptt = bptt_param_gradients(model_factory, inputs)
online = online_param_gradients(
    model_factory,
    inputs,
    algo_factory=lambda model: braintrace.D_RTRL(
        model, vjp_method="multi-step"
    ),
)
assert_param_gradients_close(online, bptt, atol=1e-4)
```

Use the private oracle for development tests, not as an application API. Choose a reduced model whose recurrent Jacobian lies in the estimator's documented exactness regime; a general D-RTRL model is not automatically BPTT-equivalent.

Check these failures explicitly:

- The hook is applied in the forward implementation but omitted from `xy_to_dw`.
- The transform derivative is applied again in `dt_to_t`.
- A convolution uses `weight_fn` instead of `kernel_fn`.
- A LoRA hook is attached to the wrong factor.
- The hook changes parameter shape.
- A fast path remains active while a transform hook is present.
- The optimizer updates a transformed copy instead of the raw `ParamState`.

## Sources

- [Customizing Parameter Transforms for ETP Operators](https://brainx.chaobrain.com/braintrace/advanced/customizing_primitive_transforms.html)
