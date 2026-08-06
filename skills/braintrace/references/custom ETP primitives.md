# Custom ETP primitives

Use this reference only when built-in BrainTrace ETP operations cannot express a required computation. Use parameter-transform hooks for reparameterizing an existing operation; do not register a new primitive for that case.

## Registration contract

A custom ETP primitive combines one plain JAX implementation, explicit trainable-input metadata, and four online-learning rules. Standard JAX transformation rules are derived automatically.

| API or contract | Description |
|---|---|
| `register_primitive(name, impl, ...)` | Use to create an `ETPPrimitive` from a plain JAX implementation. Set batching, trainable-input layout, non-trainable input index, and tail behavior during registration. |
| `trainable_invars_fn(static_params)` | Return `{trainable_name: invar_index}` for every trainable input present under the equation's static parameters. |
| `x_invar_index` | Identify the non-trainable input consumed by the operation. Use `None` only when no separate input exists. |
| `primitive.register_etp_rules(...)` | Register all four ETP-specific rules together. |
| `primitive.bind(*args, **static_params)` | Invoke the primitive directly after registration. |
| `gradient_enabled` | Leave `False` for genuinely trainable operations. Use `True` only for identity-like tail operations whose ETP rule is a passthrough. |

**Invariant:** The names returned by `trainable_invars_fn` are the keys used by trace storage, rule dictionaries, compiler relations, and final gradient routing. Keep them stable and complete.

## Rule families

The four registries encode the two trace factorizations supported by D-RTRL and pp-prop.

| Rule | Exact callable shape | Result |
|---|---|---|
| `dt_to_t` | `dt_to_t(hidden_dim, trace, **static_params)` | Return a trace dictionary after propagating the previous trace through the hidden-to-hidden term. |
| `xy_to_dw` | `xy_to_dw(x, hidden_dim, weights, **static_params)` | Return a dictionary containing the instantaneous hidden-to-raw-parameter Jacobian for every trainable key. |
| `init_drtrl` | `init_drtrl(x_var, y_var, weight_vars, num_hidden_state)` | Return zero-filled parameter-dimensional trace storage, one leaf per trainable key. |
| `init_pp` | `init_pp(x_var, y_var, weight_vars, num_hidden_state)` | Return one zero-filled output-dimensional df-trace array; pp-prop manages the matching x-trace separately. |

The backing dictionaries are `ETP_RULES_DT_TO_T`, `ETP_RULES_XY_TO_DW`, `ETP_RULES_INIT_DRTRL`, and `ETP_RULES_INIT_PP`. A primitive must register both initializers to support both `D_RTRL` and `pp_prop`.

## Canonical primitive workflow

The following scaled matrix multiplication includes optional bias and exercises every part of the dict-based rule contract.

```python
import jax
import jax.numpy as jnp
from braintrace import register_primitive


def _scaled_matmul_impl(*args, scale=1.0, has_bias=False):
    x, weight = args[0], args[1]
    output = scale * (x @ weight)
    if has_bias:
        output = output + args[2]
    return output


def _scaled_trainable_invars(params):
    trainable = {"weight": 1}
    if params.get("has_bias", False):
        trainable["bias"] = 2
    return trainable


scaled_mm_p = register_primitive(
    "etp_scaled_mm",
    _scaled_matmul_impl,
    batched=True,
    trainable_invars_fn=_scaled_trainable_invars,
    x_invar_index=0,
)
```

Register one rule for each online-learning operation:

```python
def _scaled_dt_to_t(
    hidden_dim, trace, *, scale=1.0, has_bias=False
):
    updated = {
        "weight": (
            trace["weight"]
            * jnp.expand_dims(hidden_dim, axis=-2)
            * scale
        )
    }
    if has_bias:
        updated["bias"] = trace["bias"] * hidden_dim
    return updated


def _scaled_xy_to_dw(
    x, hidden_dim, weights, *, scale=1.0, has_bias=False
):
    def forward(trainable):
        output = scale * (x @ trainable["weight"])
        if has_bias:
            output = output + trainable["bias"]
        return output

    _, pullback = jax.vjp(forward, weights)
    return pullback(hidden_dim)[0]


def _scaled_init_drtrl(
    x_var, y_var, weight_vars, num_hidden_state
):
    batch_size = x_var.aval.shape[0]
    traces = {
        "weight": jnp.zeros(
            (
                batch_size,
                *weight_vars["weight"].aval.shape,
                num_hidden_state,
            )
        )
    }
    if "bias" in weight_vars:
        traces["bias"] = jnp.zeros(
            (
                batch_size,
                *weight_vars["bias"].aval.shape,
                num_hidden_state,
            )
        )
    return traces


def _scaled_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros(
        (*y_var.aval.shape, num_hidden_state),
        dtype=y_var.aval.dtype,
    )


scaled_mm_p.register_etp_rules(
    dt_to_t=_scaled_dt_to_t,
    xy_to_dw=_scaled_xy_to_dw,
    init_drtrl=_scaled_init_drtrl,
    init_pp=_scaled_init_pp,
)
```

Bind the registered primitive and verify the forward result:

```python
x = jnp.ones((4, 3))
weight = jnp.ones((3, 5))
bias = jnp.full((5,), 0.1)

output = scaled_mm_p.bind(
    x, weight, bias, scale=2.0, has_bias=True
)
expected = 2.0 * (x @ weight) + bias
assert output.shape == (4, 5)
assert jnp.allclose(output, expected)
```

**Invariant:** Derive initializer shapes from the traced avals and the algorithm's trace factorization. Do not copy the example shapes into an operation with different batching, output, or parameter semantics.

## Compiler participation

`trainable_invars_fn` and `x_invar_index` make the primitive discoverable by the ETP compiler. The compiler then:

1. Identifies the primitive by object identity.
2. Locates every declared trainable input.
3. Traces each input back to its owning `ParamState` leaf.
4. Connects the primitive output to reachable, shape-compatible hidden groups.
5. Allocates algorithm-specific trace State through the registered initializer.

A primitive without `trainable_invars_fn` still supports direct binding and standard JAX transformations, and the compiler falls back to `{"weight": 1}`. Declare the layout explicitly for custom work, especially for optional or multiple trainable inputs.

Parameter participation remains operation-based:

| Goal | Operation |
|---|---|
| Include a parameter in online learning | Consume it through the registered ETP primitive. |
| Exclude a parameter from eligibility traces | Consume it through the corresponding regular JAX operation. |

## Tail behavior

`gradient_enabled` controls whether an ETP primitive may appear after another trainable primitive on the path to hidden State.

| Value | Use when | Compiler behavior |
|---|---|---|
| `False` | Use for normal trainable operations such as matrix multiplication, convolution, sparse multiplication, or LoRA. | Treat the primitive as a tail boundary and exclude an upstream ETP weight whose only path to hidden State crosses it. |
| `True` | Use only for identity-like operations such as element-wise gating parameters. | Allow an upstream relation to pass through the primitive. |

**Invariant:** Per-primitive rules cannot represent arbitrary weight-to-weight composition. Do not set `gradient_enabled=True` to silence a valid tail-boundary exclusion.

## Validation sequence

Validate a custom primitive in this order:

1. Compare `primitive.bind()` with an equivalent plain JAX forward computation.
2. Run `jax.jit`, `jax.grad`, `jax.vmap`, and JVP coverage appropriate to the operation.
3. Assert every declared trainable key appears in the four rule outputs or initializers where required.
4. Compile a minimal recurrent model with `D_RTRL` and inspect its included relation and trace shapes.
5. Compile the same model with `pp_prop` to exercise `init_pp` and factorized trace solving.
6. Compare online gradients with a reduced oracle in a regime where the selected estimator documents equivalence.
7. Test optional static layouts such as both `has_bias=False` and `has_bias=True`.

Keep user-facing quantity behavior in a wrapper that splits a `brainunit.Quantity` into mantissa and unit, applies the array operation to the mantissa, and recombines the result. Direct primitive registration alone does not define that wrapper contract.

## Sources

- [Creating Custom ETP Primitives](https://brainx.chaobrain.com/braintrace/advanced/etp_primitives.html)
