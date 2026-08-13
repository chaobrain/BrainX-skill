# ETP operators

Use this reference when prebuilt `braintrace.nn` layers cannot express a custom
layer or model and its trainable operations must participate in eligibility
trace propagation. Use only the built-in operators here; open
`custom ETP primitives.md` when none can express the computation.

## Control parameter participation

ETP participation is selected by the operation that consumes a parameter, not
by a special parameter class.

| Operation path | Result |
|---|---|
| A `brainstate.ParamState` value is consumed by a `braintrace.*` ETP operator. | The compiler can include that parameter in eligibility-trace computation. |
| A `brainstate.ParamState` value is consumed only by an ordinary JAX operation. | The parameter is automatically excluded from ETP participation. |

```python
import jax
import jax.numpy as jnp
import brainstate
import braintrace


class MyRNN(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_rec = brainstate.ParamState(jnp.eye(4))
        self.w_in = brainstate.ParamState(jnp.ones((3, 4)))
        self.h = brainstate.ShortTermState(jnp.zeros(4))

    def update(self, x):
        # Ordinary matmul: w_in remains trainable but is excluded from ETP.
        input_drive = x @ self.w_in.value

        # ETP matmul: w_rec participates in eligibility traces.
        recurrent_drive = braintrace.matmul(
            self.h.value,
            self.w_rec.value,
        )
        self.h.value = jax.nn.tanh(input_drive + recurrent_drive)
        return self.h.value
```

**Participation invariant:** Every `brainstate.ParamState` is eligible, but
only an ETP primitive marks its operation for online learning. Changing the
parameter subclass does not include or exclude it.

## Choose an operator

Choose the operator that matches the parameterized computation; do not replace
a direct built-in operator with a custom primitive.

| API | Use and behavior |
|---|---|
| `braintrace.matmul(x, weight, bias=None)` | Use for dense matrix multiplication. It dispatches to batched or unbatched primitives from `x.ndim`. |
| `braintrace.grouped_matmul(...)` | Use for grouped or block-diagonal matrix multiplication. |
| `braintrace.embedding(...)` | Use for an ETP-aware embedding lookup whose gather weight is trainable. |
| `braintrace.einsum(...)` | Use for a two-operand Einstein contraction that is linear in the trainable weight. |
| `braintrace.element_wise(weight, *, weight_fn=None)` | Use for diagonal element-wise parameter operations such as gates, time constants, or thresholds. `weight_fn=None` is identity; otherwise provide a JAX-differentiable transform. |
| `braintrace.conv(x, kernel, bias=None, *, strides, padding, ...)` | Use for convolution through `jax.lax.conv_general_dilated` options. The input must include a batch dimension. |
| `braintrace.sparse_matmul(x, weight_data, *, sparse_mat, bias=None)` | Use when connectivity structure is sparse. Pass a raw numeric or boolean array as `x`, trainable nonzero values as `weight_data`, and a BrainEvent `DataRepresentation` as `sparse_mat`. |
| `braintrace.lora_matmul(x, B, A, *, alpha=1.0, bias=None)` | Use for a low-rank update with `B` shaped `(in, rank)` and `A` shaped `(rank, out)`. Only the low-rank factors are trained. |

`element_wise` is the only listed primitive whose compiler registration has
`gradient_enabled=True`. The compiler descends through it while walking from
output to hidden State, so it does not form a tail boundary for upstream ETP
weights.

## Respect dispatch and shape contracts

Use a direct batched operator when the model already has a batch axis; this
keeps the primitive visible and its shape rule explicit.

```python
import jax.numpy as jnp
import braintrace

weight = jnp.ones((3, 5))

single_output = braintrace.matmul(jnp.ones(3), weight)
batched_output = braintrace.matmul(jnp.ones((4, 3)), weight)

assert single_output.shape == (5,)
assert batched_output.shape == (4, 5)

kernel = jnp.ones((4, 3, 8))
conv_output = braintrace.conv(
    jnp.ones((2, 16, 3)),
    kernel,
    strides=(1,),
    padding='SAME',
    dimension_numbers=('NWC', 'WIO', 'NWC'),
)
assert conv_output.shape == (2, 16, 8)
```

`matmul` uses its unbatched primitive when `x.ndim == 1` and its batched
primitive when `x.ndim >= 2`. `conv` is the critical exception: always supply a
batch dimension.

For sparse multiplication, keep the three roles separate:

| Argument | Contract |
|---|---|
| `x` | Pass a raw numeric or boolean array. Do not wrap spikes in `brainevent.BinaryArray`; `sparse_matmul` dispatches from array rank and expects numerical array operations. |
| `weight_data` | Pass the trainable values for the nonzero connections, with one value per stored sparse entry. |
| `sparse_mat` | Pass a `brainevent.DataRepresentation`, such as `brainevent.CSR`, that owns the fixed indices, orientation, shape, and sparse kernels. |

```python
import brainevent

indices = jnp.array([1, 2, 0], dtype=jnp.int32)
indptr = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
topology = brainevent.CSR(
    (jnp.ones(3), indices, indptr),
    shape=(3, 3),
)
spikes = jnp.array([True, False, True])
weight_data = jnp.ones(3)

current = braintrace.sparse_matmul(
    spikes,
    weight_data,
    sparse_mat=topology,
)
assert current.shape == (3,)
```

The structure's own `data` can be a construction placeholder; the separately
passed `weight_data` is the trainable operand seen by the ETP primitive.

## Preserve physical units

Every user-facing ETP operator accepts `brainunit.Quantity` inputs by separating
mantissas and units for primitive execution and recombining the result.

```python
import brainunit as u

x = jnp.ones((4, 3)) * u.volt
weight = jnp.ones((3, 5)) * u.siemens
bias = jnp.zeros(5) * u.amp

output = braintrace.matmul(x, weight, bias=bias)
assert u.get_unit(output) == u.amp
```

**Unit invariant:** A bias must be dimensionally compatible with the combined
input and weight unit. Do not strip units to bypass that check.

## Preserve compiler recognition under JAX transforms

ETP operators support `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.jvp` for
ordinary numerical transformation.

| Transform | Supported result |
|---|---|
| `jax.jit(...)` | Compiles the numerical operator. |
| `jax.grad(...)` | Differentiates with respect to operator arguments. |
| `jax.vmap(...)` | Maps the numerical operator over an added axis. |
| `jax.jvp(...)` | Produces primal and forward-mode tangent outputs. |

**Compiler invariant:** Numerical compatibility does not guarantee that the ETP
compiler still recognizes the marker. A transform can decompose it into
ordinary JAX operations. Inspect the compiled ETP graph and prefer the direct
batched BrainTrace operator when recognition matters.

## Diagnose operator failures

| Exception | Meaning |
|---|---|
| `CompilationError` | Compilation failed while discovering or constructing the ETP graph. |
| `NotSupportedError` | The requested operation or composition is unsupported. |

Open `pre-built-braintrace-layer.md` before hand-building a common layer. Open
`customizing_primitive_transforms.md` when a weight reparameterization needs
trace-aware transform hooks. Open `custom ETP primitives.md` only when the
built-in operator set cannot represent the computation.

## Sources

- [ETP Operators & Core Types](https://brainx.chaobrain.com/braintrace/apis/concepts.html)
- [Operators for Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/five_primitive_functions.html)

The participation rules, operator behaviors, shape examples, unit contract,
and transform warning above are trimmed and reorganized from the official
pages without changing their API names.
