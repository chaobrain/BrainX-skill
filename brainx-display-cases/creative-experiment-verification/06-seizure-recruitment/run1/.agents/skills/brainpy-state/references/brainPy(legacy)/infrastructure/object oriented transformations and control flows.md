# Legacy BrainPy object-oriented transformations and control flows

Use this reference when a legacy BrainPy program mutates `bm.Variable` objects
inside JIT compilation, automatic differentiation, vectorization, branching, or
loops. Use `brainpy.math` object-aware transformations; do not translate the
program to `brainpy.state` or BrainState while maintaining its legacy contract.

## Make mutable values discoverable

`BrainPyObject` owns nested objects and `Variable` leaves so BrainPy's
transformations can separate mutable dynamic values from static Python
structure.

| API | Description |
|---|---|
| `bp.BrainPyObject(name=None)` | Subclass when an object owns mutable variables or child BrainPy objects that transformations must discover. |
| `bm.Variable(value_or_size, dtype=None, batch_axis=None, *, axis_names=None)` | Use for mutable dynamical values; read or replace the stored array through `.value`. |
| `bm.TrainVar(value_or_size, dtype=None, batch_axis=None, *, axis_names=None)` | Use to mark a mutable value as trainable for legacy gradient and optimizer workflows. |
| `bm.Parameter(value_or_size, dtype=None, batch_axis=None, *, axis_names=None)` | Use for a parameter pointer that is not selected as a `TrainVar`. |
| `bm.VariableView(value, index)` | Use when a transformed object must expose a view into part of another variable. |
| `bm.NodeList(seq=())` | Store child `BrainPyObject` instances in a transformation-aware sequence. |
| `bm.NodeDict(*args, check_unique=False, **kwargs)` | Store child `BrainPyObject` instances in a transformation-aware mapping. |
| `bm.VarList(seq=())` | Store variables in a transformation-aware sequence. |
| `bm.VarDict(*args, **kwargs)` | Store variables in a transformation-aware mapping. |
| `bm.function(f=None, nodes=None, dyn_vars=None, name=None)` | Wrap a Python function as a `BrainPyObject` while declaring captured nodes or dynamic variables. |

```python
import brainpy as bp
import brainpy.math as bm


class Accumulator(bp.BrainPyObject):
    def __init__(self):
        super().__init__()
        self.total = bm.Variable(bm.zeros(1))

    @bm.cls_jit
    def __call__(self, value):
        self.total.value += value
        return self.total.value


accumulator = Accumulator()
assert bm.allclose(accumulator(2.0), bm.array([2.0]))
assert bm.allclose(accumulator(3.0), bm.array([5.0]))
```

Assign child objects and variables to attributes or the transformation-aware
containers. A plain Python list or dictionary can hide mutable leaves from
legacy object traversal.

## Choose an object-aware transformation

These APIs mirror JAX transformations while also tracking variables owned by a
legacy BrainPy object.

| API | Description |
|---|---|
| `bm.jit(func, static_argnums=None, static_argnames=None, donate_argnums=(), inline=False, keep_unused=False, **kwargs)` | JIT-compile a pure function, bound method, `DynamicalSystem`, or `BrainPyObject` call while preserving discovered variable mutation. |
| `bm.cls_jit(...)` | Decorate a class method so the compiled callable remains bound to the instance. |
| `bm.grad(func=None, grad_vars=None, argnums=None, holomorphic=False, allow_int=False, has_aux=None, return_value=False)` | Differentiate a scalar-returning function with respect to captured variables, positional arguments, or both. |
| `bm.vector_grad(...)` | Use when the differentiated function has a vector-valued result. |
| `bm.jacrev(...)` | Compute a reverse-mode Jacobian for a function or class object. |
| `bm.jacobian(...)` | Use the legacy reverse-mode Jacobian alias. |
| `bm.jacfwd(...)` | Compute a forward-mode Jacobian. |
| `bm.hessian(...)` | Compute a dense Hessian. |

```python
class Quadratic(bp.BrainPyObject):
    def __init__(self):
        super().__init__()
        self.weight = bm.TrainVar(bm.array([1.0]))

    def __call__(self, target):
        return bm.mean((self.weight - target) ** 2)


objective = Quadratic()
value_and_grad = bm.grad(
    objective,
    grad_vars=objective.weight,
    return_value=True,
)
weight_grad, loss = value_and_grad(bm.array([3.0]))

assert weight_grad.shape == objective.weight.shape
assert loss.ndim == 0
```

Use `grad_vars` for captured `Variable` objects and `argnums` for differentiable
positional arguments. When both are supplied, `bm.grad` returns variable and
argument gradients together. Set `return_value=True` only when the loss is also
needed; set `has_aux=True` only when the function returns auxiliary data in
addition to its scalar differentiable value.

## Choose a compiled conditional

Ordinary Python branching is valid only when its predicate is static during
tracing. Use structural control flow when the predicate depends on a traced
array or `bm.Variable`.

| API | Description |
|---|---|
| `bm.where(condition, x, y)` | Use for elementwise selection over scalars, vectors, or higher-dimensional arrays. |
| `bm.cond(pred, true_fun, false_fun, operands=())` | Use for one scalar boolean branch whose functions may read or mutate discovered variables. |
| `bm.ifelse(conditions, branches, operands=None)` | Use for an if/elif/else chain; provide one more branch than conditions for the final else case. |

```python
class SignedStep(bp.BrainPyObject):
    def __init__(self):
        super().__init__()
        self.total = bm.Variable(bm.zeros(1))

    def __call__(self, predicate):
        increment = bm.ifelse(
            conditions=predicate,
            branches=[lambda: 1.0, lambda: -1.0],
        )
        self.total.value += increment
        return self.total.value


step = bm.jit(SignedStep())
step(bm.asarray(True))
```

Do not write `if variable_condition:` inside `bm.jit`; traced boolean conversion
raises an error. Use `bm.where` when both choices are elementwise data and
`bm.cond` or `bm.ifelse` when branch functions own the computation.

## Choose a compiled loop

Structural loops keep tracing cost bounded and preserve legacy Variable
mutation without unrolling a long Python loop into the compiled program.

| API | Description |
|---|---|
| `bm.for_loop(body_fun, operands, reverse=False, unroll=1, jit=None, progress_bar=False)` | Iterate over axis 0 of one operand or matching operand PyTrees and stack every value returned by `body_fun`. |
| `bm.scan(body_fun, init, operands, reverse=False, unroll=1, remat=False, progress_bar=False)` | Use a `jax.lax.scan`-style explicit carry together with discovered variable mutation. |
| `bm.while_loop(body_fun, cond_fun, operands)` | Repeat while `cond_fun` is true; returned operands must preserve the input structure, shapes, and dtypes. |
| `bm.ProgressBar(freq=None, count=None, desc=None, ...)` | Configure progress reporting for a compiled `for_loop` or `scan` when callbacks are supported by the execution context. |

```python
total = bm.Variable(bm.zeros(1))


def accumulate(value):
    total.value += value
    return total.value


history = bm.for_loop(
    body_fun=accumulate,
    operands=bm.arange(1.0, 5.0),
)

assert history.shape == (4, 1)
assert bm.allclose(total.value, bm.array([10.0]))
```

`for_loop` gathers outputs; those outputs are not the loop carry. By contrast,
`while_loop` returns updated operands and therefore must return what it
receives, with stable structure and array metadata. Prefer either structural
loop over a long Python loop inside `bm.jit`, which traces each iteration and
can make first-call compilation excessive.

## Source-backed failures

- Make a differentiated loss scalar. Use `vector_grad` or a Jacobian API when
  the mathematical result is intentionally non-scalar.
- Keep array shapes and dtypes stable across every conditional branch and
  `while_loop` iteration.
- Make all mutated variables discoverable through a `BrainPyObject`, declared
  `dyn_vars`, or transformation-aware container.
- Do not mutate ordinary Python state and expect compiled execution to replay
  that side effect on every call.
- Keep optimizer-owned variables in the transformed variable set when a legacy
  optimizer update is JIT-compiled; open `../training/optimizers.md` for that lifecycle.

## Official sources

- `https://brainpy.readthedocs.io/apis/brainpy.math.oo_transform.html`
- `https://brainpy.readthedocs.io/tutorial_math/control_flows.html`
- Generated object, variable, transformation, and control-flow pages linked
  from the official API index.
