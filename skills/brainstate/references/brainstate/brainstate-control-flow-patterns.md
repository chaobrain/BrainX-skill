# BrainState Control-Flow Patterns

Use this reference when repeated or conditional execution must preserve BrainState
`State` effects and compile to JAX control-flow primitives. It covers only the
control-flow variants routed out of the main skill; use the main skill for the
fundamentals of `State`, `.value`, Modules, and state-aware transformations.

Official source used throughout: [Control Flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html).
No other source is used in this reference.

## Imports

The tutorial imports every primitive from `brainstate.transform`:

```python
import jax
import jax.numpy as jnp

import brainstate
from brainstate.transform import (
    scan,
    checkpointed_scan,
    for_loop,
    checkpointed_for_loop,
    while_loop,
    bounded_while_loop,
    cond,
    switch,
    ifelse,
)
```

```python
# Import ProgressBar
from brainstate.transform import ProgressBar
```

Source: [Imports and Setup](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#imports-and-setup).

## Choose the Primitive

| Need | Primitive | Source-grounded selection rule |
|---|---|---|
| Carry plus per-step outputs | `scan` | Use when a carry must thread through iterations, especially for recurrent patterns or an explicit accumulator. |
| Carry plus per-step outputs over a long differentiable sequence | `checkpointed_scan` | Use when storing all intermediate activations during gradient computation is memory-prohibitive. |
| Per-step outputs without an explicit carry | `for_loop` | Use when no carry is needed, including independent items with State updates. |
| No explicit carry over a long differentiable sequence | `checkpointed_for_loop` | Use when `for_loop` syntax is appropriate but backward-pass activation storage would exhaust memory. |
| Truly unknown iteration count without reverse-mode gradients | `while_loop` | Use for standard while-loop semantics when the number of iterations is unknown and gradients are not being computed. |
| Conditional iteration with an upper bound | `bounded_while_loop` | Use for reverse-mode differentiation, protection against infinite loops, or predictable compilation characteristics. |
| Two branches | `cond` | Select one lazily evaluated branch from a boolean or numeric scalar predicate. |
| Branch selected by integer index | `switch` | Select among multiple callables; the index is clamped to the valid branch range. |
| Multiple predicates or an else branch | `ifelse` | Select among mutually exclusive conditions; make the last condition `True` for a default branch. |

Sources: [scan vs for_loop](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#comparison-scan-vs-for-loop), [while_loop vs bounded_while_loop](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#comparison-while-loop-vs-bounded-while-loop), and [Conditional Control Flow summary](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#id3).

## Scans and Collected Outputs

### `scan`

`scan` iterates over the leading axis of a sequence, threads a carry through the
iterations, collects each step's output, and properly handles `State` objects.

```text
scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    pbar: ProgressBar | int | None = None,
) -> Tuple[Carry, Y]
```

- `f` has the form `(carry, x) -> (new_carry, output)`.
- `length` is inferred from `xs` when omitted.
- `reverse=True` iterates in reverse order.
- `unroll=1` disables unrolling; `unroll=True` fully unrolls.
- `pbar` accepts a `ProgressBar` or an integer update frequency.

This source example bundles a stateful update with stacked PyTree outputs. The
explicit carry is unused, so it remains `None`; the mutable statistics live in
the Module's `ShortTermState` objects.

```python
class RunningStats(brainstate.nn.Module):
    """Maintain running mean and variance."""

    def __init__(self):
        super().__init__()
        self.count = brainstate.ShortTermState(jnp.array(0))
        self.mean = brainstate.ShortTermState(jnp.array(0.0))
        self.m2 = brainstate.ShortTermState(jnp.array(0.0))  # sum of squared differences

    def update(self, x):
        """Update statistics with new value using Welford's algorithm."""
        self.count.value = self.count.value + 1
        delta = x - self.mean.value
        self.mean.value = self.mean.value + delta / self.count.value
        delta2 = x - self.mean.value
        self.m2.value = self.m2.value + delta * delta2

        variance = self.m2.value / self.count.value
        return {'mean': self.mean.value, 'var': variance}


stats = RunningStats()


def stats_body(carry, x):
    result = stats.update(x)
    return carry, result


data = jnp.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
_, history = scan(stats_body, init=None, xs=data)
```

Source: [1.1 scan: Stateful Scanning with Carry](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#scan-stateful-scanning-with-carry).

### `for_loop`

`for_loop` accepts variadic inputs sliced along axis 0 and internally uses
`scan` with `None` as the carry. It stacks the value returned at each iteration
along axis 0: scalar steps produce shape `(n,)`, and `(d,)` steps produce
`(n, d)`.

```text
for_loop(
    f: Callable[..., Y],
    *xs,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    pbar: Optional[ProgressBar | int] = None
) -> Y
```

Use State for iteration-to-iteration effects when no explicit carry is needed:

```python
class Accumulator(brainstate.nn.Module):
    """Simple accumulator that tracks total and count."""

    def __init__(self):
        super().__init__()
        self.total = brainstate.ShortTermState(jnp.array(0.0))
        self.count = brainstate.ShortTermState(jnp.array(0))

    def process(self, x):
        self.total.value = self.total.value + x
        self.count.value = self.count.value + 1
        return self.total.value / self.count.value  # running average


acc = Accumulator()

data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
running_averages = for_loop(acc.process, data)
```

Source: [1.3 for_loop: Simplified Loop Without Carry](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#for-loop-simplified-loop-without-carry).

## Checkpointed Sequence Control Flow

The checkpointed variants trade recomputation for backward-pass memory. The
tutorial states that regular `scan` stores `O(n)` intermediate activations,
whereas `checkpointed_scan` stores `O(log_base(n))` checkpoints and recomputes
intermediate values during the backward pass. A smaller `base` saves more memory
but increases recomputation.

### `checkpointed_scan`

```text
checkpointed_scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: Optional[int] = None,
    base: int = 16,
    pbar: Optional[ProgressBar | int] = None,
) -> Tuple[Carry, Y]
```

The implementation uses a hierarchical scheme in which `max_steps = base^k`
for some `k`. Use it for a long recurrent computation whose gradients would
otherwise retain every intermediate activation:

```python
class RecurrentCell(brainstate.nn.Module):
    """Simple recurrent cell with hidden state."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = brainstate.ParamState(jax.random.normal(
            jax.random.PRNGKey(0), (hidden_size, hidden_size)
        ))

    def step(self, hidden, x):
        """Single recurrent step."""
        new_hidden = jnp.tanh(jnp.dot(self.weight.value, hidden) + x)
        return new_hidden


# Create a cell and input sequence
cell = RecurrentCell(hidden_size=32)
sequence_length = 100
inputs = jax.random.normal(jax.random.PRNGKey(1), (sequence_length, 32))


def rnn_body(hidden, x):
    new_hidden = cell.step(hidden, x)
    return new_hidden, new_hidden


# Use checkpointed scan for memory efficiency during gradient computation
init_hidden = jnp.zeros(32)
final_hidden, all_hiddens = checkpointed_scan(
    rnn_body,
    init=init_hidden,
    xs=inputs,
    base=8  # Checkpoint every 8 steps
)
```

Source: [1.2 checkpointed_scan: Memory-Efficient Scanning](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#checkpointed-scan-memory-efficient-scanning).

### `checkpointed_for_loop`

```text
checkpointed_for_loop(
    f: Callable[..., Y],
    *xs: X,
    length: Optional[int] = None,
    base: int = 16,
    pbar: Optional[ProgressBar | int] = None,
) -> Y
```

This variant combines `for_loop`'s no-carry interface with checkpointing during
gradient computation:

```python
class ExpMovingAverage(brainstate.nn.Module):
    """Exponential moving average."""

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.ema = brainstate.ShortTermState(jnp.array(0.0))
        self.initialized = brainstate.ShortTermState(jnp.array(False))

    def update(self, x):
        # Initialize with first value
        self.ema.value = jnp.where(
            self.initialized.value,
            self.alpha * x + (1 - self.alpha) * self.ema.value,
            x
        )
        self.initialized.value = True
        return self.ema.value


ema = ExpMovingAverage(alpha=0.3)

# Generate noisy signal
signal = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 200)) + 0.2 * brainstate.random.normal(size=(200,))

# Process with checkpointed for_loop
smoothed = checkpointed_for_loop(ema.update, signal, base=10)
```

Source: [1.4 checkpointed_for_loop: Memory-Efficient For Loop](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#checkpointed-for-loop-memory-efficient-for-loop).

## Conditional Iteration

### `while_loop`

`while_loop` is the stateful version of `jax.lax.while_loop`:

```text
while_loop(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T
) -> T
```

The condition must be read-only: `cond_fun` cannot modify State. The loop value
must preserve its shape and dtype. `while_loop` is not reverse-mode
differentiable; select `bounded_while_loop` when reverse-mode gradients are
required. State changes belong in `body_fun`, as in this source example:

```python
class IterativeRefiner(brainstate.nn.Module):
    """Iteratively refine an estimate using Newton's method."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.iterations = brainstate.ShortTermState(jnp.array(0))

    def refine(self, x):
        """Newton's method step for computing sqrt(target)."""
        self.iterations.value = self.iterations.value + 1
        return 0.5 * (x + self.target / x)


# Compute square root of 2 using Newton's method
refiner = IterativeRefiner(target=2.0)


def cond_f(x):
    # Continue until error is small enough
    return jnp.abs(x * x - refiner.target) > 1e-6


def body(x):
    return refiner.refine(x)


result = while_loop(cond_f, body, init_val=1.0)
```

Source: [2.1 while_loop: Dynamic Conditional Iteration](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#while-loop-dynamic-conditional-iteration).

### `bounded_while_loop`

```text
bounded_while_loop(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T,
    *,
    max_steps: int,
    base: int = 16,
)
```

`max_steps` bounds iteration. For `base`, the tutorial gives the tradeoff as
larger values compiling faster but running slightly slower, and smaller values
compiling slower but running faster; compilation scales with
`math.ceil(math.log(max_steps, base))`.

The tutorial demonstrates reverse-mode differentiation with this source script:

```python
def smooth_threshold(x, threshold=5.0, lr=0.5, max_steps=20):
    """Smoothly approach threshold using gradient descent."""

    def cond_fn(val):
        return val < threshold - 0.1

    def body(val):
        # Gradient of loss = (val - threshold)^2
        grad = 2 * (val - threshold)
        return val - lr * grad

    return bounded_while_loop(cond_fn, body, x, max_steps=max_steps)


# Compute gradient
x = 0.0
value, grad = jax.value_and_grad(smooth_threshold)(x)
```

Source: [2.2 bounded_while_loop: While Loop with Maximum Steps](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#bounded-while-loop-while-loop-with-maximum-steps).

## Conditional Branches

All three branch primitives track State reads and writes. The selected branch is
executed lazily, so put branch-specific State effects inside its callable.

### `cond`

```text
cond(
    pred,
    true_fun: Callable,
    false_fun: Callable,
    *operands
)
```

`pred` is a boolean scalar, or a numeric scalar for which non-zero means `True`.
Both branches must return the same PyTree structure.

```python
class BranchTracker(brainstate.nn.Module):
    """Track which branches were taken."""

    def __init__(self):
        super().__init__()
        self.true_count = brainstate.ShortTermState(jnp.array(0))
        self.false_count = brainstate.ShortTermState(jnp.array(0))

    def true_branch(self, x):
        self.true_count.value = self.true_count.value + 1
        return x * 2

    def false_branch(self, x):
        self.false_count.value = self.false_count.value + 1
        return x / 2


tracker = BranchTracker()

# Test multiple values
values = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0])
results = []

for v in values:
    result = cond(v > 0, tracker.true_branch, tracker.false_branch, v)
    results.append(result)
```

Source: [3.1 cond: Binary Conditional](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#cond-binary-conditional-if-else).

### `switch`

```text
switch(
    index,
    branches: Sequence[Callable],
    *operands
)
```

`branches` must contain at least one callable. Negative indices are clamped to
`0`; indices at or above `len(branches)` are clamped to
`len(branches) - 1`.

```python
def operation_0(x):
    return x + 1


def operation_1(x):
    return x * 2


def operation_2(x):
    return x ** 2


def operation_3(x):
    return -x


operations = [operation_0, operation_1, operation_2, operation_3]

x = 5.0
for i in range(len(operations)):
    result = switch(i, operations, x)
```

Source: [3.2 switch: Multi-Way Branching](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#switch-multi-way-branching).

### `ifelse`

```text
ifelse(
    conditions,
    branches,
    *operands,
    check_cond: bool = True
)
```

`conditions` should be mutually exclusive, `branches` must have the same length,
and `check_cond=True` verifies that exactly one condition is true. The official
default-branch pattern is:

```python
ifelse(
    [x > 10, x > 5, True],  # last condition is always True
    [large_fn, medium_fn, small_fn],
    x
)
```

Source: [3.3 ifelse: Multi-Condition If/Elif/Else](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html#ifelse-multi-condition-if-elif-else).
