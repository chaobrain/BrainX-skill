# BrainState control-flow patterns

Use this reference when loops or branches must preserve BrainState `State`
effects under JAX transformations, or when reverse-mode differentiation through
a long rollout exhausts memory. Use the root skill for `State`, `.value`,
Modules, gradients, and JIT fundamentals.

## Choose the primitive

| API | Use when |
|---|---|
| `scan(...)` | An explicit carry must thread through recurrent steps or accumulation while per-step outputs are collected. |
| `checkpointed_scan(...)` | Reverse-mode differentiation through a long carried sequence makes activation memory the bottleneck. |
| `for_loop(...)` | Per-step outputs must be collected while iteration-to-iteration effects live in Module `State` instead of an explicit carry. |
| `checkpointed_for_loop(...)` | A long State-driven differentiated rollout exhausts memory and the body needs no explicit carry. |
| `while_loop(...)` | Iteration continues until a runtime condition becomes false and reverse-mode gradients are not required. |
| `bounded_while_loop(...)` | Conditional iteration needs reverse-mode gradients, an iteration safety bound, or predictable compilation. |
| `cond(...)` | A scalar predicate selects one of two lazily executed branches. |
| `switch(...)` | An integer selects one of several callables and out-of-range indices may be clamped. |
| `ifelse(...)` | Mutually exclusive predicates express `if`/`elif`/`else`, with a final `True` condition as the default branch. |

Source: [Control flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html).

## Run fixed-length sequences

Fixed-length loop transforms slice inputs along axis 0, preserve tracked State
updates, and stack each step's output along a new leading axis.

| API | Description |
|---|---|
| `scan(f, init, xs, length=None, reverse=False, unroll=1, pbar=None)` | Use when `f(carry, x)` must return `(new_carry, output)`; it returns the final carry and stacked outputs. |
| `for_loop(f, *xs, length=None, reverse=False, unroll=1, pbar=None)` | Use when `f(*x)` needs no explicit carry; it slices every input along axis 0 and returns stacked outputs. |
| `checkpointed_scan(f, init, xs, length=None, base=16, pbar=None)` | Use the `scan` contract with gradient checkpointing; it returns the same carry and output structure while rematerializing intermediate activations during the backward pass. |
| `checkpointed_for_loop(f, *xs, length=None, base=16, pbar=None)` | Use the `for_loop` contract with gradient checkpointing; it returns the same stacked output structure while rematerializing intermediate activations during the backward pass. |

Use `scan` when the carry is an ordinary function value. Use `for_loop` when
iteration-to-iteration effects already live in Module `State`:

```python
import jax.numpy as jnp
import brainstate


class Accumulator(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.total = brainstate.ShortTermState(jnp.array(0.0))
        self.count = brainstate.ShortTermState(jnp.array(0))

    def update(self, x):
        self.total.value = self.total.value + x
        self.count.value = self.count.value + 1
        return self.total.value / self.count.value


accumulator = Accumulator()
averages = brainstate.transform.for_loop(
    accumulator.update,
    jnp.array([1.0, 2.0, 3.0, 4.0]),
)

assert jnp.allclose(averages, jnp.array([1.0, 1.5, 2.0, 2.5]))
assert accumulator.count.value == 4
```

**Invariant:** A scalar body result produces shape `(steps,)`; a body result of
shape `(d,)` produces shape `(steps, d)`. Pass `reverse=True` to traverse inputs
in reverse, `unroll` to control loop unrolling, and either an update frequency
or `ProgressBar` through `pbar` when progress reporting is required.

## Checkpoint long training rollouts

Checkpointed loops preserve the forward result but trade backward-pass
recomputation for lower activation memory, so place the checkpointed loop
inside the differentiated loss.

| API | Description |
|---|---|
| `checkpointed_for_loop(f, *xs, base=16)` | Use as a drop-in replacement for `for_loop` when a long State-driven BPTT rollout causes out-of-memory failures. |
| `checkpointed_scan(f, init, xs, base=16)` | Use as a drop-in replacement for `scan` when the differentiated rollout carries an explicit recurrent value. |
| `brainstate.transform.grad(loss_fn, grad_states, return_value=True)` | Differentiate the loss containing the checkpointed loop; it returns gradients keyed like `grad_states` and the loss from the same pass. |
| `brainstate.transform.jit(train_step)` | Compile the complete state initialization, differentiated rollout, and parameter update as one training step. |

This training pattern keeps the source how-to's ordinary and checkpointed paths
adjacent so the memory strategy is a construction-time choice:

```python
import brainpy
import brainstate
import braintools
import brainunit as u


class SNN(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.input = brainstate.nn.Sequential(
            brainstate.nn.Linear(
                n_in,
                n_rec,
                w_init=braintools.init.KaimingNormal(unit=u.mA),
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            brainpy.state.Expon(
                n_rec,
                tau=5.0 * u.ms,
                g_initializer=braintools.init.Constant(0.0 * u.mA),
            ),
        )
        self.recurrent = brainpy.state.LIF(
            n_rec,
            tau=20.0 * u.ms,
            V_rest=0.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_th=1.0 * u.mV,
            spk_fun=braintools.surrogate.ReluGrad(),
        )
        self.readout = brainstate.nn.Linear(
            n_rec,
            n_out,
            w_init=braintools.init.KaimingNormal(),
        )
        self.output = brainpy.state.Expon(
            n_out,
            tau=10.0 * u.ms,
            g_initializer=braintools.init.Constant(0.0),
        )

    def update(self, spikes):
        spikes = self.recurrent(self.input(spikes))
        return self.output(self.readout(spikes))


def make_train_step(
    net,
    x_data,
    y_data,
    batch_size,
    *,
    use_checkpoint=True,
    base=16,
):
    params = net.states(brainstate.ParamState)
    optimizer = braintools.optim.Adam(lr=2e-3)
    optimizer.register_trainable_weights(params)

    def loss_fn():
        if use_checkpoint:
            predictions = brainstate.transform.checkpointed_for_loop(
                net.update,
                x_data,
                base=base,
            )
        else:
            predictions = brainstate.transform.for_loop(
                net.update,
                x_data,
            )

        logits = u.math.mean(predictions, axis=0)
        return braintools.metric.softmax_cross_entropy_with_integer_labels(
            logits,
            y_data,
        ).mean()

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

    return train_step


with brainstate.environ.context(dt=1.0 * u.ms):
    n_in, n_rec, n_out = 80, 8, 2
    num_steps, batch_size = 400, 64
    x_data = (
        brainstate.random.rand(num_steps, batch_size, n_in)
        < 5.0 * u.Hz * brainstate.environ.get_dt()
    ).astype(float)
    y_data = u.math.asarray(
        brainstate.random.rand(batch_size) < 0.5,
        dtype=int,
    )
    net = SNN(n_in, n_rec, n_out)
    train_step = make_train_step(
        net,
        x_data,
        y_data,
        batch_size=batch_size,
        use_checkpoint=True,
        base=16,
    )
    loss = train_step()
```

Use plain `for_loop` or `scan` by default. Switch only when reverse-mode
gradients through a long rollout make activation memory the binding constraint.
Tune `base` by measuring peak memory and step time: it controls checkpoint
granularity, so lower memory requires more backward recomputation.

**Invariant:** Initialize recurrent State before each rollout, keep the
checkpointed loop inside `loss_fn`, differentiate that loss, then JIT the whole
training step. Checkpointing changes the memory/compute profile, not the
predictions or loss contract.

Source: [How to train through long rollouts without exhausting memory](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html).

## Run conditional iteration

Conditional loops preserve a fixed loop-value structure; choose the bounded
form when reverse-mode differentiation or an explicit iteration ceiling
matters.

| API | Description |
|---|---|
| `while_loop(cond_fun, body_fun, init_val)` | Use for an unknown iteration count without reverse-mode gradients; `cond_fun` must only read State, while State writes belong in `body_fun`. |
| `bounded_while_loop(cond_fun, body_fun, init_val, *, max_steps, base=16)` | Use when the loop needs reverse-mode differentiation or a hard upper bound; it stops when the condition is false or `max_steps` is reached. |

```python
import jax.numpy as jnp
import brainstate


class Counter(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = brainstate.ShortTermState(jnp.array(0))

    def increment(self, value):
        self.steps.value = self.steps.value + 1
        return value + 1


counter = Counter()
result = brainstate.transform.while_loop(
    lambda value: value < 5,
    counter.increment,
    jnp.array(0),
)

assert result == 5
assert counter.steps.value == 5
```

**Invariant:** `cond_fun` must not mutate State, and every body result must keep
the loop value's PyTree structure, shape, and dtype stable. Do not use
`while_loop` for reverse-mode gradients; provide `max_steps` and use
`bounded_while_loop`.

For `bounded_while_loop`, `base` controls its recursive lowering rather than
gradient-checkpoint spacing. Larger values compile faster but can run slightly
slower; smaller values compile slower but can run faster.

## Select conditional branches

Branch transforms execute only the selected callable while tracking its State
reads and writes.

| API | Description |
|---|---|
| `cond(pred, true_fun, false_fun, *operands)` | Use for two branches selected by a boolean or numeric scalar; both branches must return the same PyTree structure. |
| `switch(index, branches, *operands)` | Use for integer-indexed multi-way branching; it clamps the index to `[0, len(branches) - 1]`. |
| `ifelse(conditions, branches, *operands, check_cond=True)` | Use for mutually exclusive predicates; with checking enabled, it verifies that exactly one condition is true. |

```python
import jax.numpy as jnp
import brainstate


class BranchTracker(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.positive = brainstate.ShortTermState(jnp.array(0))
        self.nonpositive = brainstate.ShortTermState(jnp.array(0))

    def double(self, x):
        self.positive.value = self.positive.value + 1
        return x * 2

    def halve(self, x):
        self.nonpositive.value = self.nonpositive.value + 1
        return x / 2


tracker = BranchTracker()
result = brainstate.transform.cond(
    jnp.array(3.0) > 0,
    tracker.double,
    tracker.halve,
    jnp.array(3.0),
)

assert result == 6
assert tracker.positive.value == 1
assert tracker.nonpositive.value == 0
```

For `ifelse`, use a final `True` predicate to encode the default branch:

```python
result = brainstate.transform.ifelse(
    [x > 10, x > 5, True],
    [large_fn, medium_fn, small_fn],
    x,
)
```

**Invariant:** Keep branch return PyTrees compatible, and put branch-specific
State writes inside the branch callable. Do not use Python `if`, `for`, or
`while` when the decision or iteration count depends on traced JAX values.

## Common failures

- Do not select a checkpointed loop for forward-only execution; its memory
  benefit applies during gradient computation.
- Do not expect checkpointing to reduce memory without extra backward
  computation.
- Do not mutate State from a `while_loop` condition.
- Do not differentiate through `while_loop` with reverse mode; use
  `bounded_while_loop`.
- Do not return incompatible structures from alternative branches.
- Do not assume `switch` rejects an invalid index; it clamps the index.

## Sources

- [Control flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html) supplies the primitive contracts, selection rules, State behavior, loop-output semantics, checkpoint tradeoff, and branch invariants.
- [How to train through long rollouts without exhausting memory](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html) supplies the BPTT checkpoint placement, compiled training-step workflow, and `base` tuning application.
