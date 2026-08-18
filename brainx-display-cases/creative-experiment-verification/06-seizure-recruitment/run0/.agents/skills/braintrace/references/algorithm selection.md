# Algorithm selection

Use this reference when choosing a BrainTrace estimator beyond the default
`D_RTRL`, expressing a learning rule with `ETraceConfig`, or selecting sequence
driver options. Use `Drtrl.md` or `pp_pprop workflow.md` for estimator-specific
training workflows; use `custom algorithms.md` only when no built-in rule or
configuration expresses the method.

## Choose an estimator

Choose by trace representation, recurrence approximation, learning signal, and
the mathematical regime in which the estimator is valid.

| API | Use when | Essential behavior and constraint |
|---|---|---|
| `D_RTRL` | Use as the general-purpose RNN default when parameter-dimensional trace memory fits. | Uses a diagonal hidden-to-hidden Jacobian approximation. Memory is `O(B * |theta|)` and computation is `O(B * I * O)`; it is not generally gradient-equivalent to BPTT outside that approximation's assumptions. |
| `pp_prop` | Use when parameter-dimensional traces are too large, especially for large or memory-constrained SNNs. | Factorizes each eligibility trace into input and output components with exponential smoothing. Memory is `O(B * (I + O))`; an integer `decay_or_rank` controls decay and does not allocate multiple rank factors. Prefer this name over its historical alias, `ES_D_RTRL`. |
| `SnAp` | Use when the recurrent position graph is structurally sparse over the requested `n`-step neighborhood. | Retains recurrent influence entries reachable within that neighborhood. Dense recurrence saturates the neighborhood immediately and removes the method's structural advantage. |
| `UORO` | Use when an unbiased RTRL-trace estimate is required and projection variance is acceptable. | Carries a rank-one random projection of the full recurrent Jacobian, trading variance for linear carrier storage. |
| `ThreeFactor` | Use for reward-modulated learning with a user-supplied modulatory signal. | Replaces the symmetric hidden-state learning signal with that external signal. |
| `DNI` | Use to carry credit across finite online windows with learned synthetic gradients. | Uses `SyntheticGradient` as a per-hidden-group predictor; update it with `train_synthetic_gradient()`. |
| `EProp` | Use for recurrent SNNs that need kappa-filtered traces or fixed random-feedback learning signals. | Implements eligibility propagation with optional kappa filtering and random feedback. |
| `OSTLRecurrent` | Use for the OSTL with-H recurrent regime. | Keeps the recurrent Jacobian and is RTRL-exact only for block-diagonal hidden-to-hidden Jacobians. |
| `OSTLFeedforward` | Use for the OSTL without-H feedforward regime. | Drops the recurrent Jacobian and uses the input/output-factorized trace path. |

**Correctness invariant:** Treat estimator guarantees as part of algorithm
selection. Exact rules must match a BPTT oracle element-wise; approximate rules
match BPTT only in the mathematical regime documented for that rule.

## Configure rule coordinates

Named algorithms are presets in a shared rule space; use `ETraceConfig` when
the required coordinate combination has no preset name.

| Axis | Values and selection rule |
|---|---|
| `trace_factorization` | Use `'per_param'` for parameter-dimensional traces or `'io_factorized'` for `O(I + O)` factorized traces. |
| `temporal_recursion` | Choose `'jacobian'`, `'scalar_leak'`, or `'none'`; under `'io_factorized'`, provide the `(x, f)` pair when the two sides differ. |
| `recurrence_scope` | Choose `'diagonal'` or `'coupled'`; `SnAp` uses `'sparse_n'` with `sparse_n=n`. |
| `learning_signal` | Use `'symmetric'` for the normal signal or `'random_feedback'` for fixed random feedback. |
| `trace_filter` | Use `'none'` normally or `'kappa'` for kappa-filtered traces. |
| `update_schedule` | Use the supported `'per_step'` schedule. |

The named presets occupy these coordinates:

| Preset | Non-default coordinates |
|---|---|
| `D_RTRL` | Default coordinate. |
| `pp_prop` / `ES_D_RTRL` | `trace_factorization='io_factorized'`, `decay=<decay_or_rank>`. |
| `EProp` | `trace_filter='kappa'`, `kappa=<kappa_filter_decay>`; with `feedback='random'`, also `learning_signal='random_feedback'`. |
| `OSTLRecurrent` | `recurrence_scope='coupled'`. |
| `OSTLFeedforward` | `trace_factorization='io_factorized'`, with `decay=1e-6` by default. |
| `SnAp` | `recurrence_scope='sparse_n'`, `sparse_n=n`. |

```python
# Use different temporal recursion on the input and output sides.
learner = braintrace.compile(
    model,
    braintrace.ETraceConfig(
        trace_factorization='io_factorized',
        temporal_recursion=('scalar_leak', 'none'),
        decay=(0.9, 0.0),
    ),
    x0,
)
```

**Configuration invariant:** Construction rejects illegal coordinate
combinations and names the legal pairings. It also canonicalizes equivalent
rules; for example, zero decay collapses to `temporal_recursion='none'`.

## Drive a sequence

Compile once, then let the learner own temporal iteration, State continuation,
loss masking, and gradient accumulation.

| API | Description |
|---|---|
| `braintrace.compile(model, algorithm, example_input, ...)` | Use as the one-call entry point. It constructs the selected algorithm, eagerly compiles its eligibility-trace graph from the example input, and returns the ready-to-run learner. |
| `learner.etrace_evolve(inputs, ...)` | Use for a loss-free prefix or other forward-only interval. It advances both hidden and eligibility State and leaves the final State installed. |
| `learner.etrace_grad(inputs, *step_args, step_fn=..., ...)` | Use for a training interval. It owns the temporal loop, calls `step_fn` once per step or chunk, accumulates online gradients, and optionally returns per-step values. |

```python
learner = braintrace.compile(model, 'd_rtrl', inputs[0], batch_size=1)

def step_loss(inp, target):
    prediction = learner(inp)
    return braintools.metric.squared_error(prediction, target).mean()

# Advance a loss-free prefix without resetting the trajectory.
learner.etrace_evolve(inputs[:n_warmup])

grads, step_losses = learner.etrace_grad(
    inputs[n_warmup:],
    targets[n_warmup:],
    step_fn=step_loss,
    return_value=True,
)
```

`step_fn` owns the learner call and must call `learner` exactly once per
invocation. Both drivers are continuations: they leave final State installed,
so consecutive calls compose into one trajectory and neither call implies a
reset.

| Driver option | Use and effect |
|---|---|
| `mask=...` | Use to gate loss contributions. The learner still runs at every step, so a zero-weighted prefix is equivalent to calling `etrace_evolve()` over that prefix. |
| `chunk_size=1` | Use for the normal single-step path. |
| `chunk_size=k` with `k >= 2` | Use when `step_fn` consumes `(k, ...)` windows. Set `vjp_method='multi-step'`. |
| `braintrace.compile(..., vmap=True)` | Use when compilation should own batch mapping. Call `etrace_grad()` and `etrace_evolve()` on the returned wrapper, not on `learner.module`, or the mapped lanes are bypassed. |

Open `batching.md` before adding a batch axis. Open `custom algorithms.md` when
the six configuration axes cannot express the research method.

## Source

[Online-Learning Algorithms](https://brainx.chaobrain.com/braintrace/apis/algorithms.html).
The selection rules, coordinate values, examples, and invariants above are
condensed from the official page's prose, API tables, and code blocks.
