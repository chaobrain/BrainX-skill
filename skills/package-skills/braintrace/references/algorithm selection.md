# Algorithm selection

Use this reference when choosing a BrainTrace `0.2.4` estimator beyond the default `D_RTRL`. Use `Drtrl.md` or `pp_pprop workflow.md` for the two canonical training workflows; open `custom algorithms.md` only when no built-in class expresses the method.

## Choose an estimator

Choose by trace representation, recurrence approximation, learning signal, and supported mathematical regime.

| API | Use when | Essential behavior and constraint |
|---|---|---|
| `D_RTRL` | Use as the general RNN default when parameter-dimensional trace memory fits. | Carries parameter-shaped traces with a diagonal hidden-recurrence approximation. |
| `pp_prop` | Use when parameter-shaped traces are too large, especially for SNNs. | Factorizes each trace into input and output components with exponential smoothing; `ES_D_RTRL` is its compatibility alias. |
| `EProp` | Use for recurrent SNNs that need kappa-filtered traces or fixed random-feedback learning signals. | Select feedback and filter decay through constructor options. |
| `OSTLRecurrent` | Use for the OSTL with-H recurrent regime. | Retains recurrent influence according to the implemented OSTL rule. |
| `OSTLFeedforward` | Use for the OSTL without-H feedforward regime. | Uses an input/output-factorized path and omits recurrent influence. |
| `OTPE` | Use for deep SNNs requiring the implemented online temporal-processing estimator. | Requires `leak`; select full or approximate mode explicitly. |
| `OTTT` | Use for very large SNNs using the implemented presynaptic trace rule. | Requires `leak`; report the selected mode. |
| `OSTTP` | Use for target-projection learning with supplied per-layer feedback matrices. | Requires `B_list`; declare target timing. |

`SnAp`, `UORO`, `ThreeFactor`, `DNI`, `SyntheticGradient`, and `ETraceConfig` are not public BrainTrace `0.2.4` APIs. Do not select them from newer documentation when targeting BrainX `v2026.7.9`.

## Compile the chosen algorithm

`braintrace.compile` is the shared entry point; options are forwarded to the selected algorithm constructor.

| API or option | Description |
|---|---|
| `braintrace.compile(model, algorithm, example_step, ...)` | Use once to initialize State, build the graph, and return a learner. |
| `vjp_method='single-step'` | Use for the normal per-step learning signal. |
| `vjp_method='multi-step'` | Use only when the algorithm and `MultiStepData` workflow require a recent-window learning signal. |
| `fast_solve=True` | Use the estimator's fast supported solve path; disable only for a documented incompatibility or comparison. |
| `trace_dtype=...` | Use when a supported estimator should store eligibility traces in a selected dtype. |
| `decay_or_rank=...` | Supply for `pp_prop`; use a float decay in `(0, 1)` or a positive integer rank parameterization. |

```python
learner = braintrace.compile(
    model,
    braintrace.pp_prop,
    example_step,
    batch_size=batch_size,
    vmap=True,
    decay_or_rank=0.95,
    vjp_method='single-step',
)
learner.module.report.show(2)
```

## Drive the sequence in `0.2.4`

BrainTrace `0.2.4` advances hidden and eligibility State on every learner call; BrainState owns the explicit temporal carry and gradient accumulation.

| API | Description |
|---|---|
| `brainstate.transform.grad(step_loss, weights, ...)` | Use around one learner call to obtain the current online gradient. |
| `brainstate.transform.scan(...)` | Use when an explicit gradient carry must sum across time. |
| `braintrace.SingleStepData(data)` | Use to mark one current input; a plain array is equivalent. |
| `braintrace.MultiStepData(sequence)` | Use when one learner call intentionally consumes a multi-step window. |

Do not use `etrace_grad`, `etrace_evolve`, `SequenceDriverMixin`, or `ETraceVmap`; they start in BrainTrace `0.2.5`. Open `batching.md` for mapped and native-batch reset patterns.

## Sources

- [BrainTrace `v0.2.4` algorithms API source](https://github.com/chaobrain/braintrace/blob/v0.2.4/docs/apis/algorithms.rst)
- [BrainTrace `v0.2.4` public API](https://github.com/chaobrain/braintrace/blob/v0.2.4/braintrace/__init__.py)
