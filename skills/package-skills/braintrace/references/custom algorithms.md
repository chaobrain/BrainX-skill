# Custom algorithms

Use this reference when built-in BrainTrace algorithms cannot express a research method and the change belongs in trace storage, trace recurrence, gradient solving, or additional algorithm State. Keep model tracing, Jacobian construction, and sequence driving in the existing BrainTrace infrastructure.

## Choose an extension base

Subclass the nearest estimator engine instead of rebuilding the compiler and executor.

| API | Use when |
|---|---|
| `ETraceAlgorithm` | Use only for a fundamentally new executor or trace representation. It owns model wrapping, graph compilation, State separation, and `running_index`; subclasses must supply trace initialization, update, and lookup behavior. |
| `ETraceVjpAlgorithm` | Use for a new VJP-based estimator representation. It owns the custom-VJP forward/backward bridge and delegates trace recurrence and gradient solving to protocol methods. |
| `ParamDimVjpAlgorithm` | Use for a D-RTRL-like estimator with one parameter-dimensional trace per weight and hidden-group relation. Memory scales with batch size and parameter count. |
| `IODimVjpAlgorithm` | Use for a pp-prop-like estimator with input/output-factorized traces. Memory scales with batch size and input plus output dimensions. |
| `EligibilityTrace` | Use to store algorithm trace State across time; it is a `brainstate.ShortTermState` specialization. |

Use `D_RTRL` as the concrete parameter-dimensional baseline and `pp_prop` as the concrete input/output-factorized baseline. `ES_D_RTRL` is a historical alias for `pp_prop`; use `pp_prop` in new code.

## Extension workflow

| Step | Action | Important result |
|---|---|---|
| 1 | Choose the nearest base estimator. | Existing graph compilation, Jacobian execution, and State separation remain intact. |
| 2 | Override the smallest protocol method that expresses the new rule. | Trace dynamics and gradient readout remain separate decisions. |
| 3 | Compile the subclass with `braintrace.compile()`. | The custom class receives the model and returns a normal learner. |
| 4 | Drive one learner call per step inside a BrainState loop or gradient-carrying scan. | BrainTrace updates eligibility State while BrainState owns temporal control and accumulation. |
| 5 | Inspect trace values and compiler diagnostics. | Storage shapes, recurrence behavior, and parameter participation become visible. |
| 6 | Reset both algorithm and model State between independent sequences. | Eligibility traces, time index, and recurrent hidden State all start cleanly. |

## Protocol methods

Override only the method responsible for the behavior being changed.

| API | Description |
|---|---|
| `init_etrace_state(*args)` | Override when the algorithm needs new trace-storage shapes at compile time. |
| `_get_etrace_data()` | Override when custom `EligibilityTrace` State must be assembled into the recurrence input. |
| `_assign_etrace_data(values)` | Override with `_get_etrace_data()` when custom recurrence results must be written back to trace State. |
| `_update_etrace_data(...)` | Override to change the eligibility-trace recurrence, such as clipping, normalization, or a decay schedule. It returns the new trace data. |
| `_solve_weight_gradients(...)` | Override to change how current traces and loss-to-hidden gradients become the final gradient dictionary. |
| `get_etrace_of(weight)` | Use to inspect trace tensors owned by one parameter. It returns `{(weight_id, leaf_index): {trainable_input_name: trace_tensor}}`. |
| `reset_state(batch_size=None)` | Use between independent episodes. It zeroes eligibility traces and resets `running_index`, optionally rebroadcasting traces for a new batch size. |

The VJP protocol signatures are:

```python
def _update_etrace_data(
    self,
    running_index,
    etrace_vals_until_t_minus_1,
    hid2weight_jac,
    hid2hid_jac,
    weight_vals,
    input_is_multi_step,
):
    ...

def _solve_weight_gradients(
    self,
    running_index,
    etrace_h2w_at_t,
    dl_to_hidden_groups,
    weight_vals,
    dl_to_nonetws_at_t,
    dl_to_etws_at_t,
):
    ...
```

`_solve_weight_gradients()` must return a dictionary mapping every parameter path, including ETP and non-ETP parameters, to its gradient pytree.

## Canonical custom recurrence

Delegate to the parent recurrence, then apply the custom trace rule. This preserves the parent estimator's storage, compiler, and executor contracts.

```python
import jax
import jax.numpy as jnp
import braintrace
class ClippedDRTRL(braintrace.ParamDimVjpAlgorithm):
    """D-RTRL with element-wise trace clipping."""

    def __init__(self, model, clip_value=1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.clip_value = clip_value

    def _update_etrace_data(
        self,
        running_index,
        history,
        hid2weight_jac,
        hid2hid_jac,
        weight_vals,
        input_is_multi_step,
    ):
        traces = super()._update_etrace_data(
            running_index,
            history,
            hid2weight_jac,
            hid2hid_jac,
            weight_vals,
            input_is_multi_step,
        )
        return jax.tree.map(
            lambda trace: jnp.clip(
                trace, -self.clip_value, self.clip_value
            ),
            traces,
        )


model = braintrace.nn.ValinaRNNCell(in_size=10, out_size=32)
learner = braintrace.compile(
    model,
    ClippedDRTRL,
    jnp.zeros(10),
    clip_value=5.0,
)
outputs = brainstate.transform.for_loop(
    learner,
    jnp.ones((5, 10)),
)
assert len(outputs) == 5
assert outputs[0].shape == (32,)
```

**Invariant:** `_update_etrace_data()` receives the previous trace and current Jacobians and must return the trace for the current step without changing its registered pytree structure.

## Gradient-solve variations

Override `_solve_weight_gradients()` when the recurrence remains valid but the final gradient dictionary needs a transformation such as global norm clipping or per-layer scaling.

Use this pattern:

```python
class GradClippedDRTRL(braintrace.D_RTRL):
    def __init__(self, model, max_norm=1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.max_norm = max_norm

    def _solve_weight_gradients(
        self,
        running_index,
        etrace_h2w_at_t,
        dl_to_hidden_groups,
        weight_vals,
        dl_to_nonetws_at_t,
        dl_to_etws_at_t,
    ):
        gradients = super()._solve_weight_gradients(
            running_index,
            etrace_h2w_at_t,
            dl_to_hidden_groups,
            weight_vals,
            dl_to_nonetws_at_t,
            dl_to_etws_at_t,
        )
        squared_norm = jax.tree.reduce(
            lambda total, grad: total + jnp.sum(grad * grad),
            gradients,
            initializer=jnp.zeros((), dtype=jnp.float32),
        )
        norm = jnp.sqrt(squared_norm)
        scale = jnp.minimum(1.0, self.max_norm / (norm + 1e-12))
        return jax.tree.map(lambda grad: grad * scale, gradients)
```

## Trace inspection and reset

Trace lookup is nested because one parameter leaf can participate in a primitive through several trainable inputs, such as `weight` and `bias`.

```python
import brainstate

brainstate.transform.for_loop(learner, sequence)

for path, weight in model.states(brainstate.ParamState).items():
    groups = learner.get_etrace_of(weight)
    for (weight_id, leaf_index), traces in groups.items():
        for input_name, trace in traces.items():
            print(path, weight_id, leaf_index, input_name, trace.shape)

# Independent sequence boundary: clear both trace and recurrent State.
learner.reset_state()
brainstate.nn.reset_all_states(model)
assert int(learner.running_index.value) == 0
```

**Invariant:** `learner.reset_state()` does not reset the model's hidden states. Call both reset operations between independent episodes. Override `reset_state()` only when the subclass carries additional State that must also be cleared.

## Execution and validation rules

- Compile the custom algorithm once for a stable model structure; do not rebuild it per step.
- Differentiate exactly one learner call per step and use `brainstate.transform.scan` when gradients must be accumulated across a sequence. Apply the optimizer after the scan.
- Compare an unmodified subclass against its parent before testing the new rule.
- Verify trace shapes, finite values, reset-to-zero behavior, parameter-gradient keys, and parameter updates.
- Use a reduced BPTT oracle only in a regime where the parent estimator documents equivalence. `D_RTRL` uses a diagonal recurrent-Jacobian approximation and is not generally equal to BPTT.
- Do not reimplement graph tracing, hidden grouping, Jacobian calculation, or State write-back unless the research method genuinely requires a new base engine.
- This reference targets BrainTrace `0.2.4`; do not use sequence-driver methods introduced in `0.2.5`.

## Sources

- [Developing Custom Algorithms](https://brainx.chaobrain.com/braintrace/advanced/custom_algorithms.html)
- [Online-Learning Algorithms](https://brainx.chaobrain.com/braintrace/apis/algorithms.html)
