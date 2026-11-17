# Optimize BrainSpike parameters

An optimizer owns persistent update State and changes only its registered parameter collection. The simulation owns hidden dynamical State and must start at the declared independence boundary for every loss evaluation.

## Select the update boundary

| API | Use |
|---|---|
| `model.states(brainstate.ParamState)` | Select native model parameters; exclude membrane, conductance, spike, random, delay, and eligibility State. |
| `braintools.optim.Adam(lr=...)` | Use as the verified adaptive baseline for native learning or a normalized Simulator candidate. |
| `optimizer.register_trainable_weights(params)` | Register the collection once, outside temporal loops and repeated optimizer steps. |
| `brainstate.transform.grad(loss, params, return_value=True)` | Use for a stateful native objective and a matching gradient tree. |
| `optimizer.update(grads)` | Apply one update at the declared cadence; use the same collection layout as registration. |

## Register and update a normalized candidate

This scalar candidate illustrates the optimizer lifecycle. The complete Simulator example maps the candidate to a physical connection weight; the native example selects the model's actual parameter collection.

```python
import brainstate
import braintools
import brainunit as u

weight = brainstate.ParamState(u.math.asarray(0.4))
params = {"weight": weight}
optimizer = braintools.optim.Adam(lr=0.05)
optimizer.register_trainable_weights(params)

def objective():
    return braintools.metric.squared_error(weight.value, u.math.asarray(1.7), reduction="mean")

differentiate = brainstate.transform.grad(objective, params, return_value=True)
grads, initial_loss = differentiate()
optimizer.update(grads)
assert objective() < initial_loss
```

## Preserve the selected execution path

For native BPTT, reset model runtime State inside the loss and compile the complete reset-gradient-update step. For pp-prop, reset both model and learner State, advance every sequence step, choose gradient reduction explicitly, and update after the sequence. Open `references/native-training.md` and `scripts/train_native_snn.py` for both complete patterns.

For direct NEST voltage fitting, keep the candidate outside a locally constructed Simulator, call `simulate()` for each independent objective, and apply the optimizer between evaluations. The verified pure candidate-to-loss boundary uses `jax.value_and_grad`; do not outer-JIT the full multi-spike-generator builder. Open `references/simulator-training.md` and `scripts/train_nest_simulator.py` for the executable boundary and finite-difference check.

Do not reuse optimizer State across independent training replicates unless the experiment requires it. Do not re-register weights at every step or reset learned parameters with runtime State. An optimizer change cannot repair a zero-gradient hard spike objective.

For a genuinely discontinuous or bounded black-box fitting objective, check the shared BrainTools optimizer tutorials for `ScipyOptimizer` or optional `NevergradOptimizer` before selecting external infrastructure. These wrappers own an objective and do not use the register/update lifecycle. The separately tested `fit_hard_count()` in `scripts/nest_boundary_workflows.py` uses bounded Powell search for a scalar NEST hard-count objective; open `references/simulator-training.md` for its scope. Verify reset independence, parameter units, bounds, plateaus, and optional dependencies before adapting a different objective.

## Sources

[BrainTools optimizer API](https://brainx.chaobrain.com/braintools/apis/optim.html), [optimization tutorials](https://brainx.chaobrain.com/braintools/optim/index.html). The canonical lifecycle targets BrainTools 0.4.0.
