# Compose native BrainSpike dynamics

Use native composition when custom equations, differentiable recurrent paths, explicit synaptic State placement, or cross-package clocks cannot be expressed through the NEST Simulator. A native graph is a BrainState Module graph containing public BrainSpike dynamics; it does not depend on the private unified kernel.

## Implement the state lifecycle

A `Neuron` owns hidden membrane/spike State; `init_state` allocates it, `reset_state` restores an independent rollout, and `update` advances one declared timestep.

| API | Use |
|---|---|
| `brainspike.Neuron(size, spk_fun=...)` | Subclass for custom point-neuron equations; implement `get_spike`, initialization, reset, and update. |
| `brainspike.Synapse(size)` | Subclass for custom temporal filtering; put the filter State on the chosen neuron or edge dimension. |
| `brainspike.Dynamics.sum_current_inputs(current)` | Consume registered current inputs in an update; persistent callables remain registered and one-shot arrays are consumed. |
| `brainstate.HiddenState(value)` | Store rollout variables that must participate in State-aware transforms. |
| `brainstate.nn.init_all_states(model)` / `reset_all_states(model)` | Allocate once and reset independent episodes without replacing learned parameters. |
| `brainstate.transform.for_loop(step, inputs)` | Advance a native graph over time while tracking State effects. |

Open `scripts/native_workflows.py` for the complete `LIF`, `ExponentialConductance`, and `RecurrentSNN` implementations. The LIF uses Euler integration, a dimensionless surrogate argument, and an explicitly implemented soft reset. Its numeric parameters are a mechanics fixture, not a claim of equivalence to every legacy LIF model.

Open `references/braintools/input-current.md` for current sampling and unit normalization; open `references/braintools/connectivity.md` when topology requires more than an existing native rule.

## Place synaptic State intentionally

Open `references/native-projections.md` when assembling chemical connections: it owns communication, temporal filtering, CUBA/COBA conversion, target update order, and the exact conditions for target/source State sharing. Run its complete E/I example before adapting a biological network.

Use the ETP-aware native training projection in `references/native-training.md` when temporal gradients are required; the fixed binary communication example does not establish the same training contract.

Open `references/native-delays.md` before inserting transmission history, and `references/model-and-readout-selection.md` when choosing dynamics or temporal decoding.

## Couple native voltages electrically

For a symmetric gap junction, snapshot both old voltages and calculate one physical current before advancing either neuron. The opposite current must use that same snapshot:

```python
current = conductance * (right.V.value - left.V.value)
left(current)
right(-current)
```

Use this with compatible current-driven native neurons; the Docker check verifies the symmetric pair's mean voltage and contraction. This does not validate every NEST gap-junction model or a general multicompartment electrical solver.

## Preserve causal ordering

Read all previous completed source values before updating any target in a recurrent step. Integrate each source once. Record whether new events affect the current or next step; never change that convention while porting a model.

Open `references/native-training.md` for BPTT and pp-prop. Open `references/multiscale-composition.md` for cellular and regional coupling, and `references/migration-boundary.md` for remaining unsupported generalizations.

## Official contracts

- [Neuron and surrogate-gradient lifecycle](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Neuron.html)
- [Synapse base and State placement](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Synapse.html)
- [Dynamics input lifecycle](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Dynamics.html)
