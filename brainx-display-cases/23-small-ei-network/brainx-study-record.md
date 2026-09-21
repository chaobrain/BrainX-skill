# BrainX study record

## Represented scale

The request names neurons and recurrent excitation/inhibition, so this is a point-neuron spiking network. `brainpy-state` owns neuron, synapse, and projection dynamics. `brainunit` keeps voltage, time, current, and conductance dimensional. `brainstate` owns Module registration, State initialization, environment settings, and transformed control flow. No `brainmass` population-rate mechanism is introduced.

## Canonical lifecycle

1. Construct one `LIFRef` population of length 20 and keep the E/I split as static Python metadata.
2. Construct one `AlignPostProj` for E sources and one for I sources. Each projection uses `EventFixedProb` communication, an exponential synaptic trace aligned to the 20-neuron target, and a COBA output with the appropriate reversal potential.
3. Initialize all registered State with `brainstate.nn.init_all_states(net)` after construction.
4. Run the complete time sequence through `brainstate.transform.for_loop` under `brainstate.environ.context(dt=...)`.
5. In each step, read the current spike vector, update E and I projections, then call the neuron with the unitful external current. Return the newly emitted spike vector.
6. Seed BrainState randomness before constructing the graph so repeated `run()` calls use the same network realization and initialization stream.

## API decisions

| Need | Decision |
|---|---|
| Firing dynamics | `brainpy.state.LIFRef`; explicit refractory timing is useful for a recurrent teaching network. |
| Connectivity | `brainstate.nn.EventFixedProb`; the graph stays sparse and event-driven without hand-written matrix logic. |
| Temporal filtering | `brainpy.state.Expon.desc(20, tau=...)`; the projection owns target-aligned synaptic State. |
| Postsynaptic effect | `brainpy.state.COBA.desc(E=...)`; E and I signs arise from reversal potentials. |
| Time execution | `brainstate.transform.for_loop`; model State is mutated once per `dt` in one transformed rollout. |
| Physical values | `brainunit as u`; keep units through construction, update, and monitor collection. |

## Update order invariant

The projections consume `self.neurons.get_spike()` before the neuron advances. This makes each emitted event available to the synaptic filter at the next model update, and the postsynaptic conductance is deposited before `self.neurons(inp)` integrates. Do not move projection calls after the neuron call without changing the event timing.

## Verification plan

Use a structural test for the 16/4 split and constants, and an integration test for the `(1000, 20)` time-major output when BrainX is installed. With the current environment, BrainPy packages are not importable, so run-time tests are pending dependency availability; Python syntax compilation remains available.
