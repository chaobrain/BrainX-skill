# Learning temporal order with BrainX

This example teaches a four-neuron spiking circuit which of two tones came
first, reverses the rewarded output association, and shows the circuit learning
the new association.

The model uses the requested BrainX layers directly:

- `brainpy-state`: two `LIFRef` sensory neurons, an exponential current
  synapse, and two `LIFRef` output neurons.
- `brainevent`: `BinaryArray` communication and post-event-triggered dense
  plasticity.
- `brainstate`: `LongTermState` weights, a `ShortTermState` eligibility trace,
  a stateful delay, `for_loop` within each trial, and `vmap2` over independent
  jittered evaluation trials.
- `brainunit`: explicit units for the integration step, tone timing, axonal
  delay, membrane parameters, currents, and trace/synapse time constants.

## Learning rule

Delayed sensory spikes build a decaying eligibility trace

```text
x <- x exp(-dt / tau_x) + s_pre
```

At the teaching event, BrainEvent updates only the selected output columns:

```text
W[:, target]     <- clip(W[:, target]     + eta x)
W[:, competitor] <- clip(W[:, competitor] - eta rho x)
```

Because the second tone has the larger trace at the teaching event, each output
learns the tone that occurred most recently. Since both tones occur once, that
recency signal uniquely identifies which tone came first. The depression term
lets the circuit erase the old association after the output contingency is
reversed.

Learning trials are causally sequential because each one consumes the weights
left by the preceding trial. Neural voltage, refractory state, synaptic current,
delay buffers, and eligibility traces reset between trials; only learned weights
persist. The final evaluation uses a stateful `vmap2` because those trials are
independent and read separate copies of the learned circuit state.

## Run

```bash
python temporal_order.py
```

The script prints acquisition, immediate-reversal, relearned, and jittered-batch
accuracy, checks them with assertions, and writes
`temporal_order_relearning.png`.
