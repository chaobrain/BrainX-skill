# Learning Temporal Order with BrainX

This example trains a small spiking circuit to report which of two tones came
first. It first receives repeated `A -> B` trials, then the stimulus order is
reversed to `B -> A`. A teacher pulse arrives after the response window, so the
accuracy curve measures the circuit's prediction before it receives feedback.

The model uses:

- `brainpy-state` for sensory, temporal-coincidence, and readout LIF neurons;
- `brainevent` for binary spike communication through CSR connectivity and
  event-triggered online weight updates;
- `brainstate` for persistent and transient state, the sequential learning
  `for_loop`, compilation, and mapped independent evaluation trials;
- `brainunit` for stimulus timing, inter-tone and feedback delays, membrane
  parameters, currents, and plasticity time constants.

Run it with the Python environment where the BrainX component packages are
installed:

```bash
python temporal_order_learning.py
```

The script prints phase and held-out accuracy, then writes
`temporal_order_learning.png`. The top panel shows acquisition before and after
the order switch, the middle panel shows the four plastic weights, and the last
panel shows the readout spike counts from a mapped batch containing both
orders.

Learning trials are intentionally sequential because each trial consumes the
weights learned on the previous trial. Held-out trials are independent, so
their complete stateful rollouts are batched with BrainState's filter-based
`vmap2` API, with separate neuron and trace state in every lane.
