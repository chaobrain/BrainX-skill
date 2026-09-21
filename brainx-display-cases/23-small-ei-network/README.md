# Small E/I network

This case builds a 20-neuron recurrent conductance-based network: neurons `0–15` are excitatory and `16–19` are inhibitory. It uses BrainPy-State LIFRef neurons, event-driven probabilistic projections, exponential synapses, and COBA outputs.

Run it from this directory after installing a compatible BrainX environment:

```bash
python small_ei_network.py
```

The `run()` function returns a unit-bearing time axis and a time-major spike array with shape `(1000, 20)` for the default 100 ms protocol. See `NeuroSpecification.md` for the locked teaching-model assumptions.
