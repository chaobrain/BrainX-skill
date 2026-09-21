# Small E/I network

This case builds a 20-neuron recurrent conductance-based network: neurons `0–15` are excitatory and `16–19` are inhibitory. It uses BrainPy-State LIFRef neurons, event-driven probabilistic projections, exponential synapses, and COBA outputs.

Run it from this directory after installing a compatible BrainX environment:

```bash
python small_ei_network.py
```

The command saves `outputs/small_ei_network_raster.png`; the `run()` function returns a unit-bearing time axis and a time-major spike array with shape `(1000, 20)` for the default 100 ms protocol. The raw array and run metadata are saved beside the figure. See `report.md` for the diagnostic result and `NeuroSpecification.md` for the locked teaching-model assumptions.
