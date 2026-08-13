# Learning temporal order with BrainX

This experiment trains a six-neuron spiking circuit to classify whether tone A
or tone B came first. It then reverses the output labels without rebuilding the
circuit and continues the same online plasticity rule until the readout is
remapped.

The implementation uses:

- `brainpy-state` for sensory, order-detector, and output LIF populations.
- `brainevent` for binary spike communication and event-triggered dense weight updates.
- `brainstate` for persistent and trial-scale state, nested `for_loop`
  training, and `vmap`-built batches evaluated with independent native batch
  state.
- `brainunit` for all timing, delays, membrane parameters, currents, voltages,
  and plasticity time constants.

Run the complete acquisition/reversal experiment:

```bash
python temporal_order_learning.py
```

It writes `artifacts/temporal_order_reversal.png` and
`artifacts/temporal_order_summary.json`. Run the regression test with:

```bash
python -m unittest -v
```

The order-sensitive layer is fixed: a decaying trace of the first tone gates a
detector when the other tone arrives. Only the detector-to-output weights are
plastic. A signed teaching event drives online potentiation of the requested
output and depression of its competitor; reversing that event remaps the same
circuit.
