# Learning Temporal Order

This example teaches a small spiking circuit which of two tones arrived first,
then swaps the tone order and shows the second decision being learned online.

The circuit uses:

- `brainpy-state` LIF populations for two sensory neurons, two order detectors,
  and two decision outputs;
- `brainevent.BinaryArray` for crossed sensory communication and a CSR readout,
  plus `update_csr_on_binary_pre()` for bounded online plasticity;
- BrainState `LongTermState` and `ShortTermState`, `for_loop` for complete
  stimulus phases, and `vmap` for independent batched AB/BA probes;
- BrainUnit quantities for the integration step, tone onsets and delay,
  membrane constants, refractory period, trace decay, and synaptic decay.

Run the experiment:

```bash
python temporal_order_learning.py
```

It prints baseline and learned output spike counts and writes
`temporal_order_relearning.png`. Run the dependency-free focused checks with:

```bash
python -m unittest -v
```
