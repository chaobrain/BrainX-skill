# Learning Temporal Order

This experiment teaches a two-output spiking circuit which of two tones came
first. It first presents `A -> B`, then reverses the sequence to `B -> A` and
shows the second output population being acquired online.

The circuit uses:

- `brainpy-state` LIF neurons for two sensory channels and two order outputs;
- `brainevent.BinaryArray` for spike-driven communication and
  `update_dense_on_binary_pre` for bounded online plasticity;
- `brainstate` State for the delay, teaching trace, and weights, `for_loop` for
  the complete time sequence, and `vmap` for batched trial encoding/scoring;
- `brainunit` quantities for every time, membrane, current, delay, and trace
  parameter.

`braintools` is used only for the unit-aware resting-voltage initializer.

The delay line is the temporal-order mechanism. At the onset of the second
tone, the delayed first-tone spike coincides with the direct second-tone spike.
Those event pairs are `[delayed A, direct B]` for `A -> B` and
`[delayed B, direct A]` for `B -> A`.

Run the experiment:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python temporal_order_learning.py
```

The script prints teacher-free probe spikes and writes
`temporal_order_relearning.png`. Run the standard-library tests with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python -m unittest -v
```
