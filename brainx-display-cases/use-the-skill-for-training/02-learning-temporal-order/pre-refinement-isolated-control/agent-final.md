Implemented the BrainX-native temporal-order learning experiment in [temporal_order_learning.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.VtMInj/workspace/temporal_order_learning.py:95), with online CSR plasticity, stateful `for_loop`, batched `vmap` probes, and explicit BrainUnit quantities.

Results:

- After A-first training: output spikes `[2, 0]`
- After B-first reversal: output spikes `[0, 2]`
- Final weights: `[[0.83, 0.05], [0.05, 0.83]]`

The generated plot is [temporal_order_relearning.png](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.VtMInj/workspace/temporal_order_relearning.png). Usage is documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.VtMInj/workspace/README.md).

Verification: `python -m unittest -v` passes both tests.