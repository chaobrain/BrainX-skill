Implemented the BrainX temporal-order reversal experiment in [temporal_order_learning.py](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.2pahiQ/workspace/temporal_order_learning.py:129). It uses six LIF neurons, event-driven communication/plasticity, unit-aware delays and time constants, nested `for_loop` training, and vmapped trial construction with independent batched evaluation state.

Default results:

- Untrained: 50%
- After acquisition: 100%
- Immediately after label reversal: 0%
- After relearning: 100%

Outputs: [figure](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.2pahiQ/workspace/artifacts/temporal_order_reversal.png), [JSON summary](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.2pahiQ/workspace/artifacts/temporal_order_summary.json), and [README](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.2pahiQ/workspace/README.md).

Verification: `python -m unittest -v` passes.