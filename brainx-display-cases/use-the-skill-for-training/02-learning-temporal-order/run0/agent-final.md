Implemented the complete BrainX experiment in [temporal_order_learning.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.g9Bzq1/workspace/temporal_order_learning.py).

The delay line converts `A→B` and `B→A` into distinct coincident event patterns. BrainEvent plasticity learns the appropriate output, while BrainState runs the sequence with `for_loop` and batches trial encoding/scoring with `vmap`.

Teacher-free results:

```text
After A→B training:
  A→B: [1, 0]
  B→A: [0, 0]

After B→A training:
  A→B: [1, 0]
  B→A: [0, 1]
```

The generated learning summary is [temporal_order_relearning.png](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.g9Bzq1/workspace/temporal_order_relearning.png). Documentation is in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.g9Bzq1/workspace/README.md), and both end-to-end tests pass via `python -m unittest -v`.