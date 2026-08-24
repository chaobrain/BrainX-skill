Implemented the temporal-order learning tutorial in [temporal_order.py](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.n7Ea3E/workspace/temporal_order.py:1), with usage and model details in [README.md](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.n7Ea3E/workspace/README.md:1).

It uses BrainPy-State LIF populations, BrainEvent communication/plasticity, BrainState `for_loop` and stateful `vmap2`, plus explicit BrainUnit timing, delay, voltage, current, and time constants.

Verified with `python temporal_order.py`:

```text
acquired=100%
immediate reversal=0%
relearned=100%
vmapped jittered batch=100%
```

The generated result is [temporal_order_relearning.png](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.n7Ea3E/workspace/temporal_order_relearning.png).