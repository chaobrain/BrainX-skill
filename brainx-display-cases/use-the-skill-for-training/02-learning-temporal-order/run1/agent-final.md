Implemented the complete BrainX example in [temporal_order_learning.py](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.X8SVpI/workspace/temporal_order_learning.py:152), with usage notes in [README.md](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.X8SVpI/workspace/README.md:1).

The default run achieved 88.9% during each acquisition phase and 100% on mixed held-out trials. Sequential learning uses `for_loop`; independent evaluations use state-aware `vmap2`. BrainEvent detector selectivity and Python compilation pass.

Generated result: [temporal_order_learning.png](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.X8SVpI/workspace/temporal_order_learning.png)

Run with:

```bash
python temporal_order_learning.py
```

`ruff` was unavailable, so linting was not run.