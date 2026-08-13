# Run notes

This frozen-parameter attempt is retained as a failed run. Unit tests passed
(`4 passed`), but the full simulation stopped during JAX tracing before any
trial was simulated: `jnp.packbits` requires boolean or integer input while the
BrainPy spike arrays use floating-point zero/one values.

No scientific result can be inferred from this run. The next attempt changes
only the trajectory-audit representation by converting spikes with `!= 0`
before lossless bit packing.
