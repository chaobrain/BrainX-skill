# Run notes

This frozen-parameter attempt is retained as a failed run. Unit tests passed
(`4 passed`). The full command created the output directory and then terminated
without a Python traceback or any result artifacts. Returning the complete
pre-photostimulation spike tensors is the suspected resource pressure, but the
available log is insufficient to identify the termination as an out-of-memory
event.

No scientific result can be inferred from this run. The next attempt keeps all
model and analysis parameters unchanged and losslessly bit-packs the boolean
pre-photostimulation trajectories before returning them from JAX.
