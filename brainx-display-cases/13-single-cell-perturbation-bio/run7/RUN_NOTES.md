# Run notes

This frozen-parameter attempt is retained as a failed run. Unit tests passed
(`4 passed`). The full process again terminated without a Python traceback and
left an empty output directory, even after lossless bit packing reduced the
trajectory-audit transfer. This indicates that the 64-lane full-network rollout
itself is the more likely resource pressure.

No scientific result can be inferred from this run. The next attempt keeps all
model and analysis parameters unchanged, uses a separate identically seeded
network instance for the 32-lane mapping rollout, and removes the 32 padding
lanes.
