# Run notes

This frozen-parameter attempt is retained as a failed run. Unit tests passed
(`4 passed`). The full process again terminated without a Python traceback and
left an empty output directory. Because the mapping model was separated in this
attempt, the remaining common failure point is the initial 64-lane tuning
rollout.

No scientific result can be inferred from this run. The next attempt keeps all
observations and parameters unchanged but evaluates the 64 tuning trials as two
sequential 32-trial chunks, maintaining one static batch shape throughout.
