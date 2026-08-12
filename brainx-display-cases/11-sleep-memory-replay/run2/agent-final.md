Implemented the complete BrainX experiment in [sleep_replay.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run2-eval.C3pVm6/workspace/sleep_replay.py:76), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run2-eval.C3pVm6/workspace/README.md).

Results:

- Replay-enabled sleep: 5 forward events, 0 backward.
- Replay-suppressed sleep: 0 complete events.
- Recall A-to-D latency: 6 ms with replay, 18 ms suppressed.
- Recall score: 76% versus 28%.

Artifacts: [summary figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run2-eval.C3pVm6/workspace/results/sleep_replay_summary.png) and [complete metrics](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run2-eval.C3pVm6/workspace/results/sleep_replay_metrics.json).

Verified with a full run and `python -m py_compile sleep_replay.py`. The model is explicitly labeled as a calibrated phenomenological demonstration; sleep replay is internally seeded while external sleep input remains zero.