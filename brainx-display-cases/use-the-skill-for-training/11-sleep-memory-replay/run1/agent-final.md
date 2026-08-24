Implemented the complete BrainX sleep-replay experiment in [sleep_replay.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run1-eval.GyLlkH/workspace/sleep_replay.py:327).

Measured result across eight matched pairs:

- Replay-enabled sleep: 2 forward, 0 backward route events
- Replay-suppressed sleep: 0 complete route events
- Ordered recall: `1.000` with replay versus `0.333` suppressed
- Pre-sleep learned weights are identical within every matched pair
- External place-cue input is zero throughout sleep

Outputs:

- [Result figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run1-eval.GyLlkH/workspace/sleep_replay_results.png)
- [Raw evidence](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run1-eval.GyLlkH/workspace/sleep_replay_evidence.npz)
- [README](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run1-eval.GyLlkH/workspace/README.md)
- [Tests](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run1-eval.GyLlkH/workspace/test_sleep_replay.py)

All four tests and syntax compilation passed. The `pytest` executable was unavailable, so the test functions were executed directly with Python.