Implemented the BrainX sleep-replay experiment in [sleep_replay.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run0-eval.4DaUuZ/workspace/sleep_replay.py:1), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run0-eval.4DaUuZ/workspace/README.md:1).

Observed results:

- Replay direction: forward
- Forward/backward transition evidence: `15 / 0`
- Pre-sleep matched-weight difference: `0`
- Sleep weight change, replay/blocked: `0.010 / 0.000`
- Ordered recall, replay/blocked: `0.625 / 0.000`

The generated figure is [sleep_replay_results.png](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/sleep-memory-replay-run0-eval.4DaUuZ/workspace/sleep_replay_results.png). Three direction and recall analysis tests pass via `python -m unittest -v`.