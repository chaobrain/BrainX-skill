# Memories Replaying During Sleep

This BrainX experiment trains four place-cell populations on route `A -> B -> C -> D`,
runs a zero-input sleep period, suppresses recurrent replay in one matched network,
and compares route completion after an `A` recall cue.

Run it with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python sleep_replay.py
```

The script prints and saves exact replay order, route-completion score and latency,
and forward-weight changes in `sleep_replay_results.json`. It also writes
`sleep_replay.png` with sleep rasters, plasticity, and recall activity.

The final wake cue to `A` is an explicit boundary-state seed. External current is
zero throughout sleep; the causal intervention is recurrent transmission enabled
versus disabled. Both networks are otherwise initialized, trained, and recalled by
the same `vmap`-executed protocol.
