# Memories Replaying During Sleep

This BrainX example teaches a recurrent spiking network the route
`A -> B -> C -> D`, runs matched replay-enabled and replay-suppressed networks
through a sleep-like period with zero external current, and compares route
recall afterward.

The model uses:

- `brainpy-state` for four LIF place-cell ensembles and recurrent synaptic
  dynamics;
- `brainevent` for binary spike transmission and event-triggered STDP;
- `brainstate` for neural/plasticity state, `for_loop` in learning, sleep, and
  recall, and state-aware `vmap2` across the matched conditions;
- `brainunit` for time, voltage, current, resistance, and plasticity timing.

During sleep, a slow intrinsic oscillator periodically depolarizes place A.
The resulting replay is network-internally seeded rather than wholly unseeded,
while the external-current protocol is identically zero. The suppression lane has
the same learned weights and intrinsic timing, but its recurrent transmission
is gated off only during sleep. Plasticity remains open in both lanes.

## Run

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python sleep_replay.py
```

Outputs are written to `results/sleep_replay_summary.png` and
`results/sleep_replay_metrics.json`. The JSON retains every event's observed
first-spike times, direction label, learned weights, and recall score.

The parameters define a calibrated phenomenological demonstration rather than
a fit to a particular animal or recording, and the output is descriptive rather
than a held-out biological result. A replay event is classified as
forward or backward from the slope of the four ensembles' first-spike times;
an event missing any ensemble is classified as incomplete. Recall scores the
fraction of downstream places activated in the expected order after cueing A,
with a fixed 25 ms completion deadline contributing a speed term.
