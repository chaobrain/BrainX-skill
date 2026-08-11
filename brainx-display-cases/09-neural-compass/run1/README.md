# An Internal Neural Compass

This BrainX experiment builds a phenomenological spiking ring attractor from
48 `brainpy.state.LIFRef` head-direction neurons. Delayed binary spikes cross
dense ring connectivity through `brainevent.BinaryArray`; a signed velocity
term biases recurrent transmission clockwise or counterclockwise.

The script runs two experiments:

1. Cue north, remove the landmark, turn at 90 degrees/s for one second, and
   compare the decoded bump with the integrated turn.
2. Cue every represented heading, silence a 60-degree wedge centered on south
   for 50 ms, then observe 550 ms of dark recovery. Matched intact controls run
   in the same stateful `vmap2` simulation.

`spared`, `recovered`, and `failed` are derived from final angular error, bump
coherence, matched-control activity, and whether a measurable departure
occurred. The thresholds are declared in the script because this is a
calibrated demonstration rather than a fit to biological data.

Run:

```bash
python neural_compass.py
```

The command writes `neural_compass_results.png` and a compressed data archive
`neural_compass_results.npz` containing the continuous lesion observables.
