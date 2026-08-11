# An Internal Neural Compass

This experiment builds a spiking head-direction ring in BrainX. Local recurrent
excitation and broad inhibition maintain an activity bump. Clockwise and
counterclockwise event pathways are shifted by one neuron and gated by angular
velocity, so the bump integrates a turn without a visual cue.

The default run performs two matched trials for every preferred direction in a
36-neuron ring: an intact control and a trial in which a 70 degree wedge around
north is silenced after the bump forms. It writes:

- `results/head_direction_compass.png`: north-turn dynamics and lesion sweep.
- `results/lesion_sweep.csv`: all continuous observables and the full recovery predicate.
- `results/summary.json`: thresholds and failed starting headings.

Run:

```bash
python head_direction_compass.py
python -m unittest -q
```

The simulation uses `brainpy.state.LIFRef` and `Expon`/`CUBA` dynamics,
`brainevent.BinaryArray` recurrent communication, `brainstate.transform.for_loop`
for time, and `brainstate.transform.vmap2` for independent heading and lesion
trials. BrainUnit quantities remain attached to time, delay, voltage, current,
resistance, and angular velocity until plotting and CSV boundaries.

Recovery requires a valid matched control, final lesion error no more than 30
degrees from that control, lesion population-vector resultant at least 0.22,
and final activity at least 35% of control. A valid control must also follow the
time-resolved turn with at most 12 degrees RMS error, which prevents a final
phase match after extra laps from being mislabeled as tracking. The parameters
and thresholds define a phenomenological demonstration regime; they are not
fitted to a particular animal or publication.
