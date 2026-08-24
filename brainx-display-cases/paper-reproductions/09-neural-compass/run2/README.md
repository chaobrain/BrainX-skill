# An Internal Neural Compass

This experiment builds a 72-neuron spiking head-direction ring. A visual cue
first places the activity bump at a known heading. The cue is then removed,
and asymmetric velocity-gated recurrent input moves the bump while the animal
turns in darkness. A second experiment silences a fixed 60-degree wedge and
runs matched control and lesion trials from all 72 preferred directions on the
ring.

The implementation uses:

- `brainpy-state` for LIF neurons, exponential synapses, current output, and a
  filtered spike readout;
- `brainevent.BinaryArray` for dense event-driven recurrent communication;
- `brainstate.transform.for_loop` for time and `vmap2` for independent control,
  lesion, and starting-heading state;
- `brainunit` for the integration step, transmission delay, time constants,
  membrane parameters, currents, and angular velocity.

The ring parameters are phenomenological. In particular, `VELOCITY_MIX`
calibrates the asymmetric recurrence so the decoded bump integrates a
90 degree/second turn. This is a mechanistic demonstration, not a fitted model
of a specific animal or recording.

Run the complete experiment:

```bash
MPLCONFIGDIR=/tmp/compass-mpl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python internal_neural_compass.py
```

Outputs are written to `outputs/internal_neural_compass.png` and
`outputs/lesion_outcomes.csv`. The table retains the continuous peak and final
angular errors, vector strength, and firing-rate ratio behind each categorical
label:

- `spared`: no departure over 12 degrees and a reliable final bump;
- `recovered`: departure over 12 degrees followed by at least 200 ms within
  10 degrees of the matched control and a reliable bump;
- `failed`: every other condition, including collapse or sustained error.

Run the focused unit tests with `python test_internal_neural_compass.py` (or
`pytest -q` when pytest is installed). The main script also rejects a collapsed
bump, excessive stationary drift, or a dark-turn error above 20 degrees before
saving results.
