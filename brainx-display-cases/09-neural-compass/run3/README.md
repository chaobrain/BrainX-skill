# An Internal Neural Compass

This example builds a spiking head-direction ring with BrainX. A localized
visual cue first points the bump north. After the cue disappears, a signed
angular-velocity input skews recurrent spike communication so the bump should
move with a 90 degree turn in darkness.

The lesion experiment cues every preferred direction represented by the ring.
For each heading, an intact control and a matched trial with a permanent
75 degree wedge silenced are simulated together. The output distinguishes:

- `spared`: the lesioned bump stays aligned with its intact control.
- `recovered`: the bump first departs or loses concentration, then remains
  aligned throughout the final 150 ms window.
- `failed`: the bump does not make that sustained return.

The simulation uses:

- `brainpy-state` LIF neurons and exponential synaptic/readout dynamics.
- `brainevent.BinaryArray` for spike-driven symmetric and velocity-skew ring
  communication.
- `brainstate.transform.for_loop` over time and filter-based stateful `vmap2`
  over independent headings and lesion conditions.
- `brainunit` quantities for time, recurrent delay, voltage, current, membrane
  parameters, and angular velocity.

Run the complete experiment:

```bash
MPLCONFIGDIR=/tmp/neural-compass-mpl python neural_compass.py
```

The command writes `results/neural_compass_results.png`, a per-heading
`results/lesion_sweep.csv`, and the aggregate metrics and decision thresholds
in `results/summary.json`.

In the included reference run, the bump completes 94% of the commanded turn
and ends 5.35 degrees from the integrated heading. All 48 intact hold controls
remain active and aligned. The permanent wedge spares 22 headings and causes
26 failures; none of the disturbed headings makes a sustained recovery under
the stated thresholds.

Run the focused analysis tests with:

```bash
MPLCONFIGDIR=/tmp/neural-compass-mpl python -m unittest -v
```
