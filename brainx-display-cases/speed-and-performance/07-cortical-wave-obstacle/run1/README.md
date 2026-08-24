# A Cortical Wave Meets an Obstacle

This example creates paired excitatory and inhibitory LIF neurons on a physical
2D sheet, starts a brief wave at the left edge, silences both populations in a
circular patch, and sweeps patch radius against inhibitory conductance.

The implementation keeps each part in its BrainX owner:

- `brainpy-state` supplies the E/I point neurons, exponential synapses, and
  conductance-based outputs.
- `brainevent` applies spikes to a geometry-derived local CSR graph.
- `brainstate` owns mutable dynamics, maps independent sweep lanes with
  `vmap2`, and advances time with `for_loop`.
- `brainunit` tracks milliseconds, millivolts, nanoamps, nanosiemens, and
  millimeters through the model.

Run the complete experiment:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python cortical_wave_obstacle.py
```

The script writes:

- `outputs/cortical_wave_obstacle.png`: intact and lesioned activity snapshots
  plus the outcome phase map.
- `outputs/phase_map.csv`: outcome labels and the downstream/upper/lower spike
  counts used to assign them.

An outcome is `dies` if the wave does not activate the far-right readout zone.
For surviving waves, significant activity on both lesion flanks is `splits`;
one dominant flank is `bends`. The same fixed topology, stimulus, and spatial
bias are used for every condition, so only patch size and inhibitory strength
change across the phase map.
