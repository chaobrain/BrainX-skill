# A Cortical Wave Meets an Obstacle

This experiment launches a brief wave across colocated excitatory and
inhibitory LIF-neuron sheets, silences a circular patch in the wave's path, and
sweeps patch radius against I-to-E conductance strength.

The implementation uses:

- `brainpy-state` for the E/I LIF populations and conductance synapses;
- `brainevent` for sparse binary-spike communication on a local CSR graph;
- `brainstate` for state initialization, a time `for_loop`, and state-aware
  `vmap2` over independent sweep conditions;
- `brainunit` for time, voltage, current, conductance, and sheet geometry.

Run:

```bash
python cortical_wave_obstacle.py
```

The command writes:

- `outputs/cortical_wave_obstacle.png`: four time-ordered activity snapshots,
  a categorical outcome map, and its continuous transmission measure;
- `outputs/phase_metrics.csv`: arrivals, regional peak activity, matched-control
  transmission, and the outcome assigned to every condition.

## Outcome rules

Each nonzero patch radius is compared with the zero-radius rollout at the same
inhibition strength. A condition counts as propagation only when source,
pre-obstacle, and far-edge activity arrive in that order.

- `dies`: the far edge is not reached causally, or transmission is below 25%
  of the matched control.
- `splits`: propagation succeeds through both flanks and wake recruitment lags
  flank arrival by at least one membrane time constant (15 ms).
- `bends`: propagation succeeds but does not meet the balanced two-flank split
  criterion.
- `control fails`: the matched zero-radius sheet does not itself propagate, so
  no lesion effect is claimed.

The weights define an unsourced phenomenological propagation regime. The CSV
retains the continuous observables so phase boundaries can be recalibrated
without rerunning the model.
