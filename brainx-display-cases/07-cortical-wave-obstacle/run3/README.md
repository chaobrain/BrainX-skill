# A Cortical Wave Meets an Obstacle

This experiment builds a two-dimensional sheet with one excitatory and one
inhibitory LIF neuron at each site. A 3 ms current spark drives the two leftmost
columns. Local eight-neighbor projections carry binary spikes through explicit
BrainEvent CSR matrices; excitatory and inhibitory conductances then drive the
BrainPy-State populations.

A circular mask makes both populations inside the patch functionally silent:
their spikes are suppressed before monitoring or communication. All patch
radii and inhibitory gains run together as independent BrainState dynamic-state
lanes. `vmap2` is the filter-based stateful `vmap` API, and its mapped step is
called by one `for_loop` over physical time.

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python cortical_wave.py
```

Outputs are written to `results/`:

- `wave_snapshots.png`: aligned control and lesion activity at five physical
  times. Each panel integrates the preceding 4 ms so the front is visible.
- `phase_map.png`: categorical outcomes plus every continuous response used at
  the phase boundaries.
- `phase_metrics.npz`: radii, inhibition gains, phase codes, raw route counts,
  matched-control ratios, and first-arrival ordering margins.

For every inhibition value, radius zero is the matched control. A lesion
"dies" if downstream activity falls below 20% of that control or if route
activity does not precede downstream activity. A surviving wave "splits" when
both the upper and lower routes retain at least 15% of their matched-control
activity and each carries at least 25% of the routed activity; otherwise it
"bends" through the dominant surviving route. If the no-lesion
wave itself does not reach the target band, the row is marked `control fails`
rather than attributing failure to the patch.

The conductances are phenomenologically calibrated to expose propagation and
blocking regimes; they are not fitted to a specific cortical preparation.
