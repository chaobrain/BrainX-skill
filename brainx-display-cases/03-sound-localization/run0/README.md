# Sound localization from timing

This example implements a compact Jeffress-style spiking circuit. Two auditory
LIF relays fire once at the left- and right-ear arrival times. A bank of 13 LIF
coincidence detectors uses opposing internal delay lines to cover preferred
interaural time differences (ITDs) from `-0.6 ms` to `+0.6 ms`. Detector spikes
are routed to two LIF readout neurons representing `LEFT` and `RIGHT`.

The sign convention is:

- positive ITD: right ear is later, so the source is `LEFT`;
- negative ITD: right ear is earlier, so the source is `RIGHT`;
- zero ITD: both directional readouts are silent, so the result is `CENTER`.

BrainPy-State owns all three neuron stages, BrainEvent performs the sparse
binary-event products, BrainState runs time with `for_loop` and independently
maps each complete circuit transition with state-aware `vmap2`, and BrainUnit
preserves the units of every time, voltage, resistance, and current.

Run the sweep:

```bash
MPLCONFIGDIR=/tmp/brainx-mpl python sound_localization.py
```

Run the tests:

```bash
MPLCONFIGDIR=/tmp/brainx-mpl python -m unittest -v
```
