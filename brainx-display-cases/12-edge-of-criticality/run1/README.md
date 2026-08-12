# Finding the Edge of Criticality

This experiment delivers one brief current pulse to an eight-neuron excitatory seed assembly in an otherwise quiescent recurrent E/I leaky integrate-and-fire network, then sweeps only the E-to-E coupling over matched sparse network realizations.

The BrainX execution path is:

- `brainpy-state` LIF populations and exponential synaptic currents
- seeded, positive `brainevent.JITCUniformR` E-to-E weights and `JITCScalarR` E/I pathways
- one stateful `vmap2` over coupling-realization lanes inside one `brainstate.transform.for_loop` over time
- `brainunit` quantities for voltage, current, and time

An avalanche is a contiguous run of nonempty 2 ms population bins. Its variability is the stable-run susceptibility `variance(size) / mean(size)`. A run is unstable when at least 80% of bins in the final 50 ms remain active. The reported critical region must contain at least two adjacent sampled couplings, remain at or below 10% unstable realizations, and reach at least 90% of the stable susceptibility peak.

The network parameters and operational criticality thresholds are phenomenological. The experiment locates this model's transition; it does not claim a calibrated biological critical point or a power-law fit.

Run the full sweep:

```bash
python edge_of_criticality.py
```

For a quick execution check:

```bash
python edge_of_criticality.py --realizations 4 --couplings 0.8 1.0 1.2
```

The default grid is coarse away from the transition and 0.01-wide near it. Its seed is held out from the calibration runs used to choose that grid. The script writes aggregate metrics, per-realization stability evidence, binned population counts, and a summary figure under `results/`.
