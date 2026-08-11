# Spike-Frequency Adaptation

This BrainX experiment applies a steady current to a conductance-based
single-compartment neuron. Spikes admit calcium, intracellular calcium recruits
a calcium-activated potassium afterhyperpolarization (AHP) current, and that
outward current makes subsequent spikes progressively harder to trigger.

The controlled ablation sets only `g_AHP` to zero. Sodium, delayed-rectifier
potassium, calcium, leak, stimulus, and solver settings remain identical. With
the AHP current removed, the inter-spike intervals stop lengthening as strongly
and the late firing rate stays closer to the early rate.

Run the experiment with:

```bash
python spike_frequency_adaptation.py
```

It prints early and late rates for a current-by-adaptation sweep and writes
`spike_frequency_adaptation.png`. Use `--show` to display the figure as well.

The implementation uses:

- `braincell.SingleCompartment`, dynamic calcium, `MixIons(k, ca)`, and
  `AHP_De1994` for the removable adaptation mechanism.
- `brainstate.transform.for_loop` for stateful time evolution and nested
  `brainstate.transform.vmap` calls for the parameter grid and per-condition
  rate comparison.
- `brainunit` quantities throughout the model and explicit unit conversion only
  at the NumPy/Matplotlib presentation boundary. Current, voltage, conductance,
  capacitance, and time remain dimensionally checked during the simulation.
