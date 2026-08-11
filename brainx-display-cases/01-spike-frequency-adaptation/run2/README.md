# Spike-Frequency Adaptation

This experiment shows how a slow calcium-activated potassium current makes a
conductance-based neuron fire rapidly at first and then progressively more
slowly during a steady input. Setting only `g_AHP` to zero removes that
adaptation current for a controlled comparison. Every condition receives the
same brief hyperpolarizing holding current outside the steady-input window so
the onset response starts from a quiet baseline.

The cell is a teaching model: classical Hodgkin-Huxley sodium and potassium
spiking are combined with dynamic calcium, `AHP_De1994`, and a leak current. It
is not a reproduction of one named biological cell type.

Run it with:

```bash
python spike_frequency_adaptation.py
```

The script:

- keeps voltage, current, conductance, capacitance, and time unit-aware with
  BrainUnit;
- evolves all cell state with `brainstate.transform.for_loop`;
- uses the BrainCell `SingleCompartment` condition grid for the dynamics and
  nested `brainstate.transform.vmap` calls for per-condition spike summaries;
- prints spike counts, first and last interspike intervals, and their ratio;
- saves `spike_frequency_adaptation.png` with the voltage, intracellular
  calcium, ISI, ablation, and current/strength sweep results.

An ISI ratio near 1 means tonic firing. A ratio above 1 means the intervals
lengthen over time, which is the operational signature of spike-frequency
adaptation used here.
