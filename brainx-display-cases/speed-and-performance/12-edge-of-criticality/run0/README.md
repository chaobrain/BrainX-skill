# Finding the Edge of Criticality

This experiment drives one forced excitatory spike into otherwise quiescent,
recurrent excitatory and inhibitory LIF populations. It scans excitatory
coupling strength across paired random network realizations, measures the
resulting avalanche, excludes persistent runs, and reports the stable gain
with the largest across-realization avalanche-size coefficient of variation.

The implementation is BrainX-native:

- `brainpy-state` owns the two LIF populations, exponential synapses, and COBA
  current conversion.
- `brainevent.JITCScalarR` regenerates reproducible sparse graphs from compact
  seeds and propagates binary spikes through the four recurrent pathways.
- `brainstate.transform.vmap2` maps the complete stateful transition over the
  gain/realization lanes; `for_loop` owns time and `jit` compiles the rollout.
- `brainunit` keeps membrane, synaptic, timing, and coupling parameters as
  explicit physical quantities.

The topology and initial voltages for a realization are held fixed across all
gains (common random numbers), so gain is the only intervention within each
paired comparison. The stability and selection thresholds are defined in
`Experiment` before simulation. This is a phenomenological critical region,
not a claim of a universal critical exponent.

Run a small end-to-end check:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python criticality_scan.py --quick --output-dir quick-results
```

Run the default 19-gain, 16-realization scan:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python criticality_scan.py --output-dir results
```

Each run writes `criticality_scan.csv`, `summary.json`, and
`criticality_scan.png`. The JSON records the selected gain, critical interval,
stability predicate, seeds, and a no-spark control at the strongest gain.

Run the focused analysis test with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache pytest -q
```
