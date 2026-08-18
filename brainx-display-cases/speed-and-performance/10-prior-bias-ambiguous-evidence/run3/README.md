# Prior bias under ambiguous evidence

This BrainX experiment asks whether a small prior changes two-choice decisions
mainly when sensory evidence is weak. Two noisy LIF populations recurrently
excite themselves and inhibit one another. Signed sensory current favors one
population while a `0.16 mA` prior favors choice A. Independent neuron noise
is combined with trial-level shared fluctuations, allowing nominally identical
trials to diverge into different attractor states.

The implementation uses:

- `brainpy-state` for the two spiking populations and four explicit synaptic
  projections;
- `brainstate.transform.for_loop` for time, `vmap2` for independent
  bias/evidence/trial lanes, and `jit` for the complete rollout;
- `brainunit` quantities for time, voltage, current, resistance, synaptic
  conductance, and decision-time reporting.

Run the experiment from this directory:

```bash
MPLCONFIGDIR=/tmp/brainx-mpl-cache python prior_bias_decision.py
```

The script writes a publication-style figure to
`results/prior_bias_decision.png` and machine-readable measurements to
`results/summary.json`. The plotted speed excludes compilation, synchronizes
device work before timing stops, and reports total simulated trial-seconds per
wall second. First-call compilation latency is reported separately.

Choices use a first-passage rule on the cumulative population spike-count
difference. The threshold is `0.8 spikes/neuron`; trials that do not cross by
`600 ms` use the sign of the final difference.
