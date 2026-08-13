# V1 single-cell perturbation, run1

This run tests three connectivity regimes in one fixed BrainPy-State spiking E/I
network realization. Parameters, random seeds, correlation bins, and the
qualitative scoring rule were fixed in the script before its first execution.

This is a correctness retry of `run0`, whose forced and ordinary excitatory
projections used matching descriptors and could therefore merge their synaptic
state. This run uses concrete synapse/output instances for independent channels
and verifies exact baseline replay after an independent-rollout state reset.

The requested perturbation description is internally inconsistent: four pulses
at 15 Hz are an approximately 200-ms pulse train, whereas six added action
potentials spread across 250 ms are not a 15-Hz six-event train. This run treats
the explicitly requested causal dose as authoritative and injects exactly six
additional events at 100, 150, 200, 250, 300, and 350 ms. Separate projection
channels make these events additive even if the target spikes spontaneously on
the same time step. The target is excluded from neighbor influence analysis.

Run with:

```bash
MPLCONFIGDIR=/home/yixinliu/brainx-runs/.matplotlib \
  conda run -n braincell-released python v1_single_cell_perturbation.py
```

Outputs are written once to `results/`. Re-running this script in this directory
is intentionally refused because experiment outputs are immutable.
