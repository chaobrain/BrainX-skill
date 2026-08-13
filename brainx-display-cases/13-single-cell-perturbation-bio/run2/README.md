# V1 feature competition, run2

This run implements the requested spatially and functionally structured V1 L2/3
E/I spiking network. Parameters, seeds, bins, and qualitative criteria are fixed
in `v1_feature_competition.py` before perturbation outcomes are inspected.

Six additive target output events at 100, 150, 200, 250, 300, and 350 ms model
six imposed action potentials. Their recurrent effect uses the target's same
outgoing weight row through an independent excitatory synaptic channel. The
normalization dose is measured per paired trial as imposed events plus the
change in the target's intrinsic spikes. Baseline and perturbed branches restore
the same complete BrainState snapshot and reuse identical sensory noise.

Run once with:

```bash
MPLCONFIGDIR=/tmp/brainx-v1-run2-mpl \
  conda run -n braincell-released python v1_feature_competition.py
```

The script refuses to overwrite a nonempty `results/` directory.
