# Seizure Recruitment Across Regions

This experiment starts a brief seizure-like population burst in region 0 of a
four-region chain, then maps when the event stays local and when it recruits
successive neighbors.

It uses:

- `brainmass.FitzHughNagumoStep` for fast-slow excitable regional population
  dynamics and `brainmass.additive_coupling` for directed inter-region input;
- `brainstate.transform.for_loop` for time and `brainstate.transform.vmap` for
  the complete independent sweep over coupling strength, per-edge delay, and
  perturbation size;
- `brainunit` quantities for the integration step, regional time constant,
  simulation/stimulation timing, and delays. The FHN state, drive, and coupling
  gain are dimensionless by the model definition.

Run the experiment from this directory:

```bash
python seizure_recruitment.py
```

The script writes:

- `outputs/seizure_recruitment.png`: local and recruited traces, recruitment
  onset versus delay, and the full regime map;
- `outputs/seizure_recruitment_metrics.csv`: one row per condition and region,
  retaining peak activity, threshold status, and onset time;
- `outputs/seizure_recruitment_data.npz`: all traces, parameter coordinates,
  condition types, and derived observables.

A region is called recruited when its fast population activity first reaches
`V = 0.5`. The continuous peak and onset observables are saved alongside that
label. No-coupling and no-perturbation controls run as extra lanes of the same
stateful parameter mapping and are checked automatically. This is a
deterministic phenomenological demonstration of excitable regional
propagation, not a calibrated clinical seizure model.
