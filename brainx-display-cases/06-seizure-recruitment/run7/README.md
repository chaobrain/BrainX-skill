# Seizure Recruitment Across Regions

This study starts a focal seizure-like burst in a directed three-region chain
and maps when it stays local versus recruits both downstream regions.

It uses:

- `brainmass.EpileptorStep` for each regional population and BrainMass
  diffusive coupling for delayed propagation;
- `brainstate.transform.for_loop` for every rollout and a state-aware
  `brainstate.transform.vmap` over coupling strength, propagation delay, and
  focal perturbation size;
- `brainunit` quantities for integration time, pulse timing, delay capacity,
  delay retrieval, event windows, and recruitment onset times.

Epileptor states, coupling gain, and input amplitude are dimensionless by the
model contract; BrainMass owns the regional model's time constants, while all
physical protocol and propagation times remain BrainUnit quantities.

Run:

```bash
python seizure_recruitment.py
```

Outputs are written to `results/seizure_recruitment.png` and
`results/seizure_recruitment.npz`. The numeric bundle includes full `x1`
trajectories, continuous event evidence, recruitment flags and onsets, all
parameter coordinates, controls, units, connectivity, and classification
metadata.

The exploratory event rule is fixed in the script: a region is recruited when
`x1 >= 0` for at least 2 cumulative milliseconds within a 40 ms window. Routed
recruitment additionally requires strictly ordered onsets from focus to
neighbor 1 to neighbor 2. The event rule and sampled demonstration regimes are
outcome-calibrated and should not be treated as confirmatory or as fitted
patient-specific parameters.
