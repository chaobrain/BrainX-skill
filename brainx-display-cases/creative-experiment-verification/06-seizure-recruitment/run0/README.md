# Seizure Recruitment Across Regions

This example starts a seizure-like population burst in one region and tests
whether it stays focal or recruits two neighbors in sequence. It uses:

- `brainmass.FitzHughNagumoStep` for fast-slow excitable regional dynamics and
  `brainmass.additive_coupling` for directed regional input;
- `brainstate.transform.for_loop` for each stateful rollout and
  `brainstate.transform.vmap` for the joint coupling, delay, and perturbation
  sweep;
- `brainunit` quantities for the regional time constant, integration step,
  stimulus timing, propagation delay, and explicitly dimensionless coupling
  and stimulation amplitudes.

Run:

```bash
python seizure_recruitment.py
```

The script writes `outputs/seizure_recruitment.png` and the complete continuous
summaries to `outputs/seizure_recruitment_results.npz`. The representative
local case uses coupling `0.2`; the recruited case uses coupling `0.4`. Both
use a `4 ms` delay and a `0.5` perturbation.

A burst onset is the first time `V > 0.5` persists for at least `2 ms`.
Recruitment requires ordered threshold crossings from the focus to Neighbor 1
and then Neighbor 2. The `.npz` file retains onset times and peak activity for
all regions at every grid point, rather than only the categorical map.

This is a deterministic, phenomenological demonstration. Its parameter values
are illustrative and are not calibrated for patient-specific or clinical use.
