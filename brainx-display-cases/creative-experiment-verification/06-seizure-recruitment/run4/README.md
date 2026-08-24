# Seizure Recruitment Across Regions

This BrainX demonstration applies a short perturbation to the first of three
directed, delay-coupled FitzHugh-Nagumo regional neural masses. It shows a
phenomenological seizure-like population event that remains local under weak
coupling and recruits both neighboring regions under stronger coupling.

Run it with:

```bash
python seizure_recruitment.py
```

The script uses `brainstate.transform.for_loop` for time and maps every
independent combination of coupling strength, edge delay, and perturbation size
with `brainstate.transform.vmap`. BrainUnit quantities retain the model time
constant, integration step, delay, protocol timing, and explicit dimensionless
coupling and stimulation contracts.

Recruitment is fixed before the sweep as regional activity `V >= 0.5` for at
least `1 ms`. The saved result bundle includes the maximum sustained-window
floor used by this predicate, event onset times, peak activity, parameter axes,
matched no-coupling and no-stimulation controls, and representative traces.

Outputs:

- `outputs/seizure_recruitment.png`
- `outputs/seizure_recruitment_results.npz`

The regimes are calibrated for illustration and are not a patient-specific or
clinical seizure prediction.
