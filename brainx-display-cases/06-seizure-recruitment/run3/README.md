# Seizure recruitment across regions

This experiment applies a finite pulse to region 1 of a directed three-region
Epileptor chain and maps when the burst stays local, recruits one neighbor, or
propagates in strict region 1 -> 2 -> 3 order.

Run it with:

```bash
python seizure_recruitment.py
```

The script uses `brainstate.transform.vmap` across coupling strengths,
propagation delays, and perturbation sizes. Each mapped condition constructs an
independent `brainmass.EpileptorStep` model, while
`brainstate.transform.for_loop` advances its mutable state through time. A
fixed-capacity `brainstate.nn.Delay` keeps mapped delay-buffer shapes static.

Time, delay, pulse timing, and minimum event duration retain `brainunit` units.
Epileptor state, coupling strength, and perturbation size are dimensionless by
the model definition. The model's internal regional time scales retain the
BrainMass defaults.

The sampled parameter values are calibrated for this reproducible demonstration
and the resulting regimes are phenomenological, not patient-specific or
clinical thresholds.

Outputs are written to `outputs/`:

- `seizure_recruitment.png`: regime maps, example traces, and recruitment
  latency versus propagation delay.
- `seizure_recruitment_results.npz`: coordinates, units, protocol, full
  trajectories, continuous boundary evidence, categorical labels, and matched
  controls.

The event predicate is fixed at `x1 > 0` for at least `1 ms`. The script also
checks the delay phase convention and asserts two causal controls: without
coupling the pulse remains local, and without the pulse coupling alone recruits
nothing.
