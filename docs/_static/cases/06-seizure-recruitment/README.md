# Seizure recruitment across regions

This example starts a finite seizure-like population burst in region 0 of a
four-region directed chain and maps whether neighboring regions are recruited.
It uses:

- `brainmass.FitzHughNagumoStep` for excitable regional population dynamics and
  `brainmass.additive_coupling` for directed inter-region input;
- `brainstate.transform.for_loop` for time and `brainstate.transform.vmap` for
  complete independent rollouts across coupling, delay, and pulse size;
- `brainstate.nn.Delay` for fixed-capacity propagation history;
- `brainunit` quantities for the integration step, duration, pulse timing,
  propagation delay, and sustained-event duration.

The model is phenomenological: a recruitment event is defined in advance as
`V >= 0.5` continuously for at least `1 ms`. The script retains peak activity
and onset time for every region so the categorical local/spreading labels can
be checked against continuous observables. FitzHugh-Nagumo activity, additive
coupling gain, and pulse amplitude are dimensionless model quantities; all
physical time parameters remain unit-bearing throughout the simulation.

Run:

```bash
python seizure_recruitment.py
```

The script prints representative local and spreading cases and creates
`seizure_recruitment.png` with their traces, recruitment extent over
coupling/pulse size, and distal onset over coupling/delay. Vertical lines mark
sustained-event onset in each recruited region; blank distal-onset cells are
sampled conditions where region 3 was not recruited.

Run the focused delay and recruitment checks with:

```bash
python -m unittest -v
```
