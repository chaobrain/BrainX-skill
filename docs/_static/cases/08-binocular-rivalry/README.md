# Binocular rivalry neural-mass study

This study models two continuously driven visual populations: one represents vertical stripes
and the other horizontal stripes. Each population suppresses the other. Its own activity builds a
slow adaptation current, weakening the current winner until the suppressed population can take
over. An Ornstein-Uhlenbeck (OU) current supplies temporally correlated fluctuations that make
switch timing irregular across observers.

The competition core is BrainMass's reduced Wong-Wang mass. Its gating states obey

```text
tau_S dS_i/dt = -S_i + (1 - S_i) gamma r_i
tau_a da_i/dt = -a_i + g_a S_i
```

where the rate `r_i` is driven by self-excitation, inhibition from the other population, and equal
continuous sensory input. The reported activity is the normalized NMDA gating `S_i`. Adaptation
`a_i` subtracts from its own population's current and OU noise `eta_i` is added to that current.
The script reuses Wong-Wang's public current, transfer-function, derivative, and exponential-Euler
APIs for this adaptation-augmented update. This is a phenomenological neural-mass regime, not a
fitted account of a particular experiment.

Run it with:

```bash
python binocular_rivalry.py
```

The script uses BrainMass for the competing Wong-Wang populations and OU noise, BrainState
`HiddenState` for adaptation, `vmap` for independent observer lanes, and `for_loop` for time.
BrainUnit keeps time constants, input, adaptation, and coupling dimensionally consistent. It simulates 12 independent
observers at every point of a 5 x 5 adaptation/noise grid and saves:

- `results/binocular_rivalry.png`: one observer trace and the dominance-duration map.
- `results/binocular_rivalry_results.npz`: observer-level metrics, grid summaries, example
  trajectories, units, parameters, seed, and analysis metadata.

Dominance uses a fixed hysteresis threshold of `S_vertical - S_horizontal = +/-0.15`. The first
5 seconds are discarded. The reported window-average interval is analysis time divided by the
number of percept segments, so it remains defined for observers with no switch; completed,
boundary-excluded episode durations and their counts are also stored.

Stronger adaptation shortens dominance because the active population fatigues faster. More noise
also shortens dominance on average by increasing the chance of escape from the current attractor,
especially when adaptation alone leaves it stable. Weak adaptation and weak noise therefore
produce the longest percepts. In this symmetric equal-drive model, neither stimulus has
a systematic duration advantage; an input imbalance would instead lengthen dominance of the
more strongly driven percept.
