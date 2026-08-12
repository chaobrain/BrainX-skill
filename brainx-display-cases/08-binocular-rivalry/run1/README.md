# Binocular rivalry

This experiment models vertical- and horizontal-selective visual populations
with `brainmass.WongWangStep`. Both receive the same continuous drive. Mutual
competition produces a winner, slow current adaptation fatigues that winner,
and independent Ornstein-Uhlenbeck current fluctuations perturb the time of
escape to the other percept.

Run:

```bash
python binocular_rivalry.py
```

The script uses `brainstate.HiddenState` for adaptation and OU noise,
`brainstate.ShortTermState` for the hysteretic percept readout,
`brainstate.transform.for_loop` for time, and a stateful
`brainstate.transform.vmap` for the complete 160-observer cohort. `brainunit`
keeps time, rate, drive, adaptation, noise, and coupling dimensionally
consistent until explicit analysis and plotting boundaries.

Outputs are written to `outputs/binocular_rivalry_results.npz` and
`outputs/binocular_rivalry.png`. The numeric bundle contains per-observer
conditions and summaries, condition-level duration/switch/censoring grids,
locked and undecided fractions, the complete protocol with units, and one
representative trajectory.

Dominance intervals touching either edge of the post-burn-in analysis window
are right/left censored and excluded from the complete-duration statistic.
Consequently, a decided no-switch run is shown by its locked fraction rather
than by an invented finite dominance duration. The exact zero-noise control is
reported separately as undecided: without a fluctuation, the perfectly
symmetric initial state gives adaptation no winner to fatigue.

The displayed parameter regime is a phenomenological, outcome-calibrated
demonstration, evaluated with a held-out random seed. It supports mechanism
illustration rather than a quantitative fit to a particular psychophysical
dataset.
