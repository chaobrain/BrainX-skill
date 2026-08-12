Implemented the binocular-rivalry experiment in [binocular_rivalry.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.cFvf6S/workspace/binocular_rivalry.py), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.cFvf6S/workspace/README.md).

It simulates 160 observers using `brainmass.WongWangStep`, BrainState adaptation/noise state, `for_loop`, and a complete stateful `vmap`. BrainUnit tracks all time, rate, current, adaptation, noise, and coupling quantities.

Held-out results show:

- At weak noise, increasing adaptation reduced median dominance from `7.454 s` to `1.315 s`.
- Strong noise produced about `157` switches/minute and largely overwhelmed adaptation effects.
- Vertical occupancy averaged `0.5095`, confirming unbiased continuous stimulation.
- With exactly zero noise, perfect symmetry remains undecided; adaptation alone cannot select a winner.
- Boundary-truncated dominance intervals are treated as censored.

Generated [figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.cFvf6S/workspace/outputs/binocular_rivalry.png) and [numeric results](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.cFvf6S/workspace/outputs/binocular_rivalry_results.npz). The script compiles, runs successfully, and the artifact assertions pass. The selected regime is explicitly labeled phenomenological and outcome-calibrated, not an empirical psychophysical fit.