Implemented the complete BrainX study in [binocular_rivalry.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.gzCkBn/workspace/binocular_rivalry.py), with methodology in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.gzCkBn/workspace/README.md).

It uses:

- BrainMass Wong-Wang competing populations and OU noise
- BrainState adaptation state, state-aware `vmap`, and `for_loop`
- BrainUnit quantities for currents and times
- 300 observers across a 5 x 5 adaptation/noise grid

Results:

- Every observer alternated.
- Mean dominance fell from `2.46 s` to `1.29 s` as adaptation increased.
- Increasing noise reduced mean dominance from `2.03 s` to `1.43 s`.
- Equal stimulation produced balanced occupancy: `50.31%` vertical.
- Stronger adaptation accelerates winner fatigue; stronger noise increases attractor escape. Stronger recurrence/inhibition would generally stabilize the winner, while unequal input would favor one percept.

Artifacts: [figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.gzCkBn/workspace/results/binocular_rivalry.png) and [numeric results](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.gzCkBn/workspace/results/binocular_rivalry_results.npz). The script compiled successfully and all scientific artifact checks passed.