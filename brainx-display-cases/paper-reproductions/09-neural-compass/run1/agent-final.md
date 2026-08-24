Built the BrainX spiking compass in [neural_compass.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.srUzpI/workspace/neural_compass.py:1), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.srUzpI/workspace/README.md:1).

Results:

- The bump moved `94.9°` during a commanded `90°` dark turn.
- Final heading was `99.2°`, with `9.2°` error and `0.59` coherence.
- Across all 48 starting headings: 11 spared, 4 recovered, 33 failed.
- Recovered headings: `52.5°`, `60°`, `300°`, `307.5°`.

Artifacts: [result figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.srUzpI/workspace/neural_compass_results.png) and [continuous metrics](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.srUzpI/workspace/neural_compass_results.npz).

Verified with the full simulation, delay impulse check, matched-control checks, recovery predicates, syntax compilation, and finite-value validation.