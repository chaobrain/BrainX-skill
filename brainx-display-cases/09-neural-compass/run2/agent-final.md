Built and ran the spiking neural compass.

The dark-turn bump followed a 90° rotation with `7.97°` final mean error, `0.00°` control drift, and `0.703` vector strength. Across all 72 starting directions, the persistent wedge lesion produced:

- 26 spared
- 0 recovered under the strict sustained-return criterion
- 46 failed, spanning approximately `-115°` through `110°`

Artifacts:

- [Experiment code](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.0jGANR/workspace/internal_neural_compass.py:133)
- [Result figure](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.0jGANR/workspace/outputs/internal_neural_compass.png)
- [Lesion outcomes CSV](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.0jGANR/workspace/outputs/lesion_outcomes.csv)
- [README](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.0jGANR/workspace/README.md:1)

Verification passed: compilation, five focused tests, full simulation, and a lesion check showing final damaged-neuron activity of `0.000001`. `pytest` was unavailable, so the pytest-compatible tests were executed directly.