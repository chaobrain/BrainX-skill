Built and ran the BrainX cortical-wave experiment.

The 42-condition sweep produced:

- 4 crossings
- 16 splits
- 4 one-sided bends
- 18 deaths

Key artifacts:

- [Wave storyboard](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.YvtaQe/workspace/results/wave_storyboard.png)
- [Phase map](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.YvtaQe/workspace/results/phase_map.png)
- [Simulation source](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.YvtaQe/workspace/cortical_wave.py:125)
- [Outcome data](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.YvtaQe/workspace/results/outcomes.csv)
- [README](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.YvtaQe/workspace/README.md)

The three focused checks passed through direct invocation. `pytest` itself was unavailable in the environment. Both generated figures were also checked as nonblank, correctly sized outputs.