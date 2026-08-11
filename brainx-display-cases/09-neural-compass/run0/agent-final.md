Built and ran the BrainX spiking compass in [head_direction_compass.py](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.SIFfph/workspace/head_direction_compass.py:105). It uses `LIFRef`, event-driven recurrence, unit-aware parameters, `for_loop`, and state-filtered `vmap2`.

Results:

- Intact ring tracked all 36 headings through a 90-degree dark turn.
- Mean control trajectory RMSE: approximately 4.1 degrees.
- Lesioned ring recovered for starts from 100 through 160 degrees: 7/36.
- Other starts retained activity but settled at incorrect lesion-pinned headings.

Artifacts: [figure](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.SIFfph/workspace/results/head_direction_compass.png), [lesion sweep](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.SIFfph/workspace/results/lesion_sweep.csv), and [summary](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.SIFfph/workspace/results/summary.json).

Verification: `python -m unittest -q` passes all 5 tests, including the complete mapped experiment. Parameters are documented as a phenomenological demonstration rather than a fitted biological regime.