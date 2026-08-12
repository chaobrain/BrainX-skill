Implemented and ran the complete BrainX experiment in [edge_of_criticality.py](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run1-eval.aVByru/workspace/edge_of_criticality.py:1).

The held-out 320-lane ensemble found:

- Critical region: E-E coupling `0.745-0.750`
- Sampled optimum: `0.750`
- Susceptibility: `2089.73`
- Unstable realizations at optimum: `0/16`
- Runaway probability rises sharply above this region

Outputs include the [summary figure](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run1-eval.aVByru/workspace/results/criticality_summary.png), [aggregate metrics](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run1-eval.aVByru/workspace/results/criticality_metrics.csv), per-realization evidence, and raw binned counts.

Compilation, artifact recomputation, and four focused tests passed. `pytest` was unavailable, so the test functions were executed directly. Usage and scientific definitions are documented in [README.md](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run1-eval.aVByru/workspace/README.md:1).