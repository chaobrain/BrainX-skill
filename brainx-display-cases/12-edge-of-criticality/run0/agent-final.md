Implemented the BrainX criticality experiment in [criticality_scan.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run0-eval.Ot6FJ9/workspace/criticality_scan.py), with usage documented in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run0-eval.Ot6FJ9/workspace/README.md).

The full 19-gain × 16-realization run found:

- Peak stable variability at excitation gain `2.1`, CV `3.97`
- Instability rising from `12.5%` at gain `2.2` to `93.75%` at `2.6`
- Zero spontaneous spikes in the strongest-gain no-spark control

Results are in [summary.json](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run0-eval.Ot6FJ9/workspace/results/summary.json), [criticality_scan.csv](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run0-eval.Ot6FJ9/workspace/results/criticality_scan.csv), and [criticality_scan.png](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/edge-criticality-run0-eval.Ot6FJ9/workspace/results/criticality_scan.png).

Verification: compilation passed, both analysis tests passed directly, and quick/default BrainX runs completed. `pytest` itself was unavailable in the environment.