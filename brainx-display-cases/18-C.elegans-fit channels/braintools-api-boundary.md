# BrainTools API boundary

## Required capabilities

| Capability | BrainTools API used | Custom implementation boundary |
|---|---|---|
| Bounded multistart fitting | `braintools.optim.ScipyOptimizer` with `L-BFGS-B` | Call the optimizer once per locked seed with the declared physical bounds. Reconstruct its documented uniform sampled start from the NumPy RNG state for complete diagnostics. |
| Robust fitting loss | `braintools.metric.huber_loss` | Per-trace scale factors are computed from the supplied traces before calling the metric. |
| Squared-error reporting | `braintools.metric.squared_error` | Square roots and per-trace aggregation convert MSE to RMSE. |

No raw SciPy optimizer or metric is called by `fit_channels.py`.

## Non-BrainTools boundaries

- Use `igor2` to parse packed experiments and their recreation records; BrainTools has no Igor reader.
- Use NumPy for baseline subtraction, deterministic sampling-grid construction, packed-wave assembly, Pearson correlation, and archive serialization.
- Use BrainCell for gate lifecycle, steady-state reset, conductance, ion reversal, and current evaluation inside every global objective.
- Use BrainUnit quantities through the BrainCell current boundary and convert explicitly to pA for comparison with the packed traces.
- Use Matplotlib only after numerical validation to draw the requested evidence.

## Optimizer diagnostics

Archive each seed, sampled physical start, initial objective, callback objective history, final objective, termination status and message, iteration and evaluation counts, final physical parameters, active physical bounds, and per-voltage loss. The BrainTools wrapper exposes only the best result when `n_iter > 1`, so call it six times with `n_iter=1` and retain all six results. Use this identical optimizer protocol for observed, recovery, and leave-one-voltage-out fits.
