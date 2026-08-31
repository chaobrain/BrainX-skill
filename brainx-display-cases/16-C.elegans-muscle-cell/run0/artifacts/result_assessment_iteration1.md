# Iteration 1 result assessment

## Deterministic outcome

Mechanical run status: done, exit code 0, all declared JSON/CSV/NPZ results parse, and every prediction is finite.

Scientific acceptance status: fail.

- Training MSE improved from 251.116987 to 82.014809 mV^2, but the fitted training trace has only 2 spikes versus 4 observed, RMSE 9.056 mV, correlation 0.666, and peak -5.994 mV versus 24.475 mV.
- Observed spike counts at 15/20/25/30 pA are 3/3/4/4 under the frozen -10 mV detector. Predicted counts are 1/1/2/1 and therefore violate the required nondecreasing current response.
- Held-out RMSE is 10.259 mV at 15 pA, 10.221 mV at 20 pA, and 10.894 mV at 30 pA.
- Zero-current behavior is quiet (0 spikes).
- dt versus dt/2 training-trace RMSE is 4.544 mV, too large to treat the fitted waveform as numerically converged.
- Synthetic recovery error is 0.010% for `g_na` and 10.42% for `g_kr` in the limited two-parameter perturbation, but this does not establish six-parameter identifiability.

## Claim-evidence matrix

| Proposed claim | Evidence | Outcome |
|---|---|---|
| Fitting improves the one-trace waveform objective | 251.116987 -> 82.014809 mV^2 | supported |
| Model behavior is consistent with held-out current responses | spike counts 1/1/2/1 vs observed 3/3/4/4; held-out RMSE 10.22-10.89 mV | rejected |
| Response strength is monotonic with current | predicted counts decrease from 2 at 25 pA to 1 at 30 pA | rejected |
| Baseline is quiet without current | zero-current spike count 0 | supported |
| Numerical result is converged at dt=0.05 ms | dt-refinement RMSE 4.544 mV | rejected |
| Fitted conductances are biologically identifiable | one trace, boundary-adjacent solutions, limited recovery | withheld |

## Proposed next action

Return to implementation in iteration 2. Correct the observation objective so event count/timing and peak amplitude participate alongside waveform error, reduce the integration step to 0.025 ms for fitting and evaluation, and narrow/fix weakly identifiable conductances instead of allowing all six to compensate at bounds. Preserve the same locked data split and held-out traces.
