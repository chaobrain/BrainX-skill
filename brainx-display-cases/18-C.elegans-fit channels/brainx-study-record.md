# BrainX study record

## Represented scale and package ownership

- BrainCell owns the two ion-channel mechanisms at one isopotential membrane. BrainUnit owns voltage, time, conductance, and current quantities. BrainState owns HH gate State and transformed lifecycle execution.
- No point-neuron network, event projection, neural mass, morphology, stochastic process, or training graph is represented.

## Relevant abstractions and invariants

- Implement each fitted mechanism as a `braincell.channel._base.HH` subclass with `Gate` declarations, `f_<gate>_inf`, `f_<gate>_tau`, and an ion-owned reversal potential supplied through `IonInfo`.
- Use `Gate("n", power=2)` for SHK-1 and `Gate("m", power=4)` for EGL-19. These powers are locked from equal-parameter local trace comparisons in the supplied data, not imported channel assumptions.
- Keep BrainCell's inward-positive current convention `g * gates * (E - V)` inside channel classes. Convert to the recordings' outward-positive convention only in the observation model.
- Initialize every independent voltage step at the fitted gate steady state for -60 mV. Do not carry gate State between voltage conditions.
- Preserve quantities in BrainCell. Convert explicitly to mV, ms, nS, and pA only at the BrainTools observation, NumPy archive, JSON, and Matplotlib boundaries.

## Fitting design

- SHK-1 target: baseline-corrected WT (`wave93:98`) minus baseline-corrected `shk-1(lf)` (`wave119:124`). Fit an `n^2` current, sigmoid activation, monotone exponential activation time constant, conductance, and reversal potential jointly. Omit a logistic time-constant midpoint because the supplied voltages do not identify it.
- EGL-19 target: baseline-corrected WT (`wave5:11`). Fit an `m^4` current, sigmoid activation, bell-shaped activation time constant, conductance, and reversal potential jointly. Compare optional `m^4h` local fits at the full 1,200-iteration budget and require a successful termination. Use BIC on the same 1,386 fitting samples to penalize the two added parameters per trace.
- Use `braintools.optim.ScipyOptimizer` directly on declared physical bounds with six locked random seeds and `braintools.metric.huber_loss`. Normalize residuals per trace so high-amplitude voltages do not erase low-amplitude kinetics. Preserve every sampled start, history, termination diagnostic, active bound, and per-voltage loss; select the successful finite minimum-objective candidate.
- Extract per-voltage activation and time-constant points by local fits with global nuisance quantities fixed. Treat these as model-conditioned experimental summaries, not independently observed gates.
- Run five deterministic interior-domain recovery cases per channel with measured baseline noise, plus leave-one-voltage-out fits. Use the same BrainCell objective, six seeds, physical bounds, budget, and selection rule as the observed fit. Archive raw observations, predictions, residuals, all starts, parameter errors, and boundary hits. Restrict interpretation to waveform prediction when recovery is weak after optimization adequacy is established.

## Execution and validation design

- Parse the two read-only packed files with `igor2`; require exact family and command groups in the Igor recreation text before loading targets.
- Evaluate every global objective through the actual BrainCell channel lifecycle: initialize and reset at -60 mV, write the exact constant-step HH gate solution into its `State`, and obtain current through `channel.current()`. Use the deterministic 0.5 ms fitting grid and preserve full resolution for evaluation and figures.
- Require analytic helper parity with the BrainCell path at nominal and near-boundary parameter sets.
- Save extracted targets, control differences, structural scores, full fitted traces, gate points, parameters, all optimizer starts, metrics, provenance, and a manifest before review.

## Sources studied

- Supplied Igor packed experiments and their embedded graph/history metadata.
- BrainX skills only for software mechanics: `brainx-general-guard`, `brainx-modeling-loop`, `braincell`, `brainunit`, `brainstate`, and the routed parameter-fitting/custom-channel references.
- No paper, external channel equation, or open-source scientific model was used.
