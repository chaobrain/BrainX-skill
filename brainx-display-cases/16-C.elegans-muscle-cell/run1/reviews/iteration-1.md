# BrainX iteration review

- **OUTCOME:** `REFUSE`
- **SCIENTIFIC_OUTCOME:** `INCONCLUSIVE`
- **LOSS_CLOSURE:** `OPEN`
- **OPTIMIZATION_ADEQUACY:** `INSUFFICIENT`
- **NEXT_ACTION:** `RETURN_TO_IMPLEMENTATION`

## Good-enough reason

The finite, leak-controlled run shows that the selected fit fails the locked held-out criterion, but every observed and recovery fit exhausted its budget without convergence, so the 15 pA failure cannot yet distinguish model inadequacy from an under-solved fitting problem.

## Findings

### FIT-001: Optimization stopped before loss closure

- **Severity:** `critical`
- **Location:** `runs/20260824-celegans-production-01/raw/fit_starts.json`; `cellegans_hh/inference.py:144`
- **Problem:** All three observed starts report `success=false` and maximum-iteration termination after only 700 candidates each. Each start found its best loss during the final three generations, while no initial-parameter objective, component-wise curve, attainable floor, or budget comparison is provided.
- **Scientific consequence:** The unfavorable held-out prediction may result from budget-limited fitting rather than the declared model's predictive limit.
- **Minimum fix:** Increase only the derivative-free evaluation budget under the unchanged objective, bounds, starts, seeds, and selection rule until a predeclared plateau or estimator convergence is demonstrated; retain cumulative-best and component-wise histories.

### CONTRACT-001: Exact fitting problem was not researcher-locked

- **Severity:** `major`
- **Location:** `NeuroSpecification.md`; `cellegans_hh/model.py:32`; `cellegans_hh/inference.py:88`; `runs/20260824-celegans-production-01/RUN_SPEC.md`
- **Problem:** The locked specification does not state parameter bounds or their provenance, nor the implemented objective weights for total RMSE, baseline RMSE, spike count, and spike timing. Source hashes show that code was frozen for execution but do not establish scientific approval of these choices.
- **Scientific consequence:** The selected solution and negative prediction may depend on undocumented bounds or arbitrary objective tradeoffs; several fitted values also lie near bounds.
- **Minimum fix:** Obtain researcher approval for the existing exact parameter map, bounds, objective formula, weights, stopping rule, and selection rule without changing them, then rerun the fitting assessment.

### FIT-002: Held-out uncertainty across equivalent starts is missing

- **Severity:** `major`
- **Location:** `runs/20260824-celegans-production-01/raw/fit_starts.json`; `scripts/run_experiment.py:199`
- **Problem:** Only the lowest-training-loss start is evaluated on held-out traces. The other starts have similar losses but materially different parameter vectors, and no per-start predictions or predictive variability are saved.
- **Scientific consequence:** It is unknown whether the missing 15 pA spikes are stable across comparably supported fits or peculiar to seed 2025.
- **Minimum fix:** Evaluate the three already completed parameter vectors through the unchanged held-out reset and observation path, retaining the original training-only selection rule, and report per-start held-out metrics.

### FIT-003: Recovery does not support parameter interpretation

- **Severity:** `major`
- **Location:** `runs/20260824-celegans-production-01/raw/recovery.json`; `scripts/run_experiment.py:110`
- **Problem:** Recovery uses only three interior truths for seven parameters, all nine fits terminate at the iteration limit, and synthetic observations, truth-parameter objective values, profiles, and tradeoff evidence are absent. Nevertheless, four parameters are classified as `interpretable`.
- **Scientific consequence:** The parameter-level claims are unsupported; the available recovery cannot resolve bias, flat directions, boundary failures, or compensating parameters.
- **Minimum fix:** After establishing optimizer closure, run a predeclared recovery set large enough to resolve parameter-specific errors and tradeoffs, save latent and noisy observations plus truth objectives, and withhold all interpretation until that gate passes.

### NUM-001: Numerical convergence is untested

- **Severity:** `major`
- **Location:** `cellegans_hh/model.py:184`; `artifacts/acceleration.json`
- **Problem:** The model uses `rk4` at 0.1 ms, but no `dt` refinement or second-solver comparison is supplied. Acceleration evidence establishes scalar/batch parity only, not temporal convergence.
- **Scientific consequence:** Near-threshold spike presence and latency, especially the decisive 15 pA failure, may be integration-step dependent.
- **Minimum fix:** Run the frozen selected parameters with `rk4` at 0.05 ms, downsample to the same observation times, and compare the locked spike and latency observables before interpreting the failure.

### VALID-001: Mechanical validation omits required State checks

- **Severity:** `major`
- **Location:** `tests/test_model.py:39`; `scripts/run_experiment.py:218`
- **Problem:** Tests and artifacts establish finite voltage but do not inspect gate or calcium State. Moreover, `spike_times_ms()` always masks to 50-300 ms, so its use for the zero-current control does not test the required pre-stimulus interval.
- **Scientific consequence:** The declared `mechanically_valid` result does not establish all finite States or the locked no-pre-stimulus-spike condition.
- **Minimum fix:** Add one full-state validation pass for nominal and boundary cases that records finite/range checks for voltage, gates, and calcium and checks pre-stimulus spikes without the stimulus-window mask.

### API-001: Fitting bypasses the BrainTools optimizer abstraction

- **Severity:** `minor`
- **Location:** `cellegans_hh/inference.py:8`
- **Problem:** The discontinuous candidate-batched objective is wired directly to `scipy.optimize.differential_evolution`; the routed BrainTools reference provides `braintools.optim.NevergradOptimizer` for bounded black-box objectives returning one loss per candidate.
- **Scientific consequence:** Unit reconstruction, candidate semantics, and optimizer bookkeeping remain manual, increasing fitting-contract risk despite current focused tests.
- **Minimum fix:** In a separate one-change parity comparison, replace only the optimizer boundary with `braintools.optim.NevergradOptimizer` while preserving the objective, physical bounds, candidate count, and evaluation budget.

## Unverified assumptions

- The custom channel kinetics, reversal potentials, calcium-pool constants, 2000 µm² area, and use of `AHP_De1994` as the SLO-2 surrogate are phenomenological assumptions whose biological fidelity cannot be established from the supplied artifacts.
- The 50-300 ms stimulus timing and 15-30 pA amplitudes are inferred because no current-monitor channel is available.
