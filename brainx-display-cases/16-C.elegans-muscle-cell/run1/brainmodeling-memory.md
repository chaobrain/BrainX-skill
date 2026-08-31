# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: locked fitting and held-out validation contract.
- `Fig4A-D.txt`: inspected read-only source data; 10,000 x 11 numeric table after ATF metadata.

### Important milestones
- Selected the single-cell ionic/channel scale and parameter-fitting execution mode.
- Predeclared Trace #8 (25 pA) for fitting and Traces #6, #7, and #9 for held-out testing.
- Resolved the contradictory channel count in favor of the six explicitly named currents and recorded the limitation.
- Step 0 complete; proceed to step 1.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: completed API, lifecycle, fitting, validation, and implementation design study.

### Important milestones
- Selected only BrainCell as the represented modeling scale, with BrainUnit and BrainState support.
- Active optional coverage is parameter fitting.
- Fixed the candidate reset, unit boundary, exact-pipeline recovery, passive-control, and held-out scoring design before implementation.
- Step 1 complete; proceed to step 2.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `cellegans_hh/data.py`: unit-bearing ATF parser, immutable data split, initial-voltage and current protocol mapping.
- `cellegans_hh/model.py`: BrainCell single-compartment model with SHK-1, EGL-19, SLO-2, Kr, Na, and leak currents.
- `cellegans_hh/inference.py`: explicit parameter map, batched candidate objective, bounded multi-start inference, passive control, and trace metrics.
- `tests/test_model.py`: split, parameter, deterministic reset, batch parity, and candidate-objective checks.
- `tests/run_checks.py`: dependency-free check runner because pytest is unavailable.

### Important milestones
- A 400- then 500-candidate pre-fit domain mechanics probe identified and corrected a non-oscillatory fixed-kinetics regime before production fitting.
- The revised domain contains stable 3-9 spike regimes; an 18-generation smoke fit matched the four training spikes, 7.1 ms first-spike error, 0.82 correlation, and 7.09 mV RMSE.
- Five focused implementation checks pass.
- Step 2 complete; proceed to step 3.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-audit.md`: hot-path inventory, State/shape contract, rewrite decision, and risks.
- `scripts/benchmark_acceleration.py`: cold/warm scalar-versus-batch benchmark.
- `artifacts/acceleration.json`: 8-candidate, 1,000-step parity and timing evidence.

### Important milestones
- Native BrainCell candidate batching preserves scalar outputs within 3.8147e-6 mV.
- Warm candidate batching is 7.55x faster than eight serial scalar rollouts on the recorded CPU environment.
- No scientific output, State ownership, unit, randomness, or objective semantics changed.
- Step 3 complete; proceed to step 4.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260824-celegans-smoke-01/`: mechanically complete target-device smoke snapshot, exit 0.
- `runs/20260824-celegans-production-01/RUN_SPEC.md`: immutable production contract.
- `runs/20260824-celegans-production-01/raw/fit_starts.json`: all three observed starts and candidate-batch histories.
- `runs/20260824-celegans-production-01/raw/fitted_parameters.json`: selected seed-2025 parameters and objective.
- `runs/20260824-celegans-production-01/raw/predictions.npz`: raw observed, predicted, and residual traces.
- `runs/20260824-celegans-production-01/raw/recovery.json`: three truths and nine exact-pipeline recovery fits.
- `runs/20260824-celegans-production-01/metrics/metrics.json`: control, mechanics, per-trace prediction, and recovery metrics.
- `runs/20260824-celegans-production-01/metrics/assessment.json`: deterministic claim boundary and evidence matrix.
- `runs/20260824-celegans-production-01/RESULT_ASSESSMENT.md`: readable deterministic assessment and proposed next action.

### Important milestones
- Production completed on CPU in 120.117 s with exit code 0 and finite parseable artifacts.
- Training Trace #8 matched 4/4 spikes, 3.7 ms first-spike error, 0.875 correlation, and 6.563 mV RMSE; passive RMSE was 17.179 mV.
- Held-out 20 and 30 pA traces passed count/latency thresholds, but the model predicted no spikes at 15 pA; the locked overall predictive result is not supported.
- Recovery classified `g_shk1`, `g_kr`, `g_na`, and capacitance as recoverable under the synthetic gate; EGL-19, SLO-2, and leak conductances were non-identifiable.
- Step 4 complete; proceed to step 5 without visualization or post-result tuning.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `reviews/iteration-1-blocked.md`: availability record for two failed fresh Codex MCP review calls.
- `runs/20260824-celegans-production-01/`: complete iteration evidence remains preserved.

### Important milestones
- Two fresh read-only `mcp__codex__codex` calls returned `user cancelled MCP tool call` immediately.
- Neither call returned a thread ID or Markdown report, so no PASS/REFUSE verdict exists.
- Step 5 is blocked by the configured Codex MCP service. No self-review was substituted, the iteration was not revised post-result, and step 6 visualization was not started.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `reviews/iteration-1.md`: authoritative verbatim Codex review report; outcome `REFUSE`.
- Reviewer thread ID: `01a03240-bfc4-7962-83aa-3ee6ba3ec5aa`.

### Important milestones
- This correction checkpoint supersedes the earlier iteration-1 step-5 MCP-blocked checkpoint: a delayed reviewer report exists and is authoritative.
- The reviewer classified the scientific outcome as inconclusive and loss closure as open, with seven findings: `FIT-001`, `CONTRACT-001`, `FIT-002`, `FIT-003`, `NUM-001`, `VALID-001`, and `API-001`.
- The researcher explicitly authorized run1 to continue iterating, overriding the earlier baseline-only stopping condition.
- Iteration 1 is refused; increment to iteration 2 and return to step 2 with parameter-fitting coverage active. Preserve the locked data split, model, objective, bounds, seeds, and training-only selection rule while applying the smallest sufficient corrections.

## Checkpoint
- Iteration: 2
- Step: 2

### Artifacts
- `NeuroSpecification.md`: exact existing parameter map, bounds, objective weights, estimator, stopping rule, and selection rule are now explicit and researcher-authorized for the correction run.
- `reviews/iteration-1.md`: authoritative correction requirements for iteration 2.

### Important milestones
- Iteration 2 step 2 started with parameter-fitting coverage active.
- No held-out trace may affect fitting, optimizer selection, convergence decisions, or candidate selection.

## Checkpoint
- Iteration: 2
- Step: 2

### Artifacts
- `cellegans_hh/model.py`: BrainCell rollout now optionally returns voltage, seven gate trajectories, and intracellular calcium State without changing model dynamics.
- `cellegans_hh/inference.py`: predeclared plateau stopping, initial/component-wise/cumulative-best histories, exact candidate evaluation counts, and explicit pre-stimulus spike windows.
- `scripts/run_experiment.py`: per-start predictions, numerical refinement, full-State checks, diagnostic recovery observations/tradeoffs, and conservative parameter claim gating.
- `tests/test_model.py`: focused full-State and pre-stimulus detector checks added.
- `artifacts/test-results.txt`: seven focused checks pass.
- `addressed-findings-iteration-2.md`: exact mapping from all seven iteration-1 findings to corrections.

### Important milestones
- The scientific model, fixed kinetics, data split, objective, bounds, seeds, and training-only candidate selection remain unchanged.
- Parameter-fitting coverage remains active; no production experiment was run during implementation.
- Step 2 complete; proceed to step 3 acceleration and parity.

## Checkpoint
- Iteration: 2
- Step: 3

### Artifacts
- `acceleration-audit-iteration-2.md`: current hot-path inventory, unchanged decision, validation, and remaining risk.
- `scripts/benchmark_acceleration.py`: full voltage/gate/calcium scalar-versus-native-batch parity and cold/warm benchmark.
- `artifacts/acceleration-iteration-2.json`: 8-candidate, 1,000-step State parity and timing evidence.
- `scripts/check_optimizer_boundary.py`: no-install unit-map and 28-candidate objective-boundary check.
- `artifacts/optimizer-boundary-iteration-2.json`: exact SciPy/direct loss parity and explicit missing-Nevergrad limitation.

### Important milestones
- Native BrainCell batching preserves voltage within 3.814697e-6 mV and every recorded gate/calcium State within 8.940697e-8 absolute error.
- Warm native batching is 8.26x faster than eight scalar rollouts on CPU.
- The valid SciPy production boundary returns one finite loss per candidate with zero direct-objective discrepancy.
- `braintools.optim.NevergradOptimizer` cannot execute because the optional `nevergrad` dependency is absent; researcher prohibited dependency installation, so the limitation is preserved and no substitute result is claimed.
- No acceleration change altered model dynamics, parameter order/units, objective, reset behavior, recovery semantics, or selection.
- Step 3 complete; proceed to step 4 immutable experiment execution.

## Checkpoint
- Iteration: 2
- Step: 4

### Artifacts
- `runs/20260824-celegans-smoke-02/`: mechanically complete target-device smoke snapshot, exit 0; strict JSON and finite NPZ checks passed.
- `runs/20260824-celegans-production-02/RUN_SPEC.md`: immutable iteration-2 production contract.
- `runs/20260824-celegans-production-02/raw/fit_starts.json`: all observed starts with initial/component/cumulative-best histories and closure evidence.
- `runs/20260824-celegans-production-02/raw/fitted_parameters.json`: unchanged training-only selection, seed 2025, objective 6.664992.
- `runs/20260824-celegans-production-02/raw/per_start_predictions.npz`: raw predictions for all starts and protocols.
- `runs/20260824-celegans-production-02/raw/objective_profiles.json`: one-dimensional component-wise profiles around the selected fit.
- `runs/20260824-celegans-production-02/raw/recovery.json`: 16 truths and all 48 exact-pipeline recovery starts, including failures and truth objectives.
- `runs/20260824-celegans-production-02/raw/recovery_observations.npz`: all latent and noisy recovery observations.
- `runs/20260824-celegans-production-02/metrics/metrics.json`: control, full-State, per-trace, per-start, closure, refinement, and recovery summaries.
- `runs/20260824-celegans-production-02/metrics/numerical_refinement.json`: frozen-parameter RK4 0.1/0.05 ms comparison.
- `runs/20260824-celegans-production-02/metrics/recovery_tradeoffs.json`: paired-error correlations, boundary rates, and 56.25% fit-failure rate.
- `runs/20260824-celegans-production-02/RESULT_ASSESSMENT.md`: deterministic result and claim-evidence matrix.
- `runs/20260824-celegans-production-02/artifact_manifest.json`: 20-entry verified manifest with no hash or size errors.

### Important milestones
- Production completed on CPU in 1,964.147 s with exit code 0; 15 JSON files parse strictly and all 59 NPZ arrays are finite.
- All three observed starts reached loss closure. The selected objective improves by 0.302275 from iteration 1.
- All three start vectors independently pass the locked held-out criteria and predict 3 spikes at 15 pA; the selected `[3, 4, 4, 5]` count and `[110.9, 94.8, 85.8, 80.9]` ms latency trends are monotone.
- RK4 refinement preserves spike counts and first-spike times exactly at reporting resolution; full nominal/lower/upper voltage, gate, and calcium State checks pass with zero pre-stimulus spikes.
- Recovery has 27/48 starts without closure and all seven parameter interpretations remain withheld as `non-identifiable-under-this-protocol`.
- The deterministic predictive result is `supported-under-tested-protocol`, limited to the tested recording series and 15-30 pA protocol.
- Step 4 complete; proceed to step 5 in reviewer thread `01a03240-bfc4-7962-83aa-3ee6ba3ec5aa` without visualization.

## Checkpoint
- Iteration: 2
- Step: 5

### Artifacts
- `reviews/iteration-2.md`: verbatim Codex review report; outcome `PASS`.
- Requested reviewer thread ID: `01a03240-bfc4-7962-83aa-3ee6ba3ec5aa`; `codex-reply` returned `Session not found for thread_id`.
- Completed fallback reviewer thread ID: `01a032d6-3801-75c2-a9ea-02db3b0c71ce`.

### Important milestones
- An initial fresh fallback review timed out at the MCP host boundary after 1,800 seconds without returning a report. A bounded fresh retry with the same authoritative prior review and addressed-findings context returned the preserved report.
- Reviewer outcome is `PASS`; scientific outcome `SUPPORTED`, loss closure `CLOSED`, and optimization adequacy `SUFFICIENT`.
- Accepted scope: prediction under the tested 15-30 pA protocol and supplied recording series. All fitted parameter interpretations remain withheld.
- Minor `FIT-004` remains: synthetic recovery reused the real Trace #8 initial voltage rather than recomputing each noisy synthetic pre-stimulus mean. The reviewer explicitly judged this non-blocking for the predictive result; recovery must be described as approximate diagnostic evidence.
- Iteration 2 is accepted and the loop position advances to step 6. The researcher explicitly prohibited visualization, so step 6 was not started.
