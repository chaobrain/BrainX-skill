# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: locked specification for a one-trace parameter fit and three held-out protocols.
- `Fig4A-D.txt`: read-only source data; 10,000 samples at 0.05 ms.

### Important milestones
- Fresh-new entry selected because no prior loop artifacts existed.
- Cellular-biophysics scale selected; BrainCell is the sole scale-owning package, supported by BrainUnit and BrainState.
- Trace 8 at 25 pA locked for fitting; traces 6, 7, and 9 locked for held-out evaluation.
- The prompt's seven-channel count conflicts with its six named currents; the six named currents are authoritative and the mismatch is an explicit limitation.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `BrainXStudyRecord.md`: selected API, lifecycle, channel mapping, fitting design, and validation evidence contract.
- `NeuroSpecification.md`: verified locked specification.

### Important milestones
- BrainCell `SingleCompartment` and custom HH channel route selected for the sole represented cellular scale.
- BrainUnit total-quantity convention selected because experimental cell area is absent.
- BrainState ParamState/reset/for-loop lifecycle selected for independent fitting candidates and protocols.
- Bounded BrainTools SciPy fitting selected; Nevergrad is unavailable and the voltage objective is smooth.
- Du et al. C. elegans-specific equations are unavailable locally and network access is disabled; the named current kinetics are explicitly classified as phenomenological HH surrogates.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `celegans_model.py`: six-current unit-bearing BrainCell model and BrainState rollout.
- `fit_and_validate.py`: ATF loading, one-trace bounded fitting, recovery, held-out metrics, and artifact writer.
- `tests/test_model.py`: six focused model/data/lifecycle tests.
- `artifacts/tests_iteration1.txt`: passing test record.

### Important milestones
- Native 0.05 ms, 10,000-step nominal simulations are finite and zero-current behavior is quiet.
- Candidate conductances are explicit ParamState quantities and runtime State reset replay is deterministic.
- Compiled nominal objective is finite at 251.12544 mV^2.
- Full-rollout reverse-mode L-BFGS-B exhausted the process; bounded BrainTools Nelder-Mead ran successfully and reduced the smoke-test loss to 95.08974 mV^2.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `artifacts/acceleration_iteration1.md`: hot-path inventory, cold/warm timing, parity, and unchanged decision.

### Important milestones
- The time axis is already owned by BrainState `for_loop`; the complete stateful loss is one stable BrainState JIT boundary.
- Cold objective time is 0.319962 s and median warm time is 0.003099 s.
- Compiled/direct nominal loss differs by 0.008455 mV^2 (3.37e-5 relative), within the locked numerical tolerance.
- No acceleration rewrite was justified; protocol batching and checkpointing would add complexity without benefiting the active derivative-free workload.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260824-celegans-production-i1/RUN_SPEC.md`: immutable production contract.
- `runs/20260824-celegans-production-i1/run_config.json`: executed run configuration and environment identity.
- `runs/20260824-celegans-production-i1/run.log`: final raw assessment output.
- `runs/20260824-celegans-production-i1/predictions.npz`: raw observed and predicted traces.
- `runs/20260824-celegans-production-i1/metrics.csv`: protocol-wise waveform and event metrics.
- `runs/20260824-celegans-production-i1/fit_starts.csv`: all optimizer starts, failures, and selected result.
- `runs/20260824-celegans-production-i1/fitted_parameters.json`: selected conductance values.
- `runs/20260824-celegans-production-i1/recovery.json`: limited exact-pipeline recovery evidence.
- `runs/20260824-celegans-production-i1/manifest.json`: run-produced file manifest.
- `artifacts/result_assessment_iteration1.md`: deterministic outcome and claim-evidence matrix.

### Important milestones
- Production run completed mechanically on CPU with exit code 0; all declared result files parse and predictions are finite.
- Training loss improved from 251.116987 to 82.014809 mV^2.
- Scientific acceptance failed: predicted 15/20/25/30 pA spike counts are 1/1/2/1 and dt-refinement RMSE is 4.544 mV.
- Unfavorable held-out and failed-start evidence was preserved; no post-fit tuning used held-out traces.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `artifacts/codex_review_iteration1.md`: two cancelled fresh-review calls and blocked outcome.
- `artifacts/result_assessment_iteration1.md`: preserved failed scientific assessment awaiting external review.

### Important milestones
- Two fresh configured Codex MCP calls were cancelled by the service before a thread or reviewer response was created.
- Step 5 is blocked; no PASS or REFUSE exists, and self-review was not substituted.
- Visualization is not entered because it requires a review PASS.
