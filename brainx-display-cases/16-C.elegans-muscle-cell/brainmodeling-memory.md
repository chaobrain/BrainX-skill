# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: locked resume specification and inspected data contract.
- `Fig4A-D.txt`: read-only raw experimental data, verified present.

### Important milestones
- Resume entry selected because implementation, tests, and result artifacts predated loop memory.
- Trace #9 (30 pA) is the only fitting observation; Traces #6-#8 are immutable held-out tests.
- Existing result inspection occurred before this reconstructed contract, so the acceptance boundary is not prospective.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: BrainX API, publication, and implementation design study.
- `celegans_muscle_inference.py`: pre-existing implementation inspected; revision pending.
- `test_celegans_muscle_inference.py`: pre-existing focused tests inspected; expansion pending.

### Important milestones
- Selected scale: single-compartment cellular biophysics, owned by BrainCell; BrainUnit owns physical quantities and BrainState owns State-aware time execution.
- Active optional coverage: fitting.
- Implementation corrections required before step 2 can complete: exact parameter naming, documented source variant, paper-bounded SLO-2 prior, recovery/sensitivity evidence, and stronger mechanics checks.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `celegans_muscle_inference.py`: revised six-current model, source-bounded ABC fitting, recovery, controls, solver parity, metrics, and artifact writer.
- `test_celegans_muscle_inference.py`: six focused tests for data split, current/state inventory, units, bounds, finite rollout, and candidate independence.
- `/tmp/celegans-muscle-smoke`: reduced 64-candidate, one-round smoke artifacts; mechanically complete and not used for scientific claims.

### Important milestones
- Expanded test suite passed: 6 tests in 15.160 s on CPU.
- Reduced smoke inference completed with finite outputs, quiet no-current control, parseable report/CSV/PNG artifacts, and matched 0.1/0.05 ms protocol spike counts.
- The lowest-discrepancy ABC sample is now named `best_fit`, not MAP.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: hot-path inventory, unchanged decision, deterministic benchmark, smoke parity, and remaining risks.

### Important milestones
- Existing native candidate batching plus `jit(for_loop)` is the accepted acceleration path.
- A repeated 64-lane rollout was deterministic (0 mV maximum difference); first/repeated times were 6.210/4.210 s on CPU.
- No State-axis or multi-device rewrite was justified before production.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260824T115614+0800-production-seed2025/RUN_SPEC.md`: immutable production contract.
- `runs/20260824T115614+0800-production-seed2025/run.log`: completed CPU log, exit code 0.
- `runs/20260824T115614+0800-production-seed2025/raw/report.json`: full metrics, controls, posterior diagnostics, parity, recovery, and assessment.
- `runs/20260824T115614+0800-production-seed2025/raw/posterior_samples.csv`: 128 retained ABC samples.
- `runs/20260824T115614+0800-production-seed2025/raw/recovery_results.csv`: three exact-budget recovery cases.
- `runs/20260824T115614+0800-production-seed2025/raw/trace_predictions.csv`: aligned observed and predicted traces.
- `runs/20260824T115614+0800-production-seed2025/raw/held_out_validation.png`: four-panel comparison, visually inspected.
- `runs/20260824T115614+0800-production-seed2025/artifact-manifest.md`: hashes and mechanical checks.
- `result-assessment.md`: deterministic assessment and claim-evidence matrix.

### Important milestones
- Production completed from 11:58:32 to 12:06:10 +08:00 with exit code 0; all required artifacts are finite and parseable.
- Held-out stimulus spike counts and ISI direction agree; waveform criteria fail on all three held-out traces.
- Best-fit SHK-1 and leak conductances hit upper bounds; SLO-2 recovery is poor; mechanistic parameter-identification claims are withheld.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `result-assessment.md`: local deterministic assessment prepared for review.
- `runs/20260824T115614+0800-production-seed2025/`: complete review evidence bundle.

### Important milestones
- Step 5 is blocked because the configured BrainX Codex MCP review tool is unavailable in this session.
- No independent `PASS` or `REFUSE` verdict exists; local checks are not substituted for the required reviewer.
- Production artifacts and the unfavorable waveform/recovery evidence are preserved unchanged.
