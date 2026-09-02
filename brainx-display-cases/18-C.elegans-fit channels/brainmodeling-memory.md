# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: Locked parameter-fitting specification based on inspected Igor wave metadata and the researcher request.

### Important milestones
- Selected the `fresh-new` entry case.
- Locked SHK-1 to processed WT-minus-`shk-1(lf)` waves and EGL-19 to pharmacologically isolated WT calcium currents.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: BrainX package, fitting, lifecycle, and implementation design record.

### Important milestones
- Selected BrainCell as the only biological-scale owner, with BrainUnit and BrainState support.
- Activated parameter-fitting coverage for steps 2-5.
- Completed the pre-implementation study and set the current position to step 2.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `fit_channels.py`: Igor extraction, bounded HH fitting, BrainCell channel classes, numerical artifact generation, and requested figures.
- `test_fit_channels.py`: Protocol, formula, lifecycle, unit, reversal, shape, and finite-output checks.
- `requirements.txt`: Igor packed-experiment reader dependency.

### Important milestones
- Four focused tests passed in 32.922 seconds in the `braincell-released` environment.
- Implemented SHK-1 as one `n^4` gate and EGL-19 as `m^2 h`, with fixed-ion ownership and BrainUnit quantities.
- Set the current position to step 3.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Hot-path inventory, unchanged acceleration decision, and parity evidence.

### Important milestones
- Retained the vectorized NumPy observation model at the explicit SciPy boundary; no stateful transform rewrite is scientifically or computationally warranted.
- Set the current position to step 4.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `fit_channels.py`: Corrected SHK-1 time-constant parameterization, identifiable EGL-19 activation-point extraction, and nominal exact-pipeline recovery.
- `runs/20260825T115108+0800-production-seed20260825/`: Preserved first production evidence with rejected gating summaries.

### Important milestones
- This checkpoint supersedes the earlier iteration-1 step-2 implementation record for gating-summary mechanics only.
- Four tests passed in 33.616 seconds after correction; nominal recovery reproduced SHK-1 traces at 7.37 pA RMSE and EGL-19 exactly.
- Set the current position to step 3.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Unchanged acceleration decision remains valid for the corrected implementation.

### Important milestones
- Set the current position to step 4 for a new frozen production run.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260825T120001+0800-production-seed20260825/`: Completed replacement production snapshot, logs, raw numerical results, metrics, provenance, and figures.
- `result-assessment.md`: Deterministic assessment and claim-evidence matrix.
- `README.md`: Fitted equations, parameters, usage, and limitations.

### Important milestones
- Production completed with exit code 0 in 58.89 seconds; all required artifacts parse and all numerical arrays are finite.
- SHK-1 aggregate RMSE is 88.08 pA and EGL-19 aggregate RMSE is 9.75 pA.
- Parameter-level biological interpretation remains withheld; current and activation predictions are limited to the measured protocols.
- Set the current position to step 5.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `runs/20260825T120001+0800-production-seed20260825/`: Complete evidence package ready for Codex review.

### Important milestones
- The configured BrainX Codex MCP review tool is unavailable in this execution environment, so the mandatory external review could not be started.
- Step 5 is blocked; requested figures are preserved because the researcher explicitly required them, but they are not labeled as externally review-accepted evidence.

## Checkpoint
- Iteration: 2
- Step: 0

### Artifacts
- `NeuroSpecification.md`: Re-locked first-principles, supplied-data-only fitting contract after auditing packed wave identities.

### Important milestones
- This checkpoint supersedes iteration-1 step 0 because the prior contract used external scientific assumptions and mislabeled the EGL-19 `n582` wave family as WT.
- Locked direct WT-minus-`shk-1(lf)` potassium preprocessing, correctly mapped WT calcium traces, data-selected gate powers, and fitted reversal potentials.

## Checkpoint
- Iteration: 2
- Step: 1

### Artifacts
- `brainx-study-record.md`: Rewritten BrainCell/BrainUnit/BrainState design using only packed-data scientific evidence.

### Important milestones
- Completed the corrected pre-implementation study with parameter-fitting coverage active.
- Set the current position to step 2.

## Checkpoint
- Iteration: 2
- Step: 2

### Artifacts
- `fit_channels.py`: Corrected data-only extraction, data-selected HH structures, fitted reversals, multi-start diagnostics, controls, and figures.
- `test_fit_channels.py`: Five protocol, mapping, control, formula, lifecycle, unit, reversal, shape, and finite-output checks.

### Important milestones
- Five tests passed in 43.586 seconds in `braincell-released` on CPU.
- Corrected the prior calcium wave-family error and removed literature-fixed scientific parameters.
- Set the current position to step 3.

## Checkpoint
- Iteration: 2
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Host-boundary hot-path inventory and unchanged acceleration decision.

### Important milestones
- Retained vectorized NumPy candidate evaluation at the SciPy boundary; no BrainState transform rewrite is warranted.
- Set the current position to step 4.

## Checkpoint
- Iteration: 2
- Step: 4

### Artifacts
- `runs/20260902T095151+0800-production-seed20260902/`: Mechanically complete run preserved with deterministic scientific rejection.

### Important milestones
- The EGL `m^4 h` fit inverted the intended inactivation asymptotes and hit multiple bounds; local inactivation reduced residual by only 1.4% while adding two parameters per trace.
- Incremented to iteration 3 and returned to step 2 without submitting invalid evidence for review.

## Checkpoint
- Iteration: 3
- Step: 2

### Artifacts
- `fit_channels.py`: Simplified EGL-19 to the data-supported activation-only `m^4` HH model.
- `test_fit_channels.py`: Updated lifecycle and formula checks for the minimal EGL gate set.

### Important milestones
- Five tests passed in 12.179 seconds on CPU.
- Preserved the rejected `m^4 h` score as a model-selection control instead of retaining its weakly supported parameters.
- Set the current position to step 3.

## Checkpoint
- Iteration: 3
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Unchanged host-boundary decision remains applicable to the simplified model.

### Important milestones
- Shape-stable vectorized candidate evaluation and the SciPy boundary are unchanged.
- Set the current position to step 4.

## Checkpoint
- Iteration: 3
- Step: 4

### Artifacts
- `runs/20260902T100508+0800-production-seed20260902/`: Mechanically complete run preserved with deterministic scientific rejection.

### Important milestones
- The SHK time-constant logistic midpoint reached its off-range lower bound, leaving its amplitude and midpoint weakly identified.
- Incremented to iteration 4 and returned to step 2 before review.

## Checkpoint
- Iteration: 4
- Step: 2

### Artifacts
- `fit_channels.py`: Replaced the SHK logistic time constant with the identifiable three-parameter monotone exponential function.
- `test_fit_channels.py`: Five checks pass against the simplified SHK parameter map.

### Important milestones
- Five tests passed in 12.585 seconds on CPU.
- Set the current position to step 3.

## Checkpoint
- Iteration: 4
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Unchanged host-boundary decision remains applicable.

### Important milestones
- The simplified parameter map preserves the same vectorized shapes and deterministic SciPy boundary.
- Set the current position to step 4.

## Checkpoint
- Iteration: 4
- Step: 4

### Artifacts
- `runs/20260902T101333+0800-production-seed20260902/`: Completed immutable production snapshot with raw arrays, report, metrics, provenance, manifest, and four figures.
- `result-assessment.md`: Deterministic outcome and claim-evidence matrix.
- `README.md`: Data-only fitted equations, parameters, usage, and limitations.

### Important milestones
- Production completed with exit code 0 in 18.153 seconds; all 14 arrays are finite and all four figures are nonblank and visually inspected.
- SHK-1 aggregate RMSE is 80.31 pA and EGL-19 aggregate RMSE is 21.55 pA.
- Set the current position to step 5.

## Checkpoint
- Iteration: 4
- Step: 5

### Artifacts
- `codex-review-iteration-4.md`: External Codex refusal and blocking findings.

### Important milestones
- Review refused the analytic/raw-SciPy production path and insufficient identifiability evidence.
- Incremented to iteration 5 and returned to step 1.

## Checkpoint
- Iteration: 5
- Step: 1

### Artifacts
- `brainx-study-record.md`: BrainCell-native objective, BrainTools boundary, metadata-verification, recovery-domain, and holdout design.
- `braintools-api-boundary.md`: Explicit BrainTools capability audit and smallest custom boundaries.

### Important milestones
- Re-studied the routed BrainCell, BrainState, BrainUnit, BrainTools optimizer, and BrainTools metric APIs.
- Locked a 0.5 ms optimization grid with full-resolution evaluation and figures.
- Set the current position to step 2.

## Checkpoint
- Iteration: 5
- Step: 2

### Artifacts
- `fit_channels.py`: BrainTools-only fitting and metrics, BrainCell production rollouts, packed-metadata enforcement, complete diagnostics, leave-one-voltage-out validation, and noisy domain recovery.
- `test_fit_channels.py`: Six protocol, mapping, lifecycle/reset, units, physical-function, finite-output, and analytic-versus-BrainCell parity tests.

### Important milestones
- Six tests passed in 19.697 seconds on CPU.
- Archived unclipped SHK gate estimates and qualified model-conditioned EGL gate points.
- Set the current position to step 3.

## Checkpoint
- Iteration: 5
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: State-aware objective, deterministic fitting-grid, and parity audit.

### Important milestones
- Retained compiled BrainCell objectives and vectorized batch/time evaluation.
- Set the current position to step 4.

## Checkpoint
- Iteration: 5
- Step: 4

### Artifacts
- `runs/20260902T115336+0800-production-seed20260902/`: Immutable production candidate with full traces, figures, optimizer histories, metadata evidence, recovery-domain results, and voltage holdouts.
- `result-assessment.md`: Claim-evidence matrix including negative holdout and identifiability results.
- `README.md`: Final equations, parameters, reproduction command, and limitations.

### Important milestones
- Production completed with exit code 0 in 253.153 seconds; all 15 arrays are finite and all four requested figures are nonblank and visually inspected.
- SHK-1 aggregate RMSE is 80.56 pA and EGL-19 aggregate RMSE is 20.86 pA.
- Set the current position to step 5.

## Checkpoint
- Iteration: 5
- Step: 5

### Artifacts
- `codex-review-iteration-5.md`: External Codex refusal and direct-bound remediation requirements.

### Important milestones
- Review refused hidden effective bounds, reduced validation optimization, failed gate-point evidence, and incomplete metadata continuation lines.
- Incremented to iteration 6 and returned to step 1.

## Checkpoint
- Iteration: 6
- Step: 1

### Artifacts
- `NeuroSpecification.md`: Declared data-scale physical fitting domain and identical observed/validation optimizer contract.
- `braintools-api-boundary.md`: Direct physical-bound BrainTools design with reconstructed sampled starts.

### Important milestones
- Removed the bounded sigmoid coordinate transform and passed physical bounds directly to BrainTools.
- Locked six optimizer seeds for every observed, recovery, and voltage-holdout fit.
- Set the current position to step 2.

## Checkpoint
- Iteration: 6
- Step: 2

### Artifacts
- `fit_channels.py`: Direct physical-bound fitting, complete metadata-wave contract, identical BrainCell validation pipelines, raw validation arrays, successful gate-point selection, and unclipped SHK plotting.
- `test_fit_channels.py`: Seven tests including complete consumed-wave sets, optimizer-domain checks, successful gate-point termination, lifecycle reset, and BrainCell parity.

### Important milestones
- Seven tests passed in 62.037 seconds on CPU.
- Every observed and validation fit retains six starts; all selected gate-point evidence terminates successfully.
- Set the current position to step 3.

## Checkpoint
- Iteration: 6
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Existing state-aware vectorization and parity decision remains applicable.

### Important milestones
- Kept the full BrainCell objective and deterministic 0.5 ms optimization grid.
- Set the current position to step 4.

## Checkpoint
- Iteration: 6
- Step: 4

### Artifacts
- `runs/20260902T124800+0800-production-seed20260902/`: Immutable direct-bound production candidate with 94 finite arrays, raw validation observations/predictions/residuals, complete optimizer records, and four inspected figures.
- `result-assessment.md`: Optimization-controlled recovery and holdout assessment.
- `README.md`: Final equations, parameters, evidence path, and claim limits.

### Important milestones
- Production completed with exit code 0 in 554.046 seconds.
- SHK-1 aggregate RMSE is 80.56 pA; EGL-19 aggregate RMSE is 20.87 pA.
- Controlled recovery is strong for most interior cases, while leave-one-voltage-out performance limits predictive claims.
- Set the current position to step 5.

## Checkpoint
- Iteration: 6
- Step: 5

### Artifacts
- `codex-review-iteration-6.md`: External Codex refusal and architecture-closure requirement.

### Important milestones
- Review confirmed every iteration-5 blocker was resolved.
- Review refused only the unfinished 600-iteration `m^4h` comparison and inconsistent historical metric wording.
- Incremented to iteration 7 and returned to step 1.

## Checkpoint
- Iteration: 7
- Step: 1

### Artifacts
- `NeuroSpecification.md`: Full-budget successful-termination and BIC architecture criterion.

### Important milestones
- Locked the same 1,200-iteration budget for global and `m^4h` comparisons.
- Required a successful `m^4h` candidate before computing its improvement.
- Set the current position to step 2.

## Checkpoint
- Iteration: 7
- Step: 2

### Artifacts
- `fit_channels.py`: Closed `m^4h` comparison, successful-candidate enforcement, RSS/BIC evidence, and explicit architecture status.

### Important milestones
- Three of six `m^4h` candidates terminated successfully.
- `m^4h` lowered the robust objective by 3.18%, but its 14 additional local parameters raised BIC by 33.60; selected `m^4`.
- Seven tests passed in 62.037 seconds on CPU before the production run.
- Set the current position to step 3.

## Checkpoint
- Iteration: 7
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Existing state-aware optimization and parity audit remains applicable.

### Important milestones
- No performance rewrite was required beyond the locked fitting grid.
- Set the current position to step 4.

## Checkpoint
- Iteration: 7
- Step: 4

### Artifacts
- `runs/20260902T132147+0800-production-seed20260902/`: Immutable production candidate with closed architecture evidence and the complete iteration-6 validation suite.
- `result-assessment.md`: Consistent robust-objective and BIC architecture evidence.
- `README.md`: Final evidence path and selected equations.

### Important milestones
- Production completed with exit code 0 in 629.743 seconds; all 94 arrays are finite.
- SHK-1 aggregate RMSE is 80.56 pA; EGL-19 aggregate RMSE is 20.87 pA.
- Set the current position to step 5.

## Checkpoint
- Iteration: 7
- Step: 5

### Artifacts
- `codex-review-iteration-7.md`: External Codex acceptance with no findings.

### Important milestones
- Review passed scientific support, loss closure, and optimization adequacy.
- Reviewer independently confirmed all 150 production/validation starts, 94 finite arrays, 23 raw residual triplets, architecture BIC, BrainCell lifecycle, packed metadata, and figures.
- Set the current position to step 6.

## Checkpoint
- Iteration: 7
- Step: 6

### Artifacts
- `runs/20260902T132147+0800-production-seed20260902/raw/shk1_current_fits.png`: Accepted SHK-1 current-time overlays.
- `runs/20260902T132147+0800-production-seed20260902/raw/shk1_gating.png`: Accepted SHK-1 activation and time-constant evidence with unclipped model-conditioned point.
- `runs/20260902T132147+0800-production-seed20260902/raw/egl19_current_fits.png`: Accepted WT calcium current-time overlays.
- `runs/20260902T132147+0800-production-seed20260902/raw/egl19_gating.png`: Accepted EGL-19 activation and time-constant evidence with model-conditioned labels.

### Important milestones
- All four requested figures were visually inspected and accepted without overlap, clipping, blank panels, or label errors.
- Completed iteration 7 within the accepted claim boundary.
