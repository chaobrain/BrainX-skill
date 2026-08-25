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
