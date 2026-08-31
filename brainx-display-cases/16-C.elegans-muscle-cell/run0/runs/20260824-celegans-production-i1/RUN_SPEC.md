# Run specification

- Run ID: `20260824-celegans-production-i1`
- Level: production
- Entry case: new
- Project root: `/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-celegans-baseline.URaiSD/workspace`
- Interpreter: `python` from the active BrainX environment
- Backend: CPU, one JAX `CpuDevice(id=0)`
- Precision: package default float32
- Randomness: deterministic model and optimizer; no RNG is used
- Checkpoint source: none
- Retry budget: zero for deterministic failures
- Timeout: 10 minutes

## Scientific contract

- Specification SHA-256: `0b7fdd75c16aa3963504c3533a68d36d4b775cd1c668e8e9669696324cd18207`
- Study record SHA-256: `cc16c65825c6d3dfdb1eec1c07459aac3d42c7fa83e5c6bb5f7d5cfda32ac190`
- Model SHA-256 before launch: `de3f3bc31bdf42c2823ebc9769f12b113a47134c96f9726c37254abdcd2b78e2`
- Runner SHA-256 before launch: `5b165d49cb4541c834a606cc5e15a0f37e08f6c27b48376fa51949a5b064b889`
- Test SHA-256: `afdc9d51b3c89856d5689540e975db84c04e5a941d02fd3e8611cd0b50e93440`
- Data SHA-256: `7f6ff6a74b17e79a7d1122dc85ea4e00daf3f30a7b07d6bc2b7f60e8a4cd7df7`
- Fit trace: Trace 8, 25 pA
- Held-out traces: Trace 6 at 15 pA, Trace 7 at 20 pA, Trace 9 at 30 pA
- Protocol: 50-250 ms current step, 500 ms total, 0.05 ms dt
- Solver: `ind_exp_euler`
- Estimator: three-start bounded BrainTools SciPy Nelder-Mead

## Expected artifacts

- `run.log`, `exit_code`, `status.json`
- `run_config.json`, `manifest.json`
- `fit_starts.csv`, `fitted_parameters.json`
- `predictions.npz`, `metrics.csv`
- `recovery.json`, `assessment.json`

Stop on non-finite simulation, data-contract failure, overwrite risk, timeout, or nonzero exit. Completion is mechanical evidence only.
