# Run specification: 20260824-celegans-smoke-01

- Run level / entry: smoke / new
- Working directory: `/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.ny74AR/workspace`
- Backend / device: CPU / `cpu:0`
- BrainState step: 0.1 ms
- Precision: JAX default float32 simulation, float64 host scoring
- Data: `Fig4A-D.txt`, SHA-256 `7f6ff6a74b17e79a7d1122dc85ea4e00daf3f30a7b07d6bc2b7f60e8a4cd7df7`
- Split: fit Trace #8 (25 pA); evaluate Trace #6, #7, and #9 without fitting
- Seeds: simulation 1701; optimizer 2025; passive 3101
- Budget: one differential-evolution generation, population size 28; no recovery cases
- Determinism: deterministic simulator, seeded host optimizer
- Checkpoint: none required for this short smoke run
- Retry budget: zero for deterministic failure; one new linked run only for a transient host interruption
- Stop conditions: non-finite output, backend mismatch, data hash/shape failure, unparseable artifact, or process error
- Expected duration: under 60 seconds

## Immutable snapshot hashes

- `config.json`: `b512f15924bca34fd7031a6876648ce7c46e06b4f6424dbde1598954ea0307e6`
- `environment.json`: `814921c67e0786a1ce059f401d28f7619d0b27e2416def1fd2eba2e793c32ba7`
- `command.txt`: `f273d09cf496ba082e755bed032c9c0e26f0c57509901e3eeb046a0b50c4239e`
- `code.diff`: `65e76e90ef96223a6771d657b0f2e839be22ec09ccd9f289c42a48990799211b`
- Source identities: listed in `code.diff`

## Expected artifacts

- Append-only `run.log`, `exit_code`, and mutable `status.json`
- `raw/fit_starts.json`, `raw/fitted_parameters.json`, `raw/passive_fit.json`
- `raw/predictions.npz`, `raw/recovery.json`, `raw/run_provenance.json`
- `metrics/metrics.json`, `metrics/assessment.json`, and `artifact_manifest.json`

This snapshot tests mechanics only. Completion is not scientific acceptance.
