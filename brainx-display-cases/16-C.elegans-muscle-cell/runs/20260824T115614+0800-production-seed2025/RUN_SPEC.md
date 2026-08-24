# Production run specification

- Run ID: `20260824T115614+0800-production-seed2025`
- Level: production
- Entry case: new
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/16-C.elegans-muscle-cell`
- Specification: `NeuroSpecification.md` (SHA-256 `89a5e6afb0db65687ccb3a4917ec22eb3af8e5d59ef4da7630595013ac62770f`)
- Entry point: `celegans_muscle_inference.py` (SHA-256 `be586c6f7e1de866ad8af846ff4b980b81c384718151a96f494a6a9adf75aa63`)
- Data: `Fig4A-D.txt` (SHA-256 `7f6ff6a74b17e79a7d1122dc85ea4e00daf3f30a7b07d6bc2b7f60e8a4cd7df7`)
- Git base: `8599500a6c2b0e7e81e0ab433bfe3a207c575ee8`; working tree is intentionally dirty and captured by file hashes/status in `code.diff`.
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python` (Python 3.11.15)
- Backend/device: JAX CPU, one `CpuDevice(id=0)` on Intel Core Ultra 7 155H under WSL2 x86_64.
- Precision: package default float32 simulation State; host summaries and serialized metrics use NumPy float64 where applicable.
- Seed: 2025 for BrainState, Latin-hypercube proposals, local ABC proposals, and deterministic recovery seed derivation.
- Scientific budget: 1,024 candidates/round, 3 ABC rounds, 3 exact-budget synthetic recovery cases.
- Protocol: 0.1 ms integration, 500 ms duration, 30 pA fitting step from 57.8 to 257.8 ms, held-out 15/20/25 pA tests.
- Expected outputs: `raw/report.json`, `raw/posterior_samples.csv`, `raw/recovery_results.csv`, `raw/trace_predictions.csv`, `raw/held_out_validation.png`, `run.log`, `exit_code`, and final `status.json`.
- Mechanical stop conditions: nonzero process exit, missing required artifact, unparsable JSON/CSV, non-finite selected simulation, or failed no-current/parity control execution.
- Scientific status: process completion is not review acceptance.

The `BrainX` meta-package import is absent. No installation was authorized or performed. The directly imported coherent working tuple is recorded in `environment.json` and passed the complete smoke path.
