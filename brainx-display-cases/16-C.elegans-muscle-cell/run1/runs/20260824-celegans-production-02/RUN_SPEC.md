# Run specification

- Run ID: `20260824-celegans-production-02`
- Level / entry: production / new iteration-2 correction run
- Locked specification: `NeuroSpecification.md`, SHA-256 `c0c737437a1ba89eb6adc6c2f9184218e77a2c47b30e9e25a433e6ddb31169ae`.
- Scientific change from iteration 1: derivative-free evaluation budget only, plus evidence collection. Model, split, objective, bounds, seeds, selection rule, solver, dt, and observation path are unchanged.
- Working directory: `/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.ny74AR/workspace`
- Interpreter: `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`
- Backend/device: CPU / `cpu:0`; `JAX_PLATFORMS=cpu`.
- Precision: installed JAX default; model quantities preserve declared BrainUnit units.
- Optimizer: SciPy differential evolution, population 28, max 100 generations, 20-generation cumulative-best plateau <= 0.01, seeds 2025/2026/2027, training-only lowest-loss selection.
- Recovery: 16 Latin-hypercube truths, truth seed 8417, noise seeds 9511-9526, same three starts and exact fitting pipeline; all parameter interpretation withheld.
- Input: read-only `Fig4A-D.txt`, SHA-256 `7f6ff6a74b17e79a7d1122dc85ea4e00daf3f30a7b07d6bc2b7f60e8a4cd7df7`.
- Command: recorded in `command.txt`.
- Expected artifacts: every fit start and component history; fitted selection and profiles; selected/per-start raw predictions; passive control; full-State and pre-stimulus checks; 0.05 ms refinement; 16 recovery truths with latent/noisy observations, truth objectives, failures and tradeoffs; metrics, assessment, provenance, manifest, log, exit, status.
- Estimate / timeout: 8-30 minutes / 45 minutes.
- Stop conditions: backend mismatch, non-finite output, unhandled exception, or 45-minute timeout. Finite unfavorable scientific results do not stop execution.
- Retry budget: zero; deterministic failure returns to implementation under a new iteration/run ID.
- Smoke gate: `runs/20260824-celegans-smoke-02`, exit 0, strict JSON parsing and finite NPZ checks passed.
