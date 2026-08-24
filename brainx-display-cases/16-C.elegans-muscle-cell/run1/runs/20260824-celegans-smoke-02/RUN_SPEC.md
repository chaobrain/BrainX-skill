# Run specification

- Run ID: `20260824-celegans-smoke-02`
- Level / entry: smoke / new
- Purpose: mechanical validation of the iteration-2 runner, convergence callback, State/refinement/per-start/recovery artifact schema, and CPU target.
- Scientific use: none; reduced budget and recovery count are not review evidence.
- Working directory: `/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.ny74AR/workspace`
- Interpreter: `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`
- Backend/device: CPU / `cpu:0`; `JAX_PLATFORMS=cpu`
- Precision: installed JAX default; model quantities preserve declared BrainUnit units.
- Seeds: simulation 4041; optimizer 2025; recovery truth 8417; recovery noise 9511; passive 3101.
- Input: read-only `Fig4A-D.txt`, SHA-256 `7f6ff6a74b17e79a7d1122dc85ea4e00daf3f30a7b07d6bc2b7f60e8a4cd7df7`.
- Command: recorded in `command.txt`.
- Expected artifacts: fit starts/selection/profiles, selected and per-start predictions, passive control, full-State checks, dt refinement, one diagnostic recovery, metrics, assessment, provenance, manifest, log, exit code, status.
- Estimate / timeout: under 3 minutes / 10 minutes.
- Stop conditions: non-finite output, backend mismatch, unhandled exception, or timeout. No scientific early stop.
- Retry budget: zero; any failure returns to implementation.
