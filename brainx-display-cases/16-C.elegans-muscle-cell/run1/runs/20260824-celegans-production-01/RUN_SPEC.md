# Run specification: 20260824-celegans-production-01

- Run level / entry: production / new
- Parent readiness evidence: `runs/20260824-celegans-smoke-01` (`done`, exit 0, finite parseable artifacts)
- Working directory: `/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.ny74AR/workspace`
- Backend / device: CPU / `cpu:0`
- BrainState step: 0.1 ms
- Precision: JAX default float32 simulation, float64 host scoring
- Data: `Fig4A-D.txt`, SHA-256 `7f6ff6a74b17e79a7d1122dc85ea4e00daf3f30a7b07d6bc2b7f60e8a4cd7df7`
- Split: fit Trace #8 (25 pA); hold out Trace #6 (15 pA), #7 (20 pA), and #9 (30 pA)
- Seeds: simulation 1701; optimizer 2025, 2026, 2027; passive 3101; recovery truth 8101; recovery noise 9100+
- Estimator: differential evolution, 24 generations, population size 28, fixed-budget selection by lowest objective then seed
- Recovery: three truths drawn 20% inside the joint bounds; identical estimator/protocol/sampling; measured baseline noise added
- Interpretation gate: median normalized error <= 0.15 and maximum normalized error <= 0.30 per parameter
- Determinism: deterministic simulator, seeded optimizer/truth/noise generators
- Checkpoint: none required; each fit is under 15 seconds and completed starts are serialized at run completion
- Retry budget: zero for deterministic failure; one new linked run only for a transient host interruption
- Stop conditions: non-finite output, backend mismatch, data hash/shape failure, unparseable artifact, process error, or runtime over 300 seconds
- Expected duration: 90-180 seconds

## Immutable snapshot hashes

- `config.json`: `1a31c2aeeedbc217c88a34c4a339153f49a8a697861df19f912174872bee7d33`
- `environment.json`: `814921c67e0786a1ce059f401d28f7619d0b27e2416def1fd2eba2e793c32ba7`
- `command.txt`: `fb73d28bd3ce6f4a9ad4916bec98962be59b9f6324fdb454d3a8727cf742c6ec`
- `code.diff`: `4262619a9d027eceff5673cbdeefed3babdcbb6edf4deb97981181ee7c751d23`
- Source identities: listed in `code.diff`

## Expected artifacts

- Append-only `run.log`, `exit_code`, and mutable `status.json`
- Every observed and recovery optimizer start with candidate-batch histories
- Fitted and passive parameters, raw predictions/residuals, recovery truths/results, and provenance
- Per-trace metrics, locked predictive assessment, parameter recovery classifications, claim-evidence matrix
- SHA-256 artifact manifest excluding mutable control/log files

Completion means mechanically `done`; scientific acceptance requires the independent step-5 Codex review.
