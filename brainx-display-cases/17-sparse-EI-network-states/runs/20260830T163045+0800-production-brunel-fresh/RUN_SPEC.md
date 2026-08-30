# Run specification

- Run ID: `20260830T163045+0800-production-brunel-fresh`
- Level: production
- Entry case: new
- Project root: `/home/yixinliu/gitcode/BrainX-skill`
- Working directory: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/17-sparse-EI-network-states`
- Entry point: `sparse_ei_network.py`
- Git HEAD: `01631e950fa654285483a79e278ae2e0a23324fb`
- Code identity: dirty case-17 scientific diff preserved in `code.diff`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend/device: CPU / `cpu:0`
- BrainState precision: 32
- Seed: 17729
- Scientific configuration: paper Model A with 10,000 E, 2,500 I, exact fan-ins 1,000/250, external indegree 1,000, `dt=0.1 ms`, 500 ms burn-in, 2,000 ms analysis, and all four Fig. 8 parameter points.
- Determinism expectation: exact replay after State/RNG reset on the same environment.
- Smoke gate: `20260830T162520+0800-smoke-brunel-fresh`, mechanically passed.
- Parent/checkpoint: none
- Expected output root: `results/`
- Expected artifacts: config, graph hashes, four parseable raw NPZ files, four finite metric rows, provenance, assessment, and artifact manifest.
- Resource estimate: about 62.5 MB topology indices and 312.5 MB Boolean spike history per sequential panel; expected peak below 2 GiB and CPU duration below 90 minutes.
- Stop conditions: non-finite required metric, malformed/missing artifact, process error, memory exhaustion, or 90 minutes without completing the run.
- Retry budget: zero unchanged retries for deterministic failure.

## Exact command

See `command.txt`. The snapshot files are immutable after launch; only `status.json`, `run.log`, and `exit_code` may change.
