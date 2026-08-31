# Run specification

- Run ID: `20260830T162520+0800-smoke-brunel-fresh`
- Level: smoke
- Entry case: new
- Project root: `/home/yixinliu/gitcode/BrainX-skill`
- Working directory: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/17-sparse-EI-network-states`
- Entry point: `sparse_ei_network.py`
- Git HEAD: `01631e950fa654285483a79e278ae2e0a23324fb`
- Code identity: dirty case-17 diff preserved in `code.diff`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend/device: CPU / `cpu:0`
- BrainState precision: 32
- Seed: 17729
- Scientific reduction: 800 E, 200 I, exact fan-ins 80/20, external indegree 80, 100 ms burn-in, 1,000 ms analysis; all four paper parameter points unchanged.
- Determinism expectation: exact replay after State/RNG reset on the same environment.
- Parent/checkpoint: none
- Expected output root: `results/`
- Expected artifacts: config, graph hashes, four parseable raw NPZ files, four finite metric rows, provenance, assessment, and artifact manifest.
- Resource estimate: less than 1 GiB result memory; CPU execution expected within 30 minutes.
- Stop conditions: non-finite required metric, malformed/missing artifact, process error, memory exhaustion, or 30 minutes without progress.
- Retry budget: zero unchanged retries for deterministic failure.

## Exact command

See `command.txt`. The snapshot files are immutable after launch; only `status.json`, `run.log`, and `exit_code` may change.
