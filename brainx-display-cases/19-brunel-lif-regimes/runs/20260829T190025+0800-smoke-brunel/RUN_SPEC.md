# Run specification

- Run ID: `20260829T190025+0800-smoke-brunel`
- Level: smoke
- Entry case: new
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/19-brunel-lif-regimes`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Entry point: `brunel_lif_regimes.py --smoke`
- Backend/device: JAX CPU, `CpuDevice(id=0)`
- BrainState platform/precision: CPU / 32 bit
- Timestep: 0.1 ms
- Seed policy: repeat seed 1729 with fixed offsets for topology, initial voltage, and external RNG
- Specification SHA-256: `a65381e684812fab05f7a379df67733a82883a6d516982b862e3f97852082dcb`
- Code SHA-256: `02513d6ecf9309e5593ebc3bbe2289395213765dc13ecd0d3c696996970b7b46`
- Expected output: `results/` with four raw NPZ files, metrics JSON/CSV, robustness JSON, graph hashes, provenance, assessment, and artifact manifest
- Resource estimate: below 1 GiB output and working memory; below 2 minutes after environment import
- Retry budget: zero unchanged retries; deterministic failure returns to implementation
- Stop conditions: non-finite State/metrics, nonzero exit, missing condition artifact, backend other than CPU, or disk exhaustion

This snapshot is immutable after launch. Mechanical completion does not imply scientific acceptance.
