# Run specification

- Run ID: `20260829T190412+0800-smoke-brunel-v2`
- Parent failed smoke: `20260829T190025+0800-smoke-brunel`
- Level: smoke
- Entry case: new corrected witness, not an unchanged retry
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/19-brunel-lif-regimes`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Entry point: `brunel_lif_regimes.py --smoke`
- Backend/device: JAX CPU, `CpuDevice(id=0)`
- BrainState platform/precision: CPU / 32 bit
- Timestep: 0.1 ms
- Reduced witness: 80 E, 20 I, exact indegrees 8/2, 100 ms burn-in, 2,000 ms analysis
- Seed policy: repeat seed 1729 with fixed topology, initial-voltage, external-RNG, and probe offsets
- Specification SHA-256: `a65381e684812fab05f7a379df67733a82883a6d516982b862e3f97852082dcb`
- Code SHA-256: `2217e881ea9e41df61307eb1f500e509088c96a69fc6a66fc7a0ba87844ddeee`
- Expected output: `results/` with four finite raw NPZ files, metrics JSON/CSV, robustness JSON, graph hashes, provenance, assessment, and artifact manifest
- Resource estimate: below 1 GiB working/output memory; below 3 minutes
- Retry budget: zero unchanged retries
- Stop conditions: non-finite State/metrics, nonzero exit, missing condition artifact, backend other than CPU, or disk exhaustion

This snapshot is immutable after launch. Mechanical completion does not imply scientific acceptance.
