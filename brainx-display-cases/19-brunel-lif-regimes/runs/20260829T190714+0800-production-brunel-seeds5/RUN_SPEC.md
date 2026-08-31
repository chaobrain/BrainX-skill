# Run specification

- Run ID: `20260829T190714+0800-production-brunel-seeds5`
- Passed smoke: `20260829T190412+0800-smoke-brunel-v2`
- Preserved failed smoke: `20260829T190025+0800-smoke-brunel`
- Level: production replication over five fixed seeds
- Entry case: new
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/19-brunel-lif-regimes`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Entry point: `brunel_lif_regimes.py --config <run>/config.json`
- Backend/device: JAX CPU, `CpuDevice(id=0)`
- BrainState platform/precision: CPU / 32 bit
- Network: 10,000 E plus 2,500 I; exact indegrees 1,000/250; external indegree 1,000
- Protocol: 0.1 ms timestep, 500 ms burn-in, 2,000 ms analysis, 1.5 ms exact delay
- Conditions: `(g,eta)=(3,2),(6,4),(5,2),(4.5,0.9)`
- Seeds: `[1729,2718,3141,5772,8119]`
- Matched policy: graph, initial voltage, and external seed restart are shared across conditions within each repeat; only `g` and `eta` vary
- Specification SHA-256: `a65381e684812fab05f7a379df67733a82883a6d516982b862e3f97852082dcb`
- Code SHA-256: `2217e881ea9e41df61307eb1f500e509088c96a69fc6a66fc7a0ba87844ddeee`
- Study SHA-256: `c49bf025dd4d472a76622b2072719bd17b09cf02af55517efd147c08f75b1093`
- Tests SHA-256: `d1751bd298039272261c47bbb021179b20f17203a18569f3e7163e899eb22a59`
- Expected output: `results/` with 20 raw NPZ files, 20 finite metric rows, five graph hashes, robustness assessment, provenance, and hashed manifest
- Memory estimate: 62.5 MB fixed-degree indices plus 312.5 MB full boolean spike history per active condition, with histories released sequentially; below 6 GiB expected working set
- Runtime estimate: five graph compilations and 20 CPU rollouts, expected below 3 hours
- Timeout: 4 hours
- Retry budget: zero unchanged retries
- Stop conditions: non-finite State/metrics, nonzero exit, missing repeat/condition artifact, backend other than CPU, memory exhaustion, disk exhaustion, or wall time beyond 4 hours

This snapshot is immutable after launch. Process completion is mechanical evidence, not result acceptance.
