# Run specification

- Run ID: `20260829T231314+0800-continuation-brunel-missing7`
- Level: production continuation
- Entry case: compatible condition-boundary continuation
- Parent run: `20260829T190714+0800-production-brunel-seeds5`
- Parent status: stopped at declared four-hour boundary, exit 130
- Parent checkpoint: 13 finite metric rows and 13 parseable raw NPZ artifacts through repeat 3 synchronous regular
- Parent partial-metrics SHA-256: `7ea1f89f021a522c659411f7203ab6145ccbcdc41df0806df6ffead5c2fb088e`
- Remaining work: repeat 3 fast-SI, AI, slow-SI; all four repeat 4 conditions
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/19-brunel-lif-regimes`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend/device: JAX CPU, `CpuDevice(id=0)`
- BrainState platform/precision: CPU / 32 bit
- Scientific config: byte-equivalent values to the parent production contract
- Specification SHA-256: `a65381e684812fab05f7a379df67733a82883a6d516982b862e3f97852082dcb`
- Continuation code SHA-256: `347114bc1bc2d132621847a8d657b47405651dbff436abd9dd116909adc9c413`
- Tests: `10 passed in 46.27s`; actual parent gate validated 13 rows/files and identified seven missing pairs
- Expected output: one new `results/` containing copied validated parent evidence plus seven newly executed conditions, totaling 20 raw files and 20 finite rows with final aggregate assessment and manifest
- Memory estimate: unchanged 62.5 MB indices plus 312.5 MB active spike history, below 6 GiB expected working set
- Runtime estimate: two needed graph compilations and seven CPU rollouts, below 3 hours
- Timeout: 4 hours
- Retry budget: zero unchanged retries
- Stop conditions: non-finite State/metrics, parent-contract mismatch, missing/corrupt parent raw file, duplicate condition, nonzero exit, missing final artifact, backend other than CPU, memory/disk exhaustion, or wall time beyond 4 hours

This snapshot is immutable after launch. Copied parent artifacts remain byte-identical; scientific completion still requires result review.
