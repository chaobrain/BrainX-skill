# Run specification

- Run ID: `20260830T142816+0800-validation-continuation-brunel`
- Level: production validation continuation
- Entry case: manifest-verified continuation from a complete source
- Parent run: `20260829T231314+0800-continuation-brunel-missing7`
- Parent status: done, exit 0, 20 finite metric rows and 20 raw NPZ artifacts
- Parent artifact-manifest SHA-256: `ccf7b092c5e6d7ee322e5104b14b68bb673239662798d93f416c0969dfb4cc4e`
- Remaining simulation work: none; verify and inherit all complete condition artifacts, then rebuild deterministic graph hashes, aggregates, assessment, manifest, and in-process provenance
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/19-brunel-lif-regimes`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend/device: JAX CPU, `CpuDevice(id=0)`
- BrainState platform/precision: CPU / 32 bit
- Scientific config: byte-equivalent values to the completed production contract
- Specification SHA-256: `a65381e684812fab05f7a379df67733a82883a6d516982b862e3f97852082dcb`
- Implementation SHA-256: `37028f8da7ecbf3f0841045064a06942318492e55002d8ba439a0bdf1c939673`
- Tests: `10 passed in 23.47s`; actual completed source passed 20 row/file gates
- Expected output: one new `results/` with 20 byte-identical raw files, 20 metric rows, five graph hashes, robustness assessment, corrected process-captured provenance, source-manifest link, and final manifest
- Runtime estimate: below 10 minutes; no stateful production rollout is required
- Timeout: 30 minutes
- Retry budget: zero unchanged retries
- Stop conditions: source-manifest mismatch, locked-contract mismatch, non-finite metric, raw-array mismatch, incorrect probe identity, nonzero exit, backend other than CPU, or missing final artifact

This snapshot is immutable after launch. Mechanical completion does not imply scientific acceptance.
