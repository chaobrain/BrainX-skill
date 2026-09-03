# Run specification

- Run ID: `production-seed-47-cpu-20260903`
- Entry case: new
- Run level: production
- Working directory: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/20-current-lif-network-first-principles`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend and device: CPU; `['cpu:0']`
- BrainState environment: `dt = 0.1 ms`, `fit = false`, precision contract 32-bit
- Seeds: `[47]`
- Seed policy: deterministic derived connectivity, initialization/sample, and external streams; exact replay expected
- Connectivity: exact fixed fan-in, no autapses, shared across conditions within each seed
- Checkpoint source: none
- Retry budget: zero unchanged retries; deterministic failure returns to implementation
- Expected artifacts: `['config.json', 'environment.json', 'command.txt', 'code.diff', 'run.log', 'status.json', 'exit_code', 'connectivity_manifest.json', 'run_metrics.csv', 'condition_assessment.json']` plus one connectivity NPZ and per-condition raw/metric files
- Resource estimate: full-scale warm estimate 1,178.6 s per condition; production seeds run on disjoint CPU core sets; approximately 100-200 MB static arrays per process plus compiled runtime State
- Stop conditions: nonzero process exit, non-finite state/metrics, invalid connectivity, failed frozen-artifact verification, parse failure, disk exhaustion, or memory exhaustion
- Scientific acceptance: not decided by process completion; requires step-5 review

## Identity

- Git commit: `a3389aeabb748e2a4bd7e0f763790b3874bd8a40`
- Git status at freeze: `M "../18-C.elegans-fit channels/prompt.md"
 M ../19-brunel-lif-regimes/brainmodeling-memory.md
 M ../19-brunel-lif-regimes/prompt.md
?? ../19-brunel-lif-regimes/FIGURE_CONTRACT.md
?? ../19-brunel-lif-regimes/figures/
?? ../19-brunel-lif-regimes/visualize_brunel_results.py
?? ./`
- Locked specification SHA-256: `01c3bd8eca0c6810c047dfe4e1a08af07605783c131f6b5d269c8cdde0df5d48`
- Active config SHA-256: `b9c0eb2aea92fb8b1b45c5343688cdaa8c4e3a83fb3a78ee82fb0b280b3058f2`
- Code snapshot SHA-256: `0e6b4baca0c3b0f549fdf3667142c4abde13f2c64aa9ac8e37b7595589e16cd0`
- Source hashes: `{"BrainXStudy.md": "d12a874bc7f557e10d5db589ec6e129206fcf5ec5885c8e2b0f2cd85490f1f48", "NeuroSpecification.md": "01c3bd8eca0c6810c047dfe4e1a08af07605783c131f6b5d269c8cdde0df5d48", "acceleration_audit.md": "eeee2917361bd8356a44371521f00d50bde858a578cccf79121478e705c05d1e", "acceleration_parity.json": "f5f3061eb09a45cb8b5ebab2b8fc9d87318a74ae79796672f2a1bef321e11123", "config.json": "0e7488782a7dda6496576df2d4cb26d086c98abf680ac426694239332115bb04", "connectivity_benchmark.json": "29c6ed26ac20fa7a666ae5d17ba34ff10216d235c94925bf4f88d2946404956e", "full_scale_benchmark.json": "47d058579d1f04690c661085badc536aad2389d95f32656170a6d78dc72b578b", "lif_network.py": "6ec7748aa4d13b8d93900c8dc26e217fa349c5f00bb693f079ba00c7acf2c93a", "test_lif_network.py": "6c25143a30dc4283fdf8b4927e3c0ba4c373f75fa563ad332bce5784355fb24a"}`

## Command

```text
taskset -c 14-20 env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib-brainx-lif PYTHONUNBUFFERED=1 /home/yixinliu/anaconda3/envs/braincell-released/bin/python /home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/20-current-lif-network-first-principles/lif_network.py --mode production --output-dir /home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/20-current-lif-network-first-principles/runs/production-seed-47-cpu-20260903 --seed 47
```
