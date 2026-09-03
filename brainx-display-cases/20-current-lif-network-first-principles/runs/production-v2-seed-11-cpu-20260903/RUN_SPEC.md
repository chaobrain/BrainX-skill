# Run specification

- Run ID: `production-v2-seed-11-cpu-20260903`
- Modeling-loop iteration: 2
- Entry case: new
- Run level: production
- Working directory: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/20-current-lif-network-first-principles`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend and device: CPU; `['cpu:0']`
- BrainState environment: `dt = 0.1 ms`, `fit = false`, precision contract 32-bit
- Seeds: `[11]`
- Seed policy: deterministic derived connectivity, initialization/sample, and external streams; exact replay expected
- Connectivity: exact fixed fan-in, no autapses, shared across conditions within each seed
- Checkpoint source: none
- Retry budget: zero unchanged retries; deterministic failure returns to implementation
- Expected artifacts: `['config.json', 'environment.json', 'command.txt', 'code.diff', 'run.log', 'status.json', 'exit_code', 'connectivity_manifest.json', 'run_metrics.csv', 'condition_assessment.json']` plus one connectivity NPZ and per-condition raw/metric files
- Resource estimate: full-scale warm estimate 1415.2 s per condition; production seeds run on disjoint CPU core sets; approximately 100-200 MB static arrays per process plus compiled runtime State
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
- Active config SHA-256: `767bbd39511401cb13dfc301c5513bc7aa973016e63892defedfb4554331ce76`
- Code snapshot SHA-256: `f7f568d3aa1526cc0a9efb77e66dcc1dbc1d861649ba9da6b97fe3a2997ebca0`
- Source hashes: `{"BrainToolsAPIGap.md": "cc73a567f6dcc2b05ef3bdf834439b60cadf78156af5b4c55cb0faa9720b2905", "BrainXStudy-iteration2.md": "2be33b4ebb12d23ef992aff9023ab7ee6fb6080bee01a0dd267ed499d4d9eb75", "BrainXStudy.md": "d12a874bc7f557e10d5db589ec6e129206fcf5ec5885c8e2b0f2cd85490f1f48", "NeuroSpecification.md": "01c3bd8eca0c6810c047dfe4e1a08af07605783c131f6b5d269c8cdde0df5d48", "acceleration_audit-iteration2.md": "18b7d1bdbb8e95fbcd0035286339bbbbe3a48adce8cc7f60febed4c48fc4fd69", "acceleration_parity-iteration2.json": "3fd76e551d3f566ec588b1ef22211e34d1167abf49bc969f4b10d780b717fb8e", "config.json": "0e7488782a7dda6496576df2d4cb26d086c98abf680ac426694239332115bb04", "connectivity_benchmark.json": "29c6ed26ac20fa7a666ae5d17ba34ff10216d235c94925bf4f88d2946404956e", "full_scale_benchmark-iteration2.json": "aaba05cb55206a3e6a7479654154a87c03b665ee2091c22fd9363bbd7a638865", "iteration2_control_evidence.json": "855b292dd355afeef1e31883455c90247188140227f56c628d739be07bb8b5a8", "lif_network.py": "9d880a0481c042a95654d0af4790f0cea5220d0248275465c7b32a546c37966e", "reviews/iteration-1.md": "da2d87b6473e59c8f36cfe615a46a6a85c28db7ad7487fd8182eb030366fca76", "test_lif_network.py": "12b28422431dd7fab7ab0cd6109fea02d340afd7ac62b6dddd017092ebfb812a"}`

## Command

```text
taskset -c 0-6 env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib-brainx-lif PYTHONUNBUFFERED=1 /home/yixinliu/anaconda3/envs/braincell-released/bin/python /home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/20-current-lif-network-first-principles/lif_network.py --mode production --output-dir /home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/20-current-lif-network-first-principles/runs/production-v2-seed-11-cpu-20260903 --seed 11
```
