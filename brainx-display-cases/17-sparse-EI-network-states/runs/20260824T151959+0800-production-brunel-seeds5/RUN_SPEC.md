# Production run specification

- Run ID: `20260824T151959+0800-production-brunel-seeds5`
- Entry case: new
- Run level: production with the approved five-seed replication included
- Frozen at: 2026-08-24T15:19:59+08:00
- Working directory: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/17-sparse-EI-network-states`
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend/device: JAX CPU / `cpu:0`
- BrainState environment: platform CPU, precision 32, `dt = 0.1 ms`
- Checkpoint source: none; each condition is an atomic independent rollout
- Retry budget: zero unchanged retries; preserve any failure and return to implementation for deterministic failures

## Frozen scientific contract

Consume `config.json` exactly. Run 10,000 E and 2,500 I neurons with exact indegrees 1,000/250, 1,000-source aggregate external Poisson drive, 0.1 mV excitatory jumps, `-g * 0.1 mV` inhibitory jumps, 1.5 ms delay, 20 ms membrane time constant, 20/10 mV threshold/reset, and 2 ms refractory time. Discard 500 ms, analyze 2,000 ms, and run all four conditions at seeds 1729, 2718, 3141, 5772, and 8119. Do not alter the prospective classifier.

## Identity and hashes

- Git commit: `e1a589815cf3d357dd8c5e65938426fe9ac7b1a6`
- Git state: case directory is untracked; exact entry-point diff is frozen as `code.diff`
- `NeuroSpecification.md`: `ffdce7f279cc4101889159c2e97744014045a4e941c9face317b19c273e13a02`
- `brainmodeling-memory.md`: `07e2154cc40eb2d5bc04c625aeffa24d0db923720a96e4f2b68daf4dff75edb8`
- `brainx-study-record.md`: `9bb3d98327f650aabe3ad6560446f9edaf136795d6f391e136db8201537316d9`
- `sparse_ei_network.py`: `a9ede50387dc1fc8c6b6ff7bef8ba767854edaf6bc2ccd0685797d6bd838b368`
- `test_sparse_ei_network.py`: `54f7803bfc43dc967f61cca539c99a14b7ed705ab4d7f5ebb5884ad8aacf845f`
- `test-results.md`: `2480d5d825f18c8e6ad3d0521a8ea0de6c10cc5821b03818aef695a117b6e78c`
- `acceleration-and-parity.md`: `308d4edd4df65b86a094f4db74b1f79e65f3d190ab396dec8b49f241c212a58b`
- `code.diff`: `73577a38c69b709d7dd4a8bf34e30a6609916f8ca761650d5a7aac2321b8b6fd`
- `config.json`: `85d7e3a1be5c176abae6a6a364396bda8c5b7157f0b6163318ed51dfdc1dd729`
- `environment.json`: `0463a5e4a1e30ef982e6406131ba2cf44afb8dfa62d1678d7688b6350466ea78`
- `command.txt`: `67a51cab2a62be4ffba3918fe8842790ae5beae6f032cf0c79a99dabef9f0d24`
- `launch.sh`: `6dc737645656340981477231b8af243b1de9da2324db400ab37c97b5a15570da`

## Command and expected artifacts

Run the exact command in `command.txt` through `launch.sh`. Append process output to `run.log` and preserve the Python exit code in `exit_code`.

Expected under `results/`: exact copied `config.json`, `metrics.json`, `metrics.csv`, `robustness.json`, `graph-hashes.json`, `provenance.json`, and 20 compressed `raw/repeat-*_*.npz` files containing fixed raster probes, E/I population counts, and spectra.

## Resource and stop contract

- Preflight resources: 11 GiB available memory, 929 GiB available disk.
- Estimated live buffers: about 62.5 MB fixed-degree indices plus 312.5 MB full boolean spike history per condition, with host analysis and XLA overhead.
- Estimated duration: approximately 1-3 hours on CPU, including five graph-specific compilations.
- Heartbeat: one JSON log row after every completed condition; a graph compilation may be silent for several minutes.
- Stop only for process failure, non-finite/corrupt required artifacts, OOM, device/backend mismatch, or less than 5 GiB remaining disk. Do not stop for an unfavorable or unexpected scientific result.
