# Small E/I network report

- **Report status:** draft
- **Scientific outcome:** pending
- **Review outcome:** pending (iteration 1 was `REFUSE`; a fresh review has not yet been run)
- **Research question:** Does the locked 20-neuron recurrent excitatory/inhibitory network construct and produce the declared time-major spike output?
- **Scientific scope:** This is a compact teaching model. The random graph and constant drive are not calibrated to reproduce a published firing regime.
- **Review status:** The preserved iteration-1 review returned `REFUSE` because runtime evidence was missing. The runtime gap is now addressed locally, but a fresh Codex review has not yet returned `PASS`, so this report remains a draft diagnostic report.

## Protocol

The network contains 16 excitatory and 4 inhibitory `LIFRef` neurons. Each population projects probabilistically to all 20 targets through exponential conductances and COBA outputs. The run uses `dt = 0.1 ms`, `100 ms`, a constant `20 mA` drive, and BrainState seed `1234`.

## Available evidence

The executed run returns a `(1000, 20)` time-major spike array with binary event values. It contains 100 events: every neuron emits five spikes, and the largest recorded step contains 20 simultaneous spikes. The saved raw arrays and metadata are in [`outputs/small_ei_network_run.npz`](outputs/small_ei_network_run.npz) and [`outputs/small_ei_network_metadata.json`](outputs/small_ei_network_metadata.json).

The diagnostic raster in [`outputs/small_ei_network_raster.png`](outputs/small_ei_network_raster.png) shows five synchronized network-wide events. It verifies the recorded event pattern and neuron ordering; it does not establish biological realism, a calibrated regime, or a causal E/I mechanism. Figure provenance and source-value checks are in [`FIGURE_MANIFEST.md`](FIGURE_MANIFEST.md).

## Validation and limitations

`pytest -q test_small_ei_network.py` passes both structural and short-rollout checks. The full run also reproduces the declared shape and binary event semantics. The remaining acceptance action is a fresh Codex review using the saved run artifacts. Until that review passes, treat all conclusions as provisional and do not call this a publication result.

## Next action

Send the locked specification, implementation, tests, runtime metadata, raw arrays, diagnostic figure, and this report to a fresh step-5 Codex review. If the reviewer passes, complete the final visualization/report step; if it refuses, retain this report and update it before the next iteration.
