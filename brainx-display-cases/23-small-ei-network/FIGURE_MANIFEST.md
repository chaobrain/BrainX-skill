# Figure manifest

## `outputs/small_ei_network_raster.png`

- Work type, evidence mode, scientific role, and question: create; diagnostic; inspect the timing and neuron ordering of the recorded spike events.
- Source run IDs, artifacts, hashes, and acceptance status: local `run()` with seed `1234`; source [`outputs/small_ei_network_run.npz`](outputs/small_ei_network_run.npz), SHA-256 `8d76607ebc4ed1ae6ff83795b3f94c76646f32db3eee5191b99e6af77fab07d3`; figure SHA-256 `5b1f0fdb2d299f424a100706077f99fd588693dcae80c6ddbef5a3efebd377b9`; not Codex-review accepted.
- Variables, axes, ordering, and units: `times_ms` and dense `spikes`; x-axis time in ms; y-axis neuron index `0–19`; excitatory neurons `0–15`, inhibitory neurons `16–19`.
- Transformations, smoothing, aggregation, and exclusions: dense binary events flattened into event times and neuron IDs for `braintools.visualize.spike_raster`; no smoothing, aggregation, thresholding, or exclusions.
- Sample size and uncertainty: 20 neurons, 1000 recorded time points, 100 events; no statistical uncertainty is estimated.
- Controls and fixed comparison settings: fixed seed `1234`, `dt = 0.1 ms`, duration `100 ms`, constant drive `20 mA`.
- Plotting source, output path, size, format, and resolution: `small_ei_network.py::plot_raster`; PNG at `outputs/small_ei_network_raster.png`; 2070 × 1016 px; 300 dpi.
- Render and source-value checks: image is nonblank and unclipped; five event columns are visible; event count and neuron ordering match `small_ei_network_metadata.json`.

Figure status: diagnostic and provisional pending a fresh Codex review.
