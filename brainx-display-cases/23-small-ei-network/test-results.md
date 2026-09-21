# Validation

- `python -m py_compile small_ei_network.py test_small_ei_network.py`: passed.
- `pytest -q test_small_ei_network.py`: collection skipped because `brainpy` is not importable in the current Python environment; the test suite is ready to run after the matched BrainX packages are installed.
- Runtime spike-shape and event checks are therefore pending dependency availability. No firing-rate or dynamical claim is made from this checkout.

## Runtime rerun after the refused review

- `pytest -q test_small_ei_network.py`: `2 passed` with six deprecation warnings.
- `MPLBACKEND=Agg python small_ei_network.py`: passed and saved `outputs/small_ei_network_raster.png`.
- Full rollout: spikes shape `(1000, 20)`, 100 binary events, five spikes per neuron, and at most 20 simultaneous events per recorded step.
- The earlier collection skip is preserved above as the evidence available at review time; a fresh Codex review is still required.
