# Validation

- `python -m py_compile small_ei_network.py test_small_ei_network.py`: passed.
- `pytest -q test_small_ei_network.py`: collection skipped because `brainpy` is not importable in the current Python environment; the test suite is ready to run after the matched BrainX packages are installed.
- Runtime spike-shape and event checks are therefore pending dependency availability. No firing-rate or dynamical claim is made from this checkout.
