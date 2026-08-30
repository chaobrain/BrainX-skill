# Test results

## Iteration 1 implementation checks

- Command: `MPLCONFIGDIR=/tmp/matplotlib-case17 JAX_PLATFORMS=cpu pytest -q test_sparse_ei_network.py`
- Result: `9 passed in 31.16s`
- Backend: CPU
- Coverage: unit/rate conversion, exact fixed fan-in, autapse exclusion, physical delay timing, threshold/reset/refractory behavior, full reset and stochastic replay, eager/JIT parity, no-recurrence external drive, acceptance predicates, and fixed stratified probes.
