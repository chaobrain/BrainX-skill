# Test results

- Command: `MPLCONFIGDIR=/tmp/matplotlib-case17-fresh JAX_PLATFORMS=cpu pytest -q test_brunel_fig8.py`
- Result: `10 passed in 42.48s`
- Backend: CPU
- Coverage: BrainPy-State neuron ownership, unit/rate conversion, fixed-fan-in orientation/uniqueness/autapse exclusion, physical delay, reset/refractory behavior, exact stochastic replay, eager/JIT parity, external-only drive, declared-only acceptance predicates, and fixed probes.
