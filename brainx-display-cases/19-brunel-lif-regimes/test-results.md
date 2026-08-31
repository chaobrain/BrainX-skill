# Focused test results

- Command: `MPLCONFIGDIR=/tmp/matplotlib-cache JAX_PLATFORMS=cpu pytest -q test_brunel_lif_regimes.py`
- Result: `9 passed in 18.39s`
- Backend: CPU

The suite verifies units and external-rate conversion, exact fan-in without duplicates or autapses, the 15-step delay, LIF threshold/reset/refractory behavior, deterministic compiled replay, eager/compiled spikes and final-voltage parity, finite external-drive-only activity, prospective classifier behavior, locked configuration loading, and the fixed 40 E plus 10 I probe sample.

## Iteration 2

- Command: `MPLCONFIGDIR=/tmp/matplotlib-cache JAX_PLATFORMS=cpu pytest -q test_brunel_lif_regimes.py`
- Result: `9 passed in 30.76s`
- Scope: Revalidated the complete focused suite after changing only the smoke duration.

## Iteration 3

- Command: `MPLCONFIGDIR=/tmp/matplotlib-cache JAX_PLATFORMS=cpu pytest -q test_brunel_lif_regimes.py`
- Result: `10 passed in 46.27s`
- Added coverage: validates and copies one completed condition-boundary artifact without changing its row or raw contents.
- Actual-parent witness: 13 rows validated, 13 raw NPZ files copied and parsed, seven locked conditions identified as missing.

## Iteration 4

- Command: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python -m pytest -q`
- Result: `10 passed in 23.47s`
- Added coverage: requires exact condition parameters, the complete metric schema, deterministic classification, fixed probe identity, every raw-array shape, and source-manifest byte count and SHA-256 before inheritance.
- Actual-source witness: all 20 rows and 20 raw NPZ files from the completed continuation passed the hardened gate.
