# Result assessment

The labels below apply the locked prospective predicates without tuning.

| Requested condition | Aggregate label | Matching repeats | Dominant frequency (median Hz) | Robust |
|---|---|---:|---:|---|
| `synchronous_regular` | `synchronous_regular` | 1/1 | 100.000 | False |
| `fast_synchronous_irregular` | `synchronous_regular` | 0/1 | 187.500 | False |
| `asynchronous_irregular` | `synchronous_regular` | 0/1 | 100.000 | False |
| `slow_synchronous_irregular` | `inconclusive` | 0/1 | 100.000 | False |

## Claim-evidence matrix

| Claim | Evidence | Boundary |
|---|---|---|
| Each requested state is reproduced robustly or not | `metrics.json` and `robustness.json` | Requires four of five repeat labels and the aggregate label; no category is forced. |
| Dominant frequencies are measured from the global rate | Per-run `frequencies_hz` and `power_hz` arrays in `raw/*.npz` | Welch estimator and search band were frozen before production. |
| E/I rates and irregularity support the labels | `metrics.csv`, `metrics.json`, and per-run ISI-CV arrays | CV excludes neurons with fewer than four analyzed spikes. |

No continuous phase boundary, exact Brunel RNG parity, or biological realism is claimed.
