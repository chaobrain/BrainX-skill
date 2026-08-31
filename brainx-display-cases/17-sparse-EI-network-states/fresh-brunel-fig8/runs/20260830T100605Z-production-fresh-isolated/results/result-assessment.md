# Result assessment

Verdicts apply the locked prospective criteria without tuning.

| Panel | Requested state | Verdict | Rate (Hz) | ISI CV | Frequency (Hz) |
|---|---|---|---:|---:|---:|
| A | `synchronous_regular` | `reproduced` | 312.495 | 0.001 | 625.000 |
| B | `fast_synchronous_irregular` | `reproduced` | 60.550 | 0.917 | 173.340 |
| C | `asynchronous_irregular` | `not_reproduced` | 37.898 | 0.408 | 114.746 |
| D | `slow_synchronous_irregular` | `not_reproduced` | 5.838 | 0.631 | 19.531 |

## Deterministic findings

- Panel A: all predicates passed.
- Panel B: all predicates passed.
- Panel C: mean ISI CV is below 0.7.
- Panel D: mean ISI CV is below 0.7.

## Claim-evidence matrix

| Claim | Evidence | Boundary |
|---|---|---|
| Each panel is or is not reproduced | `metrics.json` and `raw/*.npz` | This finite seeded realization and the locked predicates only. |
| Frequencies come from global activity | Per-panel `frequencies_hz` and `power_hz` | Frozen Welch settings and 1-1000 Hz search band. |
| Source and results are bound to the run | Parent `run-manifest.json` | Excludes mutable log/status/exit files. |

No image exists at this review stage. No phase region, pixel identity, or author-RNG parity is claimed.
