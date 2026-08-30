# Result assessment

Verdicts apply the locked prospective criteria without tuning.

| Panel | Requested state | Verdict | Rate (Hz) | ISI CV | Frequency (Hz) |
|---|---|---|---:|---:|---:|
| A | `synchronous_regular` | `reproduced` | 112.766 | 0.093 | 112.305 |
| B | `fast_synchronous_irregular` | `not_reproduced` | 171.932 | 0.100 | 180.664 |
| C | `asynchronous_irregular` | `not_reproduced` | 87.845 | 0.122 | 87.891 |
| D | `slow_synchronous_irregular` | `not_reproduced` | 3.778 | 0.483 | 14.648 |

## Deterministic findings

- Panel A: all predicates passed.
- Panel B: mean ISI CV is below 0.7; firing rate differs from 60.7 Hz by more than 20%.
- Panel C: mean ISI CV is below 0.7; firing rate differs from 37.7 Hz by more than 20%.
- Panel D: mean ISI CV is below 0.7; firing rate differs from 5.5 Hz by more than 30%; global frequency differs from 22 Hz by more than 30%.

## Claim-evidence matrix

| Claim | Evidence | Boundary |
|---|---|---|
| Each panel is or is not reproduced | `metrics.json` and `raw/*.npz` | This finite seeded realization and the locked predicates only. |
| Frequencies come from global activity | Per-panel `frequencies_hz` and `power_hz` | Frozen Welch settings and 1-1000 Hz search band. |
| Source and results are bound to the run | Parent `run-manifest.json` | Excludes mutable log/status/exit files. |

No image exists at this review stage. No phase region, pixel identity, or author-RNG parity is claimed.
