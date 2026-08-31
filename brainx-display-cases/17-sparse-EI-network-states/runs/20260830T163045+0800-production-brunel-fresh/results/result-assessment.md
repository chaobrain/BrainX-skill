# Result assessment

The verdicts apply the locked prospective criteria without tuning.

| Panel | Requested state | Verdict | Rate (Hz) | ISI CV | Dominant frequency (Hz) |
|---|---|---|---:|---:|---:|
| A | `synchronous_regular` | `reproduced` | 312.495 | 0.001 | 625.000 |
| B | `fast_synchronous_irregular` | `reproduced` | 60.550 | 0.917 | 173.340 |
| C | `asynchronous_irregular` | `not_reproduced` | 37.898 | 0.408 | 114.746 |
| D | `slow_synchronous_irregular` | `not_reproduced` | 5.838 | 0.631 | 19.531 |

## Deterministic findings

- Panel A: all locked predicates passed.
- Panel B: all locked predicates passed.
- Panel C: mean ISI CV is below 0.7; 1 ms population-rate CV is not below 0.2.
- Panel D: mean ISI CV is below 0.7.

## Claim-evidence matrix

| Claim | Evidence | Boundary |
|---|---|---|
| Each panel is or is not reproduced | `metrics.json` and `raw/*.npz` | Applies only to this finite seeded realization and the locked predicates. |
| Frequencies come from global activity | `frequencies_hz` and `power_hz` in each raw file | Welch settings and search band are fixed in source code. |
| The final image derives from accepted evidence | Figure hashes and raw-file hashes | The renderer does not alter simulation data. |

No phase region, pixel identity, or exact author-RNG parity is claimed.
