# Result assessment

The verdicts apply the locked prospective criteria without tuning.

| Panel | Requested state | Verdict | Rate (Hz) | ISI CV | Dominant frequency (Hz) |
|---|---|---|---:|---:|---:|
| A | `synchronous_regular` | `reproduced` | 112.766 | 0.093 | 112.305 |
| B | `fast_synchronous_irregular` | `not_reproduced` | 171.932 | 0.100 | 180.664 |
| C | `asynchronous_irregular` | `not_reproduced` | 87.845 | 0.122 | 87.891 |
| D | `slow_synchronous_irregular` | `not_reproduced` | 3.778 | 0.483 | 14.648 |

## Deterministic findings

- Panel A: all locked predicates passed.
- Panel B: mean ISI CV is below 0.7; firing rate differs from 60.7 Hz by more than 20%.
- Panel C: mean ISI CV is below 0.7; firing rate differs from 37.7 Hz by more than 20%; 1 ms population-rate CV is not below 0.2.
- Panel D: mean ISI CV is below 0.7; firing rate differs from 5.5 Hz by more than 30%; global frequency differs from 22 Hz by more than 30%.

## Claim-evidence matrix

| Claim | Evidence | Boundary |
|---|---|---|
| Each panel is or is not reproduced | `metrics.json` and `raw/*.npz` | Applies only to this finite seeded realization and the locked predicates. |
| Frequencies come from global activity | `frequencies_hz` and `power_hz` in each raw file | Welch settings and search band are fixed in source code. |
| The final image derives from accepted evidence | Figure hashes and raw-file hashes | The renderer does not alter simulation data. |

No phase region, pixel identity, or exact author-RNG parity is claimed.
