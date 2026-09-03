# Result assessment

All twelve immutable CPU runs completed with exit code 0. Each raw file has the frozen shapes and finite rate, CV, frequency, and power values.

| Condition | Requested | Measured | Verified | Dominant frequency | Pooled median ISI CV | Mean E/I rate |
|---|---|---|---|---:|---:|---:|
| `(g, eta) = (3, 2)` | synchronous regular | synchronous regular | yes | 333.0 Hz | 0.000 | 333.37 / 333.36 Hz |
| `(g, eta) = (6, 4)` | fast synchronous irregular | synchronous, regularity-indeterminate | no | 184.0 Hz | 0.798 | 59.23 / 59.33 Hz |
| `(g, eta) = (5, 2)` | asynchronous irregular | synchronous regular | no | 121.0 Hz | 0.410 | 37.84 / 37.84 Hz |
| `(g, eta) = (4.5, 0.9)` | slow synchronous irregular | synchronous, regularity-indeterminate | no | 22.0 Hz | 0.679 | 5.60 / 5.60 Hz |

## Claim-evidence matrix

| Claim | Evidence | Outcome |
|---|---|---|
| `synchronous regular` at `(g, eta)=(3, 2)` | seed 11: 333 Hz, CV 0.000, peak=True, seed 29: 333 Hz, CV 0.000, peak=True, seed 47: 333 Hz, CV 0.000, peak=True; pooled CV `0.000`; measured `synchronous regular` | supported |
| `fast synchronous irregular` at `(g, eta)=(6, 4)` | seed 11: 184 Hz, CV 0.808, peak=True, seed 29: 183 Hz, CV 0.792, peak=True, seed 47: 184 Hz, CV 0.794, peak=True; pooled CV `0.798`; measured `synchronous, regularity-indeterminate` | not supported |
| `asynchronous irregular` at `(g, eta)=(5, 2)` | seed 11: 119 Hz, CV 0.409, peak=True, seed 29: 121 Hz, CV 0.410, peak=True, seed 47: 130 Hz, CV 0.410, peak=True; pooled CV `0.410`; measured `synchronous regular` | not supported |
| `slow synchronous irregular` at `(g, eta)=(4.5, 0.9)` | seed 11: 22 Hz, CV 0.673, peak=True, seed 29: 22 Hz, CV 0.685, peak=True, seed 47: 21 Hz, CV 0.680, peak=True; pooled CV `0.679`; measured `synchronous, regularity-indeterminate` | not supported |

## Allowed conclusion

Report only the frozen finite-simulation classifications above. Do not relabel indeterminate or contradictory conditions to match the requested names, and do not generalize beyond this implementation, initialization, timestep, duration, or seed set.
