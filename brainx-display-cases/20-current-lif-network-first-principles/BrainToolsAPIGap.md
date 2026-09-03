# BrainTools API-gap assessment

## Exact fixed fan-in topology

Checked the documented `braintools.conn.Random`, `FixedProb`, `Regular`, and `ExcitatoryInhibitory` families.

- `Random` / `FixedProb` samples independent edges and does not guarantee exactly `CE` and `CI` inputs for every target.
- `Regular` represents a regular topology but the documented contract does not combine separate E/I source populations, distinct exact per-target indegrees, without-replacement source selection, and within-population autapse exclusion.
- `ExcitatoryInhibitory` preserves source sign but accepts connection probabilities rather than exact per-target E/I indegrees.

The minimum external boundary is therefore host-side NumPy sampling of unique source indices for each target. It returns only two `int32` arrays with frozen shapes `(12,500, 1,000)` and `(12,500, 250)`. `validate_connectivity()` checks shapes, bounds, uniqueness, exact degrees, and no autapses before the arrays cross into `brainevent.FixedNumPerPost`. BrainEvent owns all transformed event communication.

Parity evidence:

- Exact degree, uniqueness, source-bound, and autapse tests are in `test_lif_network.py`.
- BrainEvent count and voltage-unit parity against a hand sum are in `test_brainevent_fixed_fan_in_counts_and_preserves_voltage_unit`.
- Every production connectivity array is saved and SHA-256 hashed.

## Frozen global-rate spectrum

Checked `braintools.metric.power_spectral_density(lfp, dt, nperseg=None, noverlap=None, freq_range=None)`.

The helper exposes `dt`, segment length, overlap, and optional frequency range, but it does not expose the approved `window='hann'`, per-segment `detrend='constant'`, or `scaling='density'` controls. A direct probe on the iteration-1 seed-11 `(g, eta) = (5, 2)` global-rate trace produced the same frequency grid and `119 Hz` dominant peak, but PSD values differed: the maximum absolute difference was `0.0227063` even after global demeaning, and the arrays did not satisfy `rtol=1e-4, atol=1e-5`. Substituting the helper would therefore change the locked numerical spectrum contract.

The minimum external boundary remains `scipy.signal.welch` on an already unit-normalized host rate vector. All classification logic is explicit and independent of SciPy convenience functions beyond the PSD itself.

Required parity artifact for iteration 2:

- Compare BrainTools and frozen SciPy frequency grids, dominant frequencies, peak significance predicates, and final condition classifications for all twelve new raw traces.
- Preserve numerical PSD differences rather than claiming array equality.
- Require classification parity before review; otherwise return to implementation.

