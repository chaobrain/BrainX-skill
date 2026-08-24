# BrainX iteration review

- **OUTCOME:** `PASS`
- **SCIENTIFIC_OUTCOME:** `SUPPORTED`
- **LOSS_CLOSURE:** `CLOSED`
- **OPTIMIZATION_ADEQUACY:** `SUFFICIENT`
- **NEXT_ACTION:** `ADVANCE_TO_VISUALIZATION`

## Good-enough reason

All three observed-data starts satisfy the predeclared closure rule, independently pass the locked held-out spike-count, latency, and monotonicity criteria, and retain finite State, passive-control, numerical-refinement, provenance, and unfavorable recovery evidence; the supported claim is appropriately limited to prediction under the tested 15-30 pA protocol.

## Findings

### FIT-004: Recovery initial-voltage preprocessing is not exact

- **Severity:** `minor`
- **Location:** `scripts/run_experiment.py:400`
- **Problem:** Synthetic observations are fitted with the real Trace #8 `train_initial_v` instead of recomputing the declared pre-stimulus mean from each noisy synthetic observation.
- **Scientific consequence:** The recovery experiment is not literally the claimed exact observation pipeline, although this does not affect the held-out predictive result and all parameter interpretation is already withheld.
- **Minimum fix:** Describe this recovery as an approximate diagnostic, or in a future parameter-identifiability iteration recompute `initial_voltage(observed, data.time)` for each synthetic case while changing nothing else.

## Unverified assumptions

- The custom channel kinetics, reversal potentials, calcium-pool constants, 2000 µm² area, and use of `AHP_De1994` as the SLO-2 surrogate remain phenomenological assumptions whose biological fidelity is not established by the supplied artifacts.
- The 50-300 ms stimulus timing and 15-30 pA amplitudes remain inferred because the recording lacks a current-monitor channel.
