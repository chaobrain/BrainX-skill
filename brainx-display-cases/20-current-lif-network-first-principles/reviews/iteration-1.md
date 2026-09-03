# BrainX iteration review

- **OUTCOME:** `REFUSE`
- **SCIENTIFIC_OUTCOME:** `PARTIALLY_SUPPORTED`
- **LOSS_CLOSURE:** `NOT_APPLICABLE`
- **OPTIMIZATION_ADEQUACY:** `NOT_APPLICABLE`
- **NEXT_ACTION:** `RETURN_TO_STUDY`

## Good-enough reason

The raw three-seed results reproduce the reported rates, spectra, ISI-CV summaries, and classifications, but required model controls and production State-finiteness evidence are missing, and generic BrainTools-owned operations lack the mandatory API-gap assessment.

## Findings

### BX-API-001: Missing BrainTools API-gap evidence

- **Severity:** `critical`
- **Location:** `lif_network.py:64`, `lif_network.py:382`; project artifact set
- **Problem:** NumPy manually generates the fixed-degree topology and `scipy.signal.welch` computes the decisive PSD. The routed references establish BrainTools connectivity families and `braintools.metric.power_spectral_density`, but no `BrainTools API gap` artifact identifies checked APIs, missing capabilities, the minimum external boundary, or parity evidence.
- **Scientific consequence:** The connectivity realization and PSD determine the dynamics and every synchrony classification. Their departure from BrainTools-owned infrastructure is not justified under the review contract.
- **Minimum fix:** Use the suitable BrainTools APIs, or add a BrainTools API-gap artifact documenting why exact random fan-in/no-autapse generation and the frozen Hann/detrend/scaling PSD contract cannot be expressed, with unit, shape, numerical, and classification parity.

### BX-NATIVE-002: Delay duplicates a prebuilt BrainState abstraction

- **Severity:** `major`
- **Location:** `lif_network.py:185`, `lif_network.py:265`, `lif_network.py:312`
- **Problem:** `RingDelay` manually implements buffer State, pointer bookkeeping, reads, writes, and resets even though the routed delay reference provides `brainstate.nn.Delay` with step-based retrieval.
- **Scientific consequence:** The impulse test supports the current 15-step behavior, but the implementation is not the smallest suitable BrainX-native abstraction and carries avoidable lifecycle risk.
- **Minimum fix:** Replace `RingDelay` with `brainstate.nn.Delay`, preserve the documented retrieval/update convention, and require exact impulse and rollout parity before accepting existing results.

### BX-CTRL-003: Locked controls are not implemented

- **Severity:** `major`
- **Location:** `NeuroSpecification.md:22`, `test_lif_network.py:17`, `test_lif_network.py:58`, `test_lif_network.py:118`
- **Problem:** The seven tests do not verify external Poisson mean and variance, inhibitory jump sign and magnitude through the projection, delta-event summation, or a hand-checkable small-network reference transition. The external-rate test checks only the algebraic value of lambda, while the rollout test checks replay rather than transition correctness.
- **Scientific consequence:** Core input and transition mappings shared by all twelve runs remain insufficiently validated, so the measured regime claims cannot be accepted unconditionally.
- **Minimum fix:** Add focused frozen-snapshot tests for Poisson moments, signed E/I delta accumulation, and one explicit reference transition covering leak, delayed input, threshold, reset, and refractory behavior.

### BX-STATE-004: Production State finiteness is unverified

- **Severity:** `major`
- **Location:** `lif_network.py:729`, `collect_results.py:59`
- **Problem:** Mechanical validation checks rates, spectra, and valid CV values but never saves or checks production membrane State. A small-rollout test checks voltage finiteness, which does not establish it for the twelve production rollouts.
- **Scientific consequence:** The specification names non-finite State as an invalid-result condition; finite aggregate outputs do not rule out NaNs in a subset of neurons.
- **Minimum fix:** Check the complete final membrane State after each production rollout and preserve its finiteness result or hash in the immutable evidence. Re-execute the affected production snapshot if that evidence cannot be recovered.

## Unverified assumptions

- The regime thresholds are phenomenological and have no supplied external biological validation; the assessment correctly limits its claims accordingly.

