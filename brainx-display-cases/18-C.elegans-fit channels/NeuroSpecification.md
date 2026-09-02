# NeuroSpecification

- Status: locked
- Researcher approval: Direct request received 2026-09-02; the supplied-data-only constraint and requested protocols define the acceptance boundary.

## Researcher request

- Brain-modeling question or behavior: Recover phenomenological Hodgkin-Huxley SHK-1 and EGL-19 channel dynamics from the supplied C. elegans muscle voltage-clamp recordings.
- Requested model, experiment, or comparison: Fit SHK-1 to wild-type-minus-`shk-1(lf)` potassium current at 0 to +100 mV in 20 mV increments, then fit EGL-19 at -20 to +40 mV in 10 mV increments.
- Execution mode: parameter-fitting
- Required outputs: Current-time overlays, steady-state activation curves and activation time-constant functions with per-voltage experimental estimates, fitted equations and parameters, fit metrics, and reproducible BrainCell code.
- Constraints: Use a -60 mV holding potential and 100 ms steps. Derive the scientific model only from the supplied packed experiments; do not use papers or open-source channel implementations. Keep both packed files read-only.

## Inspected data contract

- Data sources and inspected contents: `Fig1C D I-V K currents.pxp` contains 12 WT potassium traces (`wave87:98`), 12 `shk-1(lf)` traces (`wave113:124`), time (`wave86`), and commands (`wave167:178`). `Fig. 3A I-V Ca currents.pxp` contains 11 WT (`wave1:11`), 11 `egl-19(n582)` (`wave12:22`), 11 `egl-19(ad1006)` (`wave23:33`), time (`wave0`), and commands (`wave34:44`). These identities come from the packed Igor graph recreation metadata.
- Shapes, axes, sampling/time base, and physical units: Potassium arrays allocate 7,500 samples with 1,500 finite values at 0.1 ms spacing; the command step is 12.3-112.3 ms. Calcium arrays contain 7,500 values at 0.02 ms spacing; the command step is 12.34-112.34 ms. Graph scales and wave values establish milliseconds, millivolts, and picoamperes.
- Required preprocessing and the subset used to fit each transform: Form SHK-1 as baseline-corrected WT minus baseline-corrected `shk-1(lf)` and retain the six 0 to +100 mV traces. Fit EGL-19 to the directly labeled WT calcium family and retain the seven -20 to +40 mV traces. Estimate each baseline over 2-10 ms, exclude the first 0.5 ms (SHK-1) or 1.0 ms (EGL-19) of each clamp step, and fit only the remaining step samples. Preserve WT-minus-each-EGL-mutant differences as controls: their low-voltage sign changes invalidate them as a single inward-conductance target over the requested range.
- Mapping from data to model inputs, targets, and observables: Commands are fixed voltage inputs and baseline-corrected currents are observations. Local trace fits select activation powers from equal-parameter candidates: SHK-1 power 2 among powers 1-5 and EGL-19 power 4 among powers 1-4. Compare optional EGL `m^4h` fits at the full global budget, require at least one successful termination, and select between `m^4` and `m^4h` by lower BIC on the same samples. Use a three-parameter exponential SHK time constant after the more general logistic midpoint fell outside the measured range. Fit reversal potentials, maximal conductances, steady-state gates, and time-constant functions jointly from the selected traces. Initialize gates at their fitted -60 mV steady state for each independent step. Optimize on a deterministic 0.5 ms grid and evaluate at the packed files' full resolution.
- Known data limitations or unresolved mismatches: Only genotype-average traces are available, not paired cells or replicate-level observations. Subtraction therefore combines different population means. Current units are pA rather than density because capacitance values are absent. Reversal potentials lie outside or at the edge of the requested voltage ranges and may be weakly identifiable. Gate powers are empirical waveform choices, not molecular stoichiometry.

## Declared fitting domain

The physical bounds are data-scale safeguards, not biological priors. They span the measured voltage windows and exceed the locally fitted 1.6-18.5 ms activation time constants while avoiding unsupported orders of magnitude.

| Channel | Parameter bounds in model order |
|---|---|
| SHK-1 | `g_max` 1-100 nS; `E_rev` -150 to -1 mV; `V_half` -60 to 40 mV; slope 2-50 mV; `tau_min` 0.05-20 ms; `tau_amp` 0-50 ms; `k_tau` 5-100 mV |
| EGL-19 | `g_max` 1-100 nS; `E_rev` 40.1-100 mV; `V_half` -40 to 20 mV; slope 2-30 mV; `tau_min` 0.05-20 ms; `tau_amp` 0-50 ms; `V_tau` -40 to 40 mV; `k_tau` 2-40 mV |

Pass these bounds directly to BrainTools. Use the same six optimizer seeds, 1,200-iteration budget, BrainCell objective, and successful-candidate rule for observed, recovery, and voltage-holdout fits.

## Acceptance boundary

- Evidence required for success, failure, or an inconclusive result: Reproduce both protocols and every requested trace; keep gate values in [0, 1] and time constants positive; preserve voltage, time, conductance, and current units; pass protocol, metadata mapping, lifecycle/reset, BrainCell-versus-analytic parity, recovery-domain, leave-one-voltage-out, and finite-output tests; report per-trace and aggregate errors; and show experimental gate estimates beside fitted functions.
- Required baselines and controls: Verify wave identity and command timing from packed metadata; verify direct SHK subtraction; report both EGL mutant-difference sign controls; compare candidate gate powers; verify zero current at fitted reversal and relaxation toward fitted steady state.
- Invalid-result conditions: Wrong wave mapping, use of literature-derived scientific parameters or kinetics, fitting clamp transients or post-step samples, current-sign inconsistency, non-finite output, gate bounds violation, unit mismatch, or labeling a mutant family as WT.
- Allowed claims and explicit non-claims: Claim only phenomenological HH waveform fits over the measured protocols. Do not claim unique molecular kinetics, channel stoichiometry, single-cell parameters, biological uncertainty, or extrapolation beyond the fitted voltage ranges.
