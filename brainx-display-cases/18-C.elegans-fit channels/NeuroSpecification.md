# NeuroSpecification

- Status: locked
- Researcher approval: Direct request received 2026-08-25; the requested protocols and outputs define the acceptance boundary.

## Researcher request

- Brain-modeling question or behavior: Recover Hodgkin-Huxley-type SHK-1 and EGL-19 channel dynamics from C. elegans body-wall-muscle voltage-clamp recordings.
- Requested model, experiment, or comparison: Fit SHK-1 to wild-type-minus-`shk-1(lf)` potassium currents at 0 to +100 mV in 20 mV increments, then fit EGL-19 to isolated wild-type calcium currents at -20 to +40 mV in 10 mV increments.
- Execution mode: parameter-fitting
- Required outputs: Current-time overlays, steady-state gate curves with per-voltage experimental estimates, activation time-constant functions with per-voltage estimates, fitted equations and parameters, metrics, and reproducible code.
- Constraints: Use a -60 mV holding potential, 100 ms voltage steps, BrainCell HH channel classes, BrainState-compatible execution, and BrainUnit quantities. Keep both packed Igor files read-only.

## Inspected data contract

- Data sources and inspected contents: `Fig1C D I-V K currents.pxp` stores 12 potassium steps, genotype-average traces, processed genotype-difference traces, voltage commands, and steady-state I-V summaries. `Fig. 3A I-V Ca currents.pxp` stores 11 calcium steps for WT, `egl-19(n582,lf)`, and `egl-19(ad1006,lf)`, plus voltage commands.
- Shapes, axes, sampling/time base, and physical units: Potassium waves contain 7,500 allocated samples but 1,500 finite samples at 0.1 ms spacing; the command step runs from 12.3 through 112.2 ms. Calcium waves contain 7,500 samples at 0.02 ms spacing; the command step runs from 12.34 through 112.32 ms. Voltage is mV and current is pA; stored potassium I-V summaries are pA/pF.
- Required preprocessing and the subset used to fit each transform: Use potassium `wave154` through `wave165`, which the packed graph identifies as processed WT-minus-`shk-1(lf)` currents, and retain the 0, 20, 40, 60, 80, and 100 mV traces. Subtract each pre-step mean over 2-10 ms and exclude the first 0.5 ms of the clamp step. Use calcium WT `wave12` through `wave22`, retain -20 through +40 mV, subtract each 2-10 ms baseline, and exclude the first 1.0 ms capacitive transient. Fit kinetics on the remaining 100 ms step without using post-step samples.
- Mapping from data to model inputs, targets, and observables: Voltage commands are fixed model inputs. Baseline-corrected currents are observations. SHK-1 uses one activation gate with power four. EGL-19 uses an activation gate with power two and one inactivation gate. Per-voltage current fits yield gate steady-state and time-constant observations; global voltage functions and maximal conductance reproduce current traces.
- Known data limitations or unresolved mismatches: Traces are genotype averages from different cells, not paired recordings. The stored SHK-1 subtraction was processed in Igor after raw genotype averaging. EGL-19 null mutants are lethal; both mutant families are partial loss-of-function, so WT calcium current under pharmacological isolation is used rather than treating either mutant subtraction as a pure channel knockout. Leak was not subtracted in the source recordings. No cell-level replicate traces or capacitances are stored in these packed files, so uncertainty and mechanistic identifiability are limited.

## Acceptance boundary

- Evidence required for success, failure, or an inconclusive result: The code must reproduce both protocols, keep gate values within [0, 1], preserve current and time units, pass lifecycle/formula tests, produce finite fitted parameters, and report per-trace RMSE and normalized RMSE. Figures must show every requested voltage trace and the extracted gate-function points.
- Required baselines and controls: Verify command timing and genotype mappings from packed graph metadata; compare processed SHK-1 waves with the raw WT and mutant families; verify zero current at reversal and exact convergence to the fitted steady state.
- Invalid-result conditions: Wrong genotype/wave mapping, inclusion of clamp transients in the objective, current-sign reversal, non-finite output, gates outside [0, 1], unit mismatch, or silent fitting to post-step samples.
- Allowed claims and explicit non-claims: Claim a phenomenological HH representation of the supplied population-average protocols. Do not claim unique molecular kinetics, single-cell parameter identification, uncertainty from biological replicates, or generalization beyond the fitted voltage range.
