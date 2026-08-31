# NeuroSpecification

- Status: locked
- Researcher approval: The supplied prompt is treated as approval of the named mechanisms, one-trace fit, and three-trace held-out protocol. The contradictory count of seven channels versus six named currents is preserved as a known limitation.

## Researcher request
- Brain-modeling question or behavior: Can a conductance-based C. elegans body-wall muscle model calibrated to one current-clamp trace predict responses to nearby held-out currents?
- Requested model, experiment, or comparison: A single-compartment HH model with SHK-1, EGL-19, SLO-2, Kr, Na, and leak currents; fit one of traces 6-9 and test the other three at 15, 20, 25, and 30 pA.
- Execution mode: parameter-fitting.
- Required outputs: Unit-safe BrainCell model and fitting code, fitted parameter table, raw training and held-out predictions, waveform and spike-feature metrics, numerical checks, and a comparison figure after review.
- Constraints: Use BrainCell, BrainUnit, and BrainState. Keep `Fig4A-D.txt` read-only. Use only one experimental trace during fitting. Treat estimates as phenomenological unless recovery establishes parameter identifiability.

## Inspected data contract
- Data sources and inspected contents: `Fig4A-D.txt`, Axon Text File with SHA-256 recorded in run provenance; 10 voltage traces plus time. Traces 6-9 map to 15, 20, 25, and 30 pA respectively.
- Shapes, axes, sampling/time base, and physical units: 10,000 rows by 11 numeric columns; time is seconds from 0 to 0.49995 s at 0.00005 s (0.05 ms); voltage columns are mV. Current is a total-current step from 50 to 250 ms, inferred from the common response onset and recovery boundary stated explicitly in the fitting contract.
- Required preprocessing and the subset used to fit each transform: No filtering, interpolation, normalization, or target-derived baseline correction. Trace 8 (25 pA) is the immutable training trace. Traces 6, 7, and 9 (15, 20, and 30 pA) are immutable held-out tests and are not inspected by the optimizer. The recorded first voltage initializes each independent rollout.
- Mapping from data to model inputs, targets, and observables: Apply the indicated total current during [50, 250) ms and zero otherwise. Compare simulated membrane voltage to the recorded mV trace at every sample. Detect upward crossings of -10 mV with a 5 ms refractory interval for spike-count and timing summaries.
- Known data limitations or unresolved mismatches: The text file contains voltage only, not the command-current waveform, temperature, cell area/capacitance, ionic concentrations, or trial replicates. The 50-250 ms protocol is inferred from voltage responses. The prompt says seven channels but names six current mechanisms; the implementation includes exactly the six named currents. Single-trace fitting cannot identify all conductances uniquely.

## Acceptance boundary
- Evidence required for success, failure, or an inconclusive result: Success requires finite simulations at all four currents, lower training loss than the fixed literature-informed initialization, correct current ordering in held-out spike count (nondecreasing with current), and held-out traces/metrics reported without post-fit tuning. Failure is any invalid simulation, optimizer leakage from held-out traces, or decreasing spike count with increasing current. Otherwise the result is inconclusive.
- Required baselines and controls: Fixed initial-parameter simulation, zero-current quiet-baseline check, parameter encode/decode round trip, State-reset replay check, solver time-step refinement at the selected fit, and synthetic exact-pipeline recovery for a small subset of fitted parameters.
- Invalid-result conditions: Non-finite State/current/loss, incompatible units, current outside the declared window, runtime State carried between independent protocols, fitted values outside bounds, or hidden use of traces 6, 7, or 9 in candidate selection.
- Allowed claims and explicit non-claims: May claim predictive consistency only for the four recorded current protocols and declared metrics. Do not claim unique biological parameter identification, exact reproduction of every equation in Du et al. (2025), or a seven-channel model because the supplied list names only six and the paper text is unavailable in the execution environment.
