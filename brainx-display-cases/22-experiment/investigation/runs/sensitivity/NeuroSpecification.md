# NeuroSpecification

- Status: locked
- Researcher approval: The user explicitly requested mechanism comparison, simplest appropriate models, simulations, predictions, and follow-up experiments. This specification implements that authorization; parameters are illustrative assumptions, not inferred experimental measurements.

## Researcher request
- Brain-modeling question or behavior: Syn-hM4Di in S2 produces seizure-like activity and convulsions, whereas CaMKIIalpha-hM4Di in S2 and Syn-hM4Di in S1 do not.
- Requested model, experiment, or comparison: Compare loss of inhibitory-cell activity, loss of inhibitory synaptic efficacy, and withdrawal of long-range recruitment of inhibition. Separately test regional circuit susceptibility versus regional intervention efficacy.
- Execution mode: forward-simulation.
- Required outputs: Literature synthesis through 2026-09-07, executable models, preserved configurations and trajectories, numerical checks, scoped conclusions, and discriminating experiments.
- Constraints: Use the simplest aggregate E/I models able to test these alternatives. No fitted biological parameters, channels, individual neurons, behavior generator, or EEG forward model. CPU, existing BrainX environment; no installation.

## Inspected data contract
- Data sources and inspected contents: User's three qualitative conditions; matching 2026 rat/DCZ report (PMID 41962964) and targeted related literature. The user specifies mice; retain that as their experimental species without treating the rat report as a mouse replication.
- Shapes, axes, sampling/time base, and physical units: No raw recordings provided. Simulations use time in ms, condition axes, and dimensionless population activity; no conversion to measured Hz, EEG volts, drug concentration, seizure severity, or dosing latency.
- Required preprocessing and the subset used to fit each transform: None; no data fitting.
- Mapping from data to model inputs, targets, and observables: Syn targets both E and I; CaMKIIalpha is an idealized E-dominant intervention; hDlx is an idealized I intervention used as an additional literature control. Record E, I, effective inhibitory output, and supplied input throughout each intervention.
- Known data limitations or unresolved mismatches: Ligand, dose, expression spread, targeting fraction, electrophysiological verification, and onset latency are unknown for the user's mice. Published promoter targeting is not equivalent to matched cell-type perturbation. S2 manipulation does not establish S2 seizure onset.

## Acceptance boundary
- Evidence required for success, failure, or an inconclusive result: A mechanism is sufficient in a phenomenological regime only if its matched baseline is stable and its declared intervention produces sustained large population oscillations or high activity, with all negative controls assessed. Multiple successful regimes imply non-identifiability, not equally likely biology.
- Required baselines and controls: Baseline, Syn-S2, CaMKIIalpha-S2, Syn-S1, hDlx-S1/S2, I-restoration, E-only suppression, equalized regional circuit/targeting, and removed long-range gate. Preserve negative outcomes.
- Invalid-result conditions: Nonfinite results, inconsistent time axes, cross-condition State leakage, unstable control mislabeled as healthy, numerical instability masquerading as a mechanism, or outcome-based parameter selection described as a prediction.
- Allowed claims and explicit non-claims: Exploratory demonstration and neighboring-parameter validation of network mechanisms. Large population oscillations are an epileptiform proxy, not a simulated electroclinical seizure or convulsion. No mechanism identification, biological parameter estimates, regional anatomy inference, or model selection probability.
