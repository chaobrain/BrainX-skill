# NeuroSpecification

- Status: locked
- Researcher approval: The user explicitly requested literature investigation, competing models, a 0%-100% knockdown-fraction sweep, simulations, and discriminating predictions. This specification implements that authorized comparison; it does not claim approval of fitted biological parameters.

## Researcher request
- Brain-modeling question or behavior: Explain increased activity of KCNT2-deficient cells in a mosaic culture but decreased culture activity at near-complete knockdown, with abnormal firing and early depolarization block.
- Requested model, experiment, or comparison: Conductance-based cells with sodium-activated potassium current; explicit recurrent excitatory networks; compare acute channel loss with reduced effective synaptic transmission; retain inhibition as a conditional alternative if inhibitory neurons are present. Model calcium separately from spikes.
- Execution mode: forward-simulation.
- Required outputs: Evidence review, executable models, numerical fraction sweeps, raw observables and controls, mechanism assessment, follow-up experiments.
- Constraints: BrainX-native integration, units, state and projections. Preserve existing files. Use an existing Python environment and CPU. No external writes or paid compute.

## Inspected data contract
- Data sources and inspected contents: User's qualitative observations; primary literature including Boggess et al., Nature Communications, 2026-08-01, DOI 10.1038/s41467-026-75882-0, full published text. No user traces or numeric dataset supplied.
- Shapes, axes, sampling/time base, and physical units: Model time in ms, voltage in mV, current density in uA/cm2, conductance density in mS/cm2; separate condition, neuron and time axes. Knockdown fraction is dimensionless and distinct from remaining conductance per targeted neuron.
- Required preprocessing and the subset used to fit each transform: No data fitting. Exploratory parameter selection is labeled; freeze subsequent validation cases and use independent seeds and nearby parameter values.
- Mapping from data to model inputs, targets, and observables: Current steps test firing and block; recurrent spike-driven conductance tests communication; synthetic calcium/reporting readouts test whether optical and electrical signs differ. No conversion from glutamate concentration to current density is calibrated.
- Known data limitations or unresolved mismatches: User culture identity, inhibitory fraction, knockdown strength and duration, and readout unspecified. Closest paper uses predominantly glutamatergic NGN2 neurons, 21-day CRISPRi cultures, glutamate stimulation and CaMPARI2; its current-clamp validation uses a separate RNP knockout preparation. These are conditional context, not assumed user facts.

## Acceptance boundary
- Evidence required for success, failure, or an inconclusive result: Demonstrate whether reversal exists within explicit model conditions, identify mediators with interventions, and report non-reversing controls. Multiple explanations fitting qualitative observations imply non-identifiability, not mechanism selection.
- Required baselines and controls: Zero knockdown, zero recurrence, low versus high drive, partial conductance loss, fixed topology and heterogeneity across fractions, timestep refinement; matched synaptic rescue and channel rescue. Separate within-culture KD/control contrast from culture mean versus all-control contrast.
- Invalid-result conditions: Nonfinite trajectories, incorrect units or signs, missing baseline, labeling mere silence as block, choosing only favorable outcomes, equating reporter with spikes, claiming a calibrated human-neuron model or a universal critical fraction.
- Allowed claims and explicit non-claims: Mechanistic existence, sufficiency within tested regimes and discriminating predictions. No proof of the culture's mechanism, no disease-treatment conclusion, no quantitative fit to CaMPARI2 or human firing rates.
