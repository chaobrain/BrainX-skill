# BrainX study record

## Scope and package selection

Represent individual isopotential cells and their spike-driven chemical synapses. Use BrainCell plus BrainPy-State, with BrainState and BrainUnit. No BrainMass rate State is introduced. Forward simulations require neither BrainTrace nor fitting.

## Studied implementation route

- BrainCell `SingleCompartment`: density-based capacitance and conductances, deliberate initial voltage, explicit solver, `init_state`/`reset_state`, current-density input, recorded voltage and spike State.
- `braincell.Channel`: custom coupled sodium gates and local sodium sensor plus KNa current, following the root-cell current interface demonstrated in the official fitting script. This combined channel keeps sodium influx and KNa activation in one declared mechanism. Built-in channel catalogue has no KNa mechanism. Use built-in delayed rectifier and leak for the remaining currents.
- BrainCell solvers advance all differential State. Use RK4 with temporal refinement; compare a suitable independent solver for decisive single-cell cases.
- BrainPy `AlignPostProj`: exponential recurrent excitation, concrete synapse and COBA output, projections before cell updates, previous-step spike events. No manually fed-back population firing rate.
- BrainState transforms own timestep loops and whole-rollout compilation. Native population/condition shapes carry independent State. Host loops may orchestrate distinct immutable runs.
- BrainTools `Constant` constructs stimulation protocols under the rollout dt. BrainState randomness generates fixed topology, heterogeneity and nested knockdown ordering. NumPy is limited to serialization and offline scientific statistics.
- Calcium is an explicitly phenomenological observation model driven by voltage and/or synaptic activation. It cannot be inferred from spike counts, nor treated as fitted CaMPARI2 kinetics.

## Execution and acceleration

Read brainx-acceleration and the modeling-loop run/monitor references. Keep timestep execution compiled from the outset. Use native batched cells for current and conductance sweeps, and shared fixed graph/heterogeneity with reset dynamics for fraction comparisons. Record cold versus warm timing and reset-repeat parity. Keep raw voltage, sodium availability and synaptic conductance for mechanism checks.

## Validation and limits

Validate KNa zero at potassium reversal, sodium loading/removal signs, bounded gates, zero-dose equivalence, finite traces, early/late rates, and a block predicate requiring depolarization with loss of sustained spikes. Test dt refinement and matched interventions. Exploratory parameter choices are demonstrations of sufficiency, not measurements of human KCNT2. A reversal is not required to appear in every regime.

## Supplementary boundary checks before review

Read BrainCell's complete BrainTools metric reference and recorded the narrow offline-summary boundary in `brain-tools-api-gap.md`. BrainTools' firing-rate output was checked against a known event train; the runtime returns an array in Hz rather than a Quantity, and smoothing can oscillate around the correct time-averaged rate. This affected only the auxiliary verification adapter, not simulation results.

Reopened the solver comparison reference and executed both coupled exponential-Euler and its timestep refinement, retaining RK4 as the main solver only after agreement. Read the perturbation-validity reference: this study is an explicitly exploratory existence comparison, not a calibrated influence/tuning or confirmatory reproduction experiment. The so-named validation runs assess numerical and seed robustness conditional on selected parameters; they do not erase selection uncertainty.
