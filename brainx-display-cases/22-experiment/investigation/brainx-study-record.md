# BrainX study record

## Scope and sources

Read brainx-general-guard, brainx-modeling-loop, bio-neuro-lit and tool contracts, BrainMass, BrainState, BrainUnit, and brainx-acceleration. Read BrainMass model library, Simulator input/monitor API, parameter sweeps/regime analysis, canonical Wilson-Cowan script, and modeling-loop experiment launch and monitor references. Retrieved official WilsonCowanStep documentation on 2026-09-07; this provides exact constructor, public derivatives, sigmoid, refractory factor, and solver options.

## Implementation design

- Use `brainmass.WilsonCowanStep` with explicit gains, thresholds, weights, refractory factor, and unit-bearing time constants. It has two dimensionless HiddenStates, `rE` and `rI`.
- Its actual sigmoid subtracts its zero-input value. Record this equation and check activity ranges; do not silently replace it with the simpler unshifted sigmoid from the gallery prose.
- `brainmass.Simulator.run` owns initialization, environment, JIT, time loop, inputs, and post-update monitors. Native `in_size` represents independent deterministic conditions, not individual neurons or random trials.
- Use a minimal child-Module wrapper only for scientific coupling or dynamic synaptic changes absent from the stock input channels. Keep all regional dynamics in stock Wilson-Cowan children. Compute all coupling from pre-update State, then update each child; a sequential call must not create asymmetric within-step coupling.
- Somatic inhibition enters population-specific external drive. Presynaptic loss scales every outgoing weight from the affected population, including inhibitory self-coupling. Preserve a monitor of efficacy times activity; firing is not transmission.
- A long-range gate supplies positive excitation to a downstream inhibitory population. Pure output withdrawal must be applied equally for matched Syn and CaMKIIalpha output suppression; failure of that negative control is scientifically informative.
- Package docs support `method="rk4"` and `method="exp_euler"`; compare timestep refinement and independent solver summaries. Use BrainTools constant initializers, and Simulator callable input protocols supported directly by its API.
- Host NumPy is restricted to parameter-grid construction, serialized outputs, statistics, and assessment. Time values cross the boundary only via `.to_decimal(u.ms)`. Population activation remains dimensionless.

## Verification and acceleration

Compare reset replay, single-condition/batched equality, deterministic analytic one-step derivative, input timing, finite/bounded activity, late oscillation persistence, timestep refinement and solver agreement. Explore parameters openly, then freeze selected parameters, thresholds and neighboring validation conditions before confirmation. All raw comparisons and discarded regimes remain available.

The Simulator already compiles the time loop and native condition arrays. Record this as unchanged acceleration unless measurement justifies a rewrite; do not claim speedup from first-call timing.

## Equation check correction before production

The independent derivative check exposed a documentation discrepancy: the official rendered equations call wEI inhibitory-to-excitatory, but the linked public implementation uses wIE in drE and wEI in drI. The executed environment agrees with the latter. Preserve target-first scientific names in the configuration, and swap only the two cross-weight constructor arguments. Default biological weights are I-to-E 15 and E-to-I 12. This preserves the original somatic exploratory trajectories. The initial synaptic variant exploration used incorrect correction weights and is excluded; rerun it under `variants_corrected`. Public source reference: https://brainx.chaobrain.com/brainmass/_modules/brainmass/wilson_cowan.html. No installed package source was inspected.
