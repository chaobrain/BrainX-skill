# Experiments that distinguish the mechanisms

## First resolve the measurement and comparison

Record exactly what increased or decreased: recruitment/rheobase, slope of the firing-current curve, sustained spike rate, network bursts, glutamate-evoked calcium, or CaMPARI2 conversion. Keep three estimands separate:

1. KD versus control cells receiving matched direct current, with recurrence controlled.
2. KD versus neighboring control cells in the same mosaic culture.
3. Mean activity per viable neuron in a mosaic or all-KD culture versus an all-control culture.

Neither comparison 1 nor 2 determines comparison 3. Obtain absolute readouts as well as within-well normalized values. Preserve cells that cease spiking in the denominator rather than analyzing only active units. Track initially identified neurons longitudinally to avoid sampling a surviving or patchable subpopulation.

Use an isogenic mixing series at 0, 25, 50, 75, 90 and 100% targeting initially; refine around a transition only after observing one. Keep total neuron density, culture age, glia, medium exchange, viral exposure and per-target-cell KCNT2 loss fixed. Measure protein/channel loss per cell, not only mean RNA per well. Randomize wells across batches, blind analysis and treat independent differentiations/cultures as biological replicates; cells within one well are not independent replicates. Include multiple independent guides, nontargeting guides and a CRISPR-resistant rescue with near-endogenous expression. Check viability, passive properties, neuronal maturation and reporter expression.

## Highest-information experiment: simultaneous voltage, spikes and calcium with acute relief

In tagged KD and untagged neighbors, record membrane voltage or calibrated voltage imaging together with calcium and population spikes during the **same** stimulus used for the phenotype. Use current clamp for a subset to establish actual voltage and spike amplitude. If reproducing the published CaMPARI2 observation, preserve its glutamate/light timing while also sampling the initial response and later steady state. Pair reporter experiments with a continuous calcium indicator and electrical measurements because photoconversion is cumulative and cannot resolve recovery by itself.

Sweep stimulus intensity from recruitment through the failure range. In an apparently silent KD neuron, test a brief hyperpolarizing pulse and a sustained reduction of drive; compare with a matched hyperpolarization in control cells. Add a calibrated outward conductance by dynamic clamp, or use genetic channel rescue. A generic outward conductance rescue identifies a membrane-state bottleneck, not KCNT2-specific gating by itself.

| Observation or intervention | Block/output failure dominant | Transmission weakening dominant without block | Chronic adaptation dominant |
|---|---|---|---|
| Voltage during low output | Sustained depolarization, diminished AP amplitude, reduced sodium availability; calcium may remain high | Repolarizing spikes or relatively normal/resting voltage despite weak recruitment of neighbors | May be normal/resting or reflect altered intrinsic currents; depends on adapted variable |
| Lower excessive drive | Can increase sustained spike output by leaving block; calcium can decrease while spikes recover | Usually reduces directly evoked spikes unless an additional overdrive mechanism is present | Usually insufficient for an immediate full reversal of a structural/synaptic deficit |
| Brief hyperpolarization | Transient recovery followed by reblock at unchanged excessive drive is expected | No special recovery of transmission if presynaptic APs were already intact | Does not immediately reverse synaptic scaling or remodeling |
| Restore outward conductance acutely | Recovery on membrane/gating timescales is possible | Does not repair an independent release or connectivity deficit | May restore intrinsic response but leave adapted connectivity impaired |
| Equal presynaptic spike trains | Transmission can normalize if the deficit was solely spike generation; axonal propagation failure may remain | Responses remain weak or unreliable after accounting for presynaptic spike count and waveform | Persists if synapses have been remodeled; timing and molecular measures then separate origins |

Assess sodium availability with an appropriate voltage-clamp protocol where feasible; waveform amplitude and hyperpolarization recovery are proxies, not direct measurements of h. A voltage plateau alone is not proof of sodium inactivation, since other ionic mechanisms can sustain depolarization.

**Why this comes first:** it distinguishes low firing caused by overdrive from low firing caused by inadequate input, and directly tests whether calcium and spikes carry opposite signs. The readouts are jointly much more informative than another population average.

## Second experiment: separate spike generation from communication

Measure unitary or optically evoked excitatory responses while controlling the presynaptic spike train and recording presynaptic voltage when feasible. Quantify release failures, postsynaptic charge, paired-pulse behavior, miniature EPSC amplitude/frequency, and synapse density. A lower miniature frequency alone cannot distinguish fewer synapses from lower release probability. Measure axonal propagation or terminal calcium if somatic spikes persist but output fails.

Test the receiving and sending sides separately: activate control senders onto KD receivers, KD senders onto control receivers, and the two homotypic pairs. This 2-by-2 design separates a sending defect from a postsynaptic response deficit. Replay matched synaptic conductances by dynamic clamp to compare intrinsic responses independently of network input.

Suppress vesicular recurrence, with suitable matched controls, while retaining a calibrated direct stimulus. Do not simply add AMPA antagonists during a glutamate assay and interpret loss of the phenotype as selective loss of recurrence: that also changes the direct stimulus. Conversely, excessive optical stimulation can itself cause block, so match actual voltage/spike output, not only delivered light.

Prediction specific to a primary/effective synaptic deficit: diminished postsynaptic responses persist at matched presynaptic spike trains, and restoring input/output efficacy can rescue network recruitment without fixing the altered intrinsic firing-current curve. Prediction specific to activity-dependent output collapse: synaptic failure tracks deteriorating spike waveform and recovers promptly when the sender is brought out of the depolarized state.

## Third experiment: distinguish fast effects from adaptation

Compare perturbation after neuronal maturation with perturbation throughout differentiation, and follow channel protein, firing, calcium, synaptic efficacy and intrinsic currents over time. Include a rapid, independently validated conductance manipulation where possible; an inducible RNA knockdown is not instantaneous channel removal. Measure the kinetics of protein depletion before interpreting a delayed phenotype as homeostasis.

Predictions: a predominantly acute channel/block mechanism emerges with channel loss and tracks stimulus intensity immediately. A compensation/remodeling mechanism shows a later change in synaptic amplitude, release, connectivity or other intrinsic conductances after an initial cellular effect. High calcium with low firing could maintain a calcium-sensed compensatory response even when spike output is already below control. This sensor-mismatch explanation is a new hypothesis requiring direct testing, not an established KCNT2 result.

Use an activity-matching intervention during the period of knockdown to ask whether preventing the initial excess activity/calcium prevents the later suppression. Match the candidate controlled variable: matching spikes while leaving calcium high does not test calcium-driven homeostasis. An ideal homeostatic controller need not overshoot; absence of a delayed undershoot does not falsify all homeostatic involvement.

## Conditional tests and confounds

If substantial inhibitory neurons are present, quantify their identity and chloride reversal/maturity, then use excitatory-only, inhibitory-only and combined KCNT2 targeting. Measure inhibitory output, not just interneuron recruitment. GABA receptor blockade alone is not decisive because disinhibition can drive excitatory cells into block. Do not introduce this mechanism into a verified glutamatergic-only culture.

Check extracellular potassium, medium/glutamate handling, ATP/metabolic stress, calcium extrusion, receptor expression and cell survival if voltage/calcium changes are not explained by the two main routes. These are unmodeled possibilities, not positive KCNT2 evidence. For a reporter-specific phenotype, calibrate conversion against calcium exposure across genotypes and verify that phototoxicity, saturation and expression differences do not create the sign change.

## Decision criterion

Favor block as the dominant immediate cause only if low sustained output coincides with depolarized/low-availability states and rapid physiological relief restores output under matched conditions. Favor an independent synaptic mechanism if communication remains weak with intact, matched presynaptic spikes. Favor adaptation as the source of that weakening only with a causal time course or prevention experiment. More than one route may operate; estimate their contributions rather than requiring a single mutually exclusive label.
