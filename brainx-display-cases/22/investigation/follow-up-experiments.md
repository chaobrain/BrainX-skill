# Follow-up experiments and new predictions

These are proposed tests, not established findings. The model parameters are illustrative and do not specify animal doses, expected seizure frequency, or biological effect sizes.

## Highest-information experiment: local versus remote initiation, with a calibrated inhibitory rescue

Record synchronized video, EMG and local multiunit/LFP activity in S2, S1 and a connected motor area during the same Syn-hM4Di protocol. Include thalamic recordings if cortical timing implicates a thalamocortical loop. Sample transduced core and border where feasible. Identify inhibitory cells by validated genetic optotagging or an equivalent cell-identity method, rather than spike waveform alone. Measure from before ligand exposure through the first abnormal event; late ictal activity cannot identify the initial perturbation.

In a matched arm, spare inhibitory cells from hM4Di expression or use an independently controlled, calibrated intervention to restore their normal inhibitory output while keeping the excitatory hM4Di perturbation. For example, a validated Cre-off hSyn construct in a GABAergic driver line can test the necessity of direct inhibitory-cell hM4Di expression. Check that E targeting, layer coverage and E suppression remain comparable after changing constructs. For a physiological rescue, restore near-baseline inhibition rather than imposing a large synchronous interneuron volley. Verify synaptic output: restoring somatic spikes alone may not reverse presynaptic Gi/o suppression.

| Hypothesis | New predicted observation | Most useful falsifying/qualifying result |
|---|---|---|
| Local inhibitory loss initiates the event | Reduced inhibitory efficacy/recruitment precedes the first local E burst; sparing/restoring local inhibition prevents initiation despite retained E suppression | Events begin remotely before local dysregulation, and persist when local inhibition is demonstrably normalized |
| S2 suppression withdraws remote feedforward inhibition | A receiving area's inhibitory drive falls before its first population burst; restoring the affected remote inhibitory recruitment prevents that burst | No change in the receiving inhibitory pathway before onset, or the earliest local event persists when the proposed route is interrupted |
| Combined local and distributed mechanism | Local perturbation changes first, but maintained ictal activity requires a returning pathway | Isolated local circuit expresses the same instability without the proposed feedback under appropriately preserved drive |

Seizure termination after remote interruption alone is insufficient evidence of remote initiation: the intervention may block propagation or maintenance. Likewise, seizure prevention by local inhibition alone is not proof of the original cause, since increasing inhibition can suppress several pathways. Combine onset order, measured mediator changes, and pathway-specific necessity tests.

## Distinguish fewer inhibitory spikes from weaker inhibitory synapses

In matched S1 and S2 preparations exposed to the same ligand, identify hM4Di-positive PV, SST and additional GABAergic subclasses alongside excitatory neurons. Measure two effects separately:

1. Somatic input-output curves, membrane potential and firing under controlled drive with recurrent synaptic effects isolated.
2. Inhibitory-to-excitatory transmission while imposing the same presynaptic action-potential train before and after ligand. Measure unitary/evoked IPSC amplitudes, failures and paired-pulse behavior; record presynaptic spike fidelity. Also measure E-to-I transmission, since loss of excitatory recruitment can cause indirect disinhibition.

Somatic-dominant model: a direct inhibitory-cell firing shift occurs, but inhibition per matched presynaptic spike is preserved in the idealized limit. Restoring the spike train should restore its output. Synaptic-dominant model: inhibition per spike decreases despite restored spike count, so a somatic rescue can fail. Mixed effects are likely in real hM4Di neurons and should be quantified rather than forced into a binary label. A fall in postsynaptic IPSC amplitude alone also allows postsynaptic changes; failures, paired-pulse measurements, quantal observations and a postsynaptic GABA response control help localize the change.

The model's late interneuron activity rises in both successful mechanisms. Therefore, increased late I firing does not falsify an initiating loss of inhibitory restraint. The decisive quantity is functional transfer under controlled input, not the sign of an uncontrolled late average.

## Distinguish regional susceptibility from regional targeting

Use the physiological measurements to match functional E suppression and inhibitory efficacy loss between S1 and S2. Match layer coverage and the fraction of each cell type affected, not just viral fluorescence or injected volume. Then measure the threshold for local epileptiform activity across a graded perturbation range with identical baseline and sensory-drive controls.

- Targeting-only explanation predicts that regional thresholds converge after the effective perturbation is matched. In the simulation, equalizing the intervention removes the S1/S2 distinction.
- Circuit-susceptibility explanation predicts that the S2 threshold remains lower after matching functional effects. The simulation realizes this using recurrent E gain, but the biological explanation could instead involve interneuron recruitment, inhibitory timing or connectivity.
- A remaining regional difference only in intact animals, weakened in appropriately matched isolated preparations, would motivate an explicit distributed model. A negative slice experiment is not conclusive if long-range tonic drive or the relevant layers were lost.

In the illustrative models, Syn-S1 can show elevated mean E activity without sustained oscillations or behavioral seizures. Thus the absence of convulsion should not be treated as proof that S1 inhibition worked as intended.

## Conditional tests for suppression-induced synchronization and slower mechanisms

Measure both mean firing and burst synchrony; a lower mean can coexist with stronger population oscillations. Use multiple activation strengths and follow onset, sustained exposure, and washout. A rebound mechanism predicts a dependence on prior suppression and release from it, whereas the successful local models generate sustained oscillations during maintained activation without a rebound channel or homeostatic State.

If identified excitatory neurons remain directly suppressed while neighboring untreated neurons initiate bursts, test loss of E-to-I recruitment and core/border organization. If the effect persists with matched inhibition and isolated intrinsic excitability changes are detected, then measure sag/Ih, input resistance, adaptation or rebound responses and build a channel/adaptation model constrained by those measurements. Current two-population simulations neither establish nor exclude HCN, T-type calcium, chloride, potassium or homeostatic mechanisms.

## Controls that change interpretation

Use ligand-only receptor-negative controls, reporter-only vector controls, saline within the relevant expression group, and verified constructs/lots. Map expression throughout nearby insular, auditory and motor cortex and across layers. Validate actual CaMKIIalpha cell-type specificity and excitatory suppression. If CNO is used, account for clozapine conversion; this is a conditional confound, not an explanation of the published DCZ result. Confirm that abnormal movements have a corresponding evolving electrographic event before treating them as seizures.

Prioritize the synchronized onset/mediator/rescue experiment and the matched-spike synaptic assay. A further uncalibrated promoter comparison or c-Fos measurement would leave the principal alternatives unresolved.
