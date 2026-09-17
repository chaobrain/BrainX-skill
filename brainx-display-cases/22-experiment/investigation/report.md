# Why inhibitory S2 manipulation can provoke pathological activity

The most plausible explanation is loss of inhibitory restraint within a recurrent circuit, potentially amplified by connected regions. hM4Di inhibits the cells or synapses that express it; if these include inhibitory interneurons, the net network effect can be increased excitatory activity. The current evidence favors this explanation but does not identify which inhibitory cells, whether their firing or transmitter output is decisive, or why the regional difference occurs.

## Established experimental evidence

The closest published match is [Yoshinaga and Sato, 2026](https://doi.org/10.1152/jn.00074.2026), in **rats treated with DCZ**, not mice. It reports the same promoter/region contrast and additionally reports convulsions with interneuron-directed hDlx manipulation in both S1 and S2. This supports inhibitory-circuit involvement and shows that S1 is not absolutely resistant. The authors explicitly leave the mechanism unresolved. Their reported behavioral phenotype should not be treated as a localization of seizure onset. The user's mouse observation remains a separate experimental report; ligand, dose, latency and EEG/LFP data were not supplied.

Related work supports several parts of the causal chain. [Lopez et al.](https://doi.org/10.1523/JNEUROSCI.3682-15.2016) demonstrated promoter-dependent disinhibition in mouse hippocampal slices. [Goldenberg et al.](https://doi.org/10.1093/cercor/bhac245) induced focal cortical electrographic events by chemogenetically suppressing GABAergic neurons, including in barrel cortex. [Stachniak et al.](https://doi.org/10.1016/j.neuron.2014.04.008) showed that hM4Di can suppress transmitter release even when presynaptic action potentials are imposed. None resolves the specific S2 mechanism.

Two cautions materially affect interpretation. [Veres et al.](https://doi.org/10.1523/ENEURO.0070-23.2023) showed that AAV CaMKIIalpha constructs can also function in cortical interneurons, so promoter labels do not prove equivalent or exclusive targeting. [Chang et al.](https://doi.org/10.1523/JNEUROSCI.0994-21.2021) established S2 recruitment of inhibitory interneurons in S1, making a distributed mechanism anatomically plausible without proving it here.

## Status of subsequent research

A targeted Europe PMC search, forward-citation check, author/title search and web recall search through **2026-09-07** found no subsequent study that mechanistically resolves this exact contrast. The original article had zero indexed citing works at retrieval. This is a bounded search result, not proof that no unpublished or unindexed work exists.

The related [Wu et al. 2025 preprint](https://doi.org/10.1101/2025.10.02.676593) reports epileptiform discharges and occasional seizures after CaMKIIalpha-hM4Di suppression in mouse dorsal cortex. It precedes the matching 2026 report and is not a subsequent S2 resolution. Its full discussion proposes disfacilitation of interneurons, altered intrinsic properties and homeostatic adaptation without isolating them. The retrieved version remains a preprint. Its observation means that a negative CaMKIIalpha control is protocol-specific; it does not establish that excitatory-cell suppression can never be ictogenic.

Reading depth and source limitations are recorded in [literature-review.md](literature-review.md).

## Competing mechanisms

| Mechanism | Why it could explain the phenotype | What remains assumed |
|---|---|---|
| Local interneuron silencing | Loss of inhibitory recruitment outweighs direct suppression of E cells; residual excitation becomes unstable | Relative E/I suppression, affected inhibitory subclasses, and whether S2 is intrinsically more susceptible |
| Reduced inhibitory synaptic efficacy | Inhibitory output per spike falls even if I firing is preserved or later recruited upward | Magnitude/selectivity of S2 synaptic suppression; hM4Di may simultaneously affect soma and terminals |
| Different effective targeting in S1 and S2 | Same construct acts on different layers, cell mixtures or functional targets | Region-specific coverage and receptor coupling must be measured; fluorescence alone is insufficient |
| Loss of remote feedforward inhibition or returning feedback | S2 drives inhibitory circuits in other regions; disrupting those pathways can destabilize a broader circuit | Where the first event begins, which projection matters, and why equally effective CaMKIIalpha output suppression would be spared |
| Suppression-induced synchronization, rebound or adaptation | Reduced excitation can move a nonlinear network into an oscillatory regime; intrinsic or slow compensatory changes may contribute | Specific HCN, rebound, ionic or homeostatic mechanisms are not demonstrated for this phenotype |

The first three are overlapping explanations at different levels, not mutually exclusive diagnoses. Off-target expression, ligand effects and non-epileptic movements also require empirical controls. Regional/promoter specificity weakens a simple drug-only explanation, but does not eliminate interactions with the manipulated circuit.

## Model assumptions

I implemented a two-population Wilson-Cowan model in BrainMass. It is the simplest useful abstraction for asking whether suppression of E and I, or suppression of their outgoing synapses, can change a stable population state into sustained oscillations. A driven receiving E/I circuit tests the pure remote-gate hypothesis. All equations, parameter values, initialization and intervention definitions are in [model-methods.md](model-methods.md).

The illustrative S2-like circuit has recurrent E gain 14 versus 12 in the S1-like circuit. These are **assumptions, not measured regional anatomy**. A second explanation instead uses identical circuitry with weaker effective inhibitory targeting in S1. A synaptic explanation reduces E output by 10% and I output by 50%. These values were selected after exploratory runs, then tested in a frozen neighborhood. No biological fitting, drug-dose conversion or inference of receptor sensitivity was performed.

The proxy requires E peak-to-trough amplitude >0.1 and at least three oscillation cycles in each of two late one-second windows, with a stable matched baseline. This measures persistent population oscillations. It does **not** reproduce recorded EEG waveforms, individual-neuron synchrony, seizure onset/termination, or convulsive behavior. The six-second model protocol is not an estimate of hM4Di's pharmacological latency.

## Model-derived results

The numbers below are dimensionless E peak-to-trough amplitudes in the final analysis window. All central baselines and trajectories passed the specified checks.

| Model | Syn-S2 | CaMKIIalpha-S2 | Syn-S1 | Interpretation |
|---|---:|---:|---:|---|
| Somatic inhibition, regional circuit difference | 0.296 | 0 | 0 | Reproduces qualitative instability/control pattern |
| Somatic inhibition, regional targeting difference | 0.296 | 0 | 0 | Same phenotype without any circuit difference |
| Synaptic inhibition, regional circuit difference | 0.383 | 0 | <0.000001 | Also reproduces pattern |
| Pure withdrawal of remote inhibitory recruitment | 0.366 | 0.366 | 0.140 | Fails both negative controls in this tested regime |

Interneuron-only suppression produced oscillations in both modeled regions, consistent with the additional hDlx observation. Sparing the inhibitory intervention prevented oscillations in the successful models while retaining E suppression; this is a **prevention control**, not a simulated post-onset rescue. Equalizing either the assumed regional circuit difference or the assumed targeting difference removed regional selectivity in the corresponding model.

In the somatic circuit model, mean E activity rose from 0.0647 to about 0.0995, but late I activity also increased. In the synaptic model, late I firing and total inhibitory output both increased despite weaker output per spike. Thus even an increase in measured late I activity does not exclude an initiating loss of inhibitory restraint. Direct cellular suppression, per-spike synaptic efficacy and recurrently recruited total activity are distinct quantities.

The E-only counterexample starts from a high stable state and develops oscillations of amplitude 0.424 when E drive is reduced. Mean E activity falls from 0.448 to 0.284. This establishes that reduced mean activity can coexist with large oscillations in a minimal nonlinear circuit. It neither reproduces the S2/S1/promoter contrast nor validates a cellular rebound or excess-inhibition mechanism in the preprint.

The three-condition pattern persisted at **13/15** sampled somatic/circuit settings, **8/9** synaptic settings and **3/3** targeting settings. These are grid counts, not biological probabilities or confidence intervals. The failed synaptic setting crossed into a different regime and also destabilized S1; the two failed somatic settings lacked a sufficiently large S2 oscillation. All outcomes are retained.

Halving dt from 0.2 to 0.1 ms preserved all 57 central classifications; maximum change in late mean E was 0.000014. A second solver also preserved every classification; maximum mean difference was 0.000924 and amplitude difference 0.00732. Focused tests verified derivatives, correct weight-direction mapping, State reset, condition independence and stock-model parity. See [assessment.json](assessment.json), [validation.json](validation.json), and [central metrics](runs/central/metrics.csv).

These results demonstrate **non-identifiability from the supplied phenotype**. They establish multiple sufficient model mechanisms and a constraint on pure output withdrawal. They do not assign posterior probabilities or identify the biological cause. Richer corticocortical or thalamocortical mechanisms remain possible because the remote model contains only a prescribed withdrawal of feedforward drive.

An [independent computational review](reviews/iteration-1.md) passed these scoped claims and independently reproduced the raw-data metrics, sensitivity counts and artifact hashes. Its fresh derivative/parity execution was blocked by its read-only cache; the successful local tests are archived. This numerical review does not validate the biological assumptions or literature ranking.

## New predictions and decisive experiments

The most informative follow-up combines **cell-type-resolved early recordings and functional synaptic measurements**, rather than another promoter-only comparison.

1. **Locate initiation and test inhibitory necessity.** Record synchronized video/EMG and S2/S1/motor-area unit/LFP activity before and during ligand exposure. Preserve or restore local inhibitory output while maintaining matched E suppression. Local disinhibition predicts a local loss of inhibitory efficacy/recruitment before the first E burst; a remote gate predicts the decisive change in the receiving circuit. Combine timing with intervention, since suppressing propagation alone does not locate initiation.
2. **Separate soma from synapse.** In S1 and S2, measure identified inhibitory-cell firing under controlled inputs and measure I-to-E currents while imposing identical presynaptic spike trains. A somatic mechanism predicts output recovery when spikes are restored; presynaptic loss predicts reduced inhibition per spike despite restored firing. Test E-to-I transmission as well. Mixed effects are an informative outcome.
3. **Match functional perturbation across regions.** Equalize measured E suppression and inhibitory efficacy loss, including affected layers/cell fractions. Targeting-only models predict that regional thresholds converge; circuit-susceptibility models predict persistent S2 vulnerability. A regional effect limited to intact networks motivates a data-constrained distributed model.

Do not infer successful S1 inhibition merely from absence of convulsions: successful models predict that S1 mean E activity may rise without oscillatory instability. Rebound, HCN or slow adaptation models become worthwhile if the recordings show the corresponding temporal and intrinsic signatures; they are not required to generate the current qualitative pattern.

Detailed protocols, discriminating outcomes and confounds are in [follow-up-experiments.md](follow-up-experiments.md).

## Reproducible artifacts

- [models.py](models.py): equations implemented through stock BrainMass dynamics.
- [experiment.py](experiment.py), [configs](configs/central.json), [artifact-manifest.json](artifact-manifest.json): runner and immutable run identity.
- `runs/central`, `runs/sensitivity`, `runs/half_dt`, `runs/solver`: complete raw E/I/activation trajectories, metrics, configurations, environment and source snapshots.
- `runs/exploration`, `runs/variants`, `runs/variants_corrected`: exploratory history; initial synaptic variants are excluded because of the documented cross-weight mismatch.

Use the existing interpreter `/home/yixinliu/anaconda3/envs/braincell-released/bin/python` with `JAX_PLATFORMS=cpu`, `MPLCONFIGDIR=/tmp/s2-mpl` and `XDG_CACHE_HOME=/tmp/s2-cache`. Run `test_models.py` for focused verification and `experiment.py --config configs/central.json --run-id <new-name>` for an immutable replication. No dependencies were installed or modified.
