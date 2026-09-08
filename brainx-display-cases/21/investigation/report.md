# KCNT2: interpretation and simulation assessment

## Main answer

There is no requirement for intrinsic recruitment, sustained firing, calcium entry and network communication to move in the same direction. KCNT2 loss can remove an outward brake at modest input while removing repolarizing reserve needed for sodium-channel recovery at strong input. A highly depolarized, poorly spiking neuron can have a large calcium signal but be a poor source of recurrent excitation. A different explanation is an intrinsically more responsive neuron that has weaker effective synaptic output. Both mechanisms are possible; the endpoint observation does not identify which is responsible.

The closest recent primary study is [Boggess et al., Nature Communications, August 2026](https://www.nature.com/articles/s41467-026-75882-0). It reports the matching KCNT2 finding, but its fraction comparison is optical and compares different estimands: KD versus neighbors in a mosaic, and an all-KD well versus a control well. It does not establish a full electrical dose-response curve. Direct current-clamp evidence supports abnormal high-drive firing and early block; the proposed network explanation is not directly established. See `literature-review.md` for the full evidence map and reading-depth limitations.

## Established evidence versus assumptions

**Established in relevant experiments:** KCNT2 is an outward sodium-activated potassium conductance; loss can increase recruitment in some neuronal preparations; high-input firing failure occurs in the matching recent study. Context matters: sensory-neuron results do not establish every human excitatory-neuron phenotype. KCNT1 GOF literature supplies examples of cell-type effects and adaptation, not direct evidence for KCNT2 KD.

**Assumed in these models:** a local sodium sensor, HH-type membrane dynamics, particular potassium reserve, fixed excitatory connectivity, and an effective synaptic defect in the alternative model. The optical mapping combines voltage- and synaptic-associated calcium proxies with an unknown weight. No human-neuron parameters, glutamate-to-current conversion, calcium channel kinetics or reporter calibration were fitted. Model selection was exploratory; held-out random seeds test conditional reproducibility, not biological validity or unbiased confirmation of a discovered mechanism.

## Model-derived conclusions

### Single-cell input-output curves cross

In the selected conductance model, the same loss of gKNa changes sign with input:

| Current density, uA/cm2 | Control rate, Hz | KD rate, Hz | KD late state |
|---|---:|---:|---|
| 20 | 66 | 106 | Sustained firing |
| 120 | 158 | 4 | Depolarized plateau near -36.5 mV; no late spikes |

Rates count the entire 500-ms step, so the 4 Hz entry consists of early events, not sustained firing. Sodium availability during late KD block is approximately 0.035. At high drive some cells first lose sufficiently large events while still oscillating; that intermediate state is **not** classified as stable block.

In an acute intervention, three identical blocked cells initially have zero sustained firing. Reducing drive restores 105 Hz; restoring outward gKNa restores 160 Hz; no intervention leaves zero firing. These are model predictions on gating timescales, not observed experimental rescue rates. See `runs/acute_rescue_native/summary.json` and raw traces; `rescue-parity.json` verifies exact equivalence to the original rescue after a native-input API correction.

### Fraction sweeps distinguish possibilities, not the biological cause

Each validation model sweeps 0%-100% targeting in 10% increments in three independent 40-cell networks. Topology, input heterogeneity, density and per-target loss are matched across fractions within a seed. All values below are illustrative; simulation-seed variability is not biological uncertainty.

Population summaries and the complete optical-weight scan are generated in `fraction-summary.csv` and `assessment.json`. The mean whole-step population rates across three seeds are:

| Model and current density | 0% KD, Hz | 50% KD, Hz | 100% KD, Hz |
|---|---:|---:|---:|
| Channel loss, moderate drive (20) | 69.6 | 92.0 | 113.2 |
| Channel loss, strong drive (120) | 66.6 | 34.5 | 7.8 |
| Transmission loss, gKNa=5 and 90% outgoing impairment (10) | 70.9 | 83.4 | 89.7 |
| Transmission loss, gKNa=2 and complete outgoing impairment (10) | 91.6 | 96.2 | 86.9 |

The different rows deliberately test different operating regimes, not alternative fits with common estimated parameters. At strong drive, 85% of mosaic KD cells and 81.7% of all-KD cells meet the late-block criterion; the control population's fraction is 0.8%. Early events and a minority of nonblocked cells explain the nonzero all-KD rate.

At alpha=0.5, the strong-drive block model's mosaic KD and neighboring-control optical signals are 0.853 and 0.770, while the all-KD population is 0.674 relative to the all-control well's 1.000. The moderate transmission-loss model gives 0.980 versus 0.805 within the mosaic and 0.716 in the all-KD well, **despite increased all-KD spiking**. Optical units are synthetic, not CaMPARI2 conversion ratios.

The channel-loss model predicts strong-drive loss of effective spikes and synaptic activation, with higher voltage-associated calcium in deficient cells. It can reproduce a higher **optical** signal in mosaic KD cells and a lower population optical signal. It does not predict higher sustained KD firing than its control neighbors at that same strong drive. At low drive, firing rises with increasing loss instead. This distinction matters if the user's mosaic phenotype is electrophysiological rather than optical.

The moderate effective-transmission-loss model increases spikes across the fraction sweep while reducing synaptic input and the mixed optical signal. Thus an optical decrease does not establish an electrical decrease, even when there is no block.

The strong effective-transmission-loss model also generates a genuine rise-then-fall in mean spike rate, without block. For the validated current-10 regime, mean population rates at 0%, 50% and 100% targeting are 91.6, 96.2 and 86.9 Hz respectively. At 50%, KD cells fire at 102.1 Hz versus 90.2 Hz in their neighbors. This counterexample assumes complete loss of targeted outgoing transmission; restoring even 10% output removes the below-control electrical endpoint in the tested seed, while full transmission rescue raises all-KD firing to 104.3 Hz versus a 92.2 Hz control. A mere decline in synaptic efficacy is therefore not automatically sufficient for electrical reversal.

### What makes reversal possible

For uncoupled cells at fixed input, the expected mean is `(1-f)*R_control + f*R_KD`. Increasing the fraction only reweights two fixed outputs. A new non-monotonic fraction curve requires feedback, altered transmission, state/history dependence or changing measurement contributions. In a recurrent culture, intrinsic responsiveness and the effective excitation delivered by the population can change in opposite directions.

For the optical model, define `C_alpha = alpha*(CV/CV_control) + (1-alpha)*(CS/CS_control)`, where CV and CS are voltage and synaptic calcium proxies and both denominators come from the same all-control well. The two experimental signs require **both** `C_KD(0.5)>C_control(0.5)` and `C_allKD(1)<1`. These inequalities are evaluated separately for every seed and drive over alpha=0...1. A voltage-only proxy need not decrease during block. The fall in population calcium requires sufficient loss of its synaptic component in the block model. This is a testable, unmeasured assumption, not a fitted reporter mechanism.

The block-model optical signs coexist for approximately alpha=0.35-0.70 across all three high-drive settings (110, 120, 130) and all three seeds. At current 120 alone the common sampled interval is 0.287-0.772. The moderate transmission-loss model's common interval is 0.205-0.733. At low drive the block model does not reverse for any tested alpha. Restoring transmission in the moderate-loss model changes the all-KD optical signal from 0.725 to 1.389 in the matched seed. Retaining half the targeted KNa conductance attenuates, but does not abolish, the strong-drive optical reversal in the tested seed. These are conditional parameter sensitivities, not estimates of channel abundance or optical mixture in a culture.

## Competing explanations and predictions

| Hypothesis | Priority from current evidence | New discriminating prediction |
|---|---|---|
| High-drive membrane/output failure with calcium/network interaction | Leading explanation for the matching optical result because block is directly observed | Low output accompanies depolarization and poor recovery of sodium availability; reducing excessive drive or adding outward conductance rapidly restores spikes, potentially while lowering calcium |
| Primary/effective transmission impairment | Plausible competing network mechanism; direct KCNT2 evidence incomplete | Weak postsynaptic responses persist after presynaptic spike count and waveform are matched; transmission rescue can restore network activity without correcting intrinsic firing-current curves |
| Delayed homeostatic or developmental change | Plausible for chronic knockdown, but not established | Delayed synaptic/intrinsic changes follow the initial perturbation; preventing the relevant excess activity/calcium during adaptation prevents later suppression |
| Increased inhibitory recruitment | Conditional on a verified inhibitory population | Cell-type-specific targeting and inhibitory output measurements identify the suppressive population; not a default explanation for glutamatergic NGN2 cultures |
| Reporter, viability or maturation effect | Must be controlled, especially with guide-specific signals | Optical/electrical mismatch persists without predicted membrane/synaptic changes, or follows reporter calibration, surviving-cell count or maturation differences |

Ordinary homeostatic negative feedback tends to restore its controlled variable. Below-control firing needs an additional condition such as overshoot, a shifted set point, structural change or sustained calcium despite reduced spiking. The model's imposed synaptic weakening does not demonstrate that homeostasis generated it. More than one mechanism may coexist.

## Most informative next experiment

Use tagged isogenic mosaic cultures and record voltage, spikes and calcium simultaneously under the exact phenotype-inducing stimulus, across both fraction and drive. In silent cells, test transient hyperpolarization, sustained drive reduction and outward-conductance rescue. Then measure communication with controlled presynaptic spike trains and separate KD sender/control receiver from control sender/KD receiver pairs. These two experiments distinguish overdriven cells from weak communication far better than another population reporter endpoint.

Follow with a mature-neuron inducible perturbation/time course, multiple guides and near-endogenous genetic rescue. Control density, survival, glia, per-cell knockdown, culture maturity and reporter response. Do not interpret AMPA blockade in a glutamate assay as a selective recurrence manipulation. Detailed protocols, confounds and decision criteria are in `follow-up-experiments.md`.

## Claim-evidence matrix

| Claim | Evidence artifact | Status and permitted inference |
|---|---|---|
| Matching published phenotype includes an optical/electrical distinction | `literature-review.md`; primary study link above | Established study context, conditional on relevance to user's assay |
| KNa loss can increase low-drive firing and reduce sustained high-drive firing | `runs/cell_fast/raw.npz`, `metrics.csv`; solver comparisons | Demonstrated in selected illustrative membrane model |
| Acute relief can restore a blocked cell | `runs/acute_rescue_native/raw.npz`, `summary.json`; `rescue-parity.json` | Model-derived rescue prediction; not an experimental observation |
| Fraction-dependent optical contrasts can arise by distinct mechanisms | `assessment.json`, fraction runs and source snapshots | Conditional sufficiency; unknown observation weight and parameters |
| Mean spike rate can rise then fall without block | `runs/release_strong_validation/metrics.csv`, raw files | Conditional counterexample requiring strong imposed transmission loss |
| Weaker transmission always produces electrical reversal | `runs/release_strong_partial/metrics.csv`; `runs/release_validation/metrics.csv` | Refuted within tested regimes |
| Homeostasis, inhibition or a primary KCNT2 synaptic defect causes the user's result | No direct causal dataset supplied; no such dynamics fitted | Unresolved; not demonstrated by these simulations |
| A biological critical fraction or quantitative CaMPARI2 effect is identified | No calibration or biological uncertainty model | Not claimed |

## Verification and remaining limits

The core checks pass for equations/units, finite/bounded trajectories, State reset, independent conditions and cell dt refinement. Exponential-Euler and RK4 agree on the decisive single-cell rates and block labels. Halving the network timestep changes compared group rates by at most 1 Hz, mean voltage by 0.021 mV and block fractions by 3.6 percentage points; optical signs persist. See `validation.json`, `metric-boundary-validation.json` and `numerical-assessment.json`. Independent review is recorded separately. Initial non-reversing explorations and the corrected conductance-unit failure remain in `runs/`; they are not omitted from the record.

These simulations omit detailed human morphology, axonal propagation, calcium-dependent inactivation/extrusion, reporter kinetics, potassium/ATP regulation, inhibitory cells, plasticity and chronic development. The strongest transmission counterexample uses an extreme assumption, and the block demonstration selects a membrane regime with appreciable KNa dependence. The biological cause remains **underdetermined**. The contribution of modeling is to show how opposite signs can coexist and specify measurements that can falsify each explanation.

Independent numerical review passed with no findings in `reviews/iteration-2.md`. It reproduced all 44 assessment conditions and 560 summary rows, verified all 180 manifest hashes and confirmed exact rescue parity. Its acceptance does not independently validate the primary literature or biological assumptions. The main agent executed the BrainX simulations; the reviewer verified saved artifacts because fresh imports in its sandbox were blocked by cache permissions.

The `brainx-modeling-loop` skill's planned visualization reference is absent. Its explicit instruction blocks that figure stage, so the deliverables are numerical tables and raw traces, not generated figures. This does not change the numerical or scientific scope accepted by review.
