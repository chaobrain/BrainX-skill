# Frozen V1 single-cell influence-mapping run

This is one preregistered point-neuron model realization. No parameter search or rerun was used to obtain the signs below.

## Outcome

- FAIL: Suppressive influence on average
- FAIL: Local center-surround distance profile
- FAIL: More negative influence with signal correlation after distance adjustment
- FAIL: Strongest suppression for target-matched stimulus
- FAIL: Target dose within published mean +/- SEM

## Effect sizes

Mean non-target E influence: -0.0001 spikes per 367 ms (95% target-bootstrap CI -0.0003, 0.0001).
Distance-adjusted signal-correlation coefficient: 0.0004 spikes per 1 SD (95% CI 0.0001, 0.0006).
Orthogonal minus matched stimulus contrast: -0.0002 spikes per non-target neuron (95% CI -0.0006, 0.0002).

Distance bins:

- 25-100 um: -0.0002 spikes (95% CI -0.0005, 0.0001; 584 pairs, 16 targets)
- 100-300 um: -0.0001 spikes (95% CI -0.0004, 0.0001; 3310 pairs, 16 targets)
- >=300 um: -0.0002 spikes (95% CI -0.0007, 0.0003; 1160 pairs, 16 targets)

Stimulus-match bins:

- 0 deg: 0.0001 spikes (95% CI -0.0001, 0.0003; 128 target-trial observations)
- 45 deg: -0.0002 spikes (95% CI -0.0005, 0.0000; 256 target-trial observations)
- 90 deg: -0.0002 spikes (95% CI -0.0004, 0.0001; 128 target-trial observations)

## Protocol and controls

The model delivered four 32-ms somatic square-current sweeps at 15 Hz, starting at visual onset. Current amplitude was frozen at 0.120 nA. The measured target increase was 0.527 spikes in 250 ms (95% target-bootstrap CI 0.326, 0.775; 512 trials). The published cell-attached benchmark was 6.38 +/- 1.01 added spikes (mean +/- SEM, n=9 cells).

All 512 baseline-perturbation pairs used bit-identical external input arrays. Pre-perturbation E and I spike-count mismatches were 0 and 0.

## Baseline activity

Unperturbed tuning trials: E mean 0.11 Hz (median 0.00, IQR 0.00-0.04); I mean 28.87 Hz (median 28.23).
Matched mapping baselines: E mean 0.11 Hz; I mean 28.85 Hz.

## Model scope and assumptions

The model contains conductance-based LIF point neurons, not dendrites. The experiment tests somatic single-cell perturbation and population spiking, so morphology and subcellular input location are not represented mechanisms in this first reproduction.

Phenomenological assumptions are explicit: orientation-tuned feedforward current; Gaussian spatial candidate connectivity; sparse thinning; and preference-modulated synaptic weights. The broader IE than EE spatial scale and preference-modulated E-I-E pathway are hypotheses, not direct anatomical fits. Weights were frozen before the influence outcome was observed.

Connectivity summary:

- EE: 4035 edges, density 0.0394, mean out-degree 12.6, mean weight 0.399 nS, mean distance 74.9 um
- EI: 2624 edges, density 0.1025, mean out-degree 8.2, mean weight 0.497 nS, mean distance 119.1 um
- IE: 8057 edges, density 0.3147, mean out-degree 100.7, mean weight 0.854 nS, mean distance 201.2 um
- II: 500 edges, density 0.0781, mean out-degree 6.2, mean weight 0.652 nS, mean distance 117.3 um

The analysis excludes target-neighbor separations below 25 um. Influence is the paired perturbed-minus-baseline spike count over 367 ms. The center-surround test requires both near-minus-middle and far-minus-middle target-bootstrap CIs to be positive. Signal correlation is computed from eight repeated-trial mean direction responses and entered with a segmented continuous distance basis. Stimulus match compares 0-deg with 90-deg orientation difference. Confidence intervals resample targets only and do not cover model-seed or parameter uncertainty.

Complete frozen parameters are in `parameters.json`; pair data are in `pair_influences.csv`; exact numerical outputs and test definitions are in `results.json`.
