# Model single-cell influence mapping in mouse V1

Build a mechanistic recurrent excitatory-inhibitory model of mouse V1 layer 2/3 and ask whether perturbing one excitatory neuron reproduces the main influence-mapping observations of Chettih and Harvey (2019).

Present a set of visual stimuli and estimate each neuron's tuning from repeated unperturbed trials. During matched trials, stimulate one excitatory neuron using the published photostimulation protocol as closely as the model permits. Keep initial state, stimulus, and noise identical within each baseline-perturbation pair, and report both the intended stimulation protocol and the measured increase in target-neuron spikes.

For every non-target excitatory neuron, measure the perturbation-induced response change. Test, without forcing the outcome, whether influence is suppressive on average, has a local center-surround dependence on cortical distance, and becomes more negative with signal correlation after accounting for distance. Also test whether suppression is strongest when the presented stimulus matches the target neuron's preference. Report effect sizes, uncertainty, sample and bin counts, baseline firing rates, and all model and analysis parameters.

Use a point-neuron network for the first reproduction unless the cited evidence makes dendritic morphology or subcellular input location part of the tested mechanism. Clearly label phenomenological connectivity assumptions. If the signatures are not reproduced, retain the run and identify which mechanistic or statistical checks failed rather than tuning solely to obtain the expected pattern.

Reference:

- Chettih, S. N. & Harvey, C. D. (2019), Nature 567:334-340. https://doi.org/10.1038/s41586-019-0997-6
