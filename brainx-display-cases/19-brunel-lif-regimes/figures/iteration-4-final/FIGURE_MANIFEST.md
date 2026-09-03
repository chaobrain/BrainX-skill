# Figure manifest

## `figures/iteration-4-final/four-condition-rate-raster.png`
- Work type, evidence mode, role, and question: create; final; Compare unsmoothed 0.1 ms global rate and fixed 50-neuron spike timing across conditions.
- Source run: `20260830T142816+0800-validation-continuation-brunel` (accepted iteration-4 evidence).
- Source manifest SHA-256: `8e690129d8927863daefbc20fba03ef5e0f21dd3aceae0a9b0ab1cd70d94e2f8`.
- PASS review SHA-256: `54f09b6ded86c3764166c4500b074125754b220d14f33ea24aa16c86da697d81`.
- Output: final evidence, report destination, PNG at 300 dpi.
- Variables and transformations: Repeat 0 (seed 1729), common first 200 ms of the analysis interval; dashed mean uses the full 2,000 ms; no smoothing.
- Comparisons: fixed condition order and encodings; requested and measured labels remain distinct.
- Render checks: 4774 x 1894 px; RGB standard deviation 47.234; nonblank and numerically cross-checked against accepted source arrays.
- Output SHA-256: `293a15725a9e63be48edc8ec7cc3a5e100142f67596b48383aeeb7a6e3f292d7`; 544171 bytes.

## `figures/iteration-4-final/ei-rates.png`
- Work type, evidence mode, role, and question: create; final; Compare excitatory and inhibitory firing rates across conditions and seeds.
- Source run: `20260830T142816+0800-validation-continuation-brunel` (accepted iteration-4 evidence).
- Source manifest SHA-256: `8e690129d8927863daefbc20fba03ef5e0f21dd3aceae0a9b0ab1cd70d94e2f8`.
- PASS review SHA-256: `54f09b6ded86c3764166c4500b074125754b220d14f33ea24aa16c86da697d81`.
- Output: final evidence, report destination, PNG at 300 dpi.
- Variables and transformations: Five paired repeat means per population with their medians; logarithmic rate axis.
- Comparisons: fixed condition order and encodings; requested and measured labels remain distinct.
- Render checks: 2494 x 1414 px; RGB standard deviation 26.098; nonblank and numerically cross-checked against accepted source arrays.
- Output SHA-256: `bbf6eaa2eacae9a1a132a23599fc6dd35967837f1e70232bd5f6f406052751c5`; 94042 bytes.

## `figures/iteration-4-final/isi-cv.png`
- Work type, evidence mode, role, and question: create; final; Compare E/I spike-train irregularity with the locked regular and irregular boundaries.
- Source run: `20260830T142816+0800-validation-continuation-brunel` (accepted iteration-4 evidence).
- Source manifest SHA-256: `8e690129d8927863daefbc20fba03ef5e0f21dd3aceae0a9b0ab1cd70d94e2f8`.
- PASS review SHA-256: `54f09b6ded86c3764166c4500b074125754b220d14f33ea24aa16c86da697d81`.
- Output: final evidence, report destination, PNG at 300 dpi.
- Variables and transformations: Five paired repeat mean CVs per population with medians; source eligibility requires at least four spikes.
- Comparisons: fixed condition order and encodings; requested and measured labels remain distinct.
- Render checks: 2494 x 1414 px; RGB standard deviation 29.266; nonblank and numerically cross-checked against accepted source arrays.
- Output SHA-256: `73745be92c546a12b4cd2bf3025d305e806cc547242fd19ff797b669e065fe9c`; 118732 bytes.

## `figures/iteration-4-final/global-rate-spectrum.png`
- Work type, evidence mode, role, and question: create; final; Compare global-rate spectral structure and report each aggregate dominant frequency.
- Source run: `20260830T142816+0800-validation-continuation-brunel` (accepted iteration-4 evidence).
- Source manifest SHA-256: `8e690129d8927863daefbc20fba03ef5e0f21dd3aceae0a9b0ab1cd70d94e2f8`.
- PASS review SHA-256: `54f09b6ded86c3764166c4500b074125754b220d14f33ea24aa16c86da697d81`.
- Output: final evidence, report destination, PNG at 300 dpi.
- Variables and transformations: Raw Welch PSD for all five repeats plus pointwise median; 1-1,000 Hz display, log-log axes, no normalization.
- Comparisons: fixed condition order and encodings; requested and measured labels remain distinct.
- Render checks: 2854 x 2074 px; RGB standard deviation 45.181; nonblank and numerically cross-checked against accepted source arrays.
- Output SHA-256: `21e319df0fe65466c895ddbd2e2b2d7925064b17430449de3ce0e1709dc73ced`; 761552 bytes.
