# Figure contract

- Question and mode: Show how the four requested parameter points differ in population timing, sampled spikes, E/I firing rate, spike-train irregularity, and global-rate spectrum. Use final evidence mode.
- Sources: Use accepted iteration-4 run `20260830T142816+0800-validation-continuation-brunel`, its verified artifact manifest, and the iteration-4 `PASS` review.
- Displayed run: Use repeat 0, seed 1729, for the four-condition instantaneous-rate and raster figure. This is the first fixed seed, selected by ordinal position rather than outcome.
- Displayed sample: Use the stored fixed probe selected with seed 8675309: 40 excitatory and 10 inhibitory neurons. Preserve its ordering in every raster.
- Data: Use the recorded 0.1 ms interval. Show the fixed first 200 ms of the 2,000 ms analysis window in every instantaneous-rate and raster panel so the 625 Hz structure remains resolvable; this is a common time slice selected by position, not outcome. Convert E, I, and global counts to Hz using their respective population sizes and the 0.1 ms bin width.
- Transformations: Do not smooth the instantaneous rate or raster. Show full-analysis temporal means as dashed lines. Show all five repeat-level E/I rate and mean ISI-CV values with their medians. Show every raw Welch spectrum plus the pointwise median; restrict display to the frozen 1-1,000 Hz search band without normalizing power.
- Exclusions: Retain the simulation's locked ISI-CV eligibility rule of at least four analyzed spikes per neuron. Do not add figure-specific exclusions.
- Comparisons: Keep condition order, time range, spectral frequency range, colors, E/I encodings, and summary axes fixed. Let instantaneous-rate and spectral power axes retain visible condition-specific numeric scales because their absolute ranges differ by orders of magnitude.
- Statistics: Repeats are the sampling unit (`n=5`). Summary centers are medians. No uncertainty interval or inferential test is added. Per-neuron CV values remain source evidence but are not treated as independent repeats.
- Labels: Keep requested condition names separate from classifications measured under the locked predicates. Do not force an unsupported requested regime.
- Destination: Archival/report PNG figures at 300 dpi with readable labels at the exported physical size.
