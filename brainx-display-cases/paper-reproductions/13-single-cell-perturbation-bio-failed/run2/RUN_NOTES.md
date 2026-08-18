# Run notes

## Outcome

The frozen model is stable at 13.39 Hz for excitatory neurons and 13.70 Hz for
inhibitory neurons. The six imposed target events produce 5--7 measured
additional target outputs per paired trial after recurrent feedback.

The complete Chettih-Harvey pattern is **not reproduced**. Mean neighbor
influence is slightly positive (`0.000734 +/- 0.000500` spikes per additional
target output), with 36.1% suppressed, 52.1% enhanced, and 11.8% unchanged.

The spatial curve is directionally center-surround-like: positive within 75 um,
near-zero negative at 150--250 um, then near zero farther away. The middle-bin
suppression is much smaller than its SEM, so classify this as partially
reproduced rather than resolved evidence.

The highest signal-correlation bin is negative while the lowest-correlation bin
is positive, and the preferred-stimulus comparison has the expected direction.
These prespecified directional criteria pass, but their effects are not
statistically resolved: uncertainty overlaps zero and the highest-correlation
bin contains only four neurons. Treat both as suggestive, not confirmatory.

Direct excitatory connection strength shows increasing positive influence in
this realization. This is reported descriptively and was not assumed in
advance.

## Verification

- Exact whole-model baseline replay passed.
- Dale signs, paired perturbation dose, influence normalization, rates, finite
  observables, and all five 119-neighbor bin reductions passed artifact checks.
- Connection probability and weight magnitude were verified to decrease with
  cortical distance and increase with receptive-field similarity.
- Both PNG figures are nonblank and correctly framed.

The stimulus panel in `influence_relationships.png` connects repeated
orientation differences from three stimulus positions, so use the
stimulus-resolved arrays in `paired_responses.npz` for quantitative analysis
rather than reading continuity from that line.

No parameter was changed after perturbation outcomes were observed.
