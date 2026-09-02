# Codex review: iteration 4

- Thread: `01a05ff2-e6ed-7213-beb9-2326eece7fb1`
- Decision: REFUSE

## Blocking findings

1. The fitting path used raw SciPy least squares and manual metrics without a BrainTools API-gap analysis.
2. Production fitting bypassed BrainCell; the BrainCell classes were exercised only in isolated tests without an explicit holding-state reset.
3. The evidence omitted complete optimizer diagnostics, noise/baseline comparison, parameter recovery across the allowed domain, and leave-one-voltage-out prediction.
4. Wave mapping relied on fixed wave numbers and labels without parsing and enforcing the packed Igor recreation metadata.

## Qualification findings

- Preserve the unclipped SHK activation estimate above one instead of hiding it in the plotted clipping step.
- Qualify the large EGL -20 mV local time-constant discrepancy rather than treating local gate summaries as direct observations.

The review returned the project to study. Iteration 5 addresses every blocking item before requesting a fresh review.
