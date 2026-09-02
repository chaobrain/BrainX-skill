# Codex review: iteration 5

- Thread: `01a06048-1cce-70a2-a203-cd6936e1490b`
- Decision: REFUSE
- Scientific outcome: partially supported
- Loss closure: unresolved
- Optimization adequacy: insufficient

## Blocking findings

1. Start-centered sigmoid coordinates bounded to `[-3, 3]` imposed hidden, center-dependent effective bounds and made several recovery truths unreachable.
2. Recovery and voltage holdouts used fewer starts/restarts than production, analytic generation/prediction helpers, incomplete termination evidence, and no raw validation arrays.
3. The plotted EGL gate points came from an abnormally terminated optimizer; the SHK plot clipped a model-conditioned estimate above one.
4. Packed-metadata verification omitted continuation lines containing WT `wave6`, n582 `wave20:22`, and ad1006 `wave31:33`.

## Iteration-6 response

- Pass declared physical bounds directly to BrainTools and archive the exact sampled starts, histories, terminations, final parameters, and bound hits.
- Use the same six seeds, bounds, budget, BrainCell objective, and selection rule in observed, recovery, and holdout fits.
- Archive raw synthetic and held-out observations, predictions, and residuals.
- Require successful local gate-point termination, plot the unclipped SHK point, and label points as model-conditioned fits.
- Verify every consumed time, WT, mutant, and command wave from exact recreation-record lines.
