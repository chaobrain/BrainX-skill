# Codex review: iteration 6

- Thread: `01a06076-73eb-7cb2-acb6-037c31648973`
- Decision: REFUSE
- Scientific outcome: partially supported
- Loss closure: open
- Optimization adequacy: insufficient for architecture selection

## Resolved prior findings

The reviewer confirmed iteration 6 resolved all four iteration-5 blockers: direct physical BrainTools bounds, exact starts and bound diagnostics, identical six-seed BrainCell pipelines for observed/recovery/holdout fits, 94 raw validation arrays, successful EGL gate points, unclipped SHK plotting, and complete metadata-wave verification.

## Remaining blocker

All six optional EGL `m^4h` candidates exhausted the 600-iteration comparison budget. Their unfinished objective could not support the activation-only architecture rationale, and older documentation still quoted an incompatible 1.4% reduction.

## Iteration-7 response

- Run `m^4h` with the full locked 1,200-iteration budget and require a successful candidate.
- Compute the objective improvement only from the best successful candidate.
- Compare `m^4` and `m^4h` using BIC on the same 1,386 samples, with 14 versus 28 local parameters.
- Use one consistent named metric and result across the report and documentation.
