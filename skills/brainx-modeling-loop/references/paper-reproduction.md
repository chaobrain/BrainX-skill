# Paper and author-code reproduction

Open during step 0 when a requested result depends on a paper, supplement or author implementation. A reproduction contract binds the scientific protocol and analysis to identified source passages; matching a curve's appearance does not establish that contract.

## Reconcile the sources

Inspect the requested publication version, relevant supplement and named author code before treating their settings as interchangeable. Use `bio-neuro-lit` → `references/required-supplements.md` when a required original artifact is missing from the default reader.

Record only scientific fields that affect the requested result:

| Field | Record |
|---|---|
| Model and protocol | Equations, units, parameter distributions, interventions, initial conditions and time averaging, with source page/equation or code location. |
| Observable | Measured activity, feedforward proxy, transformed signal or analytic prediction; retain the actual observable's name. |
| Analysis | Formula, bins, windows, reductions, normalization and exclusions, with the authoritative definition. |
| Control | The exact manipulated quantities and unchanged matched quantities; distinguish a different mechanism from a nearby parameter value. |
| Conflict or omission | Each source's value, any figure-specific override, the user's authority preference and the remaining interpretation limit. |

Honor an explicit user choice of author code or publication as authority. Apply a panel-specific setting within that panel's scope; do not silently substitute a generic Methods default. If unresolved source conflicts affect the answer, use step 0's existing clarification/delegation rule or narrow the claim. Do not combine favorable settings from different sources and call the mixture the original protocol. A stated range does not identify its sampling distribution.

## Lock the analysis contract

Before inspecting intervention results, include these details in or link them from `NeuroSpecification.md`:

| Decision | Required distinction |
|---|---|
| Samples and axes | Directed versus undirected pairs, self-exclusion, trial/subject/seed identity and the independent replication unit. |
| Binning and windows | Edges, centers, endpoint inclusion, correlation/feature windows versus temporal windows, and minimum occupancy. |
| Reductions and regression | Mean of bin means versus pooled-sample mean; regression inputs, intercept and weights. |
| Normalization | Formula, full comparison-set membership, seed/method pooling, denominator scope and degenerate-denominator handling. |
| Missingness | Invalid samples, empty bins, unavailable components, propagation into scores and any declared imputation. |
| Claims and acceptance | Published continuous score, each component needed by the claim, and the separate user-defined acceptance predicate, if any. |

Label implementation conventions not fixed by the source. Resolve consequential scientific omissions under step 0 rather than attributing agent choices to the paper. Preserve raw observations, counts, component values and normalization denominators so the analysis can be checked independently. Use a small known-input witness for non-obvious bin, axis, reduction or sign behavior.

A favorable composite score can coexist with an unfavorable component. Report the components required by the claimed phenomenon; do not infer their signs from the sum or invent an acceptance threshold. Keep a source-incomplete proxy separately named and restrict its conclusions instead of presenting it as the published metric.

## Handle later source corrections

Preserve the original specification, outputs, criterion and verdict. Record newly verified source passages before choosing follow-up work. If saved raw observations support the corrected analysis, freeze a separate offline protocol and reanalyze the complete relevant condition set before resimulating; keep outputs and claim scope distinct from the original run.

Report old/new differences, including reversals and unavailable results, without selecting a new window, comparison set, seed or parameter from the outcomes. Previously examined evaluation data is not freshly blinded evidence for a revised protocol. New dynamics or missing controls require a separately declared experiment under the existing authorization rule; offline reanalysis does not supply them or authorize a new simulation.
