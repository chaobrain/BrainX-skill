# Validate perturbation experiments before scaling

Open this reference when a point-neuron experiment estimates tuning or other
baseline features and then compares matched control and intervention trials. It
settles which scientific-validity gates must pass before a full causal mapping
run and how to calibrate intervention delivery without tuning the requested
outcome.

## Separate model validity from the tested result

Freeze the hypothesis tests only after establishing that the model produces
observable baseline responses and that the intervention reaches its declared
target. Use pilot data only for these protocol checks. Do not inspect downstream
influence signs, distance profiles, feature effects, or other requested outcomes
while choosing the operating regime or intervention strength.

| Gate | Required evidence before a formal run |
|---|---|
| Baseline operating regime | Report population and subgroup firing-rate distributions, silent fractions, and trial-count windows; reject a regime whose response statistic is almost always zero or saturated. |
| Stimulus identifiability | Show that repeated unperturbed trials yield non-degenerate response vectors, adequate reliability or dynamic range, and enough supported preferences or feature values for the planned comparisons. |
| Analysis support | Construct the exact planned eligibility filters and bins on pilot data; report finite-value, unique-value, bin, target, and effective-cluster counts, and reject empty or tie-dominated contrasts. |
| Intervention delivery | Measure the paired target-neuron response in the declared dose window and compare it with the stated protocol tolerance or cited benchmark. |
| Causal pairing | Verify identical initial State and external inputs before intervention onset and zero pre-intervention response mismatches. |

Define numerical acceptance rules from the scientific question, cited protocol,
measurement resolution, and planned estimator before running the pilot. Do not
invent universal firing-rate or bin-count thresholds. Preserve the rules and all
failed gate results in the run artifacts.

## Calibrate only the delivered intervention

When the stimulus waveform is an implementation proxy for a published
intervention, select its amplitude using target-neuron dose alone. Use a small,
predeclared candidate grid or monotone search, run matched pilot trials, and
choose according to the predeclared dose tolerance. Record every candidate,
trial count, measured target increment, uncertainty, and selection rule.

Do not select amplitude from non-target influence, suppression, feature
specificity, distance dependence, or any other reproduction outcome. If no
candidate satisfies the target-dose rule without invalidating the baseline
regime, classify the protocol as invalid and stop.

Generate each pilot's unit-aware waveform once and pass it through the same
BrainPy-State neuron and projection path used by the formal run. Measure the
executed spike increment; a requested current or pulse count is not the delivered
biological dose.

## Freeze and scale

After all gates pass, freeze model parameters, randomization rules, target
selection, intervention, eligibility filters, estimators, uncertainty method,
and success criteria. Then start a new immutable formal run and allow its result
to be positive, negative, or inconclusive.

Keep the pilot and formal artifacts distinct. A failed pilot is protocol
development evidence, not a scientific non-reproduction. A completed formal run
with adequate observability and support may legitimately fail every requested
hypothesis.

Open `skills/package-skills/brainstate/references/paired-perturbation-execution.md` for exact
State branching, fixed-shape batching, resource gates, and execution-status
classification. Open `references/braintools/input-current.md` when constructing
the unit-aware intervention waveform.

## Boundaries and common failures

- Treating a successfully executed but nearly silent network as an informative
  null experiment.
- Estimating preferences by `argmax` when most response vectors are tied at
  zero, then interpreting the tie-broken labels as tuning.
- Plotting nominal quantile bins without reporting empty bins, tied values, or
  effective target counts.
- Matching the requested current waveform but failing to measure the added
  target spikes or other delivered dose.
- Adjusting connectivity or intervention strength after seeing the sign of a
  downstream influence statistic.
