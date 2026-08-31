# Statistical and model visualization

## Purpose and boundary

Use this reference to inspect distributions and assumptions, compare groups, explore associations, diagnose regression, and evaluate classification or learning behavior. Plot the sampling structure that supports the scientific claim; a visual summary does not replace the statistical analysis.

Route BrainMass-owned FC/FCD, spectral, or dynamical summaries to `skills/package-skills/brainmass/references/visualization-analysis-api.md` before plotting them here.

## Choose the statistical family

Inspect distributions and assumptions before selecting a test or reporting a model score.

| Question | Use | Required context |
|---|---|---|
| What is the shape, spread, or tail behavior? | `distribution_plot(...)` | Sampling unit, `n`, binning or density settings, exclusions, and units. |
| Is a theoretical distribution plausible? | `qq_plot(...)` | Named reference distribution and whether deviations matter to the planned inference. |
| How do groups differ? | `box_plot(...)` or `violin_plot(...)` plus raw observations | Independent or paired structure, group sizes, center, spread, uncertainty, and effect size. |
| Which variables associate? | `correlation_matrix(...)` or `scatter_matrix(...)` | Variable orientation, method, missing-data rule, ordering, and multiple-comparison plan. |
| Is a relationship fitted adequately? | `regression_plot(...)` plus `residual_plot(...)` | Model family, train/held-out split, confidence meaning, residual definition, and units. |
| Which classification errors occur? | `confusion_matrix(...)` | Class order, counts versus normalization, prevalence, and decision threshold. |
| How does a binary ranker perform by threshold? | `roc_curve(...)` or `precision_recall_curve(...)` | Positive class, score direction, prevalence, operating costs, and held-out predictions. |
| Is performance data-limited, high-variance, or biased? | `learning_curve(...)` | Train sizes or steps, fold dimension, scoring metric, and held-out protocol. |

Do not infer causality from correlation. Do not use statistical significance as a substitute for effect size or practical importance.

## Inspect distributions and groups

Distribution and group figures expose data shape and sampling structure; show raw observations when feasible.

| API | Description |
|---|---|
| `distribution_plot(data, labels=None, plot_type='hist', bins=30, density=True, fit_normal=False, ...)` | Use for one or more distributions. Choose histogram, density, or both deliberately; treat binning and density estimation as declared transformations. |
| `qq_plot(data, distribution='norm', ...)` | Use to compare empirical and theoretical quantiles. Interpret systematic deviations, not visual proximity alone. |
| `box_plot(data, labels=None, showmeans=True, ...)` | Use for robust grouped summaries. Overlay raw observations when their count permits and state whether groups are paired. |
| `violin_plot(data, labels=None, showmeans=True, showmedians=True, ...)` | Use when distribution shape matters and sample size supports density estimation. Do not imply precision from a small-sample violin. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
btvis.distribution_plot(residuals, plot_type="hist", bins=30, ax=axes[0])
btvis.qq_plot(residuals, distribution="norm", ax=axes[1])
axes[0].set_xlabel("residual (mV)")
fig.tight_layout()
```

Report the sampling unit, `n`, center, spread, uncertainty definition, and effect size beside the plot. Correct multiple comparisons and preserve paired observations rather than plotting paired data as independent groups.

## Inspect association and regression

Association plots reveal structure; residual plots test whether a fitted relationship leaves systematic error.

| API | Description |
|---|---|
| `correlation_matrix(data, labels=None, method='pearson', ...)` | Use for pairwise associations with `(observations, variables)` data. Align labels to the variable axis and keep their order fixed. |
| `scatter_matrix(data, labels=None, diagonal='hist', ...)` | Use for pairwise multivariate inspection with ordinary `(observations, variables)` data. It returns a Matplotlib figure; reduce variables or sample density when unreadable. |
| `regression_plot(x, y, fit_line=True, confidence_interval=True, ...)` | Use for observations and an optional linear fit. A confidence band is not a prediction interval and the helper does not justify the model family. |
| `residual_plot(y_true, y_pred, ...)` | Use for aligned truth and prediction arrays. It computes residuals against predictions; preserve target units and observation order. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


assert y_true.shape == y_pred.shape
limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
axes[0].scatter(y_true, y_pred, alpha=0.6)
axes[0].plot(limits, limits, "k--", linewidth=1)
btvis.residual_plot(y_true, y_pred, ax=axes[1])
axes[0].set(xlabel="observed (Hz)", ylabel="predicted (Hz)")
fig.tight_layout()
```

Inspect residual center, trend, changing spread, tails, and outliers. Keep train and held-out residuals separate. Use a 1:1 line for observed-versus-predicted comparisons; a fitted regression line answers a different question.

## Evaluate classification and learning

Model-evaluation figures consume held-out labels, scores, predictions, or fold-wise learning results; never substitute training predictions unless the figure is explicitly diagnostic.

| API | Description |
|---|---|
| `confusion_matrix(y_true, y_pred, labels=None, normalize=None, ...)` | Use counts for workload and error totals. Use `normalize='true'` for per-true-class recall proportions or `normalize='pred'` for per-predicted-class composition; state the normalization. |
| `roc_curve(y_true, y_scores, ...)` | Use for binary ranking behavior across thresholds when both classes and ranking tradeoffs are meaningful. ROC can look optimistic under severe imbalance. |
| `precision_recall_curve(y_true, y_scores, ...)` | Prefer for imbalanced positive-class evaluation or when positive predictions and recall drive the decision. State positive-class prevalence. |
| `learning_curve(train_sizes, train_scores, validation_scores, ...)` | Use for train and validation score arrays across declared sizes or steps. Preserve fold or repeat dimensions so spread is visible. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
btvis.confusion_matrix(y_test, y_pred, normalize="true", ax=axes[0])
btvis.roc_curve(y_test, y_score, ax=axes[1])
btvis.precision_recall_curve(y_test, y_score, ax=axes[2])
fig.tight_layout()
```

Fix the positive class, score direction, decision threshold, class order, and evaluation cohort before comparing models. Use cross-validation score distributions and paired comparisons rather than presenting one split or one scalar score as definitive.

A persistent train-validation gap suggests variance or leakage and requires protocol inspection. Low converged train and validation performance suggests bias, weak features, or an unsuitable metric. A learning curve diagnoses patterns; it does not identify the cause by itself.

## Preserve statistical integrity

- Display raw observations when feasible and state `n` for every group or condition.
- Report effect sizes and confidence intervals with exact definitions. Distinguish practical from statistical significance.
- Correct multiple comparisons and state the correction family. Do not decorate uncorrected pairwise results as discoveries.
- Preserve paired, repeated-measures, nested, and cross-validation structure in both analysis and display.
- Keep axes honest. Use a truncated axis only when necessary, conspicuous, and unable to reverse the visual conclusion.
- Use the same scales, bins, ordering, metric definitions, and encodings across model or condition comparisons.
- Separate exploratory, training, validation, and held-out test evidence. Record preprocessing and threshold selection provenance.
- Treat correlation, feature importance, and fitted association as descriptive unless the design supports a causal claim.

## Common failures

- A mean and error bar hide the raw distribution, group size, or failed runs.
- Density or violin shape is overinterpreted for a small sample.
- Paired or nested observations are plotted and tested as independent.
- Correlation variables or labels use the wrong axis, or correlation is called connectivity or causality.
- A nonlinear relationship is summarized with an unjustified linear fit.
- Residuals are inspected only in aggregate, hiding bias or heteroscedasticity.
- ROC is selected for a highly imbalanced task without a precision-recall view or prevalence context.
- Confusion matrices use inconsistent class order or undisclosed normalization.
- Model comparison uses one score, one split, or tuned test-set thresholds.
- Axis truncation, changing bins, or inconsistent scales exaggerate differences.

## Tutorial sources

- [Statistical visualization](https://brainx.chaobrain.com/braintools/visualize/03_statistical_visualization.html)
- [Model evaluation plots](https://brainx.chaobrain.com/braintools/visualize/04_model_evaluation_plots.html)
