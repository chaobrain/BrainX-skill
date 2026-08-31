---
name: brainx-visualization
description: Visualize BrainX simulations, training, fitting, and analysis results as diagnostic or publication-ready scientific figures. Use for neural traces, spikes, activity, connectivity, trajectories, distributions, interactive exploration, animations, cellular topology, BrainMass summaries, or figure-integrity review after a BrainX route is selected.
---

# BrainX visualization

## Purpose and boundary

Use this skill after selecting the active BrainX modeling route and obtaining the data or run artifacts to visualize. Preserve the biological meaning, axes, units, conditions, observation mapping, and provenance established by that route.

Follow this path:

`state the figure question -> bind source evidence -> validate the data contract -> choose the highest-level visualization API -> render -> compare with raw values -> export and record provenance`

Choose the evidence mode independently from the work type:

| Decision | Choices | Rule |
|---|---|---|
| Evidence mode | `diagnostic`, `final` | Allow unaccepted runs only in diagnostic figures. Use accepted run IDs for final figures unless failed or null evidence is intentionally shown and labeled. |
| Work type | `audit`, `adapt`, `create` | Preserve originals during audits, preserve scientific meaning during adaptations, and write new plotting code and outputs in the research project rather than the installed skill. |

Visualization communicates evidence. It does not decide scientific acceptance or repair an invalid simulation, observation mapping, or statistical comparison.

## Underlying mental model

A figure maps selected BrainX evidence through declared transformations to visual encodings that answer one scientific question.

The active BrainX route defines what every State, event, region, compartment, quantity, and observation means. Never infer that meaning from array shape alone.

Define a figure contract before choosing the representation:

| Contract field | Required decision |
|---|---|
| Question and mode | State one claim, diagnostic question, or comparison and choose `diagnostic` or `final`. |
| Sources | Record run IDs, artifact paths, hashes, conditions, seeds, checkpoints, and acceptance status. |
| Data | Record each variable's meaning, shape, axis order, sampling interval, ordering, and physical unit. |
| Transformations | Declare slicing, alignment, baseline correction, normalization, smoothing, aggregation, exclusions, and observation mapping. |
| Comparisons | Name controls, interventions, aligned landmarks, and fixed display limits. |
| Statistics | State the sampling unit, sample size, center, spread, uncertainty definition, and every reduction shown. |
| Destination | Set diagnostic, report, presentation, single-column paper, or double-column paper output before size and format. |

Freeze displayed cases, thresholds, smoothing, aggregation, exclusions, normalization, and axis or color limits before inspecting intervention outcomes. Keep raw per-condition evidence available beside aggregates.

## Choose the visualization layer

Use the highest-level API that owns the scientific view; package ownership preserves domain semantics that generic plotting cannot recover.

| Layer | Use |
|---|---|
| Selected package visualization | Use first when BrainCell or BrainMass owns the represented structure, quantity, or analysis result. |
| `braintools.visualize` | Use for general neural, statistical, model-evaluation, interactive, 3D, styling, and animation workflows. |
| `matplotlib.pyplot` | Use for simple composition, labels, annotations, and export after the owning helper. Add custom artists only for a verified gap. |

Choose a general figure family by question:

| Question | Figure family | Required evidence |
|---|---|---|
| When and where do events or activity occur? | Raster, trace, population activity, histogram, rate map, or activity dashboard | Preserve event representation, time-major axes, recorded interval, units, and neuron or spatial ordering. |
| What dynamical regime or network structure appears? | Trajectory, phase portrait, tuning curve, ordered matrix, topology, or 3D view | Declare transient removal, coordinate meaning, ordering, layout semantics, and any threshold. |
| How variable is a result or does an assumption hold? | Distribution, Q-Q, raw samples with interval, box/violin, correlation, or scatter matrix | State sampling unit, `n`, uncertainty, comparison correction, and the exact statistic. |
| Does a model predict or generalize correctly? | Prediction comparison, residuals, confusion matrix, ROC or precision-recall curve, or learning curve | Separate train and held-out evidence; preserve labels, score meaning, prevalence, folds, and residual units. |
| Does hover, zoom, depth, or time evolution answer the question? | Plotly interaction, 3D view, or animation | Keep a stable final figure for archival evidence and fix encodings, camera, thresholds, and frame timing across comparisons. |

Open the matching general reference before choosing concrete functions:

- Open `references/neural-data-visualization.md` for neural events, activity, connectivity, trajectories, tuning, topology, or general 3D data.
- Open `references/statistical-and-model-visualization.md` for distributions, assumptions, grouped comparisons, regression, residuals, classification evaluation, or learning curves.
- Open `references/interactive-and-visualization-styling.md` for Plotly exploration, dashboards, themes, palettes, colormaps, export, or animation.

## Compose a static neural figure

Static helpers render into supplied axes; compose the figure once, pass `ax=` to every helper, export once, and close it after non-interactive output.

| API | Description |
|---|---|
| `braintools.visualize.spike_raster(...)` | Use for flat spike-time events with matching neuron IDs, or per-neuron spike-train sequences. It returns the Matplotlib axes. |
| `braintools.visualize.population_activity(...)` | Use for time-major `(time, neurons)` activity or an already aggregated one-dimensional rate. It returns the axes. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


assert spike_times.ndim == neuron_ids.ndim == 1
assert spike_times.shape == neuron_ids.shape
assert activity.ndim == 2
assert time.shape == (activity.shape[0],)

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
btvis.spike_raster(spike_times, neuron_ids, ax=axes[0])
btvis.population_activity(activity, time=time, ax=axes[1])
axes[0].set_ylabel("neuron index")
axes[1].set(xlabel="time (ms)", ylabel="population activity")
fig.tight_layout()
fig.savefig("neural-activity.png", dpi=600, bbox_inches="tight")
plt.close(fig)
```

Open `references/neural-data-visualization.md` when the event representation differs, the figure requires another neural family, or a 3D view is scientifically necessary.

## Prepare final figures

Make the evidence readable at its destination size without changing the encoded comparison.

- Follow the venue specification first. Otherwise use intentional physical dimensions, prefer PDF or SVG for vector line art, and use sufficient raster resolution.
- Label axes with units, omit placeholder units for dimensionless values, and use one name for the same quantity throughout a report.
- Use consistent panel labels, axes, normalization, line and marker hierarchies, and accessible colors. Do not encode a comparison only by hue.
- Use sequential colormaps for ordered magnitude, diverging colormaps only around a meaningful center, and categorical palettes for unordered conditions.
- Add labels, legends, colorbars, and panel marks before the final layout pass. Keep legends and grids secondary to the evidence.
- Reduce overplotting before increasing figure size. Use an inset only when a local difference is scientifically important and unused space is available.

Open `references/interactive-and-visualization-styling.md` for scoped styles, palette and colormap selection, interactive export, animation export, and comparison-stable visual settings.

## Verify and record the figure

Render every export and inspect it against source values before delivery.

Verify that the output is nonblank, unclipped, legible at destination size, and free of overlaps. Check axis values, units, event timing, ordering, sample count, summaries, and fixed scales against source arrays. Confirm that smoothing, aggregation, uncertainty, exclusions, interpolation, and overplotting do not hide failures or differences.

Create or update `FIGURE_MANIFEST.md` with one entry per figure:

```markdown
## <figure path>
- Work type, evidence mode, scientific role, and question:
- Source run IDs, artifacts, hashes, and acceptance status:
- Variables, axes, ordering, and units:
- Transformations, smoothing, aggregation, and exclusions:
- Sample size and uncertainty:
- Controls and fixed comparison settings:
- Plotting source, output path, size, format, and resolution:
- Render and source-value checks:
```

## Reference routing

| Route | Open when | Contains |
|---|---|---|
| `references/neural-data-visualization.md` | Selecting neural event, activity, connectivity, dynamical, topology, tuning, spatial, or general 3D figures. | Data contracts, family selection, concrete BrainTools helpers, 3D boundaries, workflows, and comparison rules. |
| `references/statistical-and-model-visualization.md` | Inspecting distributions or assumptions, comparing groups, diagnosing regression, or evaluating classification and learning behavior. | Statistical helper selection, raw-data and uncertainty rules, residual diagnostics, ROC-versus-precision-recall decisions, and learning-curve interpretation. |
| `references/interactive-and-visualization-styling.md` | Adding interactive exploration, dashboards, styles, palettes, colormaps, export, or animation. | Plotly return behavior, stable encodings, scoped Matplotlib styles, output decisions, animation data contracts, and performance boundaries. |
| `skills/package-skills/braincell/references/multicompartment/topology-building-and-visualization.md` | Visualizing BrainCell morphology branches, CVs, runtime nodes, selector placement, topology layouts, or node and mechanism values. | Branch/CV/node decisions, initialization, physical-versus-topological views, selector coverage, value coloring, and canonical compositions. |
| `skills/package-skills/brainmass/references/visualization-analysis-api.md` | Computing or plotting BrainMass trajectories, phase portraits, connectivity, FC/FCD, or spectra. | `brainmass.viz`, BrainTools metric selection, time-major and sampling rules, unit boundaries, canonical analysis, and failures. |

## Boundaries and common failures

- The figure is designed before its question, mode, evidence, and destination are fixed.
- Biological meaning, time axis, region order, or units are guessed from shape.
- Dense spike matrices and flat event arrays are sent to the wrong raster representation.
- Integration `dt` is used after recording subsampling changed the plotted interval.
- Package-owned morphology, topology, BrainMass analysis, or visualization is rebuilt with generic plotting code.
- Comparison scales, smoothing, normalization, thresholds, or exclusions vary without disclosure.
- Aggregate curves hide raw outcomes, sample size, uncertainty, failed runs, or null evidence.
- A reference figure is cloned without adapting its layout to the scientific role and real data geometry.
- Existing figures are overwritten, or plotting scripts and outputs are written into the installed skill directory.
- Diagnostic output is presented as final evidence, or a visual pattern is treated as scientific acceptance.
- Final artifacts are delivered without source provenance and render inspection.
