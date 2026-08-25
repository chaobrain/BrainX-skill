# Neural data visualization

## Purpose and boundary

Use this reference for general neural events, activity, connectivity, low-dimensional dynamics, stimulus response, topology, and spatial 3D data. Preserve the active BrainX package's variable meanings, units, ordering, and recording interval.

Route BrainCell morphology, compartments, runtime nodes, selector placement, or mechanism values directly to `skills/package-skills/braincell/references/multicompartment/topology-building-and-visualization.md`. Route BrainMass-owned trajectories, phase portraits, connectivity, FC/FCD, or spectra directly to `skills/package-skills/brainmass/references/visualization-analysis-api.md`.

## Choose the neural family

Select the figure from the data contract and question, not from visual familiarity.

| Data contract or question | Use | Preserve |
|---|---|---|
| Selected time-series columns aligned to timestamps | `line_plot(ts, val_matrix, ...)` | Time-major orientation, selected column identities, timestamps, and units. |
| Flat spike times plus matching neuron IDs, or one spike-time sequence per neuron | `spike_raster(...)` | Event times, neuron identity, filtering ranges, and time unit. |
| Dense `(time, neuron)` binary or event matrix with timestamps | `raster_plot(ts, sp_matrix, ...)` | Time-major orientation and exact timestamp vector. |
| Time-major neural activity or an aggregate rate | `population_activity(...)` | Recorded interval, aggregation method, window, and selected neurons. |
| Spike timing distribution | `isi_distribution(...)` or `spike_histogram(...)` | Whether the quantity is inter-spike interval or event count per time bin. |
| Spatial firing | `firing_rate_map(...)` | Position-to-rate mapping, grid geometry, interpolation, and rate unit. |
| Directed weights | `connectivity_matrix(...)` | Rows as presynaptic sources, columns as postsynaptic targets, labels, and normalization. |
| State evolution | `neural_trajectory(...)` or `phase_portrait(...)` | Time-major samples, selected coordinates, transient rule, and direction of time. |
| Structural relationships | `network_topology(...)` | Adjacency direction, layout meaning, and declared node or edge encodings. |
| Ordered stimulus and response | `tuning_curve(...)` | Stimulus order and unit, response unit, repeats, error meaning, and fit family. |
| Physical depth, mesh, volume, or three-coordinate geometry | Matching 3D helper | Coordinates, camera, thresholds, color normalization, and comparison view. |

Do not convert flat events to a dense matrix merely to plot them. Do not use 3D when a 2D projection answers the same question without occlusion.

## Visualize spikes and population activity

Spike helpers preserve either event-list or dense-matrix representations; population helpers reduce only along the declared neuron axis.

| API | Description |
|---|---|
| `line_plot(ts, val_matrix, plot_ids=None, ...)` | Use for selected columns of a time-by-value matrix aligned to `ts`. Preserve the meaning and unit of every selected column. |
| `spike_raster(spike_times, neuron_ids=None, ...)` | Use for flat event arrays with equal lengths or a sequence of per-neuron spike trains. Filter with explicit time or neuron ranges and receive a Matplotlib axes. |
| `raster_plot(ts, sp_matrix, ...)` | Use for dense `(time, neuron)` spikes aligned to `ts`. Do not pass flat spike-time events. |
| `population_activity(data, time=None, dt=None, method='mean', window_size=None, ...)` | Use for `(time, neurons)` activity or a one-dimensional aggregate. Prefer the recorded `time`; use `dt` only when it is the actual recorded interval. |
| `isi_distribution(spike_times, bins=50, max_isi=None, log_scale=False, ...)` | Use for sorted spike times or a list of sorted per-neuron spike-time arrays; it computes successive intervals. Use a per-neuron list to avoid intervals across neuron boundaries, and enable log scale when bin counts span orders of magnitude. |
| `spike_histogram(spike_times, bins=50, bin_size=None, density=False, ...)` | Use for event counts or density across time. Treat bin count or `bin_size` as a declared temporal-resolution choice. |
| `firing_rate_map(rates, positions=None, grid_size=None, interpolation='bilinear', ...)` | Use for position-rate pairs or an already shaped map. Declare interpolation and do not imply unsampled spatial resolution. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


assert spike_times.shape == neuron_ids.shape
assert activity.shape[0] == time.size

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
btvis.spike_raster(spike_times, neuron_ids, ax=axes[0])
btvis.population_activity(activity, time=time, ax=axes[1])
axes[0].set_ylabel("neuron index")
axes[1].set(xlabel="time (ms)", ylabel="mean activity")
fig.tight_layout()
```

Compare the raster and population trace against raw event counts in at least one known window. When smoothing population activity, retain the unsmoothed series and record the window or kernel.

## Visualize connectivity and dynamics

Matrix, trajectory, topology, and tuning figures answer different questions even when derived from the same model.

| API | Description |
|---|---|
| `connectivity_matrix(weights, pre_labels=None, post_labels=None, center_zero=True, ...)` | Use for source-by-target weights. Center a diverging colormap at zero only when zero is the meaningful neutral value; keep one normalization across comparisons. |
| `neural_trajectory(data, dims=None, time_color=True, ...)` | Use for `(time, features)` trajectories in two or three selected coordinates. Time coloring and start/end markers expose direction. |
| `phase_portrait(x, y=None, dx=None, dy=None, trajectory=True, vector_field=False, ...)` | Use for a two-variable trajectory, vector field, or both. Match vector-field and trajectory coordinates, scales, and units. |
| `network_topology(adjacency, positions=None, layout='spring', ...)` | Use for graph structure. Choose a layout for a declared structural question and state what node size, node color, edge width, and edge color encode. |
| `tuning_curve(stimulus, response, bins=20, error_bars=True, fit_curve=None, ...)` | Use for response versus ordered stimulus. Use error bars only when repeats define their statistic and use a fit only when its family is scientifically justified. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


assert trajectory.ndim == 2  # (time, features)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
btvis.neural_trajectory(trajectory, dims=(0, 1), time_color=True, ax=axes[0])
btvis.phase_portrait(trajectory[:, :2], trajectory=True, ax=axes[1])
axes[0].set(xlabel="state 1", ylabel="state 2")
axes[1].set(xlabel="state 1", ylabel="state 2")
fig.tight_layout()
```

For repeated conditions, keep coordinates, transient removal, start/end encoding, scales, matrix ordering, network positions, and tuning-curve bins fixed. A force-directed layout is not a biological coordinate system.

## Use 3D only for geometric evidence

Three-dimensional views are justified when depth, surfaces, volumes, electrodes, or three-coordinate trajectories are part of the question.

| API | Description |
|---|---|
| `neural_network_3d(layer_sizes, weights=None, activations=None, ...)` | Use for layered architecture and optional weights or activations, not anatomical morphology. |
| `brain_surface_3d(vertices, faces, values=None, ...)` | Use for a triangular surface mesh and optional vertex values. Validate vertex-face indexing and value alignment. |
| `connectivity_3d(source_positions, target_positions, connections, ...)` | Use for connections between two 3D point sets. Declare thresholding and edge-strength encoding. |
| `trajectory_3d(trajectory, time_colors=True, ...)` | Use for `(time, 3)` trajectories. Preserve direction with time color or start/end markers. |
| `volume_rendering(volume, threshold=None, ...)` | Use for a 3D scalar volume. Declare voxel geometry, threshold, alpha, and normalization. |
| `electrode_array_3d(electrode_positions, signals=None, ...)` | Use for electrode coordinates and optional aligned signals. Preserve coordinate frame and signal scaling. |
| `dendrite_tree_3d(segments, diameters=None, ...)` | Use only for generic segment geometry. Use the BrainCell route when segments represent BrainCell morphology or runtime nodes. |
| `phase_space_3d(x, y, z, time_colors=True, ...)` | Use for three declared dynamical variables. Keep their coordinate meanings and scales explicit. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


assert trajectory_xyz.shape[1] == 3
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
btvis.trajectory_3d(trajectory_xyz, time_colors=True, ax=ax)
ax.set(xlabel="state 1", ylabel="state 2", zlabel="state 3")
```

Fix the camera, axis limits, coordinate scaling, threshold, transparency, and color normalization before comparing conditions. Supply a 2D projection when occlusion could hide the relevant structure.

## Preserve comparison integrity

- Keep time bases, units, neuron and region order, matrix orientation, bins, smoothing, and aggregation fixed across comparable panels.
- Show raw observations when feasible. Do not let a smoothed population curve, fitted tuning curve, or topology layout stand in for the underlying data.
- Use a sequential colormap for nonnegative magnitude, a diverging colormap around a meaningful neutral value, and a cyclic colormap for phase.
- Label every colorbar with the encoded quantity and unit. State the meaning of node, edge, marker, and line encodings.
- Subsample dense trajectories or edges only by a declared rule. Preserve full data for numerical checks.
- Verify known events, extrema, connection directions, and start/end states against the source arrays.

## Common failures

- Time and neuron axes are swapped because shape was treated as meaning.
- Simulation `dt` is used after monitor subsampling changed the recording interval.
- Connectivity rows and columns are reversed, or matrices use different orderings across conditions.
- A diverging colormap is centered at zero when zero has no scientific role.
- Raster bins, smoothing windows, tuning bins, network layouts, or 3D thresholds change across a comparison.
- Network positions, trajectory coordinates, or interpolated maps imply biological geometry that the data do not contain.
- A generic dendrite view duplicates BrainCell-owned morphology visualization.

## Tutorial sources

- [Neural data visualization](https://brainx.chaobrain.com/braintools/visualize/01_neural_data_visualization.html)
- [Advanced neural plots](https://brainx.chaobrain.com/braintools/visualize/02_advanced_neural_plots.html)
- [3D visualization](https://brainx.chaobrain.com/braintools/visualize/06_3d_visualization.html)
