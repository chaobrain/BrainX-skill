# Braintools visualization

Use this reference to select static, interactive, three-dimensional, animated,
or publication-styled visualizations for neural and scientific data. Preserve
the input data contract first; styling does not correct swapped time, neuron,
channel, or spatial axes.

## Choose a visualization family

| Need | Family | Result |
|---|---|---|
| Fast static inspection or publication output | Neural, statistical, basic, and styling APIs | Matplotlib axes, figures, or animations |
| Zoom, hover, and dashboard exploration | Interactive APIs | Plotly figure |
| Spatial network, morphology, or volume | 3D APIs | 3D axes or interactive renderer |
| Time evolution | Animation APIs | Matplotlib animation or rendered display |

Pass `ax=` when composing static panels. Keep the returned figure or axes handle
when later styling, saving, or annotating it.

## Neural visualization

| API | Description |
|---|---|
| `spike_raster(spike_times, neuron_ids=None, time_range=None, neuron_range=None, color='black', marker='|', markersize=1.0, alpha=1.0, ax=None, figsize=(10, 6), ..., show_stats=False)` | Use for flat spike-time and neuron-ID event arrays. |
| `population_activity(data, time=None, dt=None, method='mean', window_size=None, neuron_ids=None, ax=None, ...)` | Use for population activity with data shaped `(time, neurons)` or an already aggregated one-dimensional rate. |
| `connectivity_matrix(connectivity, ...)` | Display a source-by-target connectivity matrix. |
| `neural_trajectory(trajectory, ...)` | Plot a neural-state trajectory in two or three dimensions. |
| `spike_histogram(spike_times, ...)` | Plot a peristimulus time histogram from flat spike times. |
| `isi_distribution(intervals, ...)` | Plot an inter-spike interval distribution. |
| `firing_rate_map(rate_map, ...)` | Display a two-dimensional spatial firing-rate map. |
| `phase_portrait(x, y, trajectory=True, ...)` | Plot a two-state dynamical-system portrait. |
| `network_topology(adjacency, layout='spring', ...)` | Render graph structure from an adjacency matrix. |
| `tuning_curve(stimuli, responses, ...)` | Plot response as a function of stimulus. |

Do not pass a dense spike matrix to `spike_raster`; use `raster_plot(ts,
sp_matrix)` from the basic API for that representation.

## Statistical visualization

| API | Description |
|---|---|
| `correlation_matrix(data, labels=None, ...)` | Plot a correlation heatmap from observations and variables. |
| `distribution_plot(data, plot_type=..., ...)` | Plot a histogram, density estimate, or both. |
| `qq_plot(data, distribution='norm', ...)` | Compare empirical and theoretical quantiles. |
| `box_plot(groups, labels=None, ...)` | Compare grouped distributions with box plots. |
| `violin_plot(groups, labels=None, ...)` | Compare grouped density shapes. |
| `scatter_matrix(data, labels=None, ...)` | Inspect pairwise multivariate relationships. |
| `regression_plot(x, y, fit_line=True, confidence_interval=True, ...)` | Plot observations and an optional regression fit. |
| `residual_plot(targets, predictions, ...)` | Diagnose regression residuals. |
| `confusion_matrix(y_true, y_pred, labels=None, ...)` | Display classification counts by true and predicted class. |
| `roc_curve(y_true, y_scores, ...)` | Plot a binary receiver-operating-characteristic curve. |
| `precision_recall_curve(y_true, y_scores, ...)` | Plot a binary precision-recall curve. |
| `learning_curve(train_sizes, train_scores, validation_scores, ...)` | Compare training and validation performance across data or step counts. |

## Interactive visualization

Interactive functions return Plotly figures; call `.show()` in an interactive
environment or use Plotly export methods.

| API | Description |
|---|---|
| `interactive_spike_raster(spike_times, neuron_ids=None, ..., color_by=None, title='Interactive Spike Raster', width=800, height=600)` | Explore event rasters with hover, zoom, and filtering. |
| `interactive_line_plot(time, traces, labels=None, ...)` | Explore one or more time series. |
| `interactive_heatmap(data, ...)` | Explore matrix-valued data. |
| `interactive_3d_scatter(x, y, z, color=None, ...)` | Explore three-dimensional point data. |
| `interactive_network(adjacency, ...)` | Explore network topology. |
| `interactive_histogram(data, ...)` | Explore a distribution and binning interactively. |
| `interactive_surface(z, x=None, y=None, ...)` | Explore a three-dimensional height surface. |
| `interactive_correlation_matrix(data, ...)` | Explore a correlation matrix. |
| `dashboard_neural_activity(spike_times, neuron_ids, ...)` | Build a combined neural-activity dashboard from flat events. |

## Three-dimensional visualization

| API | Description |
|---|---|
| `neural_network_3d(layer_sizes, weights=None, activations=None, layer_spacing=2.0, neuron_spacing=1.0, node_size=100, edge_alpha=0.3, ax=None, ...)` | Visualize a layered network architecture and optional weights/activations. |
| `brain_surface_3d(vertices, faces, ...)` | Render a triangular brain-surface mesh. |
| `connectivity_3d(source_positions, target_positions, connections, ...)` | Render edges between two three-dimensional point sets. |
| `trajectory_3d(trajectory, ...)` | Plot an `(time, 3)` trajectory. |
| `volume_rendering(volume, ...)` | Render a three-dimensional scalar volume with isosurfaces. |
| `electrode_array_3d(positions, signals=None, ...)` | Plot electrode locations and optional per-electrode signals. |
| `dendrite_tree_3d(segments, ...)` | Render dendritic segments as start/end point pairs. |
| `phase_space_3d(x, y, z, ...)` | Plot a three-variable phase-space trajectory. |

## Basic plots and animation

| API | Description |
|---|---|
| `line_plot(ts, val_matrix, plot_ids=None, ...)` | Plot selected columns from a time-by-value matrix. |
| `raster_plot(ts, sp_matrix, ...)` | Plot a dense `(time, neuron)` spike matrix. |
| `animate_1D(data, ...)` | Animate one-dimensional data evolving over time. |
| `animate_2D(data, net_size, ...)` | Reshape each time frame to `net_size` and animate the field. |
| `animator(data, fig, ax, ...)` | Create a Matplotlib animation from frame data and existing figure handles. |

Retain the returned animation object until it is displayed or saved; otherwise
Matplotlib may garbage-collect it.

## Styling, colormaps, and figure utilities

| API | Description |
|---|---|
| `neural_style(...)` | Apply neural-data defaults. |
| `publication_style(fontsize=..., figsize=..., dpi=...)` | Configure publication-oriented sizing and rendering. |
| `dark_style(...)` | Configure a dark presentation theme. |
| `colorblind_friendly_style()` | Apply the accessible palette preset. |
| `create_neural_colormap(colors, name=...)` | Build a custom Matplotlib colormap. |
| `brain_colormaps()` | Create/register the provided spike, membrane, calcium, inhibitory, and excitatory colormaps. |
| `apply_style(style_name, **kwargs)` | Use a named style as a context manager when changes should remain scoped. |
| `get_color_palette(name, n_colors)` | Return a categorical palette. |
| `set_default_colors(mapping)` | Replace default neural-element colors from one mapping. |
| `get_figure(row_num, col_num, row_len=3, col_len=6)` | Return a constrained-layout figure and grid specification. |
| `remove_axis(ax, *spines)` | Hide named spines or blank the panel when none are named. |

```python
import matplotlib.pyplot as plt
import numpy as np
from braintools.visualize import (
    apply_style,
    get_figure,
    population_activity,
    spike_raster,
)

spike_times = np.array([0.1, 0.4, 0.8, 1.2])
neuron_ids = np.array([0, 1, 0, 2])
activity = np.random.default_rng(0).random((100, 3))
dt = 0.1

with apply_style("publication"):
    fig, grid = get_figure(
        row_num=2,
        col_num=1,
        row_len=3,
        col_len=6,
    )
    raster_ax = fig.add_subplot(grid[0, 0])
    rate_ax = fig.add_subplot(grid[1, 0])

    spike_raster(spike_times, neuron_ids, ax=raster_ax)
    population_activity(
        activity,  # shape: (time, neurons)
        time=np.arange(activity.shape[0]) * dt,
        ax=rate_ax,
    )
    fig.savefig("neural-activity.png", dpi=300)
    plt.close(fig)
```

Use `apply_style(...)` as a context manager when plotting code shares a process
with unrelated figures. Close figures after non-interactive export.

## Official source

- `https://brainx.chaobrain.com/braintools/apis/visualize.html`
