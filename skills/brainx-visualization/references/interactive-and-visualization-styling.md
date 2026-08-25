# Interactive and visualization styling

## Purpose and boundary

Use this reference for Plotly exploration, neural dashboards, Matplotlib styles, palettes, colormaps, export, and animation. Interaction and motion must expose evidence that is difficult to inspect statically; keep a stable static figure when the destination is a paper or archival report.

## Choose interaction, styling, or animation

| Need | Use | Preserve |
|---|---|---|
| Hover, zoom, filtering, or trace toggling | Matching `interactive_*` helper | Data meaning, axis units, ordering, fixed comparison ranges, and a static evidence view. |
| A combined event-level overview | `dashboard_neural_activity(...)` | Per-neuron spike representation, population time alignment, and dashboard source data. |
| Consistent Matplotlib appearance | `apply_style(...)` or a named style helper | Destination size, readability, scientific encodings, and process-local scope. |
| Reusable categorical colors | `get_color_palette(...)` | Stable condition-to-color mapping and redundant line or marker encodings. |
| Scalar, diverging, or cyclic color encoding | Existing or custom colormap | Quantity meaning, neutral point, cyclicity, and fixed limits. |
| Evolution of a 1D profile or 2D field | `animate_1D(...)`, `animate_2D(...)`, or `animator(...)` | Time-first frames, physical `dt`, fixed axes and color limits, and retained output object. |
| Custom multi-artist temporal story | Matplotlib `FuncAnimation` | Frame update contract, stable layout, export writer, and numerical source data. |

Do not add interaction, 3D rotation, dark styling, or animation as decoration. A view that requires manual interaction to reveal the claimed result is insufficient as the only final evidence.

## Explore with interactive figures

Interactive BrainTools helpers require Plotly and return Plotly figures. Confirm that Plotly is available before selecting this route; then inspect figures with `.show()`, refine them with Plotly figure methods, and export only after fixing comparison settings.

| API | Description |
|---|---|
| `interactive_spike_raster(spike_times, neuron_ids=None, ...)` | Use for event-level hover, zoom, time or neuron filtering, and optional coloring by neuron or time. |
| `interactive_line_plot(x, y, labels=None, ...)` | Use for one or more aligned traces. Legend toggling supports exploration but must not silently redefine the final comparison. |
| `interactive_heatmap(data, x_labels=None, y_labels=None, ...)` | Use for matrix inspection. Fix `zmin`, `zmax`, label order, and colorscale across comparable matrices. |
| `interactive_3d_scatter(x, y, z, color=None, size=None, labels=None, ...)` | Use when three coordinates and hover metadata are part of the question. Declare color and size meanings. |
| `interactive_network(adjacency, positions=None, ...)` | Use for topology exploration. Preserve adjacency direction, positions, threshold, and node or edge encodings. |
| `interactive_histogram(data, labels=None, bins=30, ...)` | Use to inspect one or more distributions and bin sensitivity. Do not compare quantities with incompatible units on one axis. |
| `interactive_surface(z, x=None, y=None, ...)` | Use for a scalar surface over declared coordinates. Preserve grid orientation and fixed height/color ranges. |
| `interactive_correlation_matrix(data, labels=None, method='pearson', ...)` | Use for exact-value hover over correlation structure. Preserve variable orientation and do not infer causality. |
| `dashboard_neural_activity(spike_times, neuron_ids=None, population_activity=None, time=None, ...)` | Use for a linked overview of spikes and population summaries. Supply either matching flat events or per-neuron sequences and align population activity to `time`. |

```python
import braintools.visualize as btvis


fig = btvis.interactive_heatmap(
    connectivity,
    x_labels=target_labels,
    y_labels=source_labels,
    colorscale="RdBu_r",
    title="source-to-target connectivity",
)
fig.update_traces(zmin=-weight_limit, zmax=weight_limit)
fig.show()
fig.write_html("connectivity-explorer.html")
```

For Plotly static export, use `fig.write_image(...)` with an available image engine. Use `fig.write_html(...)` for a self-contained interactive artifact. Record the visible defaults, filtering state, camera, axis ranges, and color limits in the figure manifest.

Subsample dense interactive data only by a declared rule. Validate extrema, labels, and representative points before and after subsampling.

## Apply styles without leaking state

Matplotlib style helpers modify process-wide defaults; scope them when plotting code shares a process with unrelated figures.

| API | Description |
|---|---|
| `apply_style(style_name, **kwargs)` | Use as a context manager for `neural`, `publication`, `dark`, or `colorblind`; it restores prior `rcParams` on exit. |
| `neural_style(...)` | Use for process-wide neural-data defaults only when all subsequent figures should share them. |
| `publication_style(fontsize=10, figsize=(6, 4), dpi=300, usetex=False)` | Use for process-wide print defaults after the destination size and font constraints are known. Do not enable TeX unless the environment and glyph set are verified. |
| `dark_style(...)` | Use for screen or projection output with a dark background. Export with matching face colors and do not reuse it blindly for print. |
| `colorblind_friendly_style()` | Use for the accessible process-wide preset. Preserve redundant line, marker, or direct-label encodings. |
| `get_color_palette(palette_name, n_colors=None)` | Use for a fixed categorical palette. Save the condition-to-color mapping rather than requesting colors independently per panel. |
| `create_neural_colormap(name, colors, n_bins=256)` | Use to create a named continuous colormap only when existing maps cannot express the quantity correctly. |
| `brain_colormaps()` | Use to register the supplied neural colormap set before referring to those names in Matplotlib. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


with btvis.apply_style("publication", fontsize=9, dpi=300):
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    btvis.population_activity(activity, time=time, ax=ax)
    ax.set(xlabel="time (ms)", ylabel="population activity")
    fig.savefig("population-activity.pdf", bbox_inches="tight")
    plt.close(fig)
```

Follow the destination specification before any preset. Verify the rendered output at final physical size; preset names do not guarantee readable labels, correct fonts, or valid dimensions.

## Select colors by data meaning

- Use sequential colormaps for ordered magnitude, diverging maps around a meaningful center, and cyclic maps for phase or direction.
- Use categorical palettes for unordered conditions. Keep each condition's color fixed across all panels and figures.
- Add a line style, marker, pattern, or direct label when color carries group identity. Check grayscale and color-vision-deficiency legibility.
- Fix color limits before comparing matrices, surfaces, volumes, or animation frames. Autoscaling each panel or frame destroys magnitude comparison.
- Label colorbars with quantity and unit. Do not use decorative color variation without a declared encoding.
- Avoid rainbow maps for ordered scientific values when they introduce false boundaries or nonuniform perceptual changes.

## Animate declared temporal data

Animation maps the first array dimension to frames; physical frame time and display delay are separate quantities.

| API | Description |
|---|---|
| `animate_1D(dynamical_vars, static_vars=(), dt=None, frame_delay=50, frame_step=1, save_path=None, show=True, ...)` | Use for one or more `(frames, x)` profiles and optional static one-dimensional references. Supply physical `dt`; the function returns the created figure. |
| `animate_2D(values, net_size, dt=None, val_min=None, val_max=None, frame_step=1, save_path=None, show=True, ...)` | Use for `(frames, neurons)` values reshapeable to `net_size` or `(frames, height, width)` fields. It returns a `FuncAnimation`; retain it through display or export. |
| `animator(data, fig, ax, num_steps=False, interval=40, cmap='plasma')` | Use for time-first `(frames, height, width)` image data with existing figure and axes. It returns an `ArtistAnimation`; retain it through display or export. |
| `matplotlib.animation.FuncAnimation(...)` | Use for coordinated custom artists, sliding windows, dynamic networks, or learning diagnostics that the high-level helpers cannot express. Return every changed artist when using blitting. |

```python
import matplotlib.pyplot as plt
import braintools.visualize as btvis


assert field.ndim == 3  # (frames, height, width)
animation = btvis.animate_2D(
    field,
    net_size=field.shape[1:],
    dt=recorded_dt,
    val_min=fixed_min,
    val_max=fixed_max,
    show=False,
)
animation.save("activity.mp4", writer="ffmpeg", fps=20)
plt.close(animation._fig)
```

Set `dt` from the recorded interval, not the integration step when monitoring was subsampled. Set `frame_step` as a declared temporal decimation. Keep axis limits, camera, colormap, and value limits constant across frames and comparable animations.

Use GIF for short, simple, widely embedded motion; use MP4 for compact presentation video; use HTML for notebook or browser controls. Verify writer availability before a long render. For large animations, update existing artists, enable blitting when compatible, decimate by a declared rule, precompute expensive frame quantities when memory permits, and disable or limit frame caching for streaming data.

## Export and verify

- Export static line art as PDF or SVG when supported; use intentional pixel dimensions and raster resolution for PNG.
- Export Plotly HTML when interaction is the deliverable and a static companion when the scientific claim must be visible without interaction.
- Inspect text, math glyphs, background, transparency, colorbars, hover labels, default camera, and initial animation frame.
- Check that interaction or animation does not reveal values outside the declared source range or hide condition failures behind filters.
- Compare representative frames and hover values against source arrays. Check first, middle, and last frames plus extrema.
- Close Matplotlib figures after non-interactive export and retain animation objects until saving or display completes.

## Common failures

- Interactive controls define the result, but the saved artifact does not record their state.
- Plotly figures use different axis or color ranges across conditions.
- A process-wide style leaks into unrelated figures or overwrites venue settings.
- Color is the only condition encoding, or a colormap has no relation to the data semantics.
- Each animation frame autoscales, creating false changes in magnitude.
- Display `interval` is confused with physical `dt`, or integration `dt` is used after monitor subsampling.
- An animation object is garbage-collected before display or export completes.
- Dense data are subsampled without recording the rule or checking retained extrema.
- Dark output is exported with the wrong face color, or TeX/font settings fail only in the final render.
- Animation or interaction is used where a static comparison would be clearer and more inspectable.

## Tutorial sources

- [Interactive visualization](https://brainx.chaobrain.com/braintools/visualize/05_interactive_visualization.html)
- [Styling and themes](https://brainx.chaobrain.com/braintools/visualize/07_styling_and_themes.html)
- [Animation and dynamics](https://brainx.chaobrain.com/braintools/visualize/08_animation_and_dynamics.html)
