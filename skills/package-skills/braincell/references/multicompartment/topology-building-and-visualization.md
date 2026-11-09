# Topology building and visualization

Use this reference after `references/multicompartment/multicompartment-cell-workflow.md` when branch topology, CV coverage, runtime nodes, selector placement, or spatial values must be visualized. Keep morphology construction and validation, selector design, CV-policy choice, recording behavior, and simulation in their owning references.

## Choose the view

Morphology branches, CVs, and runtime nodes are different layers of one cell; visualize the layer that owns the question.

| API | Use when | Constraint |
|---|---|---|
| `morphology.topo()` | Check the named parent-child branch tree before discretization. | It does not show CVs or runtime nodes. |
| `morphology.select(expr)` | Confirm region intervals or locset points before plotting. | Use the filter reference when the selection itself is wrong. |
| `braincell.vis.plot_cell_topology(cell, level="branch", ...)` | Inspect compact branch topology or region coverage. | Supports region highlights. |
| `braincell.vis.plot_cell_topology(cell, level="cv", ...)` | Inspect discretization, region coverage, locset ownership, or CV values. | Initialize before reading runtime values. |
| `braincell.vis.plot_cell_topology(cell, level="node", ...)` | Inspect the runtime point graph, lowered placement, or point values. | Initialize before rendering runtime nodes. |
| `braincell.vis.plot2d()` / `plot3d()` | Inspect physical morphology geometry. | Use these instead of topology layouts when length, radius, or 3D position matters. |

The morphology defines continuous cable, the CV policy defines isopotential intervals, and `init_state()` lowers them into the execution-oriented `NodeTree`. Do not equate morphology points, branches, CVs, and runtime nodes.

## Prepare the visualization context

Supply a validated `neuron.swc` with a soma and dendrite and use an environment with Matplotlib and NetworkX, available with the BrainCell `vis` extra. Graphviz layouts require PyGraphviz. When a Graphviz layout is unavailable, the plot warns and falls back to `kamada_kawai`. Choose `layout="kamada_kawai"` directly when only NetworkX is available.

```python
import braincell
import braincell.mech as mech
import braincell.vis as vis
import brainunit as u
import matplotlib.pyplot as plt
from braincell.filter import AllRegion, RootLocation, Terminals, branch_in


morphology = braincell.Morphology.from_swc("neuron.swc")
target_region = branch_in(
    "type",
    ("dendrite", "basal_dendrite", "apical_dendrite"),
)
target_sites = RootLocation(0.5) | Terminals()

assert morphology.select(target_region).intervals
assert morphology.select(target_sites).points

cell = braincell.Cell(
    morphology,
    cv_policy=braincell.MaxCVLen(25.0 * u.um),
)
cell.paint(
    AllRegion(),
    mech.Channel(
        "IL",
        g_max=0.03 * u.mS / u.cm**2,
        E=-65.0 * u.mV,
    ),
)
cell.init_state()

assert cell.n_cv == len(cell.cvs)
assert len(cell.node_tree.edges) == len(cell.node_tree.nodes) - 1
```

Access `cell.node_tree` and node value mode only after `init_state()`. Return to the morphology reference if initialization does not produce one connected rooted tree.

## Compare structure and selector placement

Render one selection across structural levels, then show a locset at runtime level. This distinguishes a branch-selection error from CV coverage or point-lowering errors.

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=140)

vis.plot_cell_topology(
    cell,
    level="branch",
    region=target_region,
    coverage_mode="fraction",
    layout="dot",
    ax=axes[0, 0],
)
axes[0, 0].set_title("Branch region")

vis.plot_cell_topology(
    cell,
    level="cv",
    region=target_region,
    coverage_mode="fraction",
    layout="dot",
    ax=axes[0, 1],
)
axes[0, 1].set_title(f"CV region: n={cell.n_cv}")

vis.plot_cell_topology(
    cell,
    level="node",
    region=target_region,
    coverage_mode="fraction",
    layout="dot",
    ax=axes[1, 0],
)
axes[1, 0].set_title("Runtime region")

vis.plot_cell_topology(
    cell,
    level="node",
    locset=target_sites,
    highlight_color="#f97316",
    layout="dot",
    ax=axes[1, 1],
)
axes[1, 1].set_title("Runtime locset")

fig.tight_layout()
plt.show()
```

Use `coverage_mode="fraction"` to blend by overlap, `"any"` to fully highlight any overlap, and `"all"` to highlight only complete coverage. Node-level regions and locsets map to owning CV midpoint points; they do not render literal morphology coordinates.

## Compare layouts

Hold the initialized cell and data fixed while choosing the layout that makes topology easiest to inspect.

```python
fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)

vis.plot_cell_topology(
    cell,
    level="node",
    preset="dendrotweaks",
    ax=axes[0],
)
axes[0].set_title("dendrotweaks")

vis.plot_cell_topology(
    cell,
    level="node",
    layout="dot",
    ax=axes[1],
)
axes[1].set_title("dot")

vis.plot_cell_topology(
    cell,
    level="node",
    layout="kamada_kawai",
    layout_scale=1.5,
    ax=axes[2],
)
axes[2].set_title("kamada_kawai")

fig.tight_layout()
plt.show()
```

Available presets are `"dendrotweaks"`, `"mono"`, and `"depth"`; documented layouts are `"twopi"`, `"dot"`, `"neato"`, and `"kamada_kawai"`. `layout_scale` changes global spacing without changing topology.

The default topology preset ignores physical length and radius. Do not interpret graph spacing as cable distance; route physical geometry inspection to the morphology reference.

## Color runtime and mechanism values

Use value mode on an initialized or already simulated cell. Named selectors avoid manually extracting and aligning runtime arrays.

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)

vis.plot_cell_topology(
    cell,
    level="node",
    value="V",
    cmap="viridis",
    vmin=-80.0,
    vmax=-40.0,
    value_label="Voltage",
    ax=axes[0],
)
axes[0].set_title("Membrane voltage")

vis.plot_cell_topology(
    cell,
    level="node",
    value=("channel", "IL", "g_max"),
    cmap="magma",
    value_label="IL g_max",
    ax=axes[1],
)
axes[1].set_title("Leak conductance")

fig.tight_layout()
plt.show()
```

Keep `vmin`, `vmax`, and `cmap` fixed when comparing cells or time points. Run the cell through the parent workflow first when the figure must show post-stimulus rather than initialized State.

## Color a custom node metric

Pass one scalar per runtime point when the built-in value selectors do not own the diagnostic quantity. This example maps graph depth from the runtime root.

```python
from collections import deque

import numpy as np


def node_depths(tree):
    depths = np.full(len(tree.nodes), np.nan)
    depths[tree.root_node_id] = 0.0
    children = {node.id: [] for node in tree.nodes}

    for edge in tree.edges:
        children[edge.parent_node_id].append(edge.child_node_id)

    queue = deque([tree.root_node_id])
    while queue:
        parent = queue.popleft()
        for child in children[parent]:
            depths[child] = depths[parent] + 1.0
            queue.append(child)

    return depths


depth = node_depths(cell.node_tree)
assert depth.shape == (len(cell.node_tree.nodes),)
assert not np.isnan(depth).any()

fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
vis.plot_cell_topology(
    cell,
    level="node",
    value=depth,
    cmap="magma",
    value_label="Runtime graph depth",
    ax=ax,
)
fig.tight_layout()
plt.show()
```

Pass one population member as a one-dimensional array of length `n_point` or `n_cv`. In node mode the function scatters CV values into point space. Named runtime selectors handle a singleton population directly.

## Lookup

| Level | Region | Locset | Value | Best use |
|---|---|---|---|---|
| `"branch"` | Yes | No | No | Compact branch topology and region coverage. |
| `"cv"` | Yes | Yes | Yes | Discretization, owning-CV placement, and compartment values. |
| `"node"` | Yes | Yes | Yes | Full runtime topology, lowered placement, and point values. |

| `value` form | Result |
|---|---|
| Point array of length `n_point` | Color runtime nodes directly. |
| CV array of length `n_cv` | Scatter CV values into point space. |
| `"V"` or `"voltage"` | Color membrane voltage. |
| `("ion", ion_name, field)` | Color one ion field. |
| `("channel", class_name, field)` | Color one channel field. |
| `("layout_id", layout_id, field)` | Color one resolved layout field. |

| Control | Use |
|---|---|
| `highlight_color` / `coverage_mode` | Style region or locset highlights. |
| `cmap`, `vmin`, `vmax`, `norm` | Control value normalization. |
| `value_label`, `show_colorbar` | Control the colorbar. |
| `node_color`, `edge_color`, `root_color` | Style topology without entering value mode. |
| `ax=...` | Compose subplots; call `plt.show()` or `fig.savefig(...)` once afterward. |

Highlight mode and value mode are mutually exclusive. Fix selectors in `filter-function-library.md`, discretization in `cv-policy-reference.md`, morphology geometry in the IO or manual-construction reference, and recorded-trace mismatches in `probe-reference.md`.

## Sources

- [Cell topology API](https://brainx.chaobrain.com/braincell/apis/vis.html#cell-topology-plots)
- [BrainCell 0.2.0 migration](https://github.com/chaobrain/braincell/releases/tag/v0.2.0)
- [Point tree visualization](https://brainx.chaobrain.com/braincell/tutorials/vis.html)
- [Region and locset filters](https://brainx.chaobrain.com/braincell/tutorials/filter.html)
- [Morphology](https://brainx.chaobrain.com/braincell/concepts/morphology.html)
- [Cell API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.Cell.html)
