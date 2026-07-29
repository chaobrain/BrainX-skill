# Topology building and visualization

Use this reference after opening `references/multicompartment/multicompartment-cell-workflow.md` when branch topology, control-volume splitting, locset placement, runtime nodes, or spatial values must be inspected rather than inferred.

## Choose the structural view

Morphology branches, CVs, and runtime nodes are different layers of one cell; inspect the layer that owns the suspected error.

| API or object | Use when | Important behavior and result |
|---|---|---|
| `morphology.topo()` | Check imported or manually built parent-child branch structure before discretization. | It renders the named branch tree; it does not show CVs or runtime nodes. |
| `morphology.select(region_or_locset)` | Verify a selector before attaching mechanisms. | A region returns interval data; a locset returns points and `display_names`. |
| `cell.cvs` / `cell.n_cv` | Inspect the membrane-oriented discretization. | The CV preview is resolved lazily from morphology, CV policy, and paint rules and is available before runtime initialization. |
| `cell.init_state()` | Build the execution-oriented runtime. | It transitions the cell from `DECLARING` to `INITIALIZED` and materializes `node_tree`, layouts, State, and probes. |
| `cell.node_tree` | Inspect runtime point and edge connectivity after initialization. | It exposes `nodes`, `edges`, and `root_node_id`; accessing runtime topology while declaring raises `RuntimeError`. |
| `cell.vis_branch(...)` | Inspect compact morphology topology or region coverage. | It draws one node per branch and supports regions only, not locsets or value colormaps. |
| `cell.vis_cv(...)` | Inspect discretization, region coverage, locset ownership, or CV-level values. | It draws one node per CV and supports region, locset, or value mode. |
| `cell.vis_node(...)` | Inspect the full runtime graph, exact lowered placement, or point-level values. | It requires an initialized cell and draws the execution-oriented node tree. |
| `cell.vis_topology(level=...)` | Choose `"branch"`, `"cv"`, or `"node"` dynamically. | It dispatches to the matching visualization method and returns the rendered Matplotlib axes. |

Do not equate morphology sample points, branches, CVs, and runtime nodes. The morphology defines continuous cable, the CV policy defines isopotential intervals, and `init_state()` lowers those intervals and declarations into the node tree.

## Build and inspect runtime topology

Validate selectors against the morphology, inspect CVs before initialization, then inspect the node tree and visual placement after initialization.

```python
import braincell
import braincell.mech as mech
import brainunit as u
import matplotlib.pyplot as plt
from braincell.filter import Terminals, branch_in


morphology = braincell.Morphology.from_swc("neuron.swc")
dendrites = branch_in(
    "type",
    ("dendrite", "basal_dendrite", "apical_dendrite"),
)
terminal_sites = Terminals()

print(morphology.topo())
print("dendrite intervals:", morphology.select(dendrites).intervals)
print("terminal points:", morphology.select(terminal_sites).display_names)

cell = braincell.Cell(
    morphology,
    cv_policy=braincell.MaxCVLen(20.0 * u.um),
)
cell.place(terminal_sites, mech.StateProbe())

assert cell.n_cv == len(cell.cvs)
cell.init_state()

print("CVs:", cell.n_cv)
print("runtime nodes:", len(cell.node_tree.nodes))
print("runtime edges:", len(cell.node_tree.edges))
print("root node:", cell.node_tree.root_node_id)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
cell.vis_branch(region=dendrites, ax=axes[0], show=False)
cell.vis_cv(
    region=dendrites,
    coverage_mode="fraction",
    ax=axes[1],
    show=False,
)
cell.vis_node(locset=terminal_sites, ax=axes[2], show=False)
plt.tight_layout()

fig, ax = plt.subplots()
cell.vis_node(value="V", cmap="viridis", ax=ax, show=False)
plt.show()
```

Use selection inspection to verify semantic targeting and visualization to verify the lowering. A correct branch label can still produce an unsuitable CV layout, and a correct CV layout can still expose a wrong locset or mechanism placement.

## Select a visualization level

| Level | Region highlight | Locset highlight | Value color | Best use |
|---|---|---|---|---|
| `"branch"` | Yes | No | No | Compact branch topology and whole-branch or partial-region coverage. |
| `"cv"` | Yes | Yes | Yes | Spatial discretization, owning-CV placement, and compartment-level values. |
| `"node"` | Yes | Yes | Yes | Full runtime topology, lowered point placement, and point-level State or parameters. |

Use `coverage_mode="fraction"` to blend by overlap fraction, `"any"` to fully highlight any overlap, and `"all"` to highlight only complete coverage. These modes apply to region coverage; locset hits are full-intensity point selections.

## Highlight targets or color values

Highlight mode verifies where a declaration resolves. Value mode displays runtime or user-supplied scalar data.

| `value` form | Result |
|---|---|
| Point-space array of length `n_point` | Color runtime nodes directly. |
| CV-space array of length `n_cv` | Scatter each CV value into point space. |
| `"V"` or `"voltage"` | Color by membrane voltage. |
| `("ion", ion_name, field)` | Color by one ion runtime field. |
| `("channel", class_name, field)` | Color by one channel field, such as `("channel", "IL", "g_max")`. |
| `("layout_id", layout_id, field)` | Color by one resolved runtime layout field. |

Do not combine `value` with `region` or `locset`; highlight mode and value mode are mutually exclusive. At node level, regions and locsets map to the midpoint point of the owning CV, so a highlight is a runtime-placement view rather than a literal rendering of every morphology coordinate.

## Control layout and rendering

| Parameter | Use |
|---|---|
| `preset="dendrotweaks"`, `"mono"`, or `"depth"` | Choose a documented topology styling preset. |
| `layout="twopi"`, `"dot"`, `"neato"`, or `"kamada_kawai"` | Override the graph layout algorithm. |
| `layout_scale=float` | Tighten values below `1.0` or spread values above `1.0`. |
| `highlight_color=...` | Set the selection overlay color. |
| `cmap`, `vmin`, `vmax`, `norm`, `value_label`, `show_colorbar` | Control value-mode normalization and labeling. |
| `node_color`, `edge_color`, `root_color` | Override low-level topology colors. |
| `ax=...`, `show=False` | Render into an existing Matplotlib axes without calling `plt.show()`. |

For debugging, keep the selector and CV policy fixed while changing visualization levels. Keep the visualization level fixed while comparing CV policies. This isolates whether the mismatch comes from branch selection, discretization, or runtime lowering.

## Sources

- [Point tree visualization](https://brainx.chaobrain.com/braincell/tutorials/vis.html)
- [Region and locset filters](https://brainx.chaobrain.com/braincell/tutorials/filter.html)
- [Cell API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.Cell.html)
