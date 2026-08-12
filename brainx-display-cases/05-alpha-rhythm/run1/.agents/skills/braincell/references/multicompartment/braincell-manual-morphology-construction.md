# BrainCell manual morphology construction

Use this reference after opening `references/multicompartment/multicompartment-cell-workflow.md` when no existing reconstruction should be loaded and the task requires explicit branch geometry, anatomical types, or parent-child topology. Return to the workflow before choosing CVs, mechanisms, probes, or a solver.

## Mental model

A `Branch` is one immutable, unbranched cable geometry; a `Morphology` is the mutable tree that names and attaches those branches. Neither object is a list of simulation compartments.

Typed branch constructors preserve anatomical meaning for later region selectors. `Cell` creates control volumes only after the morphology is complete.

## Workflow

| Step | Action | Important result |
|---|---|---|
| Choose geometry form | Use segment lengths when coordinates are unavailable; use 3-D points when physical coordinates matter. | One unit-aware immutable branch per unbranched cable. |
| Type branches | Construct soma, dendrite, and axon branches with typed classes. | Later selectors can target anatomical `type` reliably. |
| Build the tree | Create one root and attach every child with explicit names and endpoints. | One connected, acyclic morphology with reviewable topology. |
| Inspect | Check topology, branch order, metrics, coordinates, radii, and types. | Geometry errors are found before discretization and mechanism placement. |
| Return to the parent | Pass the trusted `Morphology` into the multicompartment workflow. | CV policy, `Cell`, `paint()`, `place()`, and simulation remain outside this leaf. |

## Construct branches

Choose one geometric representation per branch.

| API | Description |
|---|---|
| `braincell.Branch.from_lengths(lengths=..., radii=..., type=...)` | Use when segment lengths and shared boundary radii are known but 3-D coordinates are not; `radii` has one more value than `lengths`. |
| `braincell.Branch.from_lengths(lengths=..., radii_proximal=..., radii_distal=..., type=...)` | Use when every segment needs an explicit proximal/distal radius pair, including a radius jump. |
| `braincell.Branch.from_points(points=..., radii=..., type=...)` | Use when ordered 3-D coordinates are known; it computes lengths from consecutive points and requires one shared radius per point. |
| `braincell.Soma.from_points(...)` | Use for a point-defined soma without passing `type`; the typed constructor assigns `soma`. |
| `braincell.BasalDendrite.from_points(...)` | Use for a point-defined basal dendrite without passing `type`. |
| `braincell.ApicalDendrite.from_points(...)` | Use for a point-defined apical dendrite without passing `type`. |
| `braincell.Axon.from_points(...)` | Use for a point-defined axon without passing `type`. |

```python
import brainunit as u
from braincell import Branch


schematic_dendrite = Branch.from_lengths(
    lengths=[12.0, 18.0, 10.0] * u.um,
    radii=[3.0, 2.4, 1.8, 1.2] * u.um,
    type="dendrite",
)

point_dendrite = Branch.from_points(
    points=[
        (0.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
        (12.0, 18.0, 0.0),
        (16.0, 34.0, 6.0),
    ] * u.um,
    radii=[3.0, 2.4, 1.8, 1.2] * u.um,
    type="dendrite",
)

assert schematic_dendrite.n_segments == 3
assert point_dendrite.points.shape == (4, 3)
```

All geometry must carry BrainUnit length units. Constructors reject bare values, inconsistent shapes, missing radii, invalid branch types, and zero total length. Consecutive equal points produce a zero-length-segment warning.

Use `from_lengths()` for schematic cable and `from_points()` when projected views, Euclidean measurements, or 3-D rendering matter. Do not invent coordinates only to satisfy `from_points()`.

## Build a typed tree

Create one root with `Morphology.from_root()`, then attach each branch exactly once.

| API | Description |
|---|---|
| `braincell.Morphology.from_root(branch, name=...)` | Use to create a mutable tree around the root branch; prefer an explicit, selector-safe root name such as `soma`. |
| `morphology.attach(parent=..., child_branch=..., child_name=..., parent_x=..., child_x=...)` | Use for explicit attachment; it returns the newly attached `MorphoBranch` and raises for an unknown parent or invalid endpoint. |
| `morphology.soma.dendrite = branch` | Use only as concise syntax when the parent and default endpoints are obvious. |
| `morphology.soma[parent_x, child_x].axon = branch` | Use as concise syntax when the attachment endpoints must remain visible. |

```python
import brainunit as u
from braincell import Axon, BasalDendrite, Morphology, Soma


soma = Soma.from_points(
    points=[(0.0, 0.0, 0.0), (12.0, 0.0, 0.0)] * u.um,
    radii=[6.0, 6.0] * u.um,
)
basal = BasalDendrite.from_points(
    points=[
        (12.0, 0.0, 0.0),
        (26.0, 8.0, 0.0),
        (40.0, 14.0, 0.0),
    ] * u.um,
    radii=[2.6, 2.0, 1.4] * u.um,
)
axon = Axon.from_points(
    points=[
        (0.0, 0.0, 0.0),
        (-12.0, -6.0, 0.0),
        (-26.0, -14.0, -2.0),
    ] * u.um,
    radii=[1.0, 0.8, 0.6] * u.um,
)

morphology = Morphology.from_root(soma, name="soma")
morphology.attach(
    parent="soma",
    child_branch=basal,
    child_name="basal",
    parent_x=1.0,
    child_x=0.0,
)
morphology.attach(
    parent="soma",
    child_branch=axon,
    child_name="axon",
    parent_x=0.0,
    child_x=0.0,
)

assert morphology.n_branches == 3
assert morphology.has_full_point_geometry
print(morphology.topo())
```

`parent_x` may be `0`, `0.5`, or `1`; midpoint attachment is allowed only on a soma. `child_x` may currently be `0` or `1`. Keep branch names valid Python identifiers and avoid names reserved by morphology methods or metrics.

Coordinates at an attachment should describe the intended physical junction. The topology API records which endpoints connect, but it does not make inconsistent point coordinates scientifically valid.

## Inspect before simulation

Construction checks structural input constraints; inspection establishes that the resulting tree represents the intended neuron.

| API | Description |
|---|---|
| `morphology.topo()` | Use to review the named parent-child tree. |
| `morphology.branches` | Use to inspect every branch in default node order. |
| `morphology.branch(name=...)` | Use to inspect one named branch and its type. |
| `morphology.branch_by_order(order=...)` | Use to compare default, type, or depth ordering. |
| `morphology.metric` | Use to inspect branch counts, stems, bifurcations, path length, area, volume, and coordinate-dependent metrics. |
| `morphology.has_full_point_geometry` | Use before coordinate-dependent analysis or 3-D rendering. |
| `morphology.vis2d(...)` | Use for a topology or projected-geometry check without constructing a `Cell`. |
| `morphology.vis3d(...)` | Use only when every branch has complete 3-D point geometry. |

Verify at least:

- exactly one intended root;
- every expected branch appears once in `topo()`;
- soma, dendrite, and axon types match later selectors;
- all lengths and radii are positive and use the intended scale;
- branch attachment endpoints and coordinates agree;
- branch counts, path distances, surface area, and volume are plausible;
- `has_full_point_geometry` is true before relying on Euclidean metrics or 3-D views.

Open `references/multicompartment/topology-building-and-visualization.md` when branch, CV, or runtime-node visualization is required. Return to `references/multicompartment/multicompartment-cell-workflow.md` when the failure appears only after CV construction or runtime lowering.

## Return to the multicompartment workflow

After inspection, return to `references/multicompartment/multicompartment-cell-workflow.md` and continue with:

`trusted Morphology -> CV policy -> Cell -> paint -> place -> init_state -> inspect runtime -> run -> verify`

Do not choose a CV count from the number of source points or branches. The CV policy owns simulation resolution.

## Common failures

- Passing bare coordinates, lengths, or radii: attach BrainUnit length units to every geometric value.
- Passing `type=` to a typed constructor: use either `Branch(..., type=...)` or a typed subclass, not both.
- Mixing the two radius forms: provide `radii` or the complete proximal/distal pair.
- Creating a disconnected or multiply attached branch: attach every non-root branch exactly once.
- Using an invalid midpoint attachment: reserve `parent_x=0.5` for a soma.
- Losing anatomical types: construct typed branches before writing type-based filters.
- Treating successful construction as biological validation: inspect topology and morphology metrics before building a `Cell`.
- Treating branches or points as compartments: select and refine a CV policy in the parent workflow.
- Adding mechanisms before geometry is trusted: finish morphology inspection first.

## Sources

- Morphology concept: https://brainx.chaobrain.com/braincell/concepts/morphology.html
- Morphology tutorial: https://brainx.chaobrain.com/braincell/tutorials/morphology.html
- Cell tutorial: https://brainx.chaobrain.com/braincell/tutorials/cell.html
- `Branch` API: https://brainx.chaobrain.com/braincell/apis/generated/braincell.Branch.html
- `Morphology` API: https://brainx.chaobrain.com/braincell/apis/generated/braincell.Morphology.html
