# Filter function library

Use this reference after opening `references/multicompartment/multicompartment-cell-workflow.md` when `paint()` or `place()` needs explicit branches, anatomical types, metric ranges, set composition, structural points, or sampled locations.

## Selection model

A region selects continuous branch intervals for `cell.paint(...)`; a locset selects branch-local points for `cell.place(...)`. Keep selection expressions separate from the mechanisms attached to them.

| Selection | Materialized result | Attach with |
|---|---|---|
| `RegionExpr` | `RegionMask.intervals`, containing `(branch_id, prox, dist)` tuples | `cell.paint(region, *density_declarations)` |
| `LocsetExpr` | `LocsetMask.points` and human-readable `display_names` | `cell.place(locset, *point_declarations)` |

Resolve either expression with `morphology.select(expression)` before declaration when imported branch types, names, or topology may differ from the intended target.

## Region selectors

Use regions for cable properties, ions, channels, and other density declarations.

| API | Use when | Important behavior and result |
|---|---|---|
| `AllRegion()` | A declaration applies to every branch interval. | It selects the entire morphology. |
| `EmptyRegion()` | A compositional identity or intentionally empty target is required. | It selects no intervals. |
| `BranchSlice(branch_index, prox, dist)` | Exact branch ids and normalized branch intervals are known. | Scalar or parallel sequence inputs materialize explicit `(branch_id, prox, dist)` intervals. |
| `branch_in(property, values)` | A branch property must equal one value or belong to a set or tuple. | It is the helper form of `BranchInFilter(property, values)` and selects whole matching branches. |
| `BranchInFilter(property, values)` | The explicit selector class is preferable to the helper. | It has the same equality or membership behavior as `branch_in(...)`. |
| `branch_range(property, bounds, closed="neither")` | A numeric or BrainUnit quantity-valued branch property must lie in a range. | It is the helper form of `BranchRangeFilter`; use `None` for an open lower or upper bound. |
| `BranchRangeFilter(property, bounds, closed="neither")` | The explicit range-selector class is preferable to the helper. | `closed` controls whether the left, right, both, or neither bound is included. |

Supported branch properties in the documented implementation are:

| Family | Properties |
|---|---|
| Metadata and topology | `branch_id`, `name`, `type`, `parent_id`, `n_children`, `branch_order`, `n_tapers` |
| Geometry | `length`, `mean_radius`, `area`, `volume` |

Use BrainUnit quantities for physical bounds:

```python
import brainunit as u
from braincell.filter import branch_in, branch_range


dendrites = branch_in(
    "type",
    ("dendrite", "basal_dendrite", "apical_dendrite"),
)
long_branches = branch_range(
    "length",
    (100.0 * u.um, None),
    closed="left",
)
target = dendrites & long_branches
```

Do not use `branch_range(...)` to select a partial interval within each branch. It filters whole branches by a property; use `BranchSlice(...)` for branch-local intervals.

## Region set algebra

Compose small selectors rather than duplicating mechanism declarations.

| Expression | Result |
|---|---|
| `left | right` | Union of both regions. |
| `left & right` | Intersection of both regions. |
| `left - right` | Parts of `left` not selected by `right`. |
| `region.complement()` | All morphology cable outside the region. |

Overlapping slices are normalized in the materialized mask. Inspect `morphology.select(region).intervals` when exact interval boundaries matter.

## Locset selectors

Use locsets for clamps, probes, synapses, junctions, and other point declarations.

| API | Use when | Important behavior and result |
|---|---|---|
| `RootLocation(x)` | Select one normalized position on the root branch. | `RootLocation(0.5)` selects the root midpoint. |
| `AtLocation(branch, x)` | Select a known branch name or id and normalized position. | It returns one explicit branch-local point. |
| `at(branch, x)` | Use the helper form of `AtLocation`. | It returns the same explicit point selector. |
| `BranchPoints()` | Select bifurcation sites. | It chooses branch-point locations on the parent side. |
| `Terminals()` | Select terminal tips. | It chooses the distal endpoint of every terminal branch. |
| `UniformSamples(region, count)` | Place a fixed number of evenly distributed samples within a region. | It returns `count` branch-local points. |
| `RandomSamples(region, count, seed)` | Place reproducible random samples within a region. | The same morphology, region, count, and seed resolve to the same points. |

Locsets support `|`, `&`, and `-` for union, intersection, and difference. A composed locset still materializes as concrete `(branch_id, x)` points, and a multi-point locset expands one placed declaration per point during lowering.

## Verify and apply selectors

Resolve semantic selectors before attaching mechanisms, then use the same expressions for declaration and topology inspection.

```python
import braincell
import braincell.mech as mech
import brainunit as u
from braincell.filter import Terminals, branch_in, branch_range


morphology = braincell.Morphology.from_swc("neuron.swc")

dendrites = branch_in(
    "type",
    ("dendrite", "basal_dendrite", "apical_dendrite"),
)
long_dendrites = dendrites & branch_range(
    "length",
    (100.0 * u.um, None),
    closed="left",
)
terminal_sites = Terminals()

region_mask = morphology.select(long_dendrites)
terminal_mask = morphology.select(terminal_sites)

assert region_mask.intervals, "no long dendrite matched the imported labels"
assert terminal_mask.points, "morphology has no terminal sites"
print(region_mask.intervals)
print(terminal_mask.display_names)

cell = braincell.Cell(morphology)
cell.paint(
    long_dendrites,
    mech.Channel(
        "IL",
        g_max=0.03 * u.mS / u.cm**2,
        E=-65.0 * u.mV,
    ),
)
cell.place(terminal_sites, mech.StateProbe())
cell.init_state()
```

Use a region only with `paint()` and a locset only with `place()`. A selector that resolves correctly against `Morphology` can still map unexpectedly after discretization; open `references/multicompartment/topology-building-and-visualization.md` to inspect CV ownership and runtime placement.

## Exported but unavailable selectors

The current filter API exports additional names, but the official filter tutorial marks these selectors as not implemented:

| Exported name | Intended family |
|---|---|
| `SubtreeRegion` | A branch and its distal subtree. |
| `RadiusRangeRegion` | Cable selected by local radius. |
| `TreeDistanceRegion` | Cable selected by along-tree distance. |
| `EuclideanDistanceRegion` | Cable selected by straight-line distance. |
| `RegionAnchors` | Region anchor construction. |
| `StepSamples` | Step-spaced locset sampling. |

Do not use an exported name as proof that the operation works. Prefer the implemented selectors above, construct equivalent explicit slices or points when practical, or inspect the installed BrainCell version before using a newly implemented surface.

## Common failures

- Do not pass a region to `place()` or a locset to `paint()`.
- Do not guess imported type strings; inspect `branch.type` values or materialize the selector first.
- Do not omit BrainUnit quantities from physical metric bounds.
- Do not confuse `branch_range(...)` property filtering with `BranchSlice(...)` interval selection.
- Do not use unseeded random placement when the model must be reproducible.
- Do not assume a nonempty morphology selector maps to the expected CV midpoint; inspect the topology after choosing the CV policy.

## Sources

- [Regions and locsets](https://brainx.chaobrain.com/braincell/concepts/regions_locsets.html)
- [Region and locset filters](https://brainx.chaobrain.com/braincell/tutorials/filter.html)
- [braincell.filter API](https://brainx.chaobrain.com/braincell/apis/filter.html)
