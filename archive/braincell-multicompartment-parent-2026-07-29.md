# BrainCell Multicompartment Cell Parent Reference

Use this reference for morphology-based BrainCell modeling: soma, dendrite, axon, branches, morphology files, `Cell`, `Morphology`, regions, locsets, `paint`, `place`, CV policies, cable equations, distributed mechanisms, point clamps/probes, SWC/ASC/NeuroML2 inputs, or geometry-dependent electrophysiology.

This parent reference is the only first-hop morphology route under `skills/braincell/SKILL.md`. Single-compartment and multicompartment BrainCell models share HH ion/channel abstractions, units, solvers, and BrainState execution rules, but geometry tasks start here with `Cell(morpho)`, not with a morphology child or `SingleCompartment`.

## Exclusive nested-reference ownership

This parent reference is the only first-hop selector for these six exclusive nested child references:

- `references/braincell/braincell-manual-morphology-construction.md`
- `references/braincell/morphology-io-loading-validation.md`
- `references/braincell/topology-building-and-visualization.md`
- `references/braincell/probe-reference.md`
- `references/libraries/filter-function-library.md`
- `references/libraries/cv-policy-reference.md`

`skills/braincell/SKILL.md` and the root bundle skill must route to this parent, never directly to those children. After the parent establishes the tree, children may cross-route within it. `references/diagnostics/common-failures-index.md` is a second-level diagnostic child reached only after the manual-morphology, topology, or probe child identifies a failure mode.

## Core Concepts

- `Morphology` describes the soma/dendrite/axon tree. Load it from SWC/ASC/NeuroML2/checkpoints or build it manually before constructing a cell.
- `Cell(morpho)` is the multicompartment front end. It discretizes morphology into control volumes and lowers declarative mechanisms into runtime layouts during `init_state()`.
- A CV is the discretized compartment interval BrainCell solves over; CV policy controls spatial resolution and runtime cost.
- `paint(region, mechanism)` is for distributed declarations over regions, including cable properties and density mechanisms.
- `place(locset, mechanism)` is for point declarations at locations, including clamps and probes.
- `CableProperty` controls passive midpoint properties such as resting potential, membrane capacitance, axial resistivity, and temperature.
- `Channel("IL", ...)`, `Channel("Na_HH1952", ...)`, and related declarations describe density mechanisms that are lowered over active CV midpoints.
- `CurrentClamp`, `StateProbe`, `MechanismProbe`, and `CurrentProbe` are point mechanisms; place them on locsets before `init_state()`.

## Canonical Workflow

1. Load or build a `Morphology`.
2. Choose a CV policy and solver.
3. Build `Cell(morpho, cv_policy=..., solver=...)`.
4. Use `paint` for cable properties and density mechanisms.
5. Use `place` for clamps, probes, and other point mechanisms.
6. Call `cell.init_state()` before inspecting runtime layouts or running.
7. Call `cell.reset_state()` when the simulation should start from initialized state.
8. Run with `cell.run(dt=..., duration=...)` or inspect layouts/probes directly when debugging.

## Minimal Patterns

Load a morphology and inspect the generated cell:

```python
from braincell import Cell, Morphology

morpho = Morphology.from_swc("../../data/morphology/example_tree.swc")
print(morpho.topo())

cell = Cell(morpho)
print(cell)
```

Paint cable properties on a branch region:

```python
import brainunit as u

from braincell import Cell, CVPerBranch
from braincell.filter import BranchSlice
from braincell.mech import CableProperty

cable_cell = Cell(morpho, cv_policy=CVPerBranch(cv_per_branch=2))
cable_cell.paint(
    BranchSlice(branch_index=0, prox=0.0, dist=1.0),
    CableProperty(
        resting_potential=-70.0 * u.mV,
        membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
        axial_resistivity=200.0 * (u.ohm * u.cm),
        temperature=u.celsius2kelvin(20.0),
    ),
)
```

Paint a density mechanism over a region:

```python
from braincell.mech import Channel

channel_cell = Cell(morpho)
channel_cell.paint(
    BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
    Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
)
```

Place point mechanisms at a locset:

```python
from braincell.filter import RootLocation
from braincell.mech import CurrentClamp, StateProbe

place_cell = Cell(morpho, cv_policy=CVPerBranch(2))
place_cell.place(
    RootLocation(x=0.5),
    CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
)
place_cell.place(RootLocation(x=0.5), StateProbe())
place_cell.init_state()
```

## CV Policy

Use CV policy references when spatial discretization matters:

```python
import braincell
import brainunit as u

policy = braincell.CVPerBranch(cv_per_branch=2)
policy = braincell.MaxCVLen(max_cv_len=20. * u.um)
```

Open `references/libraries/cv-policy-reference.md` before refining spatial resolution, comparing runtime cost, or debugging probe/CV placement.

## Existing Ions, Channels, And MixIons

Reuse the first-layer `references/libraries/ion-library.md`, `references/libraries/channel-library.md`, and `references/libraries/solver-library-with-effects.md` for ion, channel, and solver selection. Reuse does not make those files exclusive children of this parent.

If no built-in channel fits, open the first-layer sibling `references/braincell/braincell-custom-ion-channel-authoring.md`; this multicompartment parent does not own that reference.

`references/mixions-for-adaptation.md` is a first-layer BrainCell reference owned directly by `skills/braincell/SKILL.md`. Reuse it here when a multicompartment task needs calcium-dependent adaptation, AHP/KCa mechanism selection, or `MixIons(k, ca)` reasoning, then return here for morphology-specific `Cell`, `paint`, `place`, clamp, probe, and CV workflow.

## Nested Script Reference

- `references/multicompartment/references/cell_multicompartment_reference.py`
  Source mirrored: https://brainx.chaobrain.com/braincell/tutorials/cell.html
  Purpose: primary full-script reference for turning morphology into a simulation-ready multicompartment `Cell`.
  Covers: `Morphology.from_swc`, `Cell`, CVs, CV policies, `init_state`, `node_tree`, `paint`, `place`, density mechanisms, point mechanisms, `CurrentClamp`, `StateProbe`, and minimal simulation with `run`.

## Nested child references

Open these only after this multicompartment parent establishes the morphology task. Do not route to them directly from the main BrainCell skill or the bundle router.

- `references/braincell/braincell-manual-morphology-construction.md` - manual morphology construction.
- `references/braincell/morphology-io-loading-validation.md` - morphology loading, validation, NeuroMorpho, and checkpoints.
- `references/braincell/topology-building-and-visualization.md` - runtime topology, NodeTree, CV/branch/node views, and visualization.
- `references/braincell/probe-reference.md` - probe types, trace keys, sampling, and missing trace debugging.
- `references/libraries/filter-function-library.md` - region and locset selectors.
- `references/libraries/cv-policy-reference.md` - CV policy choices and discretization tradeoffs.

## Common Mistakes

- Using `SingleCompartment` after the user asks for dendrites, soma targeting, morphology import, regions, locsets, CVs, or `paint/place` -> switch to `Cell(morpho)`.
- Using `place` for a distributed density mechanism -> use `paint(region, ...)`.
- Using `paint` for a point clamp or probe -> use `place(locset, ...)`.
- Inspecting layouts, probes, or node tree before `init_state()` -> initialize first.
- Debugging a missing trace without checking locset resolution and probe key names -> open `references/braincell/probe-reference.md`.
- Choosing a cable/composite solver for a point-neuron task -> route back to the main single-compartment skill unless geometry matters.
