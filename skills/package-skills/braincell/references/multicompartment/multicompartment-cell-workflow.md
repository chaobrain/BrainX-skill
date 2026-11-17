# BrainCell multicompartment cell workflow

## Purpose and boundary

Use this reference when a BrainCell simulation depends on geometry, axial current, regional biophysics, or location-specific stimulation and recording. Use `SingleCompartment` for isopotential cells. Keep independent cells on the population axis and resolve compartments through the morphology and CV policy.

## Underlying principle of BrainCell multicompartment cells

`braincell.filter` expressions represent spatial targets. Regions select cable for `paint()`; locsets select points for `place()`.

`braincell.mech` declarations represent cellular biophysics and point processes. Paint cable properties, ions, and channels over regions. Place clamps and synapses at locsets. Declare recordings on the cell or a spatial view.

## Workflow map

```text
┌──────────────────────────────────────────────────────────────────┐
│ Declaration (what to model)                                      │
│   • Morphology          geometry: branches, radii, tree           │
│   • mech.*              channels, ions, clamps, synapses          │
│   • filter.*            regions and locsets (where)               │
└──────────────────────────────────────────────────────────────────┘
                                │
                         paint / place
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Discretization (_cv)                                             │
│   • CV                  one isopotential control volume           │
│   • CVPolicy            how many CVs each branch gets             │
└──────────────────────────────────────────────────────────────────┘
                                │
                              build
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Runtime (_compute)                                               │
│   • PointTree           execution graph over CVs                  │
│   • CellRuntimeState    frozen, JAX-friendly state                │
└──────────────────────────────────────────────────────────────────┘
                                │
                              step
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Integration (quad)                                               │
│   • DiffEqModule        defines f(t, y)                           │
│   • solver              advances y by dt                         │
└──────────────────────────────────────────────────────────────────┘
```

## Prepare and validate the morphology

`Morphology` is the continuous geometric source of truth, so validate it before selectors, discretization, or mechanisms inherit its mistakes.

| API | Description |
|---|---|
| `braincell.Morphology.from_swc(path, ...)` | Use for an SWC reconstruction; it maps structure identifiers to branch types and returns a `Morphology`. |
| `braincell.Morphology.from_asc(path, ...)` | Use for a Neurolucida ASC reconstruction; inspect its metadata and validation report before modeling. |
| `braincell.Morphology.from_neuromorpho(neuron_id, ...)` | Use for a NeuroMorpho.Org record; it downloads or reuses cached data and returns a `Morphology`. |
| `braincell.Morphology.from_swc(path, return_report=True)` | Use when parser findings or applied fixes must be reviewed; it returns `(morphology, report)`. |
| `braincell.io.SwcReadOptions(...)` | Use when SWC validation policy must change, including safe standardization, unknown-type handling, or root-soma requirements. |
| `morphology.topo()` | Use immediately after loading to inspect branch names, types, and parent-child structure before writing selectors. |

```python
import braincell
from braincell.io import SwcReadOptions


options = SwcReadOptions(
    standardize_safe_fixes=True,
    unknown_type_as_custom=True,
    require_root_type_soma=True,
)
morphology, report = braincell.Morphology.from_swc(
    "neuron.swc",
    options=options,
    return_report=True,
)

print(report)
print(morphology.topo())
```

A successful parse proves readability only; verify the root, connectivity, geometry, branch names, and anatomical types before writing selectors.

Open `references/multicompartment/morphology-io-loading-validation.md` for other loaders, validation, or checkpoints; open `references/multicompartment/braincell-manual-morphology-construction.md` for hand-built geometry.

## Choose and inspect the CV policy

A CV policy chooses the State resolution of the cable model; morphology branches and source sample points do not determine the number of simulation compartments.

| API | Description |
|---|---|
| `braincell.Cell(morphology, cv_policy=None)` | Use only when the package default is acceptable; it currently uses `CVPerBranch()`, which is predictable but is not proof of spatial convergence. |
| `braincell.CVPerBranch(cv_per_branch=N)` | Use for controlled examples, tests, or a fixed resolution per branch; it gives every branch the same CV count regardless of physical or electrical length. |
| `braincell.MaxCVLen(max_cv_len=length)` | Use when no CV may exceed a physical length; long branches receive more CVs. |
| `braincell.DLambda(d_lambda=..., frequency=..., keep_odd=True)` | Use as the normal detailed-cable starting point; it refines electrically long cable using the AC length constant and requires uniform capacitance and axial resistivity within each branch. |
| `braincell.CVPolicyByTypeRule(...)` | Use when soma, dendrite, and axon require different rules. |
| `cell.n_cv` / `cell.cvs` | Use before initialization to inspect the resolved CV count and intervals; each CV exposes branch identity, normalized boundaries, geometry, topology, and cable properties. |

```python
import braincell
import braincell.mech as mech
import brainunit as u
from braincell.filter import AllRegion


policy = braincell.DLambda(
    d_lambda=0.1,
    frequency=100.0 * u.Hz,
)
cell = braincell.Cell(
    morphology,
    cv_policy=policy,
    solver="staggered",
)

# DLambda consumes capacitance and axial resistivity from paint rules.
cell.paint(
    AllRegion(),
    mech.CableProperty(
        resting_potential=-65.0 * u.mV,
        membrane_capacitance=1.0 * u.uF / u.cm**2,
        axial_resistivity=100.0 * u.ohm * u.cm,
    ),
)

# Inspect final CVs only after recording the intended cable properties.
print("n_cv:", cell.n_cv)
for cv in cell.cvs[:3]:
    print(cv.id, cv.branch_id, cv.prox, cv.dist, cv.length, cv.area)
```

Validate every policy by refining it and comparing the target observable. Resolve cable properties before `DLambda`; split capacitance or axial-resistivity changes at branch boundaries.

Open `references/multicompartment/cv-policy-reference.md` for by-type or composite policies, recording-site mapping, and convergence.

## Declare distributed biophysics

Distributed declarations define the membrane and cable equations over regions; paint a complete passive baseline first, then add regional active mechanisms.

| API | Description |
|---|---|
| `braincell.filter.AllRegion()` | Use for passive cable values or density mechanisms that cover the entire morphology. |
| `braincell.filter.branch_in(property, value)` | Use to select branches by metadata such as `type="soma"`; it returns a region for `paint()`. |
| `braincell.filter.BranchSlice(...)` | Use to target a normalized interval of one or more branches when whole-type selection is too broad. |
| `braincell.mech.CableProperty(...)` | Use to declare resting potential, membrane-capacitance density, axial resistivity, and temperature over a region. |
| `braincell.mech.Ion(name_or_class, **parameters)` | Use when a density channel requires an explicit fixed, Nernst, or dynamic ion declaration or a named ion owner. |
| `braincell.mech.Channel(name_or_class, **parameters)` | Use to install a registered density channel with unit-aware conductance and channel parameters. |
| `cell.paint(region, *declarations)` | Use only in DECLARING; it records cable or density declarations over the selected cable and returns the cell for chaining. |

```python
import braincell.mech as mech
import brainunit as u
from braincell.filter import AllRegion, branch_in


cell.paint(
    AllRegion(),
    mech.Channel(
        "IL",
        g_max=0.03 * u.mS / u.cm**2,
        E=-54.387 * u.mV,
    ),
)
cell.paint(
    branch_in("type", "soma"),
    mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
    mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2),
)
```

BrainCell area-scales capacitance and conductance densities during lowering; keep axial resistivity in resistance-length units and all physical values as BrainUnit quantities. Paint an explicit global `CableProperty` baseline before regional overrides.

Open `references/multicompartment/filter-function-library.md` for composed or geometry-based regions, `references/ion-library.md` or `references/channel-library.md` for non-canonical mechanisms, and `references/mixions-for-adaptation.md` for calcium-dependent or multi-ion coupling.

## Place stimuli and declare recordings

Place current clamps at locsets. Register observables on a selected view before initialization.

| API | Description |
|---|---|
| `braincell.filter.RootLocation(x)` | Use for one point on the root branch; `x` is the normalized position from 0 to 1. |
| `braincell.filter.at(branch, position)` | Use for one named branch position; the branch argument identifies a branch, not an anatomical type. |
| `braincell.filter.Terminals()` | Use when a point declaration must resolve at every terminal tip. |
| `braincell.mech.CurrentClamp(...)` | Use for piecewise current injection; declare delay, durations, and amplitudes with time and current units. Currents from clamps at the same runtime point add. |
| `cell.loc(locset).record(name, observable)` | Use to record over the CVs selected by a locset. Select membrane voltage with `braincell.observe.state("v")`. |
| `cell.place(locset, *point_mechanisms)` | Use only in DECLARING; it records point declarations at every resolved location and returns the cell for chaining. |

```python
from braincell.filter import RootLocation


soma_midpoint = RootLocation(x=0.5)
cell.place(
    soma_midpoint,
    mech.CurrentClamp(
        delay=20.0 * u.ms,
        durations=60.0 * u.ms,
        amplitudes=0.2 * u.nA,
    ),
)
cell.loc(soma_midpoint).record("soma_v", braincell.observe.state("v"))
```

Open `references/multicompartment/filter-function-library.md` for composed locsets and `references/multicompartment/probe-reference.md` for observable selection, sample schedules, result schemas, or legacy probe migration.

## Initialize and inspect the lowered runtime

`init_state()` is the declaration-to-runtime boundary; call it explicitly when model-construction errors and target resolution must be checked before a long run.

| API | Description |
|---|---|
| `cell.init_state(batch_size=None)` | Use after declaring mechanisms and recordings. It lowers declarations, allocates evolving model State, and raises if the cell is already initialized. |
| `cell.node_tree` | Use after initialization to inspect the point-and-edge execution topology derived from the CV tree. |
| `cell.layouts` | Use after initialization to inspect each lowered mechanism layout, including its kind, target, active count, source CVs, and runtime point indices. |

```python
cell.init_state()

assert cell.V.value.shape == cell.pop_size + (cell.n_cv,)
print("runtime nodes:", len(cell.node_tree.nodes))
for layout in cell.layouts:
    print(layout.kind, layout.target, layout.n_active)
```

Inspect `cell.cvs` before initialization and runtime voltage and mechanism layouts afterward. In BrainCell 0.2.0 the unbatched voltage has population axes followed by the CV axis, even for a singleton population.

Open `references/multicompartment/topology-building-and-visualization.md` to visualize CV, branch, node, or placement topology; open `references/multicompartment/probe-reference.md` for sampling failures.

## Run and interpret the result

`run()` advances from the current cell time. Each recording block carries the times for its own sample schedule.

| API | Description |
|---|---|
| `cell.run(dt=..., duration=...)` | Use with positive time quantities and a duration divisible by `dt`. It initializes a standalone cell on first use. |
| `braincell.RunResult.time` | Use as the unit-aware `(n_steps,)` time axis spanning `[start_t, start_t + duration)`. |
| `braincell.RunResult.samples` | Look up a named recording block and use its `time`, `values`, and `schema` together. |
| `cell.current_time` | Use after a run to inspect the endpoint from which the next `run()` call will continue. |
| `Quantity.to_decimal(unit)` | Use only at plotting, serialization, or external-library boundaries to obtain values in an explicit compatible unit. |

```python
result = cell.run(
    dt=0.05 * u.ms,
    duration=100.0 * u.ms,
)

voltage = result.samples["soma_v"]
assert voltage.values.shape == (voltage.time.size, 1)
times_ms = voltage.time.to_decimal(u.ms)
voltage_mV = voltage.values[:, 0].to_decimal(u.mV)
```

Keep quantities unit-aware and convert only at external boundaries. Validate `dt` by comparing the target observable at `dt / 2`.

Open `references/solver-library-with-effects.md` for cable-solver choice or method-dependent traces.

## Canonical end-to-end workflow

This current-clamp pattern uses an explicit passive baseline, `DLambda` with staggered integration, somatic active channels, runtime target checks, and one voltage recording. Supply a valid SWC file containing a soma at `neuron.swc`.

```python
import braincell
import braincell.mech as mech
import brainunit as u
from braincell.filter import AllRegion, RootLocation, branch_in
from braincell.io import SwcReadOptions


# 1. Load and validate continuous cable geometry.
options = SwcReadOptions(
    standardize_safe_fixes=True,
    unknown_type_as_custom=True,
    require_root_type_soma=True,
)
morphology, report = braincell.Morphology.from_swc(
    "neuron.swc",
    options=options,
    return_report=True,
)
print(report)
print(morphology.topo())

# 2. Choose spatial and temporal integration policies.
cell = braincell.Cell(
    morphology,
    cv_policy=braincell.DLambda(
        d_lambda=0.1,
        frequency=100.0 * u.Hz,
    ),
    solver="staggered",
)

# 3. Declare passive cable everywhere and active HH channels on the soma.
cell.paint(
    AllRegion(),
    mech.CableProperty(
        resting_potential=-65.0 * u.mV,
        membrane_capacitance=1.0 * u.uF / u.cm**2,
        axial_resistivity=100.0 * u.ohm * u.cm,
    ),
    mech.Channel(
        "IL",
        g_max=0.03 * u.mS / u.cm**2,
        E=-54.387 * u.mV,
    ),
)
cell.paint(
    branch_in("type", "soma"),
    mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
    mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2),
)

# 4. Place one stimulus and one readout at the root midpoint.
soma_midpoint = RootLocation(x=0.5)
cell.place(
    soma_midpoint,
    mech.CurrentClamp(
        delay=20.0 * u.ms,
        durations=60.0 * u.ms,
        amplitudes=0.2 * u.nA,
    ),
)
cell.loc(soma_midpoint).record("soma_v", braincell.observe.state("v"))

# 5. Inspect the final spatial layout before allocating runtime State.
print("n_cv:", cell.n_cv)
for cv in cell.cvs[:3]:
    print(cv.id, cv.branch_id, cv.prox, cv.dist, cv.length, cv.area)

# 6. Lower declarations, verify resolved targets, and run.
cell.init_state()
assert cell.V.value.shape == (1, cell.n_cv)

for layout in cell.layouts:
    print(layout.kind, layout.target, layout.n_active)

result = cell.run(dt=0.05 * u.ms, duration=100.0 * u.ms)
voltage = result.samples["soma_v"]
assert voltage.values.shape == (voltage.time.size, 1)
times_ms = voltage.time.to_decimal(u.ms)
voltage_mV = voltage.values[:, 0].to_decimal(u.mV)
```

Replace the example morphology, mechanisms, stimulus, readout, `dt`, and CV policy, then validate the target observable under numerical refinement.

Open `references/scripts/cell_multicompartment_reference.py` for the full tutorial with CV, node-tree, layout, policy, and plotting inspection. Pass the path to a soma-and-dendrite SWC as the script's first argument.

## Reset, continue, or revise

Use continuation, State reset, declaration reset, or a new cell according to what must change.

| Intent | API | Result |
|---|---|---|
| Continue the same trajectory | Call `cell.run(...)` again. | The next run starts at `cell.current_time`; the cell is not reinitialized. |
| Repeat the same declaration from reset State | Call `cell.reset_state()`, then `cell.run(...)`. | The cell remains INITIALIZED while voltage, spike, time, and mechanism State are reseeded through their reset hooks. |
| Change the declaration | Call `cell.reset()` to reopen declaration before adding mechanisms or recordings. Build a fresh cell when replacing an existing recording name. | Runtime State is discarded. Existing declarations are retained. |
| Compare CV policies or independent numerical trials | Construct separate cells from one model-building function. | Each comparison has independent declarations, runtime State, and time. |

## Reference routing

Open the smallest child for the next decision.

| Reference | Open when |
|---|---|
| `references/multicompartment/morphology-io-loading-validation.md` | Loading SWC, ASC, NeuroML2, NeuroMorpho.Org, or checkpoint data; configuring validation; or checking loaded geometry. |
| `references/multicompartment/braincell-manual-morphology-construction.md` | Constructing branches, points, radii, types, and parent-child topology without an existing morphology file. |
| `references/multicompartment/filter-function-library.md` | Selecting or composing non-canonical regions and locsets for `paint()` and `place()`. |
| `references/multicompartment/cv-policy-reference.md` | Selecting advanced CV policies or evaluating spatial resolution, cost, recording-site mapping, and convergence. |
| `references/multicompartment/probe-reference.md` | Selecting voltage, mechanism State, or current, checking sampling schedules and result rows, or migrating legacy probes. |
| `references/multicompartment/network-workflow.md` | Delivering events to spatially placed synapses, registering populations, or running and resetting connected cells. |
| `references/multicompartment/topology-building-and-visualization.md` | Inspecting and visualizing morphology, branch, CV, node, region, locset, or runtime topology. |
| `references/ion-library.md` | Choosing fixed, Nernst, or dynamic ion declarations and concentration behavior. |
| `references/channel-library.md` | Choosing registered sodium, potassium, calcium, leak, HCN, or mixed-ion density channels. |
| `references/mixions-for-adaptation.md` | Adding calcium-dependent adaptation, AHP/KCa currents, rebound, or multi-ion dependencies. |
| `references/solver-library-with-effects.md` | Choosing or comparing cable-aware solvers, step sizes, stability, accuracy, and solver-dependent traces. |
| `references/braincell-custom-ion-channel-authoring.md` | Implementing a custom ion or channel after confirming that no built-in declaration fits. |
| `references/scripts/cell_multicompartment_reference.py` | Inspecting the complete official tutorial workflow and its runtime structures in one script. |
