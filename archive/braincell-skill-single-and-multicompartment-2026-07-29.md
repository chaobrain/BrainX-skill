---
name: braincell
description: BrainCell provides conductance-based single-compartment and morphology-based multicompartment neuron models in BrainX. Use this skill for `SingleCompartment`, Hodgkin-Huxley ions and channels, `Cell`, `Morphology`, control volumes, `paint` and `place`, current clamps, probes, cellular solver choice, and single-cell simulations. Do not use it for wiring cells into networks, synaptic projections between cells, or network training; route those tasks to BrainPy-State.
---

# BrainCell

## Purpose and boundary

Use BrainCell for cellular biophysics at either of its two supported spatial scales:

- Use `SingleCompartment` when voltage is uniform across one isopotential cell and the task is channel prototyping, current injection, fitting, or a batch of independent point neurons.
- Use `Cell` when soma, dendrite, axon, branch geometry, spatial channel distribution, or location-specific stimulation and recording affect the result.

Canonical path:

`choose the cell scale -> declare geometry and mechanisms -> choose units and integration -> initialize -> simulate -> verify voltage, spikes, or probe traces`

Keep network wiring, projections, and network training outside this skill. Open `skills/brainpy-state/SKILL.md` when multiple cells must communicate.

## Underlying principle of BrainCell

`HHTypedNeuron` represents conductance-based membrane dynamics. It supplies the shared ion, channel, state, and integration contract used by both cell front ends.

`SingleCompartment` represents one isopotential neuron. It collapses morphology and discretization and attaches ions and channels imperatively.

`Morphology` represents cable geometry and topology. `Cell` discretizes that geometry into isopotential control volumes whose axial coupling produces spatial voltage dynamics.

`braincell.mech` declarations represent what cellular mechanisms exist, while `braincell.filter` expressions represent where they apply. `paint()` distributes density mechanisms over cable regions; `place()` attaches point mechanisms at locations.

### API structure

| API | Use |
|---|---|
| `braincell.SingleCompartment` | Build one-compartment conductance-based cells and batches of independent point neurons. |
| `braincell.Cell` / `braincell.Morphology` | Build spatially detailed cells from cable geometry and control-volume discretization. |
| `braincell.ion` / `braincell.channel` | Select the ion species and channel implementations used by single-compartment cells. |
| `braincell.mech` | Declare cable properties, ions, channels, clamps, and probes on multicompartment cells. |
| `braincell.filter` | Select cable regions for `paint()` and point locations for `place()`. |
| `braincell.quad` | Select or inspect the integrators shared by both cell front ends. |
| `brainunit` / `brainstate` | Supply mandatory physical quantities, simulation environments, State, and transformed time loops. |

## Build and run a single-compartment cell

A `SingleCompartment` cell stores one membrane voltage per independent neuron and advances its attached ion and channel states from an external current supplied to `update()`.

| API | Description |
|---|---|
| `braincell.SingleCompartment(size, ..., solver=...)` | Use for a conductance-based point cell; `size` sets the independent population shape, `n_compartment` remains one, and `solver` selects a registered integrator. |
| `braincell.ion.*Fixed(size, E=...)` | Use when an ion's reversal potential is constant; it owns `E` and receives channels driven by that ion. |
| `ion.add(name=channel)` | Use after constructing an ion; it registers a channel under that ion so the channel reads the correct reversal potential. |
| `braincell.channel.*` | Use for membrane-current and gating dynamics; inspect `root_type` before attachment, and attach root-cell channels such as `IL` directly to the cell. |
| `cell.init_state()` / `cell.reset_state()` | Use before the first step or to reseed an existing cell; they initialize or reset voltage, spike, ion, and channel State without changing the model declaration. |
| `cell.update(I_ext)` | Use for one integration step; pass a current-density or total-current `Quantity`, let the cell area normalize total current, update all cellular State, and receive the spike output. |
| `brainstate.environ.context()` / `brainstate.transform.for_loop()` | Use to provide `dt` and `t` and to execute a transformed time loop that returns recorded State instead of mutating Python containers. |

```python
import braincell
import brainstate
import brainunit as u


class HH(braincell.SingleCompartment):
    def __init__(self, size, solver="exp_euler"):
        super().__init__(size, V_th=20.0 * u.mV, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        self.IL = braincell.channel.IL(
            size,
            E=-54.387 * u.mV,
            g_max=0.03 * u.mS / u.cm**2,
        )


cell = HH(1)
cell.init_state()
current = 5.0 * u.uA / u.cm**2


def step(t):
    with brainstate.environ.context(t=t):
        spike = cell.update(current)
    return cell.V.value, spike


with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(
        0.0 * u.ms,
        100.0 * u.ms,
        brainstate.environ.get_dt(),
    )
    voltages, spikes = brainstate.transform.for_loop(step, times)

assert voltages.shape[:1] == times.shape
assert spikes.shape[:1] == times.shape
```

Treat `V_th` only as the spike-event threshold; it does not change the membrane equation. Keep membrane capacitance and channel conductance density-based in the canonical path. `update()` accepts current density or total current and uses `SingleCompartment.area` to normalize total input; do not manually area-scale only part of the model. Use `size=N` for `N` independent cells, never for `N` compartments.

Open `references/area-scaled-hh-pattern.md` when changing cell geometry or explicitly converting density parameters to total quantities. Open `references/libraries/ion-library.md`, `references/libraries/channel-library.md`, or `references/mixions-for-adaptation.md` only when the classical fixed-ion HH path is insufficient.

## Build and run a multicompartment cell

A `Cell` lowers morphology, spatial selectors, and declarative mechanisms into coupled control-volume State, then returns only the traces defined by placed probes.

| API | Description |
|---|---|
| `braincell.Morphology` | Use to load or construct branch geometry and topology before creating a cell; it describes continuous cable, not simulation compartments. |
| `braincell.Cell(morpho, cv_policy=..., solver=...)` | Use when spatial structure matters; it clones the morphology, applies a CV policy, and owns distinct declaration and initialized runtime phases. |
| `braincell.CVPolicy` | Use to control spatial resolution; it divides branches into isopotential control volumes, trading accuracy against State size and computation. |
| `cell.paint(region, *mechanisms)` | Use for cable properties and density mechanisms; it records declarations over a cable region and returns the cell for chaining. |
| `cell.place(locset, *mechanisms)` | Use for clamps, probes, and other point mechanisms; it records declarations at selected locations and returns the cell for chaining. |
| `cell.init_state()` | Use after all `paint()` and `place()` calls; it lowers declarations and allocates runtime State, and it raises if the cell is already initialized. |
| `cell.run(dt=..., duration=...)` | Use to advance the initialized cell and collect probe traces; it initializes on first use when needed and requires at least one placed probe. |
| `cell.reset_state()` / `cell.reset()` | Use `reset_state()` to reseed State while remaining initialized; use `reset()` to discard runtime State and return to the declaration phase before changing `paint()` or `place()` rules. |

```python
import braincell
import braincell.mech as mech
import brainunit as u
from braincell.filter import AllRegion, RootLocation, at, branch_in


morphology = braincell.Morphology.from_swc("neuron.swc")
cell = braincell.Cell(
    morphology,
    cv_policy=braincell.CVPerBranch(cv_per_branch=2),
    solver="staggered",
)

cell.paint(
    AllRegion(),
    mech.CableProperty(
        resting_potential=-65.0 * u.mV,
        membrane_capacitance=1.0 * u.uF / u.cm**2,
        axial_resistivity=100.0 * u.ohm * u.cm,
    ),
    mech.Channel("IL", g_max=0.03 * u.mS / u.cm**2, E=-54.387 * u.mV),
)
cell.paint(
    branch_in("type", "soma"),
    mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
    mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2),
)
cell.place(
    RootLocation(x=0.5),
    mech.CurrentClamp(
        delay=20.0 * u.ms,
        durations=60.0 * u.ms,
        amplitudes=0.2 * u.nA,
    ),
)
cell.place(at("soma", 0.5), mech.StateProbe())

cell.init_state()
result = cell.run(dt=0.1 * u.ms, duration=100.0 * u.ms)

assert "soma(0.5)_v" in result.traces
```

Use `paint()` only with regions and density mechanisms. Use `place()` only with locsets and point mechanisms. Declare all mechanisms before `init_state()`; call `reset()` before altering the declaration of an initialized cell.

Open `references/multicompartment/multicompartment-cell-workflow.md` when the task needs morphology IO, manual geometry, selector composition, CV-policy choice, probe variants, topology inspection, or more complex spatial mechanism layouts.

## Reference routing

Open only the smallest reference that owns the non-canonical decision.

| Reference | Open when |
|---|---|
| `references/area-scaled-hh-pattern.md` | Changing single-compartment geometry or converting density parameters consistently to total capacitance and conductance. |
| `references/mixions-for-adaptation.md` | Adding calcium-dependent adaptation, AHP/KCa currents, rebound, dynamic calcium, or multi-ion channel dependencies. |
| `references/libraries/ion-library.md` | Choosing fixed, Nernst-initialized, or dynamic ions and their concentration behavior. |
| `references/libraries/channel-library.md` | Selecting a built-in sodium, potassium, calcium, leak, HCN, or mixed-ion channel. |
| `references/libraries/solver-library-with-effects.md` | Comparing registered integrators, stability, accuracy, and solver-dependent trace effects. |
| `references/braincell/braincell-custom-ion-channel-authoring.md` | Implementing a custom ion or channel after confirming that no built-in mechanism fits. |
| `references/multicompartment/multicompartment-cell-workflow.md` | Extending the canonical `Morphology -> Cell -> paint/place -> run` path with detailed morphology and spatial decisions. |

## Boundaries and common failures

- Use BrainUnit quantities for every physical parameter. A bare number where BrainCell expects a quantity raises `TypeError`.
- Treat `SingleCompartment.size` and `Cell.pop_size` as independent-cell population shapes, not compartment counts; `Cell` derives compartments from its CV policy.
- Match a single-compartment channel to its `root_type`. Put sodium, potassium, and calcium channels on the matching ion, use `MixIons` for multi-ion dependencies, and attach `IL` directly to the cell.
- Do not manually apply area conversion to isolated single-compartment terms; use the cell's current normalization or convert the entire parameter convention consistently.
- Use exponential-Euler variants as the normal single-compartment HH starting point and cable-aware solvers such as `staggered` for multicompartment cells; route comparisons to the solver reference.
- Initialize a `SingleCompartment` before `update()`. Place at least one probe before `Cell.run()`.
- Return values from transformed single-cell loops; do not append to Python lists inside `brainstate.transform.for_loop()`.
- Do not call `paint()` or `place()` after `Cell.init_state()` without first calling `reset()`.
- Keep cellular dynamics here. Route connections, projections, and training across cells to BrainPy-State.
- If the installed multicompartment runtime differs from the documented `Cell.run()` or probe surface, inspect the installed signatures; do not invent a parallel `SingleCompartment.update()` workflow for `Cell`.
