---
name: braincell
description: Use for cell modeling with conductance-based or Hodgkin-Huxley ions, channels, compartments, or morphology. Use alone for cell studies or together with BrainPy-State and/or BrainMass when cellular mechanisms participate in a multiscale model.
---

# BrainCell

## Purpose and boundary

Use BrainCell for cellular biophysics at either of its two supported spatial scales:

- Use `SingleCompartment` when voltage is uniform across one isopotential cell and the task is channel prototyping, current injection, fitting, or a batch of independent point neurons.
- Use `Cell` when soma, dendrite, axon, branch geometry, spatial channel distribution, or location-specific stimulation and recording affect the result.


Keep network wiring, projections, and network training outside this skill. Open `skills/brainpy-state/SKILL.md` when multiple cells must communicate.

## Underlying principle of BrainCell

`HHTypedNeuron` represents conductance-based membrane dynamics. It supplies the shared ion, channel, state, and integration contract used by both cell front ends.

`SingleCompartment` represents one isopotential neuron. It collapses morphology and discretization and attaches ions and channels imperatively.

`Morphology` represents cable geometry and topology. `Cell` discretizes that geometry into isopotential control volumes whose axial coupling produces spatial voltage dynamics.

`braincell.mech` declarations represent what cellular mechanisms exist, while `braincell.filter` expressions represent where they apply. `paint()` distributes density mechanisms over cable regions; `place()` attaches point mechanisms at locations.

### API structure

| Module | Use |
|---|---|
| `braincell` | Use the public modeling surface for cells, morphology and branch types, control-volume policies, ion-channel bases, clamps, and run results. |
| `braincell.morph` | Represent, manipulate, and measure neuronal morphology structures before simulation. |
| `braincell.ion` | Choose sodium, potassium, and calcium ion models and their reversal-potential or concentration dynamics. |
| `braincell.channel` | Choose built-in membrane-channel dynamics for calcium, HCN, leak, potassium, potassium-calcium, and sodium currents. |
| `braincell.synapse` | Choose built-in AMPA, GABAa, and NMDA Markov synapse models. |
| `braincell.mech` | Declare what a multicompartment `Cell` installs: cable properties, density ions and channels, point mechanisms, and probes. |
| `braincell.filter` | Build regions for `paint()` and locsets for `place()`, including composed spatial selections. |
| `braincell.io` | Load, validate, checkpoint, and retrieve morphologies from supported file formats and NeuroMorpho.Org. |
| `braincell.quad` | Define differential-equation modules and choose, inspect, or register numerical integrators. |
| `braincell.vis` | Visualize morphologies, topology, control volumes, values, traces, and morphometric analyses in 2D or 3D. |

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

Treat `V_th` only as the spike-event threshold; it does not change the membrane equation. Keep membrane capacitance and channel conductance density-based in the canonical path. `update()` accepts current density or total current and uses `SingleCompartment.area` to normalize total input; do not manually area-scale only part of the model. Use `size=N` for `N` independent cells, never for `N` compartments. For a multidimensional condition sweep, use the condition-grid shape as `size` and pass shape-aligned channel parameters and inputs; do not wrap identity functions in `vmap` merely to construct that grid.

Open `references/area-scaled-hh-pattern.md` when changing cell geometry or explicitly converting density parameters to total quantities. Open `references/ion-library.md`, `references/channel-library.md`, or `references/mixions-for-adaptation.md` only when the classical fixed-ion HH path is insufficient.

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

Open `references/multicompartment/multicompartment-cell-workflow.md` for the complete advanced morphology-to-simulation path. Let that workflow select the exclusive morphology, filter, CV-policy, probe, or topology reference; do not open those leaves directly from this root skill.

## Reference routing

Open only the smallest reference that owns the non-canonical decision.

| Reference | Open when |
|---|---|
| `references/area-scaled-hh-pattern.md` | Changing single-compartment geometry or converting density parameters consistently to total capacitance and conductance. |
| `references/braincell-custom-ion-channel-authoring.md` | Implementing a custom ion or channel after confirming that no built-in mechanism fits. |
| `references/channel-library.md` | Selecting a built-in sodium, potassium, calcium, leak, HCN, or mixed-ion channel. |
| `references/ion-library.md` | Choosing fixed, Nernst-initialized, or dynamic ions and their concentration behavior. |
| `references/mixions-for-adaptation.md` | Adding calcium-dependent adaptation, AHP/KCa currents, rebound, dynamic calcium, or multi-ion channel dependencies. |
| `references/multicompartment/multicompartment-cell-workflow.md` | Extending the canonical `Morphology -> Cell -> paint/place -> run` path and selecting the exclusive multicompartment references. |
| `references/solver-library-with-effects.md` | Comparing registered integrators, stability, accuracy, and solver-dependent trace effects. |

## Application script examples

Open one script only when its pattern matches the task; each is a complete, runnable program mirrored from the official BrainCell examples.

| Script | Open when |
|---|---|
| `references/scripts/hh_neuron_basics.py` | Building one end-to-end `SingleCompartment` current-clamp simulation with Na/K/leak currents; this is the default full-script reference. |
| `references/scripts/fi_curve.py` | Producing FI curves, current sweeps, spike counts, or firing rates, where `size=N` means N independent point neurons. |
| `references/scripts/channel_ablation.py` | Comparing intact against ablated dynamics by setting a conductance such as `g_max` to zero while preserving the ion and channel structure. |
| `references/scripts/calcium_channel_gating.py` | Inspecting voltage-dependent activation or inactivation curves through direct channel methods rather than simulating a membrane trace. |
| `references/scripts/spike_frequency_adaptation.py` | Adding spike-frequency adaptation or calcium-dependent afterhyperpolarization with dynamic calcium and `MixIons`. |
| `references/scripts/t_current_rebound.py` | Producing post-inhibitory rebound bursting from a hyperpolarizing step with T-type calcium and HCN currents. |
| `references/scripts/thalamic_neurons.py` | Comparing several thalamic point-neuron phenotypes built from different ion and channel compositions in one script. |
