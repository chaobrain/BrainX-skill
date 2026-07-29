# BrainCell ion library

Use this reference when choosing a fixed, Nernst-initialized, or dynamic BrainCell ion, or when a channel depends on more than one ion species. Use the built-in ion classes before authoring a custom ion.

## Mental model

An `Ion` owns one species' `Ci`, `Co`, `E`, and `valence`, packages them as `IonInfo`, and supplies that information to the channels whose currents it aggregates.

- A fixed ion keeps its concentrations and reversal potential constant.
- An initialized-Nernst ion keeps its concentrations fixed but computes `E` from them during initialization and reset.
- A dynamic-Nernst ion stores `Ci` as `DiffEqState`, derives `E` from the current `Ci`, and lets a concrete ion define `dCi/dt`.

Channels carry current; ions organize the shared electrochemical state that drives those currents. Do not put a shared reversal potential independently on every ion-specific channel.

## Choose an ion

| API | Use when | Important behavior |
|---|---|---|
| `braincell.ion.SodiumFixed(size, E=...)` | Use for classical sodium currents with a prescribed reversal potential. | It keeps sodium electrochemical values fixed and accepts channels whose `root_type` is `Sodium`. |
| `braincell.ion.SodiumInitNernst(size, ...)` | Use when fixed sodium concentrations should determine the initial reversal potential. | It recomputes `E` from the Nernst equation during initialization and reset; it does not integrate concentration. |
| `braincell.ion.PotassiumFixed(size, E=...)` | Use for classical potassium currents with a prescribed reversal potential. | It keeps potassium electrochemical values fixed and accepts channels whose `root_type` is `Potassium`. |
| `braincell.ion.PotassiumInitNernst(size, ...)` | Use when fixed potassium concentrations should determine the initial reversal potential. | It recomputes `E` from the Nernst equation during initialization and reset; it does not integrate concentration. |
| `braincell.ion.CalciumFixed(size, E=..., ...)` | Use when calcium current matters but intracellular calcium feedback does not. | It keeps calcium concentration and reversal potential fixed. |
| `braincell.ion.CalciumInitNernst(size, ...)` | Use when fixed calcium concentrations should initialize `E`. | It derives `E` from fixed `Ci`, `Co`, temperature, and valence during initialization and reset. |
| `braincell.ion.CalciumDetailed(size, C_rest=..., tau=..., d=...)` | Use when calcium influx must change intracellular calcium and its Nernst reversal potential. | It integrates `Ci`, receives total current from its calcium channels, and derives `E` live from the current concentration. |
| `braincell.ion.CalciumFirstOrder(size, ...)` | Use when simplified first-order calcium concentration dynamics are sufficient. | It replaces the detailed calcium-removal model with a lower-complexity concentration ODE. |

The ion API also exposes literature-derived calcium-dynamics templates such as `CdpHVA_SU2015_DCN`, `CdpLVA_SU2015_DCN`, and `CdpStC_MA2020_GoC`. Select one only when the requested model or source paper matches it; use the official `braincell.ion` API for the complete installed list.

## Attach fixed ions and channels

Construct the ion first, then add only channels whose `root_type` matches that ion.

| API | Description |
|---|---|
| `ion.add(name=channel)` | Use after constructing a single-compartment ion; it registers the channel under that ion and raises a type error when the channel's `root_type` does not match. |
| `ion.pack_info()` | Use when testing a channel directly; it returns the ion's `IonInfo(Ci, Co, E, valence)`. |
| `channel.root_type` | Inspect before attachment when the required ion owner is uncertain. |

```python
import braincell
import brainunit as u


na = braincell.ion.SodiumFixed(size=1, E=50.0 * u.mV)
na.add(INa=braincell.channel.Na_HH1952(size=1))

k = braincell.ion.PotassiumFixed(size=1, E=-77.0 * u.mV)
k.add(IK=braincell.channel.K_HH1952(size=1))

assert braincell.channel.Na_HH1952.root_type is braincell.ion.Sodium
assert braincell.channel.K_HH1952.root_type is braincell.ion.Potassium
```

Attach root-cell channels such as `IL` directly to the cell. They do not belong under a sodium, potassium, or calcium ion.

## Use dynamic calcium

`CalciumDetailed` couples the summed calcium current to intracellular concentration, then feeds the resulting Nernst potential back to calcium channels.

| API | Description |
|---|---|
| `braincell.ion.CalciumDetailed(...)` | Use for calcium-dependent adaptation, rebound, signaling, or another workflow in which `Ci` changes during integration. |
| `ca.Ci.value` | Read the current intracellular concentration after initialization; it is the value carried by the ion's `DiffEqState`. |
| `ca.E` | Read the current Nernst reversal potential; it is derived from the current `Ci`, not a fixed cache. |

```python
ca = braincell.ion.CalciumDetailed(
    size=1,
    C_rest=5.0e-5 * u.mM,
    tau=10.0 * u.ms,
    d=0.5 * u.um,
)
ca.add(
    ICaT=braincell.channel.CaT_HM1992(
        size=1,
        g_max=2.1 * u.mS / u.cm**2,
    )
)
```

The concentration ODE receives the total current from all channels owned by this calcium ion. Do not update `Ci` manually inside the cell loop or compute a second independent reversal potential in the channel.

Open `references/mixions-for-adaptation.md` when dynamic calcium participates in an AHP/KCa adaptation or rebound workflow.

## Combine ion dependencies

Use `MixIons` only when one channel's `root_type` requires several ion species.

| API | Description |
|---|---|
| `braincell.MixIons(*ions)` | Use to create a multi-ion owner in the exact order declared by the target channel's `root_type`. |
| `braincell.mix_ions(*ions)` | Use as the convenience function for the same combination. |
| `mixed.add(name=channel)` | Use to attach a channel after the ordered ion tuple has been constructed. |

```python
kca = braincell.MixIons(k, ca)
kca.add(
    IAHP=braincell.channel.AHP_De1994(
        size=1,
        g_max=0.3 * u.mS / u.cm**2,
    )
)
```

For `AHP_De1994`, the required order is potassium followed by calcium. Match `root_type` exactly; `(ca, k)` is not interchangeable with `(k, ca)`.

## Declare ions on a multicompartment cell

`braincell.mech.Ion` and `braincell.mech.Channel` are declarations; `Cell.init_state()` lowers them into runtime ion and channel objects over the selected control volumes.

```python
import braincell.mech as mech


cell.paint(
    region,
    mech.Ion(
        "CalciumDetailed",
        name="ca_dyn",
        d=0.5 * u.um,
        tau=10.0 * u.ms,
        C_rest=5.0e-5 * u.mM,
        Ci_initializer=2.4e-4 * u.mM,
    ),
)
cell.paint(
    region,
    mech.Channel(
        "CaT_HM1992",
        ion_name="ca_dyn",
        g_max=2.0 * u.mS / u.cm**2,
    ),
)
```

Use an explicit ion `name` when a channel must bind to a particular runtime ion owner. Open `references/multicompartment/multicompartment-cell-workflow.md` before applying this fragment to a `Cell`; that workflow owns morphology, regions, CV policies, initialization, and runtime inspection.

## Common failures

- Adding a channel to the wrong ion: inspect `root_type`, then attach it to the matching ion or ordered `MixIons`.
- Treating an ion as the current-producing mechanism: attach channels; the ion aggregates their current and owns shared electrochemical information.
- Using fixed calcium when the result depends on calcium accumulation: use `CalciumDetailed`, `CalciumFirstOrder`, or a matching documented template.
- Expecting `InitNernstIon` concentration to evolve: choose a dynamic ion when `Ci` must be integrated.
- Passing bare concentrations, voltages, time constants, temperature, or shell depth: use BrainUnit quantities.
- Authoring a new ion before checking `braincell.ion`: inspect the installed API and literature-derived templates first.

Open `references/braincell-custom-ion-channel-authoring.md` only when the required ion species or concentration dynamics remain unavailable after that inspection.

## Sources

- Ions & Channels: https://brainx.chaobrain.com/braincell/concepts/ions_channels.html
- Ions tutorial: https://brainx.chaobrain.com/braincell/tutorials/ion.html
- `braincell.ion` API: https://brainx.chaobrain.com/braincell/apis/braincell.ion.html
