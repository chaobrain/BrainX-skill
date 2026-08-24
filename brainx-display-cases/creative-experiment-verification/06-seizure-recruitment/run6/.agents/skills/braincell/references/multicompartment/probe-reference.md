# Probe reference

Use this reference after opening `references/multicompartment/multicompartment-cell-workflow.md` when a multicompartment `Cell` must record membrane voltage, mechanism or ion State, or membrane current, or when a probe produces a missing or unexpected trace.

## Choose the observed value

Probes are point declarations that read State or current already owned by the initialized runtime; they do not allocate evolving State of their own.

| API | Use when | Important behavior and result |
|---|---|---|
| `StateProbe(name=None, field="v")` | Record cell-owned membrane voltage at a locset. | The current multicompartment implementation supports only `field="v"` and produces one voltage probe per resolved point. |
| `MechanismProbe(mechanism, field, name=None)` | Record a runtime `brainstate.State` field owned by a named channel or ion. | Use for gate variables such as `Na_HH1952.p` or dynamic ion State such as `ca_dyn.Ci`; static parameters and derived properties are not probeable fields. |
| `CurrentProbe(ion=None, mechanism=None, name=None)` | Record one mechanism current or the total current owned by one ion. | Set `mechanism` for a mechanism current, `ion` for an ion total, or both to select a mechanism under a named ion. |
| `cell.place(locset, *probes)` | Attach probes to concrete morphology locations before initialization. | A locset resolving to several points creates one resolved probe at each point. |
| `cell.sample_probe(name)` | Read one placed probe after `init_state()`. | It returns the current snapshot for the exact resolved probe name. |
| `cell.sample_probes()` | Read every placed probe after `init_state()`. | It returns `{probe_name: current_value}`. |
| `cell.run(dt=..., duration=...)` | Record the same probes at every integration step. | It returns `RunResult(time=..., traces=...)`, where `traces` is keyed by resolved probe name. |

Use `StateProbe` for cell voltage. Use `MechanismProbe` only for stored runtime State. Use `CurrentProbe` when the requested observable is computed current rather than stored State.

## Place and read probes

Place probes with the mechanisms they observe, initialize once, inspect snapshot keys, then use the same declarations for time-series recording.

```python
import braincell
import braincell.mech as mech
import brainunit as u
from braincell.filter import AllRegion, at


morphology = braincell.Morphology.from_swc("neuron.swc")
cell = braincell.Cell(morphology, solver="staggered")

cell.paint(
    AllRegion(),
    mech.Channel(
        "IL",
        g_max=0.03 * u.mS / u.cm**2,
        E=-54.387 * u.mV,
    ),
    mech.Channel(
        "Na_HH1952",
        g_max=120.0 * u.mS / u.cm**2,
    ),
)
cell.place(
    at("soma", 0.5),
    mech.StateProbe(),
    mech.MechanismProbe(mechanism="Na_HH1952", field="p"),
    mech.CurrentProbe(mechanism="IL"),
)

cell.init_state()
samples = cell.sample_probes()

assert "soma(0.5)_v" in samples
assert "soma(0.5)_Na_HH1952_p" in samples
assert "soma(0.5)_IL_current" in samples

result = cell.run(dt=0.05 * u.ms, duration=12.0 * u.ms)
assert result.traces["soma(0.5)_v"].shape[0] == result.time.shape[0]
```

Declare probes before `init_state()`. Call `reset()` before adding or changing a probe on an initialized cell; `reset_state()` only reseeds the existing runtime and does not reopen declaration.

## Predict probe names

When `name` is omitted, the lowerer combines the resolved location with the observed target.

| Probe declaration | Generated key pattern | Documented example |
|---|---|---|
| `StateProbe()` | `<location>_v` | `soma(0.5)_v` |
| `MechanismProbe(mechanism=M, field=F)` | `<location>_<M>_<F>` | `soma(0.5)_Na_HH1952_p` |
| `CurrentProbe(mechanism=M)` | `<location>_<M>_current` | `soma(0.5)_IL_current` |
| `CurrentProbe(ion=I)` | `<location>_<I>_current` | `soma(0.5)_na_current` |

Use `sorted(cell.sample_probes())` after initialization to discover the actual keys instead of reconstructing them from memory. If `name=...` is supplied, keep it globally unique; duplicate names make snapshot and run results ambiguous.

A multi-point locset expands into separate concrete names. For example, `RootLocation(0.5) | Terminals()` creates one voltage key for the root midpoint and one for each terminal selected by the morphology.

## Probe mechanism and ion State

`MechanismProbe` resolves a named runtime owner, then reads one field that is stored as `brainstate.State`.

| Request | Use | Do not use |
|---|---|---|
| Channel gate | `MechanismProbe(mechanism="Na_HH1952", field="p")` | A parameter such as `g_max`; it is not evolving State. |
| Dynamic ion concentration | `MechanismProbe(mechanism="ca_dyn", field="Ci")` | A derived Nernst property such as `E` when it is computed rather than stored. |
| Cell voltage | `StateProbe(field="v")` | `MechanismProbe` without a mechanism owner. |

The mechanism or ion must exist at the selected runtime point. A valid field on a mechanism painted elsewhere is still invalid at the probe location.

## Probe current ownership

Choose between one mechanism current and an ion owner's total current:

| Request | Probe |
|---|---|
| One ion-bound mechanism | `CurrentProbe(ion="na", mechanism="Na_HH1952")` |
| One mechanism that can evaluate current without an explicit ion selector | `CurrentProbe(mechanism="IL")` |
| Sum of currents owned by an ion | `CurrentProbe(ion="na")` |

An ion total may equal a mechanism current when that ion owns only one channel. Do not rely on that equality: adding another channel changes the ion total but not the individual mechanism trace.

For mixed-ion channels, distinguish the current owner from ions that only modulate gating. Use `CurrentProbe(mechanism=...)` for the channel's own current and `CurrentProbe(ion=...)` for the owning ion's aggregate current.

## Diagnose missing or wrong traces

Check failures in declaration-to-runtime order:

1. Confirm that the probe was placed before `init_state()` and that `run()` has at least one placed probe.
2. Resolve the locset with `morphology.select(locset)` and inspect its `points` and `display_names`.
3. Inspect `sorted(cell.sample_probes())`; do not guess the trace key.
4. Confirm that the named mechanism or ion is painted at the resolved point.
5. For `MechanismProbe`, confirm that the field is stored runtime State rather than a parameter or derived property.
6. For `CurrentProbe`, confirm whether the request means one mechanism current or an ion-owner total.
7. Check explicit names for global uniqueness and multi-point locsets for automatic expansion.
8. Compare `result.time.shape[0]` with every requested trace length after `run()`.

Open `references/multicompartment/topology-building-and-visualization.md` when the locset-to-CV mapping or runtime placement remains uncertain.

## Sources

- [Mechanisms in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/mech.html)
- [StateProbe API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.mech.StateProbe.html)
- [MechanismProbe API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.mech.MechanismProbe.html)
- [CurrentProbe API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.mech.CurrentProbe.html)
