# Record cell observables

Open this reference when a multicompartment workflow needs a sample schedule, a mechanism or current readout, or migration from legacy probes. A recording binds a spatial selection and an observable to one named result block.

## Choose the observation

| API | Description |
|---|---|
| `cell.loc(locset).record(name, observable, period=...)` | Register a unique name before initialization. Use a locset for spatial sampling or call `record()` on another cell view to keep its existing population and CV selection. |
| `braincell.observe.state("v")` | Record membrane voltage at the selected CVs. |
| `braincell.observe.channel(name=...).state(field)` | Record a stored channel State such as a gate. Match the declared name and choose a view where the channel is present. |
| `braincell.observe.ion(species=...).current()` | Record the total current for the selected ion species. Use a named channel's `current()` for a specific contributor. |
| `braincell.observe.membrane_current()` | Record total membrane current density at the selected CVs. |

Choose `period` or `frequency`, or omit both to sample every integration step. The resolved period and `start` must align with `dt`. Use the block's own time axis for plotting and its schema to identify the result columns.

```python
# Reuse the declaring Cell and RootLocation from the multicompartment workflow.
cell.loc(RootLocation(0.5)).record(
    "voltage_2kHz", braincell.observe.state("v"), frequency=2.0 * u.kHz,
)
result = cell.run(dt=0.1 * u.ms, duration=10.0 * u.ms)
block = result.samples["voltage_2kHz"]
assert block.values.shape == (20, 1)
assert block.time.shape == (20,)
assert len(block.schema.rows) == 1
```

Use `result.samples[population_name][recording_name]` for a `Network` result. Open `references/multicompartment/network-workflow.md` for population registration and sparse events.

## Diagnose missing or misaligned samples

Check the recording name and declaration phase, then inspect the selected cells and CVs, the observed mechanism at those sites, and the time window. Read the block's schema before comparing positions across CV policies. For a layout mismatch, open `references/multicompartment/topology-building-and-visualization.md`.

## Migrate legacy probes

BrainCell 0.2.0 retains `StateProbe`, `MechanismProbe`, and `CurrentProbe` for existing declarations. Read their values by resolved key from `sample_probes()` after initialization or `RunResult.traces` after a run. For new recordings, give each view and observable an explicit name. Compare resolved sample locations, units, and time axes before treating a legacy point probe and a CV recording as equivalent.

- [BrainCell 0.2.0 recording API](https://github.com/chaobrain/braincell/blob/v0.2.0/docs/design/network/current/recording.md)
- [BrainCell 0.2.0 recording examples](https://github.com/chaobrain/braincell/blob/v0.2.0/examples/cell/recording.ipynb)
