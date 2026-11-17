# Connect detailed cells

Use `braincell.Network` when sources must deliver events to synapses placed at explicit locations on detailed cells. The network owns registration, connection pairing, event delivery, integration time, and population-keyed outputs.

| API | Description |
|---|---|
| `cell.place(locset, braincell.mech.Synapse(...))` | Declare a named synapse on the detailed cell before network initialization. |
| `network.add_population(name, model)` | Register each declaring cell or event source. Keep the population handles for connections. |
| `network.connect(name, source=..., synapse=..., weight=..., delay=...)` | Connect a source to selected synapse endpoints. Check the resolved pairs and attach physical units appropriate to the synapse and the delay. |
| `network.run(dt=..., duration=...)` | Initialize the network on first use and advance all registered populations on one clock. Read regular samples and sparse events from their separate result mappings. |

```python
import braincell
import brainunit as u
from braincell.filter import AllRegion, RootLocation

branch = braincell.Branch.from_lengths(
    lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um,
)
cell = braincell.Cell(braincell.Morphology.from_root(branch), pop_size=2)
cell.paint(
    AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
)
midpoint = RootLocation(0.5)
cell.place(midpoint, braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms))
cell.loc(midpoint).record("v", braincell.observe.state("v"))

network = braincell.Network("demo", seed=7)
source = network.add_population("drive", braincell.NetStim(size=2, start=0.1 * u.ms))
target = network.add_population("post", cell)
edges = network.connect(
    "input", source=source, synapse=target.synapses["ampa"], weight=0.001 * u.uS,
)
assert len(edges) == 2

result = network.run(dt=0.025 * u.ms, duration=0.5 * u.ms)
assert result.samples["post"]["v"].values.shape == (20, 2)
assert result.events["drive"]["spike"].source_id.shape == (2,)
```

After registration, use `network.run()` to advance time and `network.init_state()` for explicit initialization. Calls to `cell.run()`, `cell.init_state()`, `cell.reset_state()`, or `cell.reset()` on a network-managed cell raise `RuntimeError`.

Check connection views before the first run. Initialization can invalidate them, so select a fresh view for later inspection. Before a large run, check the resolved source and target indices, synapse placement, delay and weight units, and one small event-delivery response.

Use `network.reset_state()` to replay from the initial State with the same scheduled input. A noisy `NetStim` replays the same event times after this reset. For independent noisy inputs, build a fresh network and source per trial with a distinct recorded `NetStim(seed=...)`. Keep the network seed and connection declarations fixed when the study varies input noise with matched wiring.

- [BrainCell 0.2.0 Network API and minimal example](https://github.com/chaobrain/braincell/blob/v0.2.0/docs/design/network/current/api.md)
- [BrainCell 0.2.0 event sources and seeds](https://github.com/chaobrain/braincell/blob/v0.2.0/docs/design/network/current/connections.md)
- [BrainCell 0.2.0 network examples](https://github.com/chaobrain/braincell/tree/v0.2.0/examples/network)
