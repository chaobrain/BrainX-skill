# NEST-Compatible Workflow

## Purpose and boundary

Use this reference for the NEST-compatible `brainpy.state` path: create NEST-named neurons and devices, connect them with NEST-style rules and synapse specs, execute an explicit `Simulator`, inspect results, or port PyNEST code.

Canonical path:

`create Simulator -> create nodes and devices -> connect sources, populations, and recorders -> simulate -> read SimulationResult`

Keep this path separate from native BrainPy-State composition. Do not mix `Simulator` networks with native projection APIs such as `AlignPostProj`, `align_pre_projection`, or manual BrainState rollout loops. Open the routed NEST reference for model catalogs, plasticity, spatial networks, or parity details; use the parent BrainPy-State skill only for native models.

## Underlying principle of the NEST-compatible path

Use one explicit `Simulator` instead of PyNEST's global kernel; it owns creation, connectivity, rollout State, recording, and results. Preserve NEST names with BrainUnit quantities, while connection rules select edges, synapse specs define edge behavior, and source/observer roles determine device direction.

### API structure

| API family | Use |
|---|---|
| `brainpy.state.Simulator` | Own node creation, connection construction, simulation lifecycle, and result collection. |
| `brainpy.state` NEST models | Select NEST-compatible neurons, synapses, plasticity rules, generators, recorders, and detectors by NEST-style name. |
| `brainpy.state.network` | Inspect the underlying `Network`, `Builder`, `NodeView`, `SimulationResult`, projection, connection-rule, and `SynapseCollection` APIs. |
| `brainpy.state.spatial` | Place populations in 2-D or 3-D space and construct distance-dependent kernels, masks, and connection rules. |
| `brainunit` | Supply physical time, voltage, current, conductance, capacitance, rate, weight, and delay quantities. |

### 1. Create the simulator and nodes

Create every population and device through one `Simulator`; each `create()` call returns a `NodeView` that preserves population identity while supporting concatenation and slicing.

| API | Description |
|---|---|
| `bp.Simulator(dt=...)` | Use to start an independent NEST-compatible network; it owns the timestep, graph, rollout, and results. |
| `sim.create(model, size=1, *, params=None, positions=None, **kwargs)` | Use for neurons and devices; it instantiates the selected NEST-compatible model and returns a `NodeView`. Pass a large parameter set with `params=dict(...)` or individual parameters as keywords. |
| `left + right` | Use to concatenate `NodeView` objects while preserving their segment boundaries; the result can be passed to one `connect()` call. |
| `view[slice]` | Use to select a subset of a single population for connectivity or recording without creating another population. |

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)
neurons = sim.create(
    bp.iaf_psc_alpha,
    100,
    params={"I_e": 350.0 * u.pA},
)
sample = neurons[:5]
```

Open `references/nest-compatible/model-library.md` when choosing a neuron family or exact NEST-compatible model name. Open `references/nest-compatible/network-building.md` when using `Builder`, `Network`, multi-segment `NodeView` operations, or spatial placement.

### 2. Connect populations and devices

Connect source to target with a connection rule plus a unitful synapse spec, and reverse the call only for analog recorders that observe a population.

| API | Description |
|---|---|
| `sim.connect(pre, post, *, rule=..., weight=..., delay=..., ...)` | Use to build a projection, attach a source device, or register a recorder tap; it returns projection handles for realized projections and `None` for recorder taps or current injectors. |
| `bp.all_to_all` | Use when every source must connect to every target; this is the default generator fan-out rule. |
| `bp.fixed_indegree(k)` | Use when every target must draw exactly `k` presynaptic edges; set `seed` for reproducible BrainPy-State wiring and `comm="sparse"` for large event fan-in. |
| `synapse=bp.<synapse_model>(...)` | Use when the edge is plastic or needs behavior beyond static weight and delay; keep rule parameters on the synapse spec. |
| `sim.connect(generator, neuron)` | Use for current and spike sources; generators drive their targets. |
| `sim.connect(neuron, spike_recorder)` | Use to tap emitted spikes. |
| `sim.connect(multimeter, neuron)` | Use for analog observation; `multimeter` follows NEST's reversed recorder direction. |

```python
drive = sim.create(bp.poisson_generator, rate=8000.0 * u.Hz, rng_seed=0)
spike_recorder = sim.create(bp.spike_recorder)
meter = sim.create(
    bp.multimeter,
    record_from=["V_m"],
    interval=0.1 * u.ms,
)

sim.connect(
    drive,
    neurons,
    rule=bp.all_to_all,
    weight=10.0 * u.pA,
    delay=1.0 * u.ms,
)
sim.connect(neurons, spike_recorder)
sim.connect(meter, sample)
```

Open `references/nest-compatible/synapse-and-connectivity.md` when selecting a static, special, STP, STDP, or voltage-based synapse; choosing a rule; or inspecting realized edges. Open `references/nest-compatible/devices.md` when selecting a generator, recorder, detector, or result-readback method.

### 3. Simulate and read results

`simulate()` starts from freshly initialized State and returns one in-memory `SimulationResult` whose common time axis indexes spike and analog recordings.

| API | Description |
|---|---|
| `sim.simulate(duration, *, dt=None)` | Use for an independent run from initialized State and time zero; it returns a `SimulationResult`. |
| `result.times` | Use for the run's common `(n_steps,)` time axis. |
| `result.spikes(recorder)` | Use for a spike recorder's `(n_steps, n_recorded)` spike matrix. |
| `result.rate(recorder)` | Use for the mean firing rate in spikes per second over the recorded neurons. |
| `result.trace(recorder, recordable)` | Use for a multimeter trace in the model State's natural unit; it raises `KeyError` when the recordable was not registered. |

```python
result = sim.simulate(100.0 * u.ms)

spikes = result.spikes(spike_recorder)
voltage = result.trace(meter, "V_m")
rate = result.rate(spike_recorder)

assert spikes.shape[0] == result.times.shape[0]
assert voltage.shape[0] == result.times.shape[0]
```

Open `references/nest-compatible/network-building.md` when continuing State across windows with `cont()`, resetting persistent rollout State, recording plastic weights, inspecting `SynapseCollection`, or selecting spatial primitives.

## Reference routing

Open only the smallest reference that owns the decision.

| Reference | Open when |
|---|---|
| `references/nest-compatible/model-library.md` | Selecting or identifying a NEST-compatible neuron family or exact model name, or locating its neuron API entry |
| `references/nest-compatible/synapse-and-connectivity.md` | Selecting static or special synapses, STP/STDP rules, connection rules, weight/delay specs, or realized-connectivity inspection |
| `references/nest-compatible/devices.md` | Selecting generators, recorders, detectors, source semantics, connection direction, or result readback |
| `references/nest-compatible/network-building.md` | Using `Builder`, `Network`, `Simulator`, `NodeView`, `SimulationResult`, `SynapseCollection`, projection classes, connection-rule helpers, or spatial networks |
| `references/nest-compatible/divergence-and-parity.md` | Porting NEST code, locating STDP parameters, explaining a mismatch, choosing trace versus distributional validation, or checking documented parity bands |
| `references/nest-compatible/integration-categories.md` | Determining how a model family is numerically updated and which validation category to consult next |

## Full workflow scripts

Use these standalone scripts when the canonical inline workflow is too small for the task. Keep the scripts separate from the Markdown references so their complete source remains directly runnable.

| Script | Open when |
|---|---|
| `references/nest-compatible/scripts/brunel_alpha.py` | Building an alpha-synapse Brunel network |
| `references/nest-compatible/scripts/brunel_delta.py` | Checking delta-synapse voltage-weight semantics in a Brunel network |
| `references/nest-compatible/scripts/brette_et_al_2007.py` | Reproducing the comparative network workflow from Brette et al. (2007) |
| `references/nest-compatible/scripts/synapsecollection.py` | Inspecting or manipulating realized synapses through `SynapseCollection` |
| `references/nest-compatible/scripts/evaluate_tsodyks2_synapse.py` | Evaluating short-term plasticity behavior against the Tsodyks2 protocol |
| `references/nest-compatible/scripts/clopath_synapse_spike_pairing.py` | Running the Clopath voltage-based plasticity spike-pairing protocol |
| `references/nest-compatible/scripts/spatial_gaussex.py` | Building spatial connectivity with a Gaussian distance-dependent rule |

## Boundaries and common failures

- Do not mix this workflow with native BrainPy-State projections or manual rollout code; keep the whole NEST-compatible graph under one `Simulator`.


## Official sources

- https://brainx.chaobrain.com/brainpy-state/nest-style/models.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/devices.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/03-connect-network.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-network.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/index.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/validation-status.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/stdp.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/integration-categories.html
