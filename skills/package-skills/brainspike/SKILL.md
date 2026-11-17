---
name: brainspike
description: Use for point-neuron modeling, spiking networks, synapses, plasticity, devices, spatial connectivity, and PyNEST migration. Use as the sole point-neuron skill with BrainCell or BrainMass in multiscale tasks; use public Neuron/Synapse composition for custom dynamics, gradient training, and cross-package coupling.
---

# BrainSpike

## Purpose and boundary

Use this skill for point-neuron networks through either the NEST-compatible `Simulator` or a native graph built from public `brainspike.Neuron` and `Synapse` bases. Use the Simulator for NEST-named models, devices, and the verified continuous-voltage fitting path; use native composition for custom equations, surrogate-gradient recurrent training, and cross-package coupling.

The examples target BrainSpikes `0.2.0` (import `brainspike`), bundled with
BrainX `2026.10.25`. Use the original skill snapshot for experiments frozen
on the older `brainpy.state` NEST API.

Enable 64-bit JAX before creating a `Simulator`, for example
`jax.config.update("jax_enable_x64", True)` or
`brainstate.environ.set(precision=64)`. With 32-bit JAX, the default
`poisson_generator` stop sentinel overflows its int64 comparison and simulation
fails during tracing. The bundled NEST-compatible scripts set this at import.

Canonical path:

`create Simulator -> create nodes and devices -> connect sources, populations, and recorders -> simulate -> read SimulationResult`

Keep each NEST-compatible graph under its Simulator. For native composition, construct a BrainState Module graph, initialize hidden State, reset independent rollouts, and execute State-aware transforms. Open `references/simulator-training.md` for direct NEST Simulator voltage-objective training, `references/native-composition.md` for custom neurons and synapses, `references/native-training.md` for BPTT or pp-prop, and `references/multiscale-composition.md` for cellular/regional exchange. These public-base workflows do not require the private unified kernel.


## Underlying mental model

For the NEST path, use one explicit `Simulator` instead of PyNEST's global kernel; it owns creation, connectivity, rollout State, recording, and results. Preserve NEST names with BrainUnit quantities, while connection rules select edges, synapse specs define edge behavior, and source/observer roles determine device direction.

### API structure

| API family | Use |
|---|---|
| `brainspike.Simulator` | Own node creation, connection construction, simulation lifecycle, and result collection. |
| `brainspike` NEST models | Select NEST-compatible neurons, synapses, plasticity rules, generators, recorders, and detectors by NEST-style name. |
| `brainspike.Neuron`, `brainspike.Synapse` | Define native dynamics with explicit hidden State and compose them in BrainState Modules. |
| `brainstate.transform` | Execute and differentiate native Module graphs; the Simulator owns its own rollout. |
| `brainspike.spatial` | Place populations in 2-D or 3-D space and construct distance-dependent kernels, masks, and connection rules. |
| `brainunit` | Supply physical time, voltage, current, conductance, capacitance, rate, weight, and delay quantities. |

### Create the simulator and nodes

Create every population and device through one `Simulator`; each `create()` call returns a `NodeView` that preserves population identity while supporting concatenation and slicing.

| API | Description |
|---|---|
| `brainspike.Simulator(dt=...)` | Use to start an independent NEST-compatible network; it owns the timestep, graph, rollout, and results. |
| `sim.create(model, size=1, *, params=None, positions=None, **kwargs)` | Use for neurons and devices; it instantiates the selected NEST-compatible model and returns a `NodeView`. Pass a large parameter set with `params=dict(...)` or individual parameters as keywords. |
| `left + right` | Use to concatenate `NodeView` objects while preserving their segment boundaries; the result can be passed to one `connect()` call. |
| `view[slice]` | Use to select a subset of a single population for connectivity or recording without creating another population. |

```python
import brainstate
import brainunit as u
import brainspike

brainstate.environ.set(precision=64)

sim = brainspike.Simulator(dt=0.1 * u.ms)
neurons = sim.create(
    brainspike.iaf_psc_alpha,
    100,
    params={"I_e": 350.0 * u.pA},
)
sample = neurons[:5]
```

Open `references/braintools/parameter-initializer.md` when choosing initial voltages or weight distributions. Open `references/braintools/input-current.md` when a timed stimulation protocol must enter a neuron or NEST device.

Open `references/nest-compatible/model-library.md` when choosing a neuron family or exact NEST-compatible model name. Open `references/nest-compatible/network-building.md` when composing multi-segment `NodeView` objects or spatial placement.

### Connect populations and devices

Connect source to target with a connection rule plus a unitful synapse spec, and reverse the call only for analog recorders that observe a population.

| API | Description |
|---|---|
| `sim.connect(pre, post, *, rule=..., weight=..., delay=..., ...)` | Use to build a projection, attach a source device, or register a recorder tap; it returns projection handles for realized projections and `None` for recorder taps or current injectors. |
| `brainspike.all_to_all` | Use when every source must connect to every target; this is the default generator fan-out rule. |
| `brainspike.fixed_indegree(k)` | Use when every target must draw exactly `k` presynaptic edges; set `seed` for reproducible BrainSpike wiring and `comm="sparse"` for large event fan-in. |
| `synapse=brainspike.<synapse_model>(...)` | Use when the edge is plastic or needs behavior beyond static weight and delay; keep rule parameters on the synapse spec. |
| `sim.connect(generator, neuron)` | Use for current and spike sources; generators drive their targets. |
| `sim.connect(neuron, spike_recorder)` | Use to tap emitted spikes. |
| `sim.connect(multimeter, neuron)` | Use for analog observation; `multimeter` follows NEST's reversed recorder direction. |

```python
drive = sim.create(brainspike.poisson_generator, rate=8000.0 * u.Hz, rng_seed=0)
spike_recorder = sim.create(brainspike.spike_recorder)
meter = sim.create(
    brainspike.multimeter,
    record_from=["V_m"],
    interval=0.1 * u.ms,
)

sim.connect(
    drive,
    neurons,
    rule=brainspike.all_to_all,
    weight=10.0 * u.pA,
    delay=1.0 * u.ms,
)
sim.connect(neurons, spike_recorder)
sim.connect(meter, sample)
```

Do not interchange default and explicit synapse delay paths; open `references/nest-delay-boundaries.md` for supported array shapes and the release-scoped amplitude counterexample.

Open `references/nest-compatible/synapse-and-connectivity.md` when selecting a static, special, STP, STDP, or voltage-based synapse; choosing a rule; or inspecting realized edges. Open `references/nest-compatible/devices.md` when selecting a generator, recorder, detector, or result-readback method.

### Simulate and read results

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
| `references/simulator-training.md` | Differentiating NEST Simulator voltage recordings, fitting connection weights/current, or checking hard-spike and outer-JIT limitations |
| `references/nonlinear-state-sharing.md` | Sharing nonlinear source traces or rejecting invalid target/heterogeneous aggregation |
| `references/nest-delay-boundaries.md` | Choosing tested delay-array/spec paths, fractional grid transmission, or checking the explicit-spec amplitude anomaly |
| `references/multirate-coupling.md` | Exchanging held regional signals and conserved window rates on integer-related fine/coarse clocks |
| `references/model-and-readout-selection.md` | Choosing physiology, synaptic current/conductance meaning, or instantaneous versus dynamic/windowed readouts |
| `references/native-projections.md` | Building a complete native E/I conductance network and checking communication/filter/current/target order |
| `references/native-delays.md` | Adding native transmission history and proving event, conductance, and voltage arrival times |
| `references/temporal-task-training.md` | Training independent batched cue-delay-report trials with masks or checkpointed BPTT |
| `references/native-composition.md` | Implementing custom point-neuron or synaptic equations, explicit State placement, or native event communication |
| `references/native-training.md` | Training a native recurrent spiking graph with BPTT, a trainable readout, or BrainTrace pp-prop |
| `references/multiscale-composition.md` | Exchanging signals between point neurons, detailed cells, and regional models on an explicit clock |
| `references/braintools/input-current.md` | Generating unit-aware stimulation or choosing native current arrays versus NEST source devices |
| `references/braintools/parameter-initializer.md` | Initializing membrane State, physical weights, or dimensionless native-layer parameters reproducibly |
| `references/braintools/metric.md` | Choosing a voltage, classification, spike-timing, or rate metric and its unit/axis reduction |
| `references/braintools/optimizer.md` | Selecting the registered parameter collection and native versus Simulator optimizer boundary |
| `references/braintools/surrogate.md` | Choosing and checking the backward spike derivative in the actual training path |
| `references/braintools/data-preprocessing.md` | Encoding analog features as time-major spikes or a declared current representation |
| `references/braintools/connectivity.md` | Generating topology beyond native Simulator rules and preserving edge/weight/delay ordering |
| `references/perturbation-experiment-validity.md` | Calibrating matched point-neuron interventions or validating baseline activity before scaling a causal experiment |
| `references/migration-boundary.md` | Replacing BrainPy or BrainPy-State, deciding whether their skills or runtime dependencies can be removed, or checking the tested migration scope |
| `references/nest-compatible/model-library.md` | Selecting or identifying a NEST-compatible neuron family or exact model name, or locating its neuron API entry |
| `references/nest-compatible/synapse-and-connectivity.md` | Selecting static or special synapses, STP/STDP rules, connection rules, weight/delay specs, or realized-connectivity inspection |
| `references/nest-compatible/devices.md` | Selecting generators, recorders, detectors, source semantics, connection direction, or result readback |
| `references/nest-compatible/network-building.md` | Using `Simulator`, `NodeView`, `SimulationResult`, `SynapseCollection`, projection classes, connection-rule helpers, or spatial networks |
| `references/nest-compatible/divergence-and-parity.md` | Porting NEST code, locating STDP parameters, explaining a mismatch, choosing trace versus distributional validation, or checking documented parity bands |
| `references/nest-compatible/integration-categories.md` | Determining how a model family is numerically updated and which validation category to consult next |

## Full workflow scripts

Use these standalone scripts when the canonical inline workflow is too small for the task. Plotting examples save PNG files in the current working directory. Keep the scripts separate from the Markdown references so their complete source remains directly runnable.

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

- Keep NEST Simulator execution separate from native Module execution; do not assume the two graphs are interchangeable.
- Native training and custom dynamics use public BrainSpike bases; do not require private kernel APIs or restore retired BrainPy projections.


## Official sources

- https://brainx.chaobrain.com/brainspike/nest-style/models.html
- https://brainx.chaobrain.com/brainspike/nest-style/connectivity.html
- https://brainx.chaobrain.com/brainspike/nest-style/devices.html
- https://brainx.chaobrain.com/brainspike/nest-style/tutorials/03-connect-network.html
- https://brainx.chaobrain.com/brainspike/apis/nest-network.html
- https://brainx.chaobrain.com/brainspike/nest-style/divergences/index.html
- https://brainx.chaobrain.com/brainspike/nest-style/validation-status.html
- https://brainx.chaobrain.com/brainspike/nest-style/divergences/stdp.html
- https://brainx.chaobrain.com/brainspike/nest-style/integration-categories.html
