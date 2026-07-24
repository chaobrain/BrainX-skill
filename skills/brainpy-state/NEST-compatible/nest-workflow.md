# NEST-Compatible Workflow

## Purpose And Boundary

Use this workflow when the user wants to write, port, debug, explain, or organize BrainPy-State NEST-compatible models.

This is the compatibility path for users coming from NEST or PyNEST. The NEST-compatible family is built around JAX reimplementations of NEST neuron, synapse, plasticity, and device models, preserving NEST parameter names, defaults, and unit conventions where possible.

Keep the operational workflow explicit:

```text
Simulator -> create -> connect -> simulate -> read result
```

Do not turn this file into a full model encyclopedia. Keep model libraries, API lists, divergence details, and long gallery examples as progressive-disclosure references when those references exist or are supplied by the user.

## P0 Concepts

- **Explicit Simulator, not global NEST kernel**
  Source: https://brainx.chaobrain.com/brainpy-state/apis/nest-network.html
  PyNEST often feels global, but the NEST-compatible BrainPy-State path is explicit. The `Simulator` owns created populations, devices, connections, recording setup, rollout state, and the resulting `SimulationResult`.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)
pop = sim.create(bp.iaf_psc_alpha, 100)
res = sim.simulate(1000.0 * u.ms)
```

Use this rewrite first when porting from PyNEST:

```text
nest.SetKernelStatus({"resolution": dt})
->
sim = bp.Simulator(dt=dt * u.ms)
```

- **NEST model names, parameter names, and units**
  Source: https://brainx.chaobrain.com/brainpy-state/nest-style/models.html
  Use NEST-style model names directly from `brainpy.state`, and attach explicit `brainunit` units to parameters.

```python
params = dict(
    C_m=250.0 * u.pF,
    tau_m=20.0 * u.ms,
    t_ref=2.0 * u.ms,
    E_L=-70.0 * u.mV,
    V_reset=-70.0 * u.mV,
    V_th=-55.0 * u.mV,
    I_e=0.0 * u.pA,
)

pop = sim.create(bp.iaf_psc_alpha, 100, params=params)
```

When the user asks for model physiology, keep the answer short and route to model-library material or upstream NEST semantics instead of reproducing a full model textbook.

- **NodeView algebra**
  Source: https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html
  `sim.create(...)` returns a `NodeView`, not a raw array. Use NodeView concatenation and slicing for population composition.

```python
exc = sim.create(bp.iaf_psc_alpha, 800, params=params)
inh = sim.create(bp.iaf_psc_alpha, 200, params=params)
all_cells = exc + inh
sample = exc[:50]

sim.connect(exc, all_cells, rule=bp.fixed_indegree(80), weight=0.1 * u.pA)
sim.connect(sample, sr)
```

- **NEST-style connection rule plus synapse spec**
  Source: https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html
  The NEST-style connect surface has two conceptual parts: a connection rule and a synapse spec.

```python
sim.connect(
    exc,
    exc + inh,
    rule=bp.fixed_indegree(80),
    weight=0.1 * u.pA,
    delay=1.5 * u.ms,
    seed=1,
    comm="sparse",
)
```

Common connection rules:

- `bp.all_to_all`
- `bp.one_to_one`
- `bp.fixed_indegree(K)`
- `bp.fixed_total_number(N)`
- `bp.pairwise_bernoulli(p)`
- `bp.explicit_edges(pre_idx, post_idx)`

Use `synapse=...` when the connection model is not only static weight and delay:

```python
syn = bp.tsodyks2_synapse(...)
proj = sim.connect(pre, post, synapse=syn, weight=..., delay=...)
```

- **Weight-unit exception for delta neurons**
  Source: https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html
  Most current-based postsynaptic-current examples use synaptic current weights in `pA`, but `iaf_psc_delta` uses an instantaneous membrane-voltage jump.

```python
pop = sim.create(bp.iaf_psc_delta, 100)
sim.connect(pre, pop, weight=0.1 * u.mV, delay=1.5 * u.ms)
```

This is one of the most important porting traps: `iaf_psc_delta` weights are voltage in `mV`, not current in `pA`.

- **Device connection direction**
  Source: https://brainx.chaobrain.com/brainpy-state/nest-style/devices.html
  Device direction is intentionally NEST-like and easy to get wrong.

| Device type | Direction |
|---|---|
| Current generator | `sim.connect(generator, neuron)` |
| Spike generator or Poisson generator | `sim.connect(generator, neuron)` |
| Spike recorder | `sim.connect(neuron, spike_recorder)` |
| Voltmeter | `sim.connect(voltmeter, neuron)` |
| Multimeter | `sim.connect(multimeter, neuron)` |

Analog recorders are reversed because they observe the neuron rather than receive spikes from it.

- **Current sources versus spike sources**
  Source: https://brainx.chaobrain.com/brainpy-state/apis/nest-devices.html
  Current generators inject current in `pA` through the neuron current ring buffer. Spike sources deliver delayed spike events.

Current sources:

- `bp.dc_generator`
- `bp.ac_generator`
- `bp.noise_generator`
- `bp.step_current_generator`
- `bp.step_rate_generator`

Spike sources:

- `bp.poisson_generator`
- `bp.poisson_generator_ps`
- `bp.inhomogeneous_poisson_generator`
- `bp.sinusoidal_poisson_generator`
- `bp.spike_generator`
- `bp.spike_train_injector`
- `bp.spike_dilutor`
- `bp.mip_generator`
- `bp.pulsepacket_generator`

- **STDP state and parameter location**
  Source: https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/stdp.html
  Do not assume every NEST STDP parameter lives on the same object in BrainPy-State.

Main rule:

```text
Parameter names may match, but parameter location may differ.
```

The canonical example is `tau_minus`: in NEST it is a neuron parameter, while in BrainPy-State it belongs on the synapse-spec side. Setting a parameter on the wrong object is a common STDP porting mistake.

## Minimal General Workflow

Use this skeleton before adding model-specific details.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)

pop = sim.create(bp.iaf_psc_alpha, 100)
noise = sim.create(bp.poisson_generator, rate=8000.0 * u.Hz)
sr = sim.create(bp.spike_recorder)
mm = sim.create(bp.multimeter, record_from=["V_m"], interval=0.1 * u.ms)

sim.connect(noise, pop, weight=1.2 * u.pA, delay=1.5 * u.ms)
sim.connect(pop, sr)
sim.connect(mm, pop[:5])

res = sim.simulate(1000.0 * u.ms)

times = res.times
spikes = res.spikes(sr)
rate = res.rate(sr)
vm = res.trace(mm, "V_m")
```

The important decisions are the explicit `Simulator`, NEST-style model creation, unitful parameters and weights, correct device direction, and result readback through `SimulationResult`.

## Canonical Workflow Scripts

### Run One Neuron

Use this when the user needs the smallest complete model: create one neuron, record voltage, simulate, and read a trace.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)

params = dict(
    C_m=250.0 * u.pF,
    tau_m=20.0 * u.ms,
    t_ref=2.0 * u.ms,
    E_L=-70.0 * u.mV,
    V_reset=-70.0 * u.mV,
    V_th=-55.0 * u.mV,
    I_e=400.0 * u.pA,
)

cell = sim.create(bp.iaf_psc_alpha, 1, params=params)
mm = sim.create(bp.multimeter, record_from=["V_m"], interval=0.1 * u.ms)

sim.connect(mm, cell)

res = sim.simulate(200.0 * u.ms)
vm = res.trace(mm, "V_m")
times = res.times
```

Concepts taught: `Simulator`, model creation, unitful parameters, analog recorder direction, and `SimulationResult.trace`.

### Build Populations And Devices

Use this when the user needs populations, combined groups, slices, source devices, and recorder devices.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)

params = dict(
    C_m=250.0 * u.pF,
    tau_m=20.0 * u.ms,
    t_ref=2.0 * u.ms,
    E_L=-70.0 * u.mV,
    V_reset=-70.0 * u.mV,
    V_th=-55.0 * u.mV,
    I_e=0.0 * u.pA,
)

exc = sim.create(bp.iaf_psc_alpha, 800, params=params)
inh = sim.create(bp.iaf_psc_alpha, 200, params=params)
all_cells = exc + inh
sample = exc[:50]

drive = sim.create(bp.poisson_generator, rate=8000.0 * u.Hz)
sr = sim.create(bp.spike_recorder)
mm = sim.create(bp.multimeter, record_from=["V_m"], interval=0.1 * u.ms)

sim.connect(drive, all_cells, weight=1.2 * u.pA, delay=1.5 * u.ms)
sim.connect(sample, sr)
sim.connect(mm, sample[:5])
```

Concepts taught: `NodeView` algebra, population/device distinction, source devices, recorder devices, and NEST-style names and parameters.

### Connect A Network

Use this when the user needs a minimal E/I or Brunel-like network.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)

params = dict(
    C_m=250.0 * u.pF,
    tau_m=20.0 * u.ms,
    t_ref=2.0 * u.ms,
    E_L=-70.0 * u.mV,
    V_reset=-70.0 * u.mV,
    V_th=-55.0 * u.mV,
    I_e=0.0 * u.pA,
)

exc = sim.create(bp.iaf_psc_alpha, 800, params=params)
inh = sim.create(bp.iaf_psc_alpha, 200, params=params)
all_cells = exc + inh

drive = sim.create(bp.poisson_generator, rate=8000.0 * u.Hz)
sr = sim.create(bp.spike_recorder)

sim.connect(drive, all_cells, rule=bp.all_to_all, weight=1.2 * u.pA, delay=1.5 * u.ms)
sim.connect(exc, all_cells, rule=bp.fixed_indegree(80), weight=0.1 * u.pA, delay=1.5 * u.ms, seed=1, comm="sparse")
sim.connect(inh, all_cells, rule=bp.fixed_indegree(20), weight=-0.4 * u.pA, delay=1.5 * u.ms, seed=2, comm="sparse")
sim.connect(exc[:100], sr)

res = sim.simulate(1000.0 * u.ms)
rate = res.rate(sr)
spikes = res.spikes(sr)
```

Concepts taught: `all_to_all`, `fixed_indegree`, seeded random connectivity, sparse communication, signed weights, delay, population-level recording, and rate readback.

### Record And Analyze

Use this when the user asks about spike trains, firing rate, voltage traces, or basic readback.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)

pop = sim.create(
    bp.iaf_psc_alpha,
    10,
    params=dict(I_e=350.0 * u.pA),
)
sr = sim.create(bp.spike_recorder)
mm = sim.create(bp.multimeter, record_from=["V_m"], interval=0.1 * u.ms)

sim.connect(pop, sr)
sim.connect(mm, pop[:3])

res = sim.simulate(500.0 * u.ms)

times = res.times
spikes = res.spikes(sr)
rate = res.rate(sr)
vm = res.trace(mm, "V_m")
```

Concepts taught: spike-recorder direction, multimeter direction, `SimulationResult.spikes`, `SimulationResult.rate`, `SimulationResult.trace`, and `SimulationResult.times`.

## Embedded Reference Map

These six areas are compact lookups embedded in this parent file, not child Markdown routes. Do not create separate files for them as part of this architecture pass.

### Model Library

Sources:

- https://brainx.chaobrain.com/brainpy-state/nest-style/models.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-neurons.html

Purpose: model selection and neuron-model API lookup.

### Synapse And Connectivity

Sources:

- https://brainx.chaobrain.com/brainpy-state/apis/nest-synapses.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-plasticity.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html

Purpose: static synapses, special synapses, plasticity models, connection rules, synapse specs, and realized-connectivity inspection.

### Devices

Sources:

- https://brainx.chaobrain.com/brainpy-state/nest-style/devices.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-devices.html

Purpose: generators, recorders, detectors, source-device semantics, recorder direction, and result readback.

### Network Building

Sources:

- https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/03-connect-network.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-network.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-spatial.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/spatial.html

Purpose: network builder API, simulator execution, `NodeView`, `SimulationResult`, `SynapseCollection`, projection classes, connection rules, and spatial network primitives.

### Divergence And Parity

Sources:

- https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/index.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/validation-status.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/stdp.html

Purpose: porting differences, STDP parameter placement, recording conventions, stochastic parity, validation logic, and NEST mismatches.

### Integration Categories

Sources:

- https://brainx.chaobrain.com/brainpy-state/nest-style/integration-categories.html

Purpose: numerical and integration behavior by model family.

## Selected Full-Script Inventory

Sources:

- https://brainx.chaobrain.com/brainpy-state/examples/nest-gallery.html
- `brunel_alpha.py`: https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_alpha.py
- `brunel_delta.py`: https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_delta.py
- `brette_et_al_2007.py`: https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brette_et_al_2007.py
- `synapsecollection.py`: https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/synapsecollection.py
- `evaluate_tsodyks2_synapse.py`: https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/evaluate_tsodyks2_synapse.py
- `clopath_synapse_spike_pairing.py`: https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/clopath_synapse_spike_pairing.py
- `spatial_gaussex.py`: https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_gaussex.py

These seven scripts are the complete selected NEST-compatible inventory for this branch, not the whole ported-script gallery. Do not add unselected gallery scripts unless the source plan is revised.

## Common Mistakes -> Fix

- Raw numbers without units -> attach `brainunit` units to times, voltages, capacitances, currents, rates, weights, and delays.
- Treating `Simulator` like a global NEST kernel -> keep all `create`, `connect`, `simulate`, and readback calls tied to one `sim`.
- Wrong analog recorder direction -> use `sim.connect(multimeter_or_voltmeter, neuron)`.
- Wrong spike recorder direction -> use `sim.connect(neuron, spike_recorder)`.
- Confusing current and spike sources -> current devices inject `pA`; spike sources deliver delayed spike events.
- Using `pA` weight for `iaf_psc_delta` -> use `mV` for delta-neuron weights.
- Forgetting NodeView algebra -> use `exc + inh` and `exc[:50]` instead of manually rebuilding populations.
- Wrong connection rule -> verify `all_to_all`, `one_to_one`, `fixed_indegree`, `pairwise_bernoulli`, or `explicit_edges`.
- Expecting stochastic sample identity -> compare stochastic drives and connectivity distributionally across seeds.
- Copying STDP parameter locations blindly -> check where the learning state and rule parameters live before concluding the model is wrong.
- Treating the model library as full physiology docs -> summarize the BrainPy-State API and route detailed equations to upstream NEST semantics.
- Loading the whole script gallery -> use only selected representative scripts.

## Default Answer Pattern

When using this workflow:

1. Identify whether the task is create, port, debug, analyze, or organize reference docs.
2. Start from `Simulator -> create -> connect -> simulate -> read result`.
3. Mention the relevant NEST-specific concept: explicit simulator, units, NodeView, connection rule plus synapse spec, delta-weight exception, device direction, current versus spike source, or STDP state location.
4. Use only one reference area unless the task requires more.
5. Use one representative full script when examples are needed.
6. For mismatch or parity questions, check divergence material before giving a conclusion.

## Load Strategy

Start from this file for NEST-compatible tasks. Consult only the embedded lookup area that matches the request.

| User need | Open or consult |
|---|---|
| Choose or identify a NEST-compatible neuron model | model-library material |
| Understand synapse models, plasticity models, connection rules, weight or delay specs | synapse and connectivity material |
| Use generators, recorders, detectors, current sources, or spike sources | device material |
| Use Simulator, NodeView, SimulationResult, SynapseCollection, projections, connection-rule APIs, or spatial APIs | network-building material |
| Port NEST code and resolve semantic mismatches | divergence material |
| Understand numerical or integration behavior | integration-categories material |
| Find representative official gallery scripts | full-script reference material |
