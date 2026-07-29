# Devices

Use this reference when selecting a NEST-compatible generator, recorder, detector, or broadcast device; deciding connection direction; or reading device output from `SimulationResult`.

## Family map

Use this map to choose a device family before locating its constructor.

| Family | Use when | Key constraint |
|---|---|---|
| Current generators | Injecting a current waveform or noise in `pA` | Current enters through the neuron's ring buffer with a one-step delay. |
| Direct spike generators | Delivering explicit or transformed spike trains | Events use the connection's weight and delay. |
| Poisson generators | Producing constant, precise, scheduled, or sinusoidal Poisson spikes | NEST and JAX streams require distributional parity. |
| Other generators | Producing gamma, correlated, dead-time, or pulse-packet trains | One independent train is realized per target. |
| Recording devices | Recording spikes, analog State, or plastic weights | Direction and registration depend on recorder type. |
| Detectors | Computing correlation or binary-state statistics | Several detectors execute eagerly on the host. |
| Broadcast devices | Delivering a shared modulating signal | Bind the broadcast node to the compatible plasticity rule. |

## Current generators

Use these constructors when the device must inject a current rather than deliver weighted spike events.

| Constructor | Use when |
|---|---|
| `sim.create(bp.dc_generator, size=1, params=parameters)` | Use a constant current over a `start`/`stop` window. |
| `sim.create(bp.ac_generator, size=1, params=parameters)` | Use a sinusoidal current. |
| `sim.create(bp.noise_generator, size=1, params=parameters)` | Use Gaussian white-noise current with independent target realizations. |
| `sim.create(bp.step_current_generator, size=1, params=parameters)` | Use a piecewise-constant current schedule. |
| `sim.create(bp.step_rate_generator, size=1, params=parameters)` | Use a piecewise-constant rate schedule. |

Current generators inject through the target's current ring buffer. Do not treat them as weighted spike-event sources.

## Direct spike generators

Use these constructors for explicit spike schedules, supplied trains, or spike dilution.

| Constructor | Use when |
|---|---|
| `sim.create(bp.spike_generator, size=1, params=parameters)` | Use explicit `spike_times`. |
| `sim.create(bp.spike_train_injector, size=1, params=parameters)` | Use a supplied spike train as an event source. |
| `sim.create(bp.spike_dilutor, size=1, params=parameters)` | Use NEST-compatible spike dilution. |

## Poisson generators

Use these constructors when spike times follow a constant, precise, scheduled, or sinusoidally modulated Poisson process.

| Constructor | Use when |
|---|---|
| `sim.create(bp.poisson_generator, size=1, params=parameters)` | Use a constant-rate Poisson train over a configured window. |
| `sim.create(bp.poisson_generator_ps, size=1, params=parameters)` | Use precise-time Poisson spikes with dead time. |
| `sim.create(bp.inhomogeneous_poisson_generator, size=1, params=parameters)` | Use a time-varying Poisson schedule. |
| `sim.create(bp.sinusoidal_poisson_generator, size=1, params=parameters)` | Use a sinusoidally modulated Poisson rate. |

## Other generators

Use these constructors for gamma-process, correlated, dead-time, or pulse-packet sources.

| Constructor | Use when |
|---|---|
| `sim.create(bp.gamma_sup_generator, size=1, params=parameters)` | Use a superposition of independent gamma processes. |
| `sim.create(bp.sinusoidal_gamma_generator, size=1, params=parameters)` | Use a sinusoidally modulated gamma source. |
| `sim.create(bp.mip_generator, size=1, params=parameters)` | Use correlated trains from a multiple-interaction process. |
| `sim.create(bp.ppd_sup_generator, size=1, params=parameters)` | Use a superposition of Poisson processes with dead time. |
| `sim.create(bp.pulsepacket_generator, size=1, params=parameters)` | Use a Gaussian pulse packet. |

A generator fans out to one independent train per target neuron. For a multi-channel generator, pass one signed connection weight per channel.

## Recording devices

Use these constructors to capture spikes, named analog State, or per-edge plastic weights.

| Constructor | Use when |
|---|---|
| `sim.create(bp.spike_recorder, size=1)` | Use to record emitted spikes, event counts, or mean population rate. |
| `sim.create(bp.multimeter, size=1, record_from=recordables, interval=interval)` | Use to sample named analog recordables such as `V_m`. |
| `sim.create(bp.weight_recorder, size=1, params=parameters)` | Use the NEST-compatible weight-recorder device. |

## Recorder readback

Use these methods after `simulate()` or `cont()` to read the in-memory recordings.

| API | Use when |
|---|---|
| `result.spikes(node)` | Use to return the `(n_steps, n_recorded)` spike matrix for a recorder or source. |
| `result.rate(node)` | Use to return mean firing rate in spikes per second for a recorder or source. |
| `result.n_events(node)` | Use to return the number of recorded spike events for a recorder or source. |
| `result.trace(recorder, recordable="V_m")` | Use to return an analog trace in the recordable's natural unit. |
| `result.times` | Use to read the common time axis. |
| `sim.record_weight(proj)` | Use before simulation to register a plastic projection for per-step weight capture. |
| `result.weight_trace(proj)` | Use after registration to return the `(n_steps, n_edges)` weight trajectory. |

## Detectors

Use these constructors to compute correlation or binary-State statistics from recorded trains.

| Constructor | Use when |
|---|---|
| `sim.create(bp.correlation_detector, size=1, params=parameters)` | Use pairwise correlation statistics. |
| `sim.create(bp.correlomatrix_detector, size=1, params=parameters)` | Use a correlation matrix. |
| `sim.create(bp.correlospinmatrix_detector, size=1, params=parameters)` | Use spin-based correlation matrices. |
| `sim.create(bp.spin_detector, size=1, params=parameters)` | Use binary-State decoding from spikes. |

Obtain spike data before driving a detector that uses NumPy randomness or a Python host loop; do not force imperative detector logic into the compiled simulation loop.

## Broadcast devices

Use this constructor when dopamine-modulated plasticity needs one shared broadcast signal.

| Constructor | Use when |
|---|---|
| `sim.create(bp.volume_transmitter, size=1, params=parameters)` | Use to broadcast dopamine State to `stdp_dopamine_synapse`. |

## Device direction

Use device role to determine the `connect()` direction.

| Connection | Use when |
|---|---|
| `sim.connect(generator, neuron)` | Use for current, spike, Poisson, and other source devices. |
| `sim.connect(neuron, spike_recorder)` | Use to tap emitted spikes. |
| `sim.connect(multimeter, neuron)` | Use for analog observation; `multimeter` follows NEST's reversed recorder direction. |
| `sim.connect(dopamine_source, volume_transmitter)` | Use to register the modulating source on a volume transmitter. |

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)
neuron = sim.create(bp.iaf_psc_exp, 1)
drive = sim.create(bp.poisson_generator, rate=8000.0 * u.Hz, rng_seed=0)
spike_recorder = sim.create(bp.spike_recorder)
meter = sim.create(bp.multimeter, record_from=["V_m"], interval=0.1 * u.ms)

sim.connect(drive, neuron, weight=10.0 * u.pA, delay=1.0 * u.ms)
sim.connect(neuron, spike_recorder)
sim.connect(meter, neuron)

result = sim.simulate(100.0 * u.ms)
spikes = result.spikes(spike_recorder)
voltage = result.trace(meter, "V_m")
```

**Invariant:** Recording is in memory only. Do not port `record_to` file or ASCII backends; read spikes, traces, rates, counts, and weights from `SimulationResult`.

## Common failures

- Do not connect `multimeter` as `neuron -> multimeter`; use `multimeter -> neuron`.
- Do not connect `spike_recorder` as a source; use `neuron -> spike_recorder`.
- Do not model a current source as a spike source. Current devices inject `pA`; spike sources deliver delayed weighted events.
- Do not assume one generator object emits one shared train to every target; fan-out realizes independent target trains.
- Do not call `weight_trace()` without first registering the plastic projection.
- Do not request an unregistered analog variable; `result.trace()` raises `KeyError`.

## Official sources

- https://brainx.chaobrain.com/brainpy-state/nest-style/devices.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-devices.html
