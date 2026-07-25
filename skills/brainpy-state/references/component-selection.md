# Component selection

Use this reference after choosing the native `brainpy.state` path and before constructing the Module graph. Select the smallest model that preserves the phenomenon, input semantics, and output semantics the task requires.

## Neuron models

All native neuron models share the same operational contract: construct a population, initialize its State, call it once per `dt` with input current, and read the current spike with `get_spike()`.

| Need | APIs | Decision |
|---|---|---|
| Minimal integrate-and-fire | `IF`, `LIF`, `LIFRef` | Use `IF` without leak, `LIF` for canonical leaky dynamics, and `LIFRef` when refractory timing must be explicit. |
| Adaptation | `ALIF`, `AdExIF`, `AdExIFRef`, `AdQuaIF`, `AdQuaIFRef`, `Gif`, `GifRef` | Use when firing history changes excitability; select a `Ref` variant when the model requires refractory State. |
| Nonlinear spike onset | `ExpIF`, `ExpIFRef`, `AdExIF`, `AdExIFRef`, `QuaIF`, `AdQuaIF`, `AdQuaIFRef` | Use when the onset shape or adaptation matters beyond LIF dynamics. |
| Rich phenomenology | `Izhikevich`, `IzhikevichRef` | Use for diverse firing patterns at lower cost than channel-based models. |
| Conductance-based biophysics | `HH`, `MorrisLecar`, `WangBuzsakiHH` | Use when voltage-dependent membrane conductances are part of the scientific question; reduce `dt` as required by the numerical dynamics. |

Do not select a more detailed neuron only because it is available. Match the model to the behavior the experiment must preserve, then verify its firing response under the intended input and `dt`.

## Synaptic dynamics

Synapses filter presynaptic events over time. Their linearity determines whether AlignPost is valid.

| API | Use |
|---|---|
| `Expon` | Use for a single exponential decay and the canonical AlignPost path. |
| `DualExpon` | Use when distinct rise and decay time constants matter; it remains an exponential-family AlignPost choice. |
| `Alpha` | Use for an alpha-shaped response when one characteristic timescale is sufficient. |
| `AMPA` | Use for nonlinear AMPA receptor kinetics; place its State on the presynaptic side with AlignPre. |
| `GABAa` | Use for nonlinear GABA-A receptor kinetics; place its State on the presynaptic side with AlignPre. |
| `BioNMDA` | Use for biological NMDA receptor kinetics; pair with an appropriate output such as `MgBlock` and use AlignPre. |

Use `.desc(size, **parameters)` when a projection owns construction and size inference. Instantiate the synapse directly when it is called independently or explicitly shared outside a projection.

## Synaptic outputs

The output converts the synaptic variable into current received by the target neuron.

| API | Use |
|---|---|
| `COBA(E=...)` | Use conductance-based current `g * (E - V)` when reversal potential and voltage dependence matter. Excitatory and inhibitory paths normally use different `E`. |
| `CUBA(scale=...)` | Use voltage-independent current when simpler current-based dynamics are intended. Preserve compatible units across synaptic State, weights, and `scale`. |
| `MgBlock(...)` | Use voltage-dependent magnesium block for NMDA-like conductance. |

Changing `COBA` to `CUBA` is not merely a syntax change: weight signs, units, and calibrated magnitudes may also change.

Subclass `brainpy.state.SynOut` only when these built-ins cannot express the required conductance-to-current conversion. Preserve the binding contract used by the target neuron's registered current inputs.

## Short-term plasticity

| API | Use |
|---|---|
| `STP(size, U=..., tau_f=..., tau_d=...)` | Use Tsodyks-Markram facilitation and depression; it tracks utilization `u` and available resources `x` and returns modulated efficacy. |
| `STD(size, tau=..., U=...)` | Use depression without facilitation; it tracks available resources `x`. |

Call either model directly to inspect its dynamics. Use `.desc(...)` through a functional projection builder's `stp=` argument when plasticity belongs inside a projection.

## Input generators

| API | Use |
|---|---|
| `SpikeTime` | Use when exact spike times are already specified. |
| `PoissonSpike` | Use a stateful Poisson spike source with fixed rates. |
| `PoissonEncoder` | Use to encode rates or continuous values as Poisson spikes. |
| `PoissonInput` | Use a Module that applies Poisson input to a target State variable. |
| `poisson_input` | Use the functional Poisson-input form when a standalone function fits the update workflow. |

Keep time-dependent inputs inside the same `brainstate.environ.context(dt=..., t=...)` and transform loop as the model.

## Readouts

| API | Use |
|---|---|
| `LeakyRateReadout(in_size, out_size, tau=...)` | Use a trainable weight plus leaky low-pass dynamics to decode spikes or rates into a continuous output. |
| `brainstate.nn.Linear` followed by `Expon` | Use when the readout's weight transform and temporal filter must remain explicit, independently replaceable Modules. |
| Time reduction over outputs | Use `mean`, `max`, final-step, or task-specific temporal reduction when no separate trainable dynamical readout is required. Validate that the reduction matches the target definition. |

## Official sources

- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-choose-neuron.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-neurons.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synapses.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synouts.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-plasticity.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-inputs.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-readouts.html`
