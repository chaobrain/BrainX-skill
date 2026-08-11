# Component selection

Use this reference after choosing the native `brainpy.state` path and before constructing the Module graph. Select the least detailed component set that preserves the phenomenon, signal semantics, and decoded result required by the experiment.

## Selection order

Choose components in dependency order so later decisions preserve earlier semantics:

| Step | Decide | Result |
|---|---|---|
| 1 | Which firing mechanism must the neuron reproduce? | Neuron family and required numerical resolution |
| 2 | Is the presynaptic source scheduled, stochastic, encoded, or injected directly into State? | Input generator or input function |
| 3 | Which temporal response must a presynaptic event create? | Synaptic dynamics and the valid projection alignment |
| 4 | Should synaptic activity become voltage-dependent conductance or voltage-independent current? | Synaptic output, weight units, signs, and scale |
| 5 | Must efficacy change over recent presynaptic history? | Optional `STP` or `STD` |
| 6 | What tensor and temporal statistic does the loss or analysis consume? | Readout Module or explicit time reduction |

Complete steps 3 through 5 whenever a point-neuron task names recurrent excitation, mutual inhibition, or spike-driven connectivity. Use explicit synapses, outputs, communication, and projections; do not substitute a scalar mean-spike feedback `State`. If scalar population rates and rate coupling are the scientific variables, route that scale to BrainMass or open both skills for an explicit hybrid model.

After these choices, open `references/projection-patterns.md` to choose AlignPre versus AlignPost, communication, delay integration, and projection construction form.

## Neuron models

All native point-neuron models use the same lifecycle: construct a population, initialize its State, call it once per `dt` with input current, and read the current spike with `get_spike()`.

### Minimal integrate-and-fire models

Use this family when threshold, reset, leak, and optional refractoriness are sufficient to preserve the required firing behavior.

| API | Use when |
|---|---|
| `brainpy.state.IF(in_size, R=..., tau=..., V_th=..., ...)` | Use when membrane voltage should integrate input without a leak-to-rest term. |
| `brainpy.state.LIF(in_size, R=..., tau=..., V_th=..., V_reset=..., V_rest=..., ...)` | Use as the canonical low-cost neuron when leaky voltage integration is required. |
| `brainpy.state.LIFRef(in_size, R=..., tau=..., tau_ref=..., V_th=..., V_reset=..., V_rest=..., ...)` | Use when the LIF model must enforce an explicit refractory interval after each spike. |

### Nonlinear integrate-and-fire models

Use this family when the spike-onset nonlinearity affects the result but adaptation is not required.

| API | Use when |
|---|---|
| `brainpy.state.ExpIF(in_size, R=..., tau=..., V_T=..., delta_T=..., ...)` | Use when exponential spike initiation must replace the hard LIF onset. |
| `brainpy.state.ExpIFRef(in_size, R=..., tau=..., tau_ref=..., V_T=..., delta_T=..., ...)` | Use when exponential spike initiation and an explicit refractory interval are both required. |
| `brainpy.state.QuaIF(in_size, R=..., tau=..., V_c=..., c=..., ...)` | Use when quadratic spike initiation is the required reduced nonlinear model. |

### Adaptive integrate-and-fire models

Use this family when recent firing must change threshold or excitability through an adaptation State.

| API | Use when |
|---|---|
| `brainpy.state.ALIF(in_size, R=..., tau=..., tau_a=..., beta=..., ...)` | Use for low-cost LIF dynamics with a spike-driven adaptive threshold. |
| `brainpy.state.AdExIF(in_size, R=..., tau=..., tau_w=..., V_T=..., delta_T=..., a=..., b=..., ...)` | Use when exponential spike onset and adaptation current are both required. |
| `brainpy.state.AdExIFRef(in_size, R=..., tau=..., tau_w=..., tau_ref=..., V_T=..., delta_T=..., a=..., b=..., ...)` | Use when adaptive exponential dynamics must also enforce refractoriness. |
| `brainpy.state.AdQuaIF(in_size, R=..., tau=..., tau_w=..., V_c=..., c=..., a=..., b=..., ...)` | Use when quadratic spike onset and adaptation current are both required. |
| `brainpy.state.AdQuaIFRef(in_size, R=..., tau=..., tau_w=..., tau_ref=..., V_c=..., c=..., a=..., b=..., ...)` | Use when adaptive quadratic dynamics must also enforce refractoriness. |

### Generalized integrate-and-fire models

Use this family when dynamic threshold and spike-triggered aftercurrents must be represented explicitly.

| API | Use when |
|---|---|
| `brainpy.state.Gif(in_size, V_th_inf=..., V_th_reset=..., k1=..., k2=..., A1=..., A2=..., ...)` | Use when a generalized integrate-and-fire model must capture threshold adaptation and multiple aftercurrents. |
| `brainpy.state.GifRef(in_size, tau_ref=..., V_th_inf=..., V_th_reset=..., k1=..., k2=..., A1=..., A2=..., ...)` | Use when generalized integrate-and-fire dynamics must also enforce an explicit refractory interval. |

### Phenomenological firing models

Use this family when diverse firing regimes matter but channel-level conductance dynamics do not.

| API | Use when |
|---|---|
| `brainpy.state.Izhikevich(in_size, a=..., b=..., c=..., d=..., V_th=..., ...)` | Use for rich single-neuron firing patterns at lower cost than conductance-based models. |
| `brainpy.state.IzhikevichRef(in_size, a=..., b=..., c=..., d=..., V_th=..., tau_ref=..., ...)` | Use when an Izhikevich model must additionally enforce a refractory interval. |

### Conductance-based neuron models

Use this family only when voltage-dependent membrane conductances are part of the scientific question; expect higher cost and verify a sufficiently small `dt`.

| API | Use when |
|---|---|
| `brainpy.state.HH(in_size, ENa=..., gNa=..., EK=..., gK=..., EL=..., gL=..., C=..., ...)` | Use for canonical Hodgkin-Huxley sodium, potassium, and leak conductances. |
| `brainpy.state.MorrisLecar(in_size, V_Ca=..., g_Ca=..., V_K=..., g_K=..., V_leak=..., g_leak=..., C=..., ...)` | Use for reduced calcium-potassium conductance dynamics with a two-variable phase-plane interpretation. |
| `brainpy.state.WangBuzsakiHH(in_size, ENa=..., gNa=..., EK=..., gK=..., EL=..., gL=..., C=..., phi=..., ...)` | Use for Wang-Buzsaki fast-spiking interneuron conductance dynamics. |

Do not increase model detail as a default. Validate each candidate under its natural parameter scale and the intended driving protocol:

```python
# Given a fully constructed candidate and a unitful drive:
with brainstate.environ.context(dt=0.1 * u.ms):
    brainstate.nn.init_all_states(candidate)
    times = u.math.arange(
        0.0 * u.ms,
        200.0 * u.ms,
        brainstate.environ.get_dt(),
    )

    def step(t):
        with brainstate.environ.context(t=t):
            spike = candidate(drive)
            return candidate.V.value, spike

    voltage, spikes = brainstate.transform.for_loop(step, times)

assert voltage.shape[0] == times.shape[0]
assert spikes.shape == voltage.shape
```

Compare firing rate, adaptation, spike onset, and stability at the intended `dt`; do not choose from the class name alone.

## Input generators

Choose the source from what is already known and where the generated input should be applied.

### Scheduled and autonomous spike sources

Use this family when the source owns either a prescribed event schedule or a persistent stochastic firing-rate configuration.

| API | Use when |
|---|---|
| `brainpy.state.SpikeTime(in_size, indices=..., times=..., weights=True, time_as_step="round", ...)` | Use when event indices and times are data rather than stochastic samples. |
| `brainpy.state.PoissonSpike(in_size, freqs, spk_type=bool, ...)` | Use when a stateful source should sample one spike vector per update from fixed or callable rates. |

### Rate encoders

Use this family when firing rates arrive as call-time data and must be converted into stochastic spikes.

| API | Use when |
|---|---|
| `brainpy.state.PoissonEncoder(in_size, spk_type=bool, ...)` | Use when each call supplies the rate tensor to encode instead of storing a fixed source rate. |

### Direct Poisson input

Use this family when Poisson events should be accumulated directly into a target State rather than emitted as an intermediate spike population.

| API | Use when |
|---|---|
| `brainpy.state.PoissonInput(target=..., num_input=..., freq=..., weight=..., indices=None, ...)` | Use the Module form when the input configuration and target binding belong in the Module graph. |
| `brainpy.state.poisson_input(freq, num_input, weight, target, indices=None, refractory=None)` | Use the functional form when the surrounding update workflow already owns the target State and call site. |

Keep stochastic and time-dependent inputs inside the same `brainstate.environ.context(dt=..., t=...)` and transformed time loop as the model. Use BrainState randomness when reproducibility or mapped random streams matter.

## Synaptic dynamics

Synaptic dynamics determine the temporal filter and constrain valid State alignment.

### Linear synaptic filters

Use this family when superposition permits exact AlignPost State and the required response is exponential or alpha-shaped.

| API | Use when |
|---|---|
| `brainpy.state.Expon(in_size, name=None, tau=..., g_initializer=...)` | Use for the canonical single-exponential decay and AlignPost path. |
| `brainpy.state.DualExpon(in_size, name=None, tau_decay=..., tau_rise=..., amplitude=..., normalize=True, g_initializer=...)` | Use when distinct rise and decay constants are required while retaining linear-filter alignment. |
| `brainpy.state.Alpha(in_size, name=None, tau=..., g_initializer=...)` | Use when one time constant and an alpha-shaped response are required. |

### Receptor-kinetic synapses

Use this family when transmitter binding introduces nonlinear receptor State; use AlignPre rather than assuming linear AlignPost equivalence.

| API | Use when |
|---|---|
| `brainpy.state.AMPA(in_size, name=None, alpha=..., beta=..., T=..., T_dur=..., g_initializer=...)` | Use for AMPA receptor binding and unbinding kinetics. |
| `brainpy.state.GABAa(in_size, name=None, alpha=..., beta=..., T=..., T_dur=..., g_initializer=...)` | Use for GABA-A receptor binding and unbinding kinetics. |
| `brainpy.state.BioNMDA(in_size, name=None, alpha1=..., beta1=..., alpha2=..., beta2=..., T=..., T_dur=..., g_initializer=..., x_initializer=...)` | Use for biological NMDA kinetics; pair it with a compatible voltage-dependent output when magnesium block matters. |

Use `.desc(size, **parameters)` when a projection should construct, own, or merge a compatible component. Instantiate the synapse directly when it is updated independently or intentionally shared outside projection construction.

## Synaptic outputs

The output converts the communicated or filtered synaptic value into current received by the target.

### Conductance-based outputs

Use this family when postsynaptic voltage and reversal potential must determine synaptic current.

| API | Use when |
|---|---|
| `brainpy.state.COBA(E=...)` | Use for current `g * (E - V)`; choose the reversal potential for the intended excitatory or inhibitory path. |
| `brainpy.state.MgBlock(E=..., cc_Mg=..., alpha=..., beta=..., V_offset=...)` | Use for NMDA-like conductance when voltage-dependent magnesium block is part of the response. |

### Current-based outputs

Use this family when synaptic current must remain independent of postsynaptic voltage.

| API | Use when |
|---|---|
| `brainpy.state.CUBA(scale=...)` | Use for voltage-independent current injection; calibrate `scale`, weights, signs, and units for the current-based model. |

Changing `COBA` to `CUBA` is not a syntax-only substitution. Recheck weight units, inhibitory sign, magnitude, and network operating point.

Subclass `brainpy.state.SynOut` only when the built-ins cannot express the required conductance-to-current rule. Preserve the target neuron's registered-current input contract.

## Short-term plasticity

Use this family only when recent presynaptic activity must modulate transmission.

| API | Use when |
|---|---|
| `brainpy.state.STP(in_size, name=None, U=..., tau_f=..., tau_d=...)` | Use when both facilitation and depression are required; it tracks utilization `u` and available resources `x`. |
| `brainpy.state.STD(in_size, name=None, tau=..., U=...)` | Use when depression is required without facilitation; it tracks available resources `x`. |

Use `.desc(...)` through a functional projection builder's `stp=` argument when plasticity belongs inside projection construction:

```python
# Given constructed pre/post populations:
projection = brainpy.state.align_post_projection(
    pre.prefetch("V"),
    lambda voltage: pre.get_spike(voltage) != 0.0,
    comm=brainstate.nn.EventFixedProb(
        n_pre,
        n_post,
        conn_num=0.1,
        conn_weight=0.5 * u.mS,
    ),
    syn=brainpy.state.Expon.desc(n_post, tau=5.0 * u.ms),
    out=brainpy.state.COBA.desc(E=0.0 * u.mV),
    post=post,
    stp=brainpy.state.STP.desc(
        n_pre,
        U=0.2,
        tau_f=1500.0 * u.ms,
        tau_d=200.0 * u.ms,
    ),
)

# Inside update(), before post integrates the current step:
projection()
```

Use `STP.desc(n_pre, ...)` for plasticity State aligned to the presynaptic population. Call `STP(...)` or `STD(...)` directly only when inspecting their isolated dynamics. Open `references/projection-patterns.md` for projection selection and update order.

## Readouts and temporal reduction

Choose the readout from the target definition, not from convenience.

### Trainable dynamical readouts

Use this family when the model must learn a continuous decoding map, optionally with an independently replaceable temporal filter.

| API | Use when |
|---|---|
| `brainpy.state.LeakyRateReadout(in_size, out_size, tau=..., w_init=..., name=None)` | Use for a single trainable linear decoder with built-in leaky dynamics and per-step continuous output. |
| `brainstate.nn.Linear(in_size, out_size, w_init=..., b_init=..., ...)` | Use as the mapping stage when decoder weights and temporal filtering must remain separate. |
| `brainpy.state.Expon(in_size, name=None, tau=..., g_initializer=...)` | Use after a separate linear mapping when the filtered State and its lifecycle must remain independently replaceable. |

### Explicit temporal reductions

Use this family when the loss should consume a statistic of per-step spikes or logits rather than a dynamical readout State.

| API or expression | Use when |
|---|---|
| `u.math.sum(outputs, axis=0)` | Use when the target depends on total spike count or accumulated evidence across time. |
| `u.math.mean(outputs, axis=0)` | Use when the target depends on a time-normalized rate or average logit. |
| `outputs[-1]` | Use when the decision is encoded at the final valid step; handle padding and masking before selecting it. |

Do not default to time averaging when the label depends on latency, precise timing, or the final State. Open `references/braintools/metric.md` when selecting the loss-facing reduction, `references/braintools/surrogate.md` when gradients cross a spiking nonlinearity, and `references/braintools/brainstate-control-flow-patterns.md` for long-rollout BPTT.

## Boundaries and official sources

- Open `references/projection-patterns.md` for communication, alignment, descriptor ownership, delays, short-term plasticity integration, direct input, and electrical coupling.
- Open `references/braintools/metric.md` for loss and reduction choice, `references/braintools/surrogate.md` for surrogate choice, and `references/braintools/brainstate-control-flow-patterns.md` for loop form and checkpointing.
- Open `references/nest-compatible/nest-workflow.md` instead when the task uses NEST/PyNEST model names or `Simulator`; do not mix those component APIs into the native path.

Official sources:

- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-choose-neuron.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-coba-cuba-synapses.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-short-term-plasticity.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-neurons.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synapses.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synouts.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-plasticity.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-inputs.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-readouts.html`
