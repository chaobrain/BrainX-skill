# Projection patterns

Use this reference after selecting neuron, synapse, output, and communication semantics. Choose alignment from synapse linearity and reuse direction before choosing the class or functional API form.

## Projection anatomy

Every native chemical projection separates four roles:

| Role | Decision |
|---|---|
| `comm` | Choose how presynaptic values are weighted and mapped to postsynaptic values. Use event operators for sparse binary spikes and dense linear operators for dense continuous values. |
| `syn` | Choose temporal filtering and determine whether its dynamics satisfy AlignPost's linear exponential condition. |
| `out` | Choose conductance-based, current-based, or voltage-blocked conversion into postsynaptic current. |
| `post` | Supply the receiving neuron population; projections accumulate input into it before its update. |

Call each chemical projection before `post(...)` in the step. Read the presynaptic spike from the previous completed update, route it through the projection, then update the source and target populations.

Subclass `brainpy.state.Projection` only when the concrete projections cannot express the required lifecycle. Register communication, dynamics, and output Modules as attributes and preserve the postsynaptic current/delta-input contract.

## Alignment decision

| Condition | Use | State and result |
|---|---|---|
| Exponential-family linear synapse with many sources converging on a target | `AlignPostProj` or `align_post_projection` | Stores exact synaptic State on the postsynaptic dimension, with memory proportional to `N_post`. |
| Nonlinear receptor kinetics | `align_pre_projection` | Stores exact synaptic State on the presynaptic dimension when outgoing synapses share parameters per source. |
| One source population fans out to several targets | `align_pre_projection` | Reuses presynaptic traces across targets, with memory proportional to `N_pre`. |
| Continuous input needs communication and output conversion but no synapse dynamics | `CurrentProj` | Applies `comm`, then `out`, and deposits current into `post`. |
| An event should produce an instantaneous target increment | `DeltaProj` | Communicates the delta without temporal synapse State. |
| Electrical coupling | `SymmetryGapJunction` or `AsymmetryGapJunction` | Computes reciprocal or directed gap-junction current rather than a chemical synapse. |

AlignPre and AlignPost are exact only under their stated conditions. They change State ownership and memory, not the intended dynamics.

## AlignPost API forms

Use `AlignPostProj` for the direct push form:

```python
proj = brainpy.state.AlignPostProj(
    comm=brainstate.nn.EventFixedProb(
        n_pre,
        n_post,
        conn_num=0.1,
        conn_weight=0.5 * u.mS,
    ),
    syn=brainpy.state.Expon.desc(n_post, tau=5.0 * u.ms),
    out=brainpy.state.COBA.desc(E=0.0 * u.mV),
    post=post,
)

proj(pre.get_spike() != 0.0)
```

Use the functional builder when presynaptic extraction or short-term plasticity belongs to projection construction:

```python
proj = brainpy.state.align_post_projection(
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

proj()
```

Do not pass both a pulled presynaptic source and a second explicit spike array at update time.

When `syn` and `out` are matching `.desc(...)` objects, compatible AlignPost projections targeting the same population can merge and share the postsynaptic synapse/output instances. When `syn` is a concrete synapse instance, pass a concrete output instance as well; that pair is not merged.

## AlignPre form

Use the function `align_pre_projection`; there is no AlignPre projection class:

```python
proj = brainpy.state.align_pre_projection(
    pre,
    syn=brainpy.state.AMPA(n_pre),
    comm=brainstate.nn.Linear(
        n_pre,
        n_post,
        w_init=braintools.init.KaimingNormal(unit=u.mS),
        b_init=None,
    ),
    out=brainpy.state.COBA(E=0.0 * u.mV),
    post=post,
)
```

Use AlignPre only when synaptic parameters are homogeneous for the outgoing synapses represented by each presynaptic trace. If connection-specific nonlinear kinetics are required, the shared-trace premise does not hold.

## Delayed projections

Use the projection's `delay=` argument when the spike or continuous signal is passed directly at update time:

```python
proj = brainpy.state.AlignPostProj(
    comm=brainstate.nn.EventFixedProb(
        n_pre,
        n_post,
        conn_num=0.1,
        conn_weight=0.5 * u.mS,
    ),
    syn=brainpy.state.Expon.desc(n_post, tau=5.0 * u.ms),
    out=brainpy.state.COBA.desc(E=0.0 * u.mV),
    post=post,
    delay=5.0 * u.ms,
)

proj(pre.get_spike() != 0.0)
```

`AlignPostProj` and `CurrentProj` accept a scalar time for one homogeneous delay or a `(N_pre,)` array for per-presynaptic axonal delays. Sub-`dt` delays are linearly interpolated. The delay buffer is sized during State initialization from the largest delay, so do not change `dt` afterward.

Built-in neurons store membrane potential rather than a separate historical spike State. For the pull-based form, delay the prefetched voltage, reconstruct the spike from that delayed value, and call the projection without an explicit signal:

```python
proj = brainpy.state.AlignPostProj(
    pre.prefetch("V").delay.at(5.0 * u.ms),
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
)

proj()
```

Reinitialize the containing Module graph before an independent run so delay buffers and neural State reset together.

## Communication boundary

Use `brainstate.nn.EventFixedProb` for the canonical probabilistic sparse-spike connection shown in BrainPy-State tutorials. Open `skills/brainevent/SKILL.md` when the decision involves `EventLinear`, fixed-number event connectivity, sparse formats, plasticity operators, custom kernels, or dense-to-event performance conversion.

## Official sources

- `https://brainx.chaobrain.com/brainpy-state/concepts/model-anatomy.html`
- `https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/02-synapse-and-projection.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-short-term-plasticity.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-projections.html`
