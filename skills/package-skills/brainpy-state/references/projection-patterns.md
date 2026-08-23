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

Choose alignment from synapse linearity and trace reuse; use a filter-free or electrical projection only when the connection does not require aligned chemical-synapse State.

### Linear convergent fan-in

Use this family for exponential-family linear synapses when many sources converge on a target and exact State should scale with the postsynaptic dimension.

| API | Use when |
|---|---|
| `brainpy.state.AlignPostProj(*modules, comm=..., syn=..., out=..., post=..., label=None, delay=None)` | Use the direct push form when the caller supplies presynaptic events at update time; it stores exact synaptic State with memory proportional to `N_post`. |
| `brainpy.state.align_post_projection(*spike_generator, comm=..., syn=..., out=..., post=..., stp=None)` | Use the builder form when presynaptic extraction or short-term plasticity belongs to projection construction. |

### Nonlinear or reusable fan-out

Use this family for nonlinear receptor kinetics or when one source population fans out to several targets and outgoing parameters are homogeneous per source.

| API | Use when |
|---|---|
| `brainpy.state.align_pre_projection(*spike_generator, syn=..., comm=..., out=..., post=..., stp=None)` | Use when exact synaptic traces should live on and be reusable from the presynaptic dimension, with memory proportional to `N_pre`. |

### Filter-free chemical projections

Use this family when communication should deposit continuous current or an instantaneous event without temporal synapse dynamics.

| API | Use when |
|---|---|
| `brainpy.state.CurrentProj(*prefetch, comm=..., out=..., post=..., delay=None)` | Use when continuous input requires communication and output conversion before current is deposited into `post`. |
| `brainpy.state.DeltaProj(*prefetch, comm=..., post=..., label=None)` | Use when an event should produce an instantaneous target increment without temporal synapse State. |

### Electrical coupling

Use this family when voltage differences should drive reciprocal or directed gap-junction current rather than a chemical synapse.

| API | Use when |
|---|---|
| `brainpy.state.SymmetryGapJunction(couples, states, conn, weight, param_type=brainstate.ParamState)` | Use when the same conductance applies reciprocally in both coupling directions. |
| `brainpy.state.AsymmetryGapJunction(pre, pre_state, post, post_state, conn, weight, param_type=brainstate.ParamState)` | Use when the two coupling directions require distinct conductances. |

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

Use the functional builder when presynaptic extraction or short-term plasticity belongs to projection construction. Open `references/component-selection.md` for the canonical `stp=STP.desc(...)` composition.

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

Choose projection alignment before adding a delay. Open `references/brain-dynamics-delay-protocol.md` when selecting direct `delay=` versus delayed prefetch, applying scalar or per-presynaptic delays, preserving the fixed-`dt` buffer invariant, or separating BrainPy projection delays from general BrainState delay buffers.

## Communication boundary

Use `brainstate.nn.EventFixedProb` for the canonical probabilistic sparse-spike connection shown in BrainPy-State tutorials. Open `skills/package-skills/brainevent/SKILL.md` when the decision involves `EventLinear`, fixed-number event connectivity, sparse formats, plasticity operators, custom kernels, or dense-to-event performance conversion.

Open `references/scripts/103_COBA_2005.py` for a complete E/I COBA projection workflow. Open `references/scripts/109_fast_global_oscillation.py` when the projection uses `DeltaProj` with delayed recurrent input.

Open `references/scripts/sound_localization.py` when independent conditions require a mapped bank of heterogeneous integer event taps feeding BrainEvent fixed-fan-out communication and coincidence detectors.

## Official sources

- `https://brainx.chaobrain.com/brainpy-state/concepts/model-anatomy.html`
- `https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/02-synapse-and-projection.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-short-term-plasticity.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/brainpy-projections.html`
