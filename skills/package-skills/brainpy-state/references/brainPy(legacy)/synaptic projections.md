# Synaptic projections

Use this reference when a legacy BrainPy model needs a composed `brainpy.dyn`
projection. Select the projection from who supplies presynaptic activity, where
synaptic state is stored, whether compatible state should be merged, and whether
the response is instantaneous. Open `synpase properties.md` after making these
structural decisions to select the synaptic dynamics and output.

## Selection map

| Need | Use | Critical invariant |
|---|---|---|
| The projection owns `pre`, delay, communication, synapse, output, and `post` | A `FullProj...` class | Call it with no spike argument; it reads the registered presynaptic system. |
| The caller already has the presynaptic value | A `HalfProj...` class | Pass the value to the projection on every update. |
| Synaptic state should have postsynaptic size | An align-post class | Internal order is `spikes -> comm -> syn -> out`; prefer event-driven communication. |
| Synaptic state should have presynaptic size | An align-pre class | Internal order places `syn` before `comm`; choose whether delay occurs before or after `syn`. |
| Compatible delay and synapse state should be shared | A class ending in `Mg` | `Mg` means automatic merging here; it does not mean magnesium block. |
| Spikes should produce instantaneous target increments | `HalfProjDelta` or `FullProjDelta` | Input must be spiking data; these projections omit rise and decay state. |

## Projection anatomy

A chemical projection composes distinct roles; keep them separate so changing
connectivity does not silently change the synaptic waveform or current law.

| Role | Decision |
|---|---|
| `pre` | Supplies presynaptic state to a full-chain projection. |
| `delay` | Delays presynaptic spikes or synaptic state, depending on the align-pre class. |
| `comm` | Maps values from presynaptic to postsynaptic dimensions and applies connection weights. |
| `syn` | Evolves the conductance or other synaptic state over time. |
| `out` | Converts synaptic state into the input accumulated on `post`. |
| `post` | Receives the projected input before its own state update. |

Use full-chain projections when the projection belongs inside a network graph.
Use half-chain projections for external inputs, trainable layers, or another
workflow that already owns presynaptic extraction.

## Align-post projections

Use align-post when synaptic variables should scale with the postsynaptic
population. These projections compute communication from spikes before updating
the synapse, so use event-driven `comm` implementations for sparse spike input.

| API | Description |
|---|---|
| `HalfProjAlignPostMg(comm, syn, out, post, out_label=None, ...)` | Use with caller-supplied spikes when compatible postsynaptic synapse state should be merged; it implements `comm -> syn -> out -> post`. |
| `FullProjAlignPostMg(pre, delay, comm, syn, out, post, out_label=None, ...)` | Use for the complete registered chain with automatic delay and synapse merging; pass mergeable `syn` and `out` descriptors. |
| `HalfProjAlignPost(comm, syn, out, post, out_label=None, ...)` | Use with caller-supplied spikes when the projection must own independent postsynaptic synapse state. |
| `FullProjAlignPost(pre, delay, comm, syn, out, post, out_label=None, ...)` | Use for a complete registered chain with independent postsynaptic synapse state. |

The current legacy API uses `FullProjAlignPostMg`; do not copy the older
`ProjAlignPostMg2` spelling from historical tutorial cells.

```python
import brainpy as bp


class EINet(bp.DynSysGroup):
    def __init__(self, n_exc=3200, n_inh=800):
        super().__init__()
        self.exc = bp.dyn.LifRef(n_exc)
        self.inh = bp.dyn.LifRef(n_inh)
        self.e2i = bp.dyn.FullProjAlignPostMg(
            pre=self.exc,
            delay=0.1,
            comm=bp.dnn.EventJitFPHomoLinear(
                n_exc,
                n_inh,
                prob=0.02,
                weight=0.6,
            ),
            syn=bp.dyn.Expon.desc(size=n_inh, tau=5.0),
            out=bp.dyn.COBA.desc(E=0.0),
            post=self.inh,
        )

    def update(self, drive):
        self.e2i()
        self.exc(drive)
        self.inh(drive)
        return self.exc.spike, self.inh.spike
```

Preserve the grouped-network update order shown by the generated API example:
apply the full projection before updating the receiving population so its input
is available to that update.

## Align-pre projections

Use align-pre when synaptic variables should scale with the presynaptic
population. Communication receives floating-point synaptic state rather than
spikes, so these classes do not provide the event-driven communication advantage
of align-post projections.

| API | Description |
|---|---|
| `VanillaProj(comm, out, post, name=None, mode=None)` | Use when the caller supplies an already prepared presynaptic value and no delay or dynamic synapse belongs in the projection; it applies communication and output conversion. |
| `FullProjAlignPreSDMg(pre, syn, delay, comm, out, post, ...)` | Use when synaptic state must update before it is delayed; it implements `pre -> syn -> delay -> comm -> out -> post` and merges compatible state. |
| `FullProjAlignPreDSMg(pre, delay, syn, comm, out, post, ...)` | Use when presynaptic output must be delayed before synaptic state updates; it implements `pre -> delay -> syn -> comm -> out -> post` and merges compatible state. |
| `FullProjAlignPreSD(pre, syn, delay, comm, out, post, ...)` | Use the synapse-then-delay order when every projection must own independent state. |
| `FullProjAlignPreDS(pre, delay, syn, comm, out, post, ...)` | Use the delay-then-synapse order when every projection must own independent state. |

Choose `DS` when delay represents axonal transmission before receptor dynamics.
Choose `SD` only when the modeled delay applies to the evolved synaptic state.

## Delta projections

Delta projections transmit an instantaneous weighted jump and do not construct a
temporal synapse or synaptic output object.

| API | Description |
|---|---|
| `HalfProjDelta(comm, post, name=None, mode=None)` | Use for caller-supplied spikes and an instantaneous target increment. |
| `FullProjDelta(pre, delay, comm, post, name=None, mode=None)` | Use when the registered presynaptic group and delay belong to the instantaneous projection. |

Do not pass continuous rates or currents to a delta projection. Use `VanillaProj`
or an align-post projection with explicit dynamics for those signals.

## Projection-adjacent inputs

The projection API also exposes input helpers. Use them when an input should
target a variable directly rather than arrive through a neuron-to-neuron
chemical projection.

| API | Description |
|---|---|
| `PoissonInput(target_var, num_input, freq, weight, mode=None, name=None)` | Use for many independent homogeneous Poisson inputs to one target variable; it generates events during simulation without storing spike trains. |
| `InputVar(size, keep_size=False, ..., name=None, mode=None)` | Use to represent an explicit input variable inside a dynamical-system graph. |
| `SynConn(pre, post, conn=None, name=None, mode=None)` | Subclass only when implementing a new two-end connection abstraction; it binds the pre/post systems and optional connector. |

## Common failures

- Do not interpret an `Mg` projection suffix as `MgBlock`; merging and
  magnesium-dependent output are independent choices.
- Do not use an align-pre class merely because the source is presynaptic. Choose
  alignment from synaptic-state dimension and internal execution order.
- Do not use non-event communication in an align-post spike path without a
  concrete reason; the API explicitly prefers event-driven communication.
- Do not update `post` before the projection has deposited the current step's
  input.
- Do not copy `brainpy.dyn` projection names or construction patterns from a
  different BrainPy implementation. This reference covers the legacy API only.

## Sources

- Phenomenological Synaptic Models: https://brainpy.readthedocs.io/tutorial_building/phenon_synapse_models.html
- Synaptic Projections API: https://brainpy.readthedocs.io/apis/brainpy.dyn.projections.html
