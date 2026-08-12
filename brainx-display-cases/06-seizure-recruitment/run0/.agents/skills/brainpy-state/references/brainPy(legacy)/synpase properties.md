# Synapse properties

Use this reference when a legacy BrainPy projection needs synaptic dynamics,
short-term filtering, output-current semantics, coupling, or long-term
plasticity. Open `synaptic projections.md` first when alignment, merging, delay
order, or half-chain versus full-chain ownership is still undecided.

## Selection map

| Modeled property | Use | Key constraint |
|---|---|---|
| One exponential decay after each spike | `Expon` | Use as the canonical low-state phenomenological filter. |
| An alpha-shaped response | `Alpha` | Rise and decay are represented by the alpha limit. |
| Separate rise and decay constants | `DualExpon` or `DualExponV2` | `DualExpon` is align-pre-only; `DualExponV2` supports align-pre and align-post. |
| Saturating slow NMDA gating | `NMDA` | Pair with `MgBlock` when voltage-dependent magnesium block must affect current. |
| Receptor binding and unbinding | `AMPA`, `GABAa`, or `BioNMDA` | Use align-pre for these kinetic state models. |
| Recent spikes change release efficacy | `STD` or `STP` | These filter transmission state; they do not persistently rewrite connection weights. |
| Voltage-dependent versus independent current | `COBA`, `MgBlock`, or `CUBA` | Select output semantics separately from synaptic dynamics. |
| Spike timing persistently changes weights | `STDP_Song2000` | The projection updates the communication weight during simulation. |

## Phenomenological dynamics

Phenomenological models prescribe a conductance waveform from spike events; use
the least complex waveform that preserves the timing required by the experiment.

| API | Description |
|---|---|
| `Expon(size, tau=8.0, ...)` | Use for instantaneous activation followed by a single exponential decay; `tau` is the decay time constant in milliseconds. |
| `Alpha(size, tau_decay=10.0, ...)` | Use for an alpha-shaped response when a distinct rise constant is unnecessary. |
| `DualExpon(size, tau_decay=10.0, tau_rise=1.0, A=None, ...)` | Use for separate rise and decay phases in align-pre projections; it implements the waveform through two coupled linear equations. |
| `DualExponV2(size, tau_decay=10.0, tau_rise=1.0, A=None, ...)` | Use for the same dual-exponential choice when align-post compatibility is required; it also works with align-pre. |
| `NMDA(size, a=0.5, tau_decay=100.0, tau_rise=2.0, ...)` | Use for phenomenological NMDA gating with a slow decay and saturating rise; use `MgBlock` to add postsynaptic-voltage dependence to the output. |

The alpha response is the limiting dual-exponential case when rise and decay
constants become equal. Do not substitute `DualExpon` into an align-post
projection; use `DualExponV2` there.

## Receptor-kinetic dynamics

Kinetic models track fractions of receptors in closed and open states; choose
them when receptor saturation under spike trains is part of the modeled result.

| API | Description |
|---|---|
| `AMPA(size, alpha=0.98, beta=0.18, T=0.5, T_dur=0.5, ...)` | Use for AMPA binding and unbinding with transmitter concentration `T` applied for `T_dur`. |
| `GABAa(size, alpha=0.53, beta=0.18, T=1.0, T_dur=1.0, ...)` | Use for GABA-A kinetics; select inhibitory current through the output reversal potential rather than by changing this state equation. |
| `BioNMDA(size, alpha1=2.0, beta1=0.01, alpha2=1.0, beta2=0.5, T=1.0, T_dur=0.5, ...)` | Use for second-order biological NMDA kinetics with separate conversion rates for the `g` and `x` variables. |

```python
import brainpy as bp


class AMPAProjection(bp.Projection):
    def __init__(self, pre, post, delay, probability, g_max, E=0.0):
        super().__init__()
        self.proj = bp.dyn.FullProjAlignPreDSMg(
            pre=pre,
            delay=delay,
            syn=bp.dyn.AMPA.desc(
                pre.num,
                alpha=0.98,
                beta=0.18,
                T=0.5,
                T_dur=0.5,
            ),
            comm=bp.dnn.CSRLinear(
                bp.conn.FixedProb(
                    probability,
                    pre=pre.num,
                    post=post.num,
                ),
                g_max,
            ),
            out=bp.dyn.COBA(E=E),
            post=post,
        )

    def update(self):
        return self.proj()
```

This uses delay-before-synapse align-pre composition: presynaptic spikes are
delayed, receptor state is updated, communication maps that state, and `COBA`
converts it to postsynaptic input.

## Short-term efficacy

Short-term models filter current according to recent activity without changing
the stored long-term connection weight.

| API | Description |
|---|---|
| `STD(size, tau=200.0, U=0.07, ...)` | Use for depression only; it tracks the available-resource fraction `x`. |
| `STP(size, U=0.15, tau_f=1500.0, tau_d=200.0, ...)` | Use for facilitation and depression; it tracks release utilization `u` and available resources `x`. |

Keep `STD` and `STP` aligned to the dimension at which presynaptic release
history is represented. Do not describe their changing effective transmission
as long-term weight learning.

## Synaptic outputs

Synaptic dynamics produce a conductance-like state; a `SynOut` converts it into
the signal accumulated by the receiving model.

| API | Description |
|---|---|
| `SynOut(name=None, scaling=None)` | Subclass only when the built-in current laws cannot express the required output; it is the base output descriptor. |
| `COBA(E, sharding=None, name=None, scaling=None)` | Use for conductance-based current whose magnitude and direction depend on postsynaptic voltage and reversal potential `E`. |
| `CUBA(name=None, scaling=None)` | Use for current-based transmission; it passes the communicated synaptic value through an identity output. |
| `MgBlock(E=0.0, cc_Mg=1.2, alpha=0.062, beta=3.57, V_offset=0.0, ...)` | Use for NMDA-like conductance with postsynaptic-voltage-dependent magnesium block. |

Changing `COBA` to `CUBA` is a modeling change, not a syntax change. Recheck
weight sign, magnitude, and network operating point after changing output law.

## Coupling models

Use coupling models for direct state-to-state coupling rather than a chemical
spike-to-conductance projection.

| API | Description |
|---|---|
| `DiffusiveCoupling(coupling_var1, coupling_var2, var_to_output, conn_mat, delay_steps=None, ...)` | Use when output depends on a connected difference between two coupling variables; the first variable owns optional delay. |
| `AdditiveCoupling(coupling_var, var_to_output, conn_mat, delay_steps=None, ...)` | Use when connected source values should be added directly to one or more target variables. |

## Long-term spike-timing plasticity

`STDP_Song2000` is a complete full-chain plastic projection, not a standalone
synaptic filter.

| API | Description |
|---|---|
| `STDP_Song2000(pre, delay, syn, comm, out, post, tau_s=16.8, tau_t=33.7, A1=0.96, A2=0.53, W_max=None, W_min=None, ...)` | Use for Song et al. pair-based STDP; it tracks pre/post traces and updates `comm.weight`, optionally clipping it with `W_min` and `W_max`. |

Use an event-driven communication object with mutable weights, pass the temporal
filter and output as descriptors, and inspect `projection.comm.weight` to verify
learning. Call the STDP projection before updating its registered pre- and
postsynaptic groups, matching the generated API example.

## Common failures

- Do not choose a kinetic receptor model when a single exponential waveform is
  sufficient; kinetic state is justified only when saturation or binding
  dynamics affect the result.
- Do not pair `DualExpon` with align-post; use `DualExponV2`.
- Do not expect `NMDA` alone to apply postsynaptic-voltage-dependent magnesium
  block; select `MgBlock` as the output.
- Do not treat reversal potential as a cosmetic parameter; it controls the
  direction and voltage dependence of conductance-based current.
- Do not conflate `STP` or `STD` with `STDP_Song2000`; only the latter rewrites
  long-term communication weights.
- Do not copy these `brainpy.dyn` APIs into a different BrainPy implementation.
  This reference covers the legacy API only.

## Sources

- Synaptic Dynamics API: https://brainpy.readthedocs.io/apis/brainpy.dyn.synapses.html
- Synaptic Outputs API: https://brainpy.readthedocs.io/apis/brainpy.dyn.outs.html
- Synaptic Plasticity API: https://brainpy.readthedocs.io/apis/brainpy.dyn.plasticity.html
- Kinetic Synaptic Models: https://brainpy.readthedocs.io/tutorial_building/kinetic_synapse_models.html
