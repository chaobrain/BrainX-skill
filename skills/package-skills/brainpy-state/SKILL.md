---
name: brainpy-state
description: Use for point-neuron modeling, from individual neurons to spiking networks with synapses and projections. Use alone for point-neuron studies or together with BrainCell and/or BrainMass when point-neuron dynamics participate in a multiscale model. Use as the legacy BrainPy routing entry only when an existing codebase uses legacy `brainpy`, uses neither units nor BrainTrace, and the user explicitly asks to preserve it.
---

# BrainPy-State

## Purpose and boundary

Use this skill for the native `brainpy.state` modeling path: compose point neurons, synapses, communication operators, synaptic outputs, projections, inputs, plasticity, and readouts into simulations or trainable spiking neural networks.

Route NEST-compatible work to `references/nest-compatible/nest-workflow.md`, and online-learning execution to BrainTrace.

Open `references/brainPy(legacy)/brainpy legacy workflow.md` only when the workspace already contains legacy `brainpy` code, contains no BrainUnit quantities, unit-aware APIs, and when the user explicitly asks to preserve old BrainPy. Use native BrainPy-State for new project is HIGHLY RECOMMENDED; use BrainPy-State with BrainCell for all cellular modeling.

Treat that workflow as the only first-level legacy reference. Do not open or route directly to any other Markdown under `references/brainPy(legacy)/`; let the workflow select those second-level references.

## Underlying principle of BrainPy-State

BrainPy-State composes unit-aware, stateful neuron and synapse `Dynamics` into projections that separate `comm`, `syn`, `out`, and `post`; BrainState transforms run the same Module graph for simulation or surrogate-gradient training.
Align synaptic State to a neuron dimension instead of each connection, reduce that state from per-synapse to per-neuron, which makes simulation memory-efficient.

### API structure

| API family | Use |
|---|---|
| Base Classes | Find the shared dynamics, neuron, and synapse abstractions used to implement custom BrainPy-State components. |
| BrainPy-style Neurons | Select native point-neuron dynamics, from integrate-and-fire models to conductance-based models. |
| BrainPy-style Synapses | Select native synaptic dynamics, from simple temporal filters to biological receptor models. |
| BrainPy-style Projections | Connect populations through projection classes, projection helper functions, or gap-junction projections. |
| BrainPy-style Synaptic Outputs | Convert synaptic state into current- or conductance-based postsynaptic input. |
| BrainPy-style Plasticity | Add short-term facilitation or depression to synaptic transmission. |
| BrainPy-style Readouts | Decode spiking activity through native readout models. |
| BrainPy-style Input Generators | Generate spike trains or time-varying simulation inputs. |

### 1. Choose and run neuron dynamics

A neuron call advances one `dt`, mutates its registered dynamical State, and exposes the current spike through `get_spike()`; initialize before every independent rollout, set `V_initializer` whenever startup voltage affects the result, and lower the complete time loop through `brainstate.transform`.

| API | Description |
|---|---|
| `brainpy.state.LIFRef(..., V_initializer=...)` | Use for the canonical point-neuron rollout when leak and refractory timing are required; each call integrates one `dt`, updates membrane/refractory State, and returns the current spike. `V_rest` sets the resting dynamics but does not set the initial membrane State, so pass an intentional initializer when startup spikes or transients matter. |
| `braintools.init.Constant(value)` | Use as `V_initializer` when every neuron must start at one known voltage; initialization creates membrane State at that value. Use a documented distribution instead when initial heterogeneity is part of the model. |
| `brainstate.nn.init_all_states(model, batch_size=...)` | Use after construction and before a rollout; it allocates or resets dynamical State across the Module graph and adds a leading batch dimension when `batch_size` is given. |
| `brainstate.environ.context(dt=..., t=...)` | Use around the rollout for `dt` and inside the step for `t`; dynamics read the active values and the context restores previous settings on exit. |
| `brainstate.environ.get_dt()` | Use when constructing the time axis or a numerical update from the active simulation step; it returns `dt` and raises when no value is set. |
| `u.math.mean(current, axis=...)` | Use when a drive current should be the arithmetic mean of unit-aware samples; it reduces the selected axes and returns a quantity with the input current unit. |
| `brainstate.transform.for_loop(step, *xs)` | Use by default for a multi-step rollout whose model State carries hidden variables; it slices leading input axes and stacks returned monitors. |
| `neuron.get_spike()` | Use after advancing or when wiring the previous completed step into a projection; it returns the neuron's current spike output without advancing State. |

```python
import brainpy
import brainstate
import braintools
import brainunit as u

with brainstate.environ.context(dt=0.1 * u.ms):
    neuron = brainpy.state.LIFRef(
        1,
        R=1.0 * u.ohm,
        tau=20.0 * u.ms,
        V_rest=-60.0 * u.mV,
        V_th=-50.0 * u.mV,
        V_reset=-60.0 * u.mV,
        tau_ref=5.0 * u.ms,
        V_initializer=braintools.init.Constant(-60.0 * u.mV),
    )
    brainstate.nn.init_all_states(neuron)
    times = u.math.arange(
        0.0 * u.ms,
        200.0 * u.ms,
        brainstate.environ.get_dt(),
    )

    def step(t):
        with brainstate.environ.context(t=t):
            neuron(25.0 * u.mA)
            return neuron.V.value, neuron.get_spike()

    voltages, spikes = brainstate.transform.for_loop(step, times)

assert voltages.shape == (2000, 1)
assert spikes.shape == (2000, 1)
```

Open `references/braintools/input-current.md` when a direct current protocol needs timed sections, pulses, waveforms, stochastic processes, or composition. Generate the complete time-major protocol once under the rollout `dt` and pass it as an input to `for_loop`; do not rebuild named sections or pulses from time predicates inside the model step. Open `references/array-creation.md` when constructing specialized unit-aware current ranges, grids, filled arrays, template-shaped arrays, or matrix patterns. Open `references/component-selection.md` when choosing among documented neuron, input, synapse, output, plasticity, or readout variants, for the decision boundary and complete category list. Open `references/braintools/brainstate-control-flow-patterns.md` when the rollout needs explicit carry, branching, or checkpointed control flow.

### 2. Compose synapses and projections

A projection deposits current into `post` before the postsynaptic neuron integrates the step, while its alignment determines which neuron dimension owns synaptic State and which kinetics remain exact; when recurrent excitation, mutual inhibition, or spike-driven connectivity is a point-neuron mechanism, represent it with explicit projections instead of a hand-written population-rate feedback `State`, and route aggregate rate coupling to BrainMass.

| API | Description |
|---|---|
| `brainstate.nn.EventFixedProb(...)` | Use for the canonical sparse binary-spike communication path; it maps presynaptic events to postsynaptic weighted input without materializing dense all-to-all activity. |
| `brainevent.BinaryArray(spikes) @ connectivity` | Use when the network must choose an explicit BrainEvent connectivity representation; it processes active presynaptic events through dense, CSR, generated, or fixed-degree storage and returns input for BrainPy synaptic dynamics. |
| `brainpy.state.Expon.desc(...)` | Use for the canonical linear exponential synaptic filter; the descriptor lets the projection construct and align the concrete synapse on the target. |
| `brainpy.state.COBA.desc(E=...)` | Use when synaptic current must depend on postsynaptic voltage and reversal potential; the descriptor lets the projection bind the output to `post`. |
| `brainpy.state.AlignPostProj(...)` | Use for exponential-family fan-in; it composes `comm`, `syn`, `out`, and `post`, stores exact synaptic State on the postsynaptic dimension, and deposits current before `post(...)` integrates. |
| `brainpy.state.align_pre_projection(...)` | Use instead for nonlinear synaptic kinetics or reusable one-to-many fan-out; it stores exact shared traces on the presynaptic dimension when outgoing parameters are homogeneous per source. |

```python
import brainpy
import brainstate
import braintools
import brainunit as u


class TwoPop(brainstate.nn.Module):
    def __init__(self, n_pre=20, n_post=10):
        super().__init__()
        self.pre = brainpy.state.LIFRef(
            n_pre,
            tau=20.0 * u.ms,
            tau_ref=5.0 * u.ms,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        self.post = brainpy.state.LIFRef(
            n_post,
            tau=20.0 * u.ms,
            tau_ref=5.0 * u.ms,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        self.proj = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                n_pre,
                n_post,
                conn_num=0.5,
                conn_weight=0.5 * u.mS,
            ),
            syn=brainpy.state.Expon.desc(n_post, tau=5.0 * u.ms),
            out=brainpy.state.COBA.desc(E=0.0 * u.mV),
            post=self.post,
        )

    def update(self, t, drive):
        with brainstate.environ.context(t=t):
            pre_spikes = self.pre.get_spike() != 0.0
            self.proj(pre_spikes)
            self.pre(drive)
            self.post(0.0 * u.mA)
            return self.post.V.value, self.proj.syn.g.value


with brainstate.environ.context(dt=0.1 * u.ms):
    net = TwoPop()
    brainstate.nn.init_all_states(net)
    times = u.math.arange(0.0 * u.ms, 200.0 * u.ms, brainstate.environ.get_dt())
    post_voltage, conductance = brainstate.transform.for_loop(
        lambda t: net.update(t, 30.0 * u.mA),
        times,
    )

assert post_voltage.shape == (2000, 10)
assert conductance.shape == (2000, 10)
```

Open `references/braintools/connectivity.md` first when geometry, distance, degree, modules, or another named topology determines which neuron pairs connect; generate the topology before choosing its BrainEvent representation. Open `references/projection-patterns.md` before implementing recurrent excitation, mutual inhibition, or other spike-driven connectivity; it owns AlignPre versus AlignPost, communication, delays, direct/delta input, short-term plasticity, and electrical coupling. Open `skills/package-skills/brainevent/references/scripts/coba_ei_teaching.py` when a BrainPy network should use BrainEvent for efficient event-driven communication; it keeps BrainPy neuron and synapse dynamics while replacing the communication step with `BinaryArray @ connectivity` over fixed-degree, CSR, or dense storage.

### 3. Train a spiking network

Training reuses the simulation graph: place a surrogate on each trained spiking nonlinearity, unroll with a differentiable transform loop, differentiate only `ParamState`, reset dynamical State for each independent batch, and update inside a compiled train step.

| API | Description |
|---|---|
| `neuron(..., spk_fun=braintools.surrogate.ReluGrad())` | Use on a spiking layer crossed by the loss gradient; it preserves hard forward spikes and supplies the backward derivative. |
| `model.states(brainstate.ParamState)` | Use to select trainable parameters; it excludes membrane voltage, conductance, spike history, and other rollout State. |
| `brainstate.transform.grad(loss_fn, params, return_value=True)` | Use to differentiate the stateful loss with respect to the selected State collection; calling the returned transform yields gradients and the loss. |
| `braintools.optim.Adam(...).register_trainable_weights(params)` | Use for the canonical optimizer lifecycle; register the same State collection used by `grad`, then call `optimizer.update(grads)`. |
| `brainstate.transform.jit(train_step)` | Use around the complete reset-gradient-update operation; compatible calls reuse the compiled training step while BrainState preserves State effects. |

```python
# Given a stateful net whose update() returns per-step logits [batch, classes].
params = net.states(brainstate.ParamState)
optimizer = braintools.optim.Adam(lr=3e-3)
optimizer.register_trainable_weights(params)

def loss_fn():
    logits = brainstate.transform.for_loop(net.update, inputs)
    logits = u.math.mean(logits, axis=0)
    return braintools.metric.softmax_cross_entropy_with_integer_labels(
        logits,
        labels,
    ).mean()

@brainstate.transform.jit
def train_step():
    brainstate.nn.init_all_states(net, batch_size=batch_size)
    grads, loss = brainstate.transform.grad(
        loss_fn,
        params,
        return_value=True,
    )()
    optimizer.update(grads)
    return loss

loss = train_step()
```

Choose training variations from the first-level references below; open only the smallest reference that owns the decision. Use `references/scripts/training-snn.py` when a complete runnable training workflow is required.

## Reference routing

Open only the smallest reference that owns the decision.

### Training variation choices

| Reference | Open when |
|---|---|
| `references/braintools/data-preprocessing.md` | Encoding continuous values, events, or temporal features into time-major spike inputs before training |
| `references/braintools/parameter-initializer.md` | Selecting unit-aware parameter, weight, delay, variance-scaling, orthogonal, or spatial initialization |
| `references/braintools/surrogate.md` | Selecting a surrogate API style or gradient family, implementing a custom surrogate, or validating its gradient |
| `references/braintools/metric.md` | Selecting the training loss or evaluation metric and validating its input orientation, units, and reduction |
| `references/braintools/optimizer.md` | Selecting an optimizer, learning-rate scheduler, Optax bridge, SciPy optimizer, or Nevergrad optimizer |
| `references/braintools/brainstate-control-flow-patterns.md` | Choosing `for_loop` versus `scan`, adding transformed branches, or checkpointing a long differentiated rollout |

### Modeling variation choices

| Reference | Open when |
|---|---|
| `references/array-creation.md` | Constructing specialized unit-aware current ranges, grids, filled arrays, template-shaped arrays, or matrix patterns |
| `references/braintools/connectivity.md` | Selecting spatial, topological, biological E/I, kernel, or compartment-aware connection pairs and exporting a `ConnectionResult` to BrainEvent storage |
| `references/braintools/input-current.md` | Generating or composing direct unit-aware current sections, pulses, waveforms, or stochastic processes |
| `references/component-selection.md` | Choosing a neuron, synapse, synaptic output, plasticity model, input generator, or readout from the documented native API families |
| `references/projection-patterns.md` | Choosing projection alignment or API form, adding delays or short-term plasticity, using direct/delta projections, or adding gap junctions |
| `references/perturbation-experiment-validity.md` | Estimating baseline tuning or features before matched point-neuron perturbations, calibrating delivered target dose, or deciding whether activity and analysis support justify a formal causal run |
| `references/nest-compatible/nest-workflow.md` | Using NEST/PyNEST model names, `Simulator`, devices, connection rules, spatial networks, parity, porting, or the bundled NEST-compatible full scripts |
| `references/brainPy(legacy)/brainpy legacy workflow.md` | Use as the only first-level legacy entry when existing legacy `brainpy` has no BrainUnit, unit-aware, or BrainTrace code and the user explicitly requests it; this workflow routes all second-level legacy references |
| `skills/package-skills/brainevent/SKILL.md` | Selecting sparse event representations, connectivity formats, event operators, plasticity kernels, or custom kernels |
| `skills/package-skills/brainevent/references/scripts/coba_ei_teaching.py` | Learning how to incorporate efficient BrainEvent communication into a complete BrainPy COBA E/I network while preserving BrainPy dynamics and BrainState execution |


## Application script examples
| Reference | Open when |
|---|---|
| `references/scripts/103_COBA_2005.py` | Needing a complete canonical E/I COBA network built from native BrainPy-State projections |
| `references/scripts/106_COBA_HH_2007.py` | Reproducing a conductance-based E/I network with a custom Hodgkin-Huxley neuron |
| `references/scripts/107_gamma_oscillation_1996.py` | Reproducing gamma oscillations with custom neuron and synapse dynamics |
| `references/scripts/109_fast_global_oscillation.py` | Needing a complete `DeltaProj` network with delayed recurrent input |
| `references/scripts/sound_localization.py` | Needing a mapped delay-line coincidence network with heterogeneous integer event taps and BrainEvent fixed fan-out |
| `references/scripts/201_surrogate_grad_lif_fashion_mnist.py` | Needing a complete real-data surrogate-gradient LIF training workflow |
| `references/scripts/training-snn.py` | Needing a complete runnable surrogate-gradient SNN training workflow |
