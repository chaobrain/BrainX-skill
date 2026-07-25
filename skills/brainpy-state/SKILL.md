---
name: brainpy-state
description: Use for native BrainPy-State point-neuron simulations and trainable spiking networks: neuron and synapse selection, projections, synaptic outputs, short-term plasticity, delays, unitful rollouts, readouts, and surrogate-gradient training.
---

# BrainPy-State

## Purpose and boundary

Use this skill for the native `brainpy.state` modeling path: compose point neurons, synapses, communication operators, synaptic outputs, projections, inputs, plasticity, and readouts into simulations or trainable spiking neural networks.

Canonical path:

`choose dynamics -> choose projection alignment -> construct the Module graph -> initialize State -> set dt and t -> transform the rollout -> inspect or differentiate outputs`

Use the BrainState skill for general `State`, Module graphs, environments, randomness, and transforms. Use the BrainUnit skill for quantity algebra and unit conversion. Route sparse operator or kernel work to `skills/brainevent/SKILL.md`, NEST-compatible work to `NEST-compatible/nest-workflow.md`, and online-learning execution to BrainTrace. Do not invent BrainTrace APIs in this skill.

## Underlying principle of BrainPy-State

BrainPy-State is the point-neuron layer built on BrainState: every neuron and synapse is a stateful `Dynamics`, every physical value is a BrainUnit quantity, and every rollout is driven by a State-aware `brainstate.transform` loop.

Build networks by separating communication from dynamics. A projection combines `comm` (connectivity and weights), `syn` (temporal filtering), `out` (conductance/current conversion), and `post` (the receiving population). Align synaptic State to a neuron dimension instead of storing it per connection: use AlignPost for linear exponential-family fan-in and AlignPre for nonlinear kinetics or reusable fan-out.

The same Module graph supports simulation and training. Surrogate spike functions supply the backward derivative, while differentiable transform loops provide backpropagation through time over the ordinary simulation workflow.

### API structure

| API family | Use |
|---|---|
| `brainpy.state.Dynamics`, `Neuron`, `Synapse` | Implement time-evolving components with State allocation, one-step `update()`, and neuron spike output. |
| `brainpy.state` neuron models | Choose point-neuron dynamics from integrate-and-fire through conductance-based biophysical models. |
| `brainpy.state` synapses and outputs | Filter spikes with synaptic dynamics, then convert the result to postsynaptic current. |
| `brainpy.state` projections | Connect populations with AlignPre, AlignPost, direct-current, delta, or electrical coupling. |
| `brainpy.state.STP`, `STD` | Apply short-term facilitation/depression directly or inside functional projection builders. |
| `brainpy.state` inputs and readouts | Generate spikes/current and decode spike or rate activity. |
| `brainstate.nn` | Build the containing Module graph, communication layers, and trainable linear layers. |
| `brainstate.environ` | Supply `dt`, `t`, and other run-scoped values read by dynamics. |
| `brainstate.transform` | Compile, loop, batch, checkpoint, and differentiate the complete stateful operation. |
| `brainunit` | Attach units to model values and convert outputs explicitly at external boundaries. |
| `braintools` | Initialize values, choose surrogate gradients, optimize parameters, compute losses, and visualize results. |

### 1. Choose and run neuron dynamics

A neuron call advances one `dt`, mutates its registered dynamical State, and exposes the current spike through `get_spike()`; initialize before every independent rollout and lower the complete time loop through `brainstate.transform`.

| API | Description |
|---|---|
| `brainpy.state.IF`, `LIF`, `LIFRef` | Use for minimal integrate-and-fire dynamics; choose `LIF` for leak and `LIFRef` when an explicit refractory period matters. Each returns the current step's spike when called and exposes it through `get_spike()`. |
| `ALIF`, `ExpIF`, `ExpIFRef`, `AdExIF`, `AdExIFRef`, `QuaIF`, `AdQuaIF`, `AdQuaIFRef`, `Gif`, `GifRef` | Use when adaptation, nonlinear spike onset, or generalized integrate-and-fire behavior is required; refractory-suffixed variants add refractory State. |
| `Izhikevich`, `IzhikevichRef` | Use for rich phenomenological firing patterns at lower cost than conductance-based channel models; use the refractory variant when that lifecycle is required. |
| `HH`, `MorrisLecar`, `WangBuzsakiHH` | Use when membrane conductances and biophysical spike generation are required; expect greater numerical cost and smaller `dt` than integrate-and-fire models. |
| `brainstate.nn.init_all_states(model, batch_size=...)` | Use after construction and before a rollout; it allocates or resets dynamical State across the Module graph and adds a leading batch dimension when `batch_size` is given. |
| `brainstate.environ.context(dt=..., t=...)` | Use around construction/rollout for `dt` and inside the step for `t`; nested contexts restore the previous values on exit. |
| `brainstate.transform.jit(fn)` | Use for one complete repeated operation or compiled train step; it compiles compatible calls but does not remove Python dispatch from an external time loop. |
| `brainstate.transform.for_loop(step, *xs)` | Use by default for a multi-step rollout whose model State carries hidden variables; it slices leading input axes and stacks returned monitors. |
| `brainstate.transform.scan(step, carry, xs)` | Use when the rollout also requires explicit non-Module carry with body `f(carry, x) -> (carry, y)`. |
| `checkpointed_for_loop`, `checkpointed_scan` | Use only for long reverse-mode-differentiated rollouts; they preserve loop semantics while trading recomputation for lower activation memory. |
| `brainstate.transform.vmap(fn, ...)` | Use when a single-sample computation or State graph must be mapped or shared explicitly; prefer `batch_size=` initialization when the built-in dynamics already operate on batched leading axes. |
| `Quantity.to_decimal(unit)` | Use only at plotting, serialization, or non-unit-aware library boundaries; it returns plain values expressed in the requested unit. |

```python
import brainpy
import brainstate
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

Open `references/component-selection.md` when choosing among all documented neuron, input, synapse, output, plasticity, or readout variants, for the decision boundary and complete category list.

### 2. Compose synapses and projections

A projection deposits current into `post` before the postsynaptic neuron integrates the step, while its alignment determines which neuron dimension owns synaptic State and which kinetics remain exact.

| API | Description |
|---|---|
| `brainpy.state.Expon`, `DualExpon`, `Alpha` | Use for simple linear synaptic filters; `Expon` is the canonical decay, `DualExpon` adds distinct rise/decay behavior, and `Alpha` gives an alpha-shaped response. |
| `AMPA`, `GABAa`, `BioNMDA` | Use for nonlinear receptor kinetics; choose AlignPre because AlignPost is exact only for exponential-family linear dynamics. |
| `brainpy.state.SynOut` | Subclass only when `COBA`, `CUBA`, and `MgBlock` cannot express the required conductance-to-current rule; bind the computed current through the postsynaptic input contract. |
| `COBA.desc(E=...)` | Use for conductance-based current `g * (E - V)`; it depends on postsynaptic voltage and saturates near the reversal potential. |
| `CUBA.desc(scale=...)` | Use for voltage-independent current injection; choose it for simpler analysis or brain-inspired models and preserve compatible weight/output units. |
| `MgBlock.desc(...)` | Use when the synaptic output requires voltage-dependent magnesium block, typically with NMDA-like conductance. |
| `brainpy.state.Projection` | Subclass only for a projection lifecycle not covered by the concrete chemical, direct-input, or gap-junction APIs; preserve registered child Modules and the postsynaptic input contract. |
| `brainpy.state.AlignPostProj(...)` | Use for the common exponential-family fan-in path; it stores State on postsynaptic neurons, accepts direct spike input, and can also pull from supplied prefetch/spike-generator arguments. Pass `delay=` for scalar or per-presynaptic axonal delay on direct input. |
| `brainpy.state.align_post_projection(...)` | Use the functional AlignPost builder when the projection should own presynaptic spike generation, delay, or an `stp=` describer; it returns the constructed projection. |
| `brainpy.state.align_pre_projection(...)` | Use for nonlinear synapses or one source reused by many targets; it is a function, not a class, and stores one synaptic trace per presynaptic neuron when parameters are homogeneous per source. |
| `brainpy.state.CurrentProj(...)` | Use when a continuous presynaptic value should pass through `comm` and `out` without separate synapse dynamics; it deposits the resulting current into `post` and accepts the same direct-input `delay=` form. |
| `brainpy.state.DeltaProj(...)` | Use for instantaneous delta input without temporal synapse State; it deposits the communicated event directly into the target. |
| `SymmetryGapJunction`, `AsymmetryGapJunction` | Use for electrical coupling; choose the symmetric or directed form according to whether coupling is reciprocal. |
| `brainpy.state.STP`, `STD` | Use directly to inspect short-term plasticity or pass `.desc(...)` through `stp=` in a functional projection builder; `STP` tracks facilitation and resources, while `STD` tracks depression only. |
| `Component.desc(...)` | Use when a parent projection must infer or own a component's size/lifecycle; paired `syn`/`out` descriptors also let compatible AlignPost projections share postsynaptic instances, while paired concrete instances remain unmerged. |
| `delay=` / `pre.prefetch("V").delay.at(delay)` | Pass `delay=` to `AlignPostProj` or `CurrentProj` for direct scalar or `(N_pre,)` input delay; use delayed prefetch when a pull-based projection must delay stored voltage and re-derive spikes with `pre.get_spike(v)`. Keep `dt` fixed after delay-buffer initialization. |
| `brainstate.nn.EventFixedProb` | Use for the canonical sparse binary-spike connection; route broader event connectivity and kernel selection to the BrainEvent skill. |

```python
import brainpy
import brainstate
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
        )
        self.post = brainpy.state.LIFRef(
            n_post,
            tau=20.0 * u.ms,
            tau_ref=5.0 * u.ms,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
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

Open `references/projection-patterns.md` when selecting AlignPre versus either AlignPost API, adding short-term plasticity or delays, using direct/delta input, or composing electrical coupling.

### 3. Train a spiking network

Training reuses the simulation graph: place a surrogate on each trained spiking nonlinearity, unroll with a differentiable transform loop, differentiate only `ParamState`, reset dynamical State for each independent batch, and update inside a compiled train step.

| API | Description |
|---|---|
| `neuron(..., spk_fun=surrogate)` | Use on spiking layers traversed by gradients; it keeps hard spikes in the forward pass and uses the surrogate derivative in the backward pass. |
| `braintools.surrogate.ReluGrad()` | Use as the robust canonical surrogate; select another class or its lowercase functional counterpart only when gradient shape changes optimization behavior. |
| `model.states(brainstate.ParamState)` | Use to select trainable parameters; it excludes membrane voltage, conductance, spike history, and other rollout State. |
| `brainstate.transform.grad(loss_fn, params, return_value=True)` | Use to differentiate the stateful loss with respect to the selected State collection; calling the returned transform yields gradients and the loss. |
| `braintools.optim.Adam(...).register_trainable_weights(params)` | Use for the canonical optimizer lifecycle; register the same State collection used by `grad`, then call `optimizer.update(grads)`. |
| `brainpy.state.LeakyRateReadout` | Use when spikes require a trainable, low-pass continuous decoder; use a linear layer plus `Expon` when that explicit composition is preferable. |
| `checkpointed_for_loop(..., base=...)`, `checkpointed_scan(...)` | Use only when long-rollout BPTT exhausts memory; tune `base` to trade stored checkpoints against recomputation. |

```python
import brainpy
import brainstate
import braintools
import brainunit as u


class SNN(brainstate.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.input = brainstate.nn.Sequential(
            brainstate.nn.Linear(
                n_in,
                n_hidden,
                w_init=braintools.init.KaimingNormal(unit=u.mA),
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            brainpy.state.Expon(
                n_hidden,
                tau=5.0 * u.ms,
                g_initializer=braintools.init.Constant(0.0 * u.mA),
            ),
        )
        self.hidden = brainpy.state.LIF(
            n_hidden,
            tau=20.0 * u.ms,
            V_rest=0.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_th=1.0 * u.mV,
            spk_fun=braintools.surrogate.ReluGrad(),
        )
        self.readout = brainpy.state.LeakyRateReadout(
            n_hidden,
            n_out,
            tau=10.0 * u.ms,
        )

    def update(self, spike):
        return self.readout(self.hidden(self.input(spike)))


with brainstate.environ.context(dt=1.0 * u.ms):
    num_steps, batch_size, num_inputs = 100, 128, 100
    net = SNN(num_inputs, 8, 2)
    inputs = (
        brainstate.random.rand(num_steps, batch_size, num_inputs)
        < 5.0 * u.Hz * brainstate.environ.get_dt()
    ).astype(float)
    labels = u.math.asarray(
        brainstate.random.rand(batch_size) < 0.5,
        dtype=int,
    )
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

Open `references/training-variations.md` when choosing a surrogate API, readout form, batched State layout, BPTT loop, or checkpointing policy. Use `references/brainstate-dynamics/scripts/training-snn.py` for a complete runnable training script and `references/braintools-optimizer.md` when optimizer or scheduler selection goes beyond Adam.

## Reference routing

Open only the smallest reference that owns the decision.

| Reference | Open when |
|---|---|
| `references/component-selection.md` | Choosing a neuron, synapse, synaptic output, plasticity model, input generator, or readout from the documented native API families |
| `references/projection-patterns.md` | Choosing projection alignment or API form, adding delays or short-term plasticity, using direct/delta projections, or adding gap junctions |
| `references/training-variations.md` | Selecting surrogate class/function forms, readout structure, BPTT loops, batching, or checkpoint granularity |
| `references/brainstate-dynamics/scripts/training-snn.py` | Needing a complete runnable surrogate-gradient SNN training workflow |
| `references/braintools-optimizer.md` | Selecting an optimizer, learning-rate scheduler, Optax bridge, SciPy optimizer, or Nevergrad optimizer |
| `NEST-compatible/nest-workflow.md` | Using NEST/PyNEST model names, `Simulator`, devices, connection rules, spatial networks, parity, or porting |
| `skills/brainevent/SKILL.md` | Selecting sparse event representations, connectivity formats, event operators, plasticity kernels, or custom kernels |

## Boundaries and common failures

- Do not time-step with a bare Python `for` or `while`; use `for_loop` by default, `scan` for explicit carry, and checkpointed variants only for long BPTT.
- Do not call a freshly constructed model before `brainstate.nn.init_all_states()`; call it again to reset an independent rollout and pass `batch_size` when the run is batched.
- Do not omit `dt` or use a stale `t`; scope `dt` around the run and set `t` inside the per-step function when dynamics, inputs, or delays read time.
- Do not pass unitless membrane, time, current, or conductance values; attach BrainUnit units and convert to plain arrays only at explicit boundaries.
- Do not update the postsynaptic neuron before its projections; projections must deposit their current before `post(...)` integrates the step.
- Do not use AlignPost for nonlinear receptor kinetics; use AlignPre for `AMPA`, `GABAa`, or `BioNMDA`.
- Do not look for an AlignPre class; use `brainpy.state.align_pre_projection`.
- Do not assume `AlignPostProj` and `align_post_projection` are interchangeable syntax; use the class for the direct canonical path and the functional builder when spike generation, delay, or `stp=` is part of construction.
- Do not differentiate all State; select `model.states(brainstate.ParamState)` and leave dynamical State to the rollout.
- Do not put an epoch loop into the time-step transform; a Python optimization loop may repeatedly call the compiled `train_step`, while model time remains inside a transform loop.
- Do not implement online learning with guessed BrainPy APIs; use BrainTrace.
- Do not mix the native composition path with NEST-compatible `Simulator` APIs; open the NEST workflow and stay within that model family.
