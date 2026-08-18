# Customize legacy BrainPy neurons and synapses

Use this reference when a legacy BrainPy 2.x model needs custom neuron equations, spike/reset logic, synapse dynamics, synaptic output, communication, or projection composition. Use `built-in dynamic neuron model.md` first when a supplied `brainpy.dyn` model already matches the required mechanism.

## Choose the extension point

A custom neuron owns state evolution and event transitions; a custom projection composes presynaptic events, optional delay, communication, synapse dynamics, output current, and postsynaptic input binding.

| Need | Extend or compose |
|---|---|
| New neuron equations or spike/reset rule | `bp.dyn.NeuDyn` |
| New synaptic state equations | `bp.dyn.SynDyn` |
| New conductance-to-current rule | `bp.dyn.SynOut` |
| New weighted connectivity operation | `bp.dnn.Layer` or another `DynamicalSystem` used as `comm` |
| State aligned to postsynaptic neurons | `bp.dyn.ProjAlignPostMg2` |
| State aligned to presynaptic neurons | `bp.dyn.ProjAlignPreMg2` |
| Reusable projection wrapper | `bp.Projection` |

Keep these mechanisms separate. A projection chooses their order and ownership; it should not duplicate neuron or synapse differential equations.

## Implement a custom neuron

`NeuDyn` supplies population shape and model lifecycle, while the subclass declares every evolving value as `bm.Variable` and implements one time step in `update()`.

| API | Description |
|---|---|
| `bp.dyn.NeuDyn(size, keep_size=False, method='exp_auto', ...)` | Initialize population metadata and common neuronal-dynamics behavior. |
| `bm.Variable(initial_value)` | Register voltage, gating, spike, refractory, and last-event state for transformed mutation. |
| `bp.odeint(f, method=...)` | Build an integrator for continuous state equations. |
| `bp.share['t']` | Read the current simulation time supplied by the runner or `step_run`. |
| `bp.share['dt']` | Read the active integration step. |
| `bm.where(condition, x, y)` | Apply discontinuous reset or refractory transitions without Python branching on traced arrays. |
| `update(input=None)` | Advance exactly one step, update Variables, and return the step output if callers need it. |

```python
import brainpy as bp
import brainpy.math as bm


class LegacyLIF(bp.dyn.NeuDyn):
    def __init__(
        self,
        size,
        V_rest=0.0,
        V_reset=-5.0,
        V_th=20.0,
        R=1.0,
        tau=10.0,
        tau_ref=5.0,
        **kwargs,
    ):
        super().__init__(size=size, **kwargs)
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.tau_ref = tau_ref

        self.V = bm.Variable(bm.ones(self.num) * V_reset)
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        self.integral = bp.odeint(self.dV, method='exp_auto')

    def dV(self, V, t, current):
        return (-V + self.V_rest + self.R * current) / self.tau

    def update(self, current=None):
        t = bp.share['t']
        dt = bp.share['dt']
        current = 0.0 if current is None else current

        refractory = (t - self.t_last_spike) <= self.tau_ref
        next_V = self.integral(self.V, t, current, dt=dt)
        next_V = bm.where(refractory, self.V, next_V)
        spike = next_V >= self.V_th

        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, next_V)
        self.refractory.value = bm.logical_or(refractory, spike)
        return self.spike.value
```

This example separates continuous integration from the discontinuous threshold/reset transition. For a continuous Hodgkin-Huxley-style model, create one integrator per equation or a joint integrator, update gating variables from the old state consistently, detect threshold crossings, then commit the new Variables.

**Invariant:** Compute the next state before assigning any dependent `Variable.value` unless the equations intentionally use a sequential update. Premature assignment silently changes which time level later equations read.

### Run the custom neuron

```python
neurons = LegacyLIF(10)
runner = bp.DSRunner(neurons, monitors=['V', 'spike'])
runner.run(inputs=bm.ones(1000) * 22.0)

assert runner.mon['V'].shape == (1000, 10)
assert runner.mon['spike'].shape == (1000, 10)
```

Open `integrators.md` when choosing a solver, implementing coupled ODEs/SDEs, passing delays to integrators, or validating equation argument order.

## Decompose a custom chemical synapse

A legacy chemical projection is a chain: presynaptic event -> optional delay -> communication -> synapse state -> output current -> postsynaptic input.

| Component | Role | Typical base/API |
|---|---|---|
| `pre` | Produces spike or graded presynaptic activity. | `NeuDyn`, `SpikeTimeGroup`, or another `DynamicalSystem` |
| `delay` | Selects an earlier presynaptic event. | Projection `delay=` and pre delay support |
| `comm` | Maps pre-aligned activity to the post population with weights and connectivity. | `EventCSRLinear`, `CSRLinear`, `AllToAll`, `OneToOne`, or custom `bp.dnn.Layer` |
| `syn` | Evolves conductance, gating, or transmitter state. | `bp.dyn.SynDyn` descriptor |
| `out` | Converts synaptic state to current. | `bp.dyn.SynOut` descriptor or instance, according to the projection signature |
| `post` | Receives the bound current before its own update. | Postsynaptic neuron group |

Use `.desc(...)` for `syn` in both merging projections. `ProjAlignPostMg2` also takes an `out` descriptor; `ProjAlignPreMg2` takes a concrete `SynOut` instance because it binds condition data directly.

## Implement postsynaptic-aligned exponential conductance

The tutorial's postsynaptic-aligned path is for exponential synapse dynamics whose incoming events can be communicated first and accumulated into one conductance state per postsynaptic neuron.

| API | Description |
|---|---|
| `bp.dyn.SynDyn` | Base class for evolving synapse state. |
| `bp.mixin.AlignPost` | Mark dynamics that support postsynaptic alignment and accumulated current. |
| `add_current(x)` | Add communicated event increments to the postsynaptic-aligned state. |
| `return_info()` | Expose the state information required by projection merging. |
| `bp.dyn.ProjAlignPostMg2(...)` | Compose `delay -> comm -> postsynaptic synapse -> output -> post`. |

```python
class ExponentialConductance(bp.dyn.SynDyn, bp.mixin.AlignPost):
    def __init__(self, size, tau=5.0):
        super().__init__(size=size)
        self.tau = tau
        self.g = bm.Variable(bm.zeros(self.num))
        self.integral = bp.odeint(
            lambda g, t: -g / self.tau,
            method='exp_auto',
        )

    def update(self, increment=None):
        self.g.value = self.integral(
            self.g,
            bp.share['t'],
            dt=bp.share['dt'],
        )
        if increment is not None:
            self.add_current(increment)
        return self.g.value

    def add_current(self, increment):
        self.g += increment

    def return_info(self):
        return self.g


class COBAOutput(bp.dyn.SynOut):
    def __init__(self, E=0.0):
        super().__init__()
        self.E = E

    def update(self, conductance, potential):
        return conductance * (self.E - potential)


class ExponentialCOBA(bp.Projection):
    def __init__(
        self,
        pre,
        post,
        prob,
        g_max,
        tau=5.0,
        E=0.0,
        delay=None,
        seed=None,
    ):
        super().__init__()
        connector = bp.conn.FixedProb(
            prob,
            pre=pre.num,
            post=post.num,
            seed=seed,
        )
        self.proj = bp.dyn.ProjAlignPostMg2(
            pre=pre,
            delay=delay,
            comm=bp.dnn.EventCSRLinear(connector, weight=g_max),
            syn=ExponentialConductance.desc(post.num, tau=tau),
            out=COBAOutput.desc(E=E),
            post=post,
        )
```

The communication result is postsynaptic-sized, so `ExponentialConductance.desc(post.num, ...)` must use the postsynaptic population size.

## Use presynaptic alignment for event-history dynamics

Use `ProjAlignPreMg2` when each presynaptic neuron owns kinetic state that must update before communication, as in the tutorial's AMPA model with spike-arrival time.

| API | Description |
|---|---|
| `bp.dyn.ProjAlignPreMg2(pre, syn, delay, comm, out, post, ...)` | Compose `presynaptic synapse -> delay -> communication -> output -> post`. |
| `AMPA.desc(pre.num, ...)` | Allocate one custom AMPA state per presynaptic neuron. |
| `bp.dnn.CSRLinear(connector, weight=...)` | Transform graded pre-aligned synapse state into postsynaptic conductance. |
| `MgBlock(...)` | Pass a concrete custom output instance for postsynaptic condition binding. |

```python
# Given a custom AMPA SynDyn and MgBlock SynOut:
self.proj = bp.dyn.ProjAlignPreMg2(
    pre=pre,
    syn=AMPA.desc(pre.num, alpha=0.98, beta=0.18, T=0.5, T_dur=0.5),
    delay=delay,
    comm=bp.dnn.CSRLinear(connector, weight=g_max),
    out=MgBlock(E=0.0),
    post=post,
)
```

The tutorial restricts postsynaptic alignment to exponential synapse dynamics. Do not move AMPA-like event-history state after communication merely to reduce its size; doing so changes the represented mechanism.

## Execute projection before the postsynaptic neuron

Projection output must be bound before the postsynaptic group sums inputs and integrates its current step.

```python
class TwoNeuronNetwork(bp.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.pre = bp.dyn.SpikeTimeGroup(
            1,
            indices=(0, 0, 0, 0),
            times=(10.0, 30.0, 50.0, 70.0),
        )
        self.post = bp.dyn.LifRef(
            1,
            V_rest=-60.0,
            V_reset=-60.0,
            V_th=-50.0,
            tau=20.0,
            tau_ref=5.0,
            V_initializer=bp.init.Constant(-60.0),
        )
        self.syn = ExponentialCOBA(
            self.pre,
            self.post,
            prob=1.0,
            g_max=0.1,
            E=0.0,
            seed=123,
        )

    def update(self):
        self.pre()
        self.syn()
        self.post()
        return self.post.V.value


network = TwoNeuronNetwork()
runner = bp.DSRunner(network, monitors=['post.V', 'post.spike'])
runner.run(100.0)
```

**Invariant:** Call `pre`, then the chemical projection, then `post` for each step. Calling `post` first delays the new synaptic current by one integration step.

## Common failures

- Do not store evolving voltage, gating, conductance, spike, or event-time state as an ordinary array.
- Do not use Python `if` on traced per-neuron spike conditions; use `bm.where` or boolean array operations.
- Do not update one coupled state before computing derivatives that require all old states.
- Do not confuse postsynaptic alignment with presynaptic alignment; descriptor size and operation order must agree.
- Do not guess descriptor ownership: both merging projections take a `syn` descriptor, `ProjAlignPostMg2` takes an `out` descriptor, and `ProjAlignPreMg2` takes a concrete `out` instance.
- Do not call the postsynaptic neuron before its chemical projection.
- Do not embed a dense communication matrix in custom synapse dynamics; keep connectivity and kinetics independently selectable.

## Routing

Open `integrators.md` for numerical methods and coupled equation signatures. Open `connecting neurons.md` for connector representations and JIT connectivity. Open `synaptic projections.md` for built-in projection families. Open `synpase properties.md` for built-in synapse dynamics and output choices. Open `infrastructure/delays.md` for delayed presynaptic events.

## Sources mirrored

- https://brainpy.readthedocs.io/tutorial_building/customize_neuron_models.html
- https://brainpy.readthedocs.io/tutorial_building/how_to_customze_a_synapse.html
