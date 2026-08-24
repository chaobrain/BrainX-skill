# Coupling and network API

Use this reference when configuring structural connectivity, delays, coupling families, self-connections, network-level noise, trainable coupling parameters, or direct coupling objects. Use `brainmass.Network` unless custom wiring or an independent drive on the coupled input channel requires direct composition.

## Construct the network

`Network` reads one `N`-region node bank, computes its coupling current, and passes that current as the node's first positional input on every update.

| API | Description |
|---|---|
| `brainmass.Network(node, *, conn, distance=None, speed=None, coupling="diffusive", coupled_var, k=1.0, delay_init=..., self_connection=False, noise=None)` | Use for the high-level whole-brain path; it validates `coupled_var`, constructs delays and coupling, and exposes the wrapped dynamics as `.node`. |
| `Network.init_state(...)` | Use indirectly through `Simulator`; it initializes the node, coupling, delay history, and optional network noise. |
| `Network.update(*node_inputs)` | Use for one custom step; it inserts coupling plus optional noise as the node's first positional input, forwards `node_inputs` after it, and returns the node update result. |
| `brainmass.Simulator(network, dt=...)` | Use for the canonical execution path; monitor wrapped State through `network.node`. |

```python
import brainmass
import brainstate
import brainunit as u

dt = 0.1 * u.ms
brainstate.environ.set(dt=dt)

connectome = brainmass.datasets.load_dataset("example_connectome")
n_region = connectome.weights.shape[0]
network = brainmass.Network(
    brainmass.HopfStep(in_size=n_region, a=0.2, w=0.3),
    conn=connectome.weights,
    distance=connectome.distances,
    speed=10.0 * u.mm / u.ms,
    coupling="diffusive",
    coupled_var="x",
    k=0.5,
)
result = brainmass.Simulator(network, dt=dt).run(
    400.0 * u.ms,
    monitors=lambda model: model.node.x.value,
    transient=100.0 * u.ms,
)

assert result["output"].shape == (3000, n_region)
```

**Invariant:** set global `dt` before constructing a delayed network and use the same value in `Simulator`. The delay buffer is sized at construction.

### Route coupling and external inputs

Choose the path from the node's positional input contract.

| Need | Use |
|---|---|
| Coupling alone on the first node input | `Network`; do not pass another input. |
| Coupling on the first input plus an independent drive on the second or later input | `Network`; pass the drive through `Simulator.run(inputs=...)`, which `Network.update()` forwards after coupling. |
| Coupling plus an independent drive summed on the same first input | Direct composition; retrieve the delayed source, call the functional coupling kernel, add the drive, and pass the sum to the node. |

Do not expect a caller-supplied first value to be added to `Network` coupling. It becomes the node's second positional input.

## Preserve connectivity semantics

Use the convention `W[i, j] = source j -> target i`. Each row therefore contains the incoming weights for one target.

| Input | Behavior |
|---|---|
| Plain `(N, N)` connectivity array | `Network` zeroes the diagonal unless `self_connection=True`. |
| Directed array | Preserved; symmetry is not required. |
| `Param` or `LaplacianConnParam` connectivity | Passed through without diagonal rewriting; the caller owns its structure. |
| Missing `distance` or `speed` | Coupling is instantaneous with zero delays. |
| Unit-aware `distance` and `speed` | `distance / speed` produces unit-correct conduction delays. |
| Plain `distance` and `speed` | Their quotient is interpreted as milliseconds by the documented convention. |

Use structural connectivity for wiring. Functional connectivity is an output statistic, not a substitute for anatomical weights.

## Choose a coupling

Choose where the interaction should be relative, absolute, or saturated.

| API | Description |
|---|---|
| `Network(..., coupling="diffusive")` / `DiffusiveCoupling(...)` | Use for `k * sum_j W[i,j] * (source_j - target_i)`; it vanishes at consensus and is the canonical synchronization path. |
| `Network(..., coupling="additive")` / `AdditiveCoupling(...)` | Use for direct weighted input `k * sum_j W[i,j] * source_j + b`; it does not vanish when nodes agree. |
| `Network(..., coupling="laplacian")` / `LaplacianConnParam(...)` | Use for additive application of a graph Laplacian, including normalized variants. |
| `Network(..., coupling="tanh")` / `HyperbolicTangentCoupling(...)` | Use for symmetric saturation of the summed input toward `+/-k`. |
| `Network(..., coupling="sigmoidal")` / `SigmoidalCoupling(...)` | Use for a bounded one-sided post-sum firing-rate transfer with configurable slope and midpoint. |
| `Network(..., coupling="sigmoidal_jansen_rit")` / `SigmoidalJansenRitCoupling(...)` | Use for the Jansen-Rit pre-synaptic transfer, where each source passes through the sigmoid before weighting and summation. |

The functional forms expose stateless kernels for custom composition:

| API | Description |
|---|---|
| `brainmass.diffusive_coupling(source, target, conn, k)` | Compute relative diffusive input from delayed source reads and each target's current value. |
| `brainmass.additive_coupling(source, conn, k, b=...)` | Compute a direct weighted source sum and optional bias. |
| `brainmass.sigmoidal_coupling(...)` | Apply the logistic nonlinearity after the network sum. |
| `brainmass.hyperbolic_tangent_coupling(...)` | Apply symmetric tanh saturation after the network sum. |
| `brainmass.sigmoidal_jansen_rit_coupling(...)` | Apply the Jansen-Rit sigmoid to each source before the weighted sum. |
| `brainmass.laplacian_connectivity(conn, normalize=...)` | Build unnormalized, symmetric-normalized, or random-walk graph Laplacians for explicit Laplacian workflows. |

Use the `Network` string selector for standard defaults. Construct a coupling class directly only when its shape parameters, prefetched sources, or trainable parameters need explicit control.

## Configure delays and coupling parameters

`distance / speed` defines each edge delay. `Network` zeroes self-delay and stores enough history for the largest discretized delay.

| Parameter | Decision |
|---|---|
| `coupled_var` | Name the node State that supplies the delayed source and receives the coupling path; an unknown State raises `ValueError`. |
| `k` | Set the global coupling strength; pass `Param(..., fit=True)` only when it belongs in the fitting target. |
| `delay_init` | Set the history initializer when startup dynamics depend on pre-run delayed State. |
| `self_connection` | Keep `False` unless the structural matrix intentionally contains coupling-level recurrent edges distinct from local node dynamics. |
| `noise` | Add network-level noise to the coupling current; keep population-specific noise on the node. |

Start with normalized or otherwise scientifically calibrated weights and a modest `k`. Changing weight normalization changes the meaning of `k`.

## Compose custom source and target wiring

Use direct construction only when a built-in `Network` cannot represent the source/target wiring. A diffusive coupling requires one delayed `(target, source)` read and one current target read.

```python
import jax.numpy as jnp
import braintools

n_region = connectome.weights.shape[0]
nodes = brainmass.HopfStep(in_size=n_region, a=0.2, w=0.3)
delays = jnp.ones((n_region, n_region)) * (1.0 * u.ms)
indices = brainmass.delay_index(n_region)

source = nodes.prefetch_delay(
    "x",
    (delays, indices),
    init=braintools.init.Constant(0.0),
)
target = nodes.prefetch("x")
coupling = brainmass.DiffusiveCoupling(
    source,
    target,
    conn=connectome.weights,
    k=0.5,
)
```

Do not replace the delay prefetch protocol with a manually shifted Python container.

## Vary delay with a same-channel drive

Use a fixed-capacity `brainstate.nn.Delay` when mapped conditions vary retrieval delay while State shape must remain static; construct it under the final `dt`, insert the current source once per step, then retrieve the declared integer offset.

| API | Description |
|---|---|
| `brainstate.nn.Delay(target_info, time=max_delay, init=...)` | Use for package-owned short-term history with capacity fixed by `max_delay`; construct it while the rollout's `dt` is active and initialize the intended prehistory explicitly. |
| `Delay.update(current)` | Use exactly once at the start of each model step to insert the source State at the current simulation time. |
| `Delay.retrieve_at_step(delay_steps)` | Use after insertion with a scalar or traced integer offset no greater than the fixed capacity; step 0 is current and step `d` is exactly `d` completed updates earlier. |
| `brainmass.<coupling>_coupling(...)` | Use the functional kernel to compute coupling before adding an independent same-channel drive. |

Initialize delay history from the intended pre-run source State, not from an
arbitrary numeric zero. For diffusive coupling, startup history should reproduce
the source baseline that the current target sees; a homogeneous equilibrium then
produces zero startup current. If no defensible prehistory exists, warm up the
same model and coupling path and exclude the warm-up from classification. Verify
the first retrieved value and resulting coupling current before interpreting
timing.

```python
import brainmass
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

dt = 0.1 * u.ms
max_delay = 12.0 * u.ms
conn = jnp.asarray([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


class DrivenChain(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.node = brainmass.FitzHughNagumoStep(
            in_size=3,
            init_V=braintools.init.Constant(0.0),
            init_w=braintools.init.Constant(0.0),
        )
        self.history = brainstate.nn.Delay(
            jnp.zeros(3),
            time=max_delay,
            init=braintools.init.Constant(0.0),
        )

    def update(self, stimulus, k, delay):
        self.history.update(self.node.V.value)
        delay_steps = jnp.rint(delay / dt).astype(jnp.int32)
        delayed = self.history.retrieve_at_step(delay_steps)
        sources = jnp.broadcast_to(delayed, conn.shape)
        current = brainmass.additive_coupling(sources, conn, k)
        return self.node(current + stimulus)


def run_condition(k, delay):
    with brainstate.environ.context(dt=dt):
        model = DrivenChain()
        brainstate.nn.init_all_states(model)
        indices = jnp.arange(200)

        def step(i):
            stimulus = jnp.where(
                i < 20,
                jnp.asarray([0.5, 0.0, 0.0]),
                jnp.zeros(3),
            )
            with brainstate.environ.context(i=i, t=i * dt):
                return model(stimulus, k, delay)

        return brainstate.transform.for_loop(step, indices)


traces = brainstate.transform.vmap(run_condition)(
    jnp.asarray([0.2, 0.4]),
    jnp.asarray([4.0, 8.0]) * u.ms,
)
assert traces.shape == (2, 200, 3)
```

Keep `max_delay` static. Constructing `Network` or `Delay` with a mapped delay can make buffer capacity depend on a tracer and fail during shape allocation. Map the complete independent condition, vary only the integer retrieval step, and preserve the same `dt` across construction, initialization, and execution.

Lock the phase convention with an impulse check before interpreting neural output:

```python
with brainstate.environ.context(dt=1.0 * u.ms):
    delay = brainstate.nn.Delay(jnp.zeros(()), time=3.0 * u.ms)
    brainstate.nn.init_all_states(delay)

    def delayed_impulse(value):
        delay.update(value)
        return delay.retrieve_at_step(jnp.asarray(3, dtype=jnp.int32))

    observed = brainstate.transform.for_loop(
        delayed_impulse,
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
    )

assert jnp.array_equal(observed, jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0]))
```

If retrieval occurs before insertion, the buffer still ends at the previous simulation time and the same integer offset shifts the physical delay by one `dt`. Do not compensate with an unexplained `delay_steps - 1`; keep one call order and verify it directly.

## Application script

Open `scripts/resting-state-meg-whole-brain-pipeline.py` when the task needs the complete connectome-to-network-to-MEG-to-functional-connectivity workflow.

## Diagnose network failures

- Exploding activity: lower `k`, verify weight normalization and units, or select an appropriate saturating coupling.
- No collective effect: inspect isolated rows, directionality, `coupled_var`, simulation duration, and `k`.
- Shape error in direct diffusive coupling: provide an `(N, N)` source read and an `(N,)` target read.
- Unexpected self-drive: inspect whether the input is a plain array, a `Param`, or a Laplacian wrapper and confirm ownership of diagonal handling.
- Missing or wrong delays: confirm global `dt`, distance units, speed units, and use of the same `dt` during simulation.
- Missing monitor State: traverse `lambda model: model.node.<state>.value`.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/coupling.html`
- `https://brainx.chaobrain.com/brainmass/tutorials/04_building_a_network.html`
- `https://brainx.chaobrain.com/brainmass/concepts/coupling_and_delays.html`
- `https://brainx.chaobrain.com/brainmass/howto/custom_coupling.html`
