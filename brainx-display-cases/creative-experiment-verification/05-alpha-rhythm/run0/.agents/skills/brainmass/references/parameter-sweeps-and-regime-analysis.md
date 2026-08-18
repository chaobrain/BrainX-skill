# Parameter sweeps and regime analysis

Use this reference when mapping regimes, sensitivities, or identifiability over a fixed parameter grid. Use `Fitter` instead when the task is to minimize one target objective rather than characterize the landscape.

## Choose sweep or fit

| Goal | Use |
|---|---|
| Visualize a bifurcation, regime boundary, sensitivity, or multiple equivalent solutions | Parameter sweep |
| Recover parameters that minimize one differentiable target | Gradient `Fitter` |
| Optimize a non-differentiable target without exhaustively mapping the grid | Gradient-free `Fitter` |

Return one scalar or fixed-shape scientific summary per grid point. Avoid retaining full trajectories unless they are required downstream.

## Run a one-dimensional sweep

Construct the model inside the mapped function so BrainState can trace one independent simulation per parameter value.

```python
import brainmass
import brainstate
import brainunit as u
import braintools
import jax.numpy as jnp

def amplitude_for(a):
    node = brainmass.HopfStep(
        in_size=1,
        a=a,
        w=0.3,
        init_x=braintools.init.Constant(0.5),
    )
    result = brainmass.Simulator(node, dt=0.1 * u.ms).run(
        150.0 * u.ms,
        monitors=["x"],
        transient=50.0 * u.ms,
    )
    x = u.get_magnitude(result["x"])[:, 0]
    return jnp.sqrt(jnp.mean(x**2)) * jnp.sqrt(2.0)

a_values = jnp.linspace(0.0, 1.5, 16)
amplitudes = brainstate.transform.vmap(amplitude_for)(a_values)

assert amplitudes.shape == a_values.shape
```

The mapped function must return the same PyTree structure and leaf shapes for every point.

## Run a two-dimensional grid

Flatten coordinate grids, map all coordinate arrays together, then reshape the fixed-shape result back to grid order.

```python
a_grid = jnp.linspace(0.1, 1.5, 6)
w_grid = jnp.linspace(0.1, 0.6, 5)
aa, ww = jnp.meshgrid(a_grid, w_grid, indexing="ij")

def amplitude_for_pair(a, w):
    node = brainmass.HopfStep(
        in_size=1,
        a=a,
        w=w,
        init_x=braintools.init.Constant(0.5),
    )
    result = brainmass.Simulator(node, dt=0.1 * u.ms).run(
        150.0 * u.ms,
        monitors=["x"],
        transient=50.0 * u.ms,
    )
    x = u.get_magnitude(result["x"])[:, 0]
    return jnp.sqrt(jnp.mean(x**2)) * jnp.sqrt(2.0)

flat = brainstate.transform.vmap(amplitude_for_pair)(
    aa.reshape(-1),
    ww.reshape(-1),
)
grid = flat.reshape(aa.shape)
assert grid.shape == (6, 5)
```

Preserve `indexing="ij"` and record axis values with the reshaped result. A heatmap without physical axis values is not a reproducible regime map.

## Sweep a delayed network

Set global `dt` before `vmap` traces construction of each delayed `Network`.

```python
brainstate.environ.set(dt=0.1 * u.ms)
connectome = brainmass.datasets.load_dataset("example_connectome")

def mean_activity_for(k):
    n_region = connectome.weights.shape[0]
    node = brainmass.HopfStep(
        in_size=n_region,
        a=0.2,
        w=0.3,
        init_x=braintools.init.Constant(0.3),
    )
    network = brainmass.Network(
        node,
        conn=connectome.weights,
        distance=connectome.distances,
        speed=10.0 * u.mm / u.ms,
        coupling="diffusive",
        coupled_var="x",
        k=k,
    )
    output = brainmass.Simulator(network, dt=0.1 * u.ms).run(
        600.0 * u.ms,
        monitors=lambda model: model.node.x.value,
        transient=100.0 * u.ms,
    )["output"]
    return jnp.mean(jnp.abs(u.get_magnitude(output)))

k_values = jnp.linspace(0.0, 1.5, 8)
summary = brainstate.transform.vmap(mean_activity_for)(k_values)
assert summary.shape == k_values.shape
```

Use FC, FCD, spectrum, amplitude, synchrony, or another domain summary when mean absolute activity does not answer the scientific question.

## Store results

Keep coordinate arrays, summary arrays, units, model identity, fixed parameters, `dt`, duration, transient, seed, and code version together. Convert flattened grids to long-form columns only after preserving the original axis ordering.

Seed stochastic sweeps deliberately. One common seed across parameter values reduces random differences but couples their noise realization; independent seeds estimate uncertainty instead. State which design was used.

## Official source

- `https://brainx.chaobrain.com/brainmass/howto/parameter_sweeps.html`
