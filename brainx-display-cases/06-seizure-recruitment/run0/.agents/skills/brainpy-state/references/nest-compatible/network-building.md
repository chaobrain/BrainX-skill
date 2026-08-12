# Network building

Use this reference when the canonical `Simulator` workflow needs lifecycle control, `NodeView` composition, `SimulationResult`, `SynapseCollection`, projection or connection-rule lookup, declarative `Network`/`Builder` APIs, or spatially structured populations and connectivity.

## Network construction

Use these APIs to choose the declarative or imperative network surface and create its populations, devices, and connections.

| API | Use when | Important result |
|---|---|---|
| `bp.Simulator(*, dt)` | Use for the NEST-flavored create/connect/run workflow. | Owns the graph, timestep, rollout State, recording, and result construction. |
| `bp.network.Network(*args, **kwargs)` | Use when subclassing the lower-level network base and registering populations, projections, and devices as attributes. | `update()` walks immediate children in projection-first, then dynamics order. |
| `bp.network.Builder(*args, **kwargs)` | Use for the imperative variant of `Network`. | `add()` registers named modules and `connect()` registers projections. |
| `sim.create(model_cls, size=1, *, params=None, positions=None, **kw)` | Use to instantiate a population or device. | Returns a `NodeView`; spatial `positions` attach coordinates. |
| `sim.connect(pre, post, *, rule=bp.all_to_all, weight=None, delay=None, comm="dense", receptor_type=None, synapse=None, vt=None, allow_autapses=True, allow_multapses=True, seed=None)` | Use to create a projection, attach a source, or register a recorder. | Returns a projection handle, a list for multi-segment fan-out, or `None` for taps/injectors. |

## Simulation execution

Use these methods to choose whether a run starts from fresh State or continues a persistent biological timeline.

| API | Use when | Important result |
|---|---|---|
| `sim.simulate(duration, *, dt=None)` | Use for an independent run. | Reinitializes all State, starts at time zero, and returns `SimulationResult`. |
| `sim.cont(duration, *, dt=None)` | Use to continue biological time across host-controlled windows. | Preserves State and returns a window result with absolute times. |
| `sim.reset_rollout(*, dt=None)` | Use before restarting a persistent `cont()` workflow. | Reinitializes all State and resets accumulated time and step index to zero. |

Do not interchange `simulate()` and `cont()` casually. Use `simulate()` for independent trials; use `cont()` only when State and biological time must persist across windows.

## NodeView composition

Use `NodeView` composition when one connection or recording operation must address a full population, a slice, or several preserved population segments.

| Operation | Use |
|---|---|
| `left + right` | Concatenate views while preserving segment boundaries, such as `exc + inh` for a common target. |
| `view[start:stop]` | Slice one population segment for recording or partial connectivity. |
| `bp.network.NodeView.of(population)` | Build a full-population view over an existing population. |
| `view.segments` | Inspect the population-and-local-index segments when an API needs the concrete underlying population. |

Source and target indices reported by connectivity inspection are population-local. Do not reinterpret them as NEST global node IDs.

## Connection-rule lookup

Pass rule objects to `Simulator.connect()`; let the rule select edges and the synapse spec select edge behavior.

| Rule helper | Choose when |
|---|---|
| `bp.all_to_all` | Every source-target pair must exist. |
| `bp.one_to_one` | Source `i` must connect to target `i`. |
| `bp.fixed_indegree(k)` | Every target must receive exactly `k` incoming edges. |
| `bp.fixed_total_number(n)` | Exactly `n` edges must be drawn over the source-target grid. |
| `bp.pairwise_bernoulli(p)` | Every possible pair must receive an independent Bernoulli draw. |
| `bp.explicit_edges(pre_idx, post_idx)` | The exact source and target edge-index arrays are already known. |
| `bp.third_factor_bernoulli_with_pool(p=..., pool_size=..., pool_type=...)` | A tripartite pre-post-third-factor network needs Bernoulli pairing with an astrocyte pool. |

Use `seed` to reproduce one BrainPy-State realization, `allow_autapses` and `allow_multapses` to control self and repeated edges, and `comm="sparse"` for memory-light event communication.

## Projection-class lookup

For standard construction, pass a rule or synapse spec to `Simulator.connect()` and inspect the returned handle. Use the classes below when reading API behavior or extending the network layer.

| Projection class | Represents |
|---|---|
| `bp.network.OneToOneProj(*args, **kwargs)` | Edge `(i, i)` for every paired index. |
| `bp.network.AllToAllProj(*args, **kwargs)` | Full all-to-all connectivity. |
| `bp.network.PairwiseBernoulliProj(*args, **kwargs)` | Independent Bernoulli sampling for every source-target pair. |
| `bp.network.SymmetricPairwiseBernoulliProj(*args, **kwargs)` | Symmetric Bernoulli connectivity where an edge implies its reverse. |
| `bp.network.FixedIndegreeProj(*args, **kwargs)` | Exactly `k` incoming edges per postsynaptic neuron. |
| `bp.network.FixedOutdegreeProj(*args, **kwargs)` | Exactly `k` outgoing edges per presynaptic neuron. |
| `bp.network.FixedTotalNumberProj(*args, **kwargs)` | Exactly `n` edges over the full source-target grid. |
| `bp.network.PairwisePoissonProj(*args, **kwargs)` | A Poisson-distributed edge multiplicity for each source-target pair. |
| `bp.network.EventProjection(*args, **kwargs)` | Delayed, weighted static event transmission. |
| `bp.network.EventPlasticProj(*args, **kwargs)` | Event-driven plastic transmission. |
| `bp.network.VoltageCoupledPlasticProj(*args, **kwargs)` | Plasticity coupled to postsynaptic voltage State. |

## Canonical sparse E/I construction

Use `NodeView` algebra and a seeded fixed-indegree rule to express Brunel-style wiring without materializing a dense edge list.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)
params = {
    "C_m": 1.0 * u.pF,
    "tau_m": 20.0 * u.ms,
    "t_ref": 2.0 * u.ms,
    "E_L": 0.0 * u.mV,
    "V_reset": 0.0 * u.mV,
    "V_th": 20.0 * u.mV,
}
exc = sim.create(bp.iaf_psc_delta, 400, params=params)
inh = sim.create(bp.iaf_psc_delta, 100, params=params)
all_neurons = exc + inh

sim.connect(
    exc,
    all_neurons,
    rule=bp.fixed_indegree(40),
    weight=0.1 * u.mV,
    delay=1.5 * u.ms,
    comm="sparse",
    allow_multapses=True,
    seed=1,
)
sim.connect(
    inh,
    all_neurons,
    rule=bp.fixed_indegree(10),
    weight=-0.5 * u.mV,
    delay=1.5 * u.ms,
    comm="sparse",
    allow_multapses=True,
    seed=2,
)
```

**Invariant:** Delta-neuron weights are voltage jumps in `mV`. Do not reuse current weights from an alpha- or exponential-PSC network.

## Result readback

Use these APIs after simulation to read the common time axis, spike observables, analog traces, or registered plastic-weight trajectories.

| API | Use |
|---|---|
| `result.times` | Read the common time axis. |
| `result.spikes(node)` | Read a per-step spike matrix. |
| `result.rate(node)` | Read the mean population firing rate. |
| `result.trace(recorder, recordable="V_m")` | Read a registered analog trace. |
| `sim.record_weight(proj)` | Register the returned plastic projection for per-step weight capture. |
| `result.weight_trace(proj)` | Read the registered `(n_steps, n_edges)` weight trajectory. |

## Realized-connection inspection

Use these APIs after construction to obtain a lazy view over realized edges and read or update supported live connection attributes.

| API | Use |
|---|---|
| `sim.get_connections(source=None, target=None, synapse=None)` | Obtain a filtered, lazy `SynapseCollection` over realized edges. |
| `connections.get(key)` | Read live `source`, `target`, `weight`, or `delay` values. |
| `connections.set(key, value)` | Write supported static weight or delay values; plastic and homogeneous constraints can reject the write. |

## Spatial layouts

Use these constructors to build concrete or deferred 2-D/3-D node positions.

| Constructor | Use when |
|---|---|
| `bp.spatial.Layer(coords, ndim, shape=None, extent=None, center=None, sampler=None)` | Use when constructing a layer directly from coordinates or a sampler. |
| `bp.spatial.grid(shape, extent=None, center=None)` | Use a regular cell-centered lattice. |
| `bp.spatial.free(positions, extent=None, num_dimensions=None)` | Use explicit or sampled free positions. |

## Spatial position expressions

Use these expression roots when a kernel depends on absolute, source, or target coordinates.

| API | Use when |
|---|---|
| `bp.spatial.pos` | Use `.x`, `.y`, or `.z` for the current position expression. |
| `bp.spatial.source_pos` | Use `.x`, `.y`, or `.z` for source-coordinate expressions. |
| `bp.spatial.target_pos` | Use `.x`, `.y`, or `.z` for target-coordinate expressions. |

## Spatial distance and displacement

Use these APIs to compute pairwise geometric relationships between source and target nodes or layers.

| API | Use when |
|---|---|
| `bp.spatial.displacement(pre_pos, post_pos)` | Use the vector displacement `post - pre`. |
| `bp.spatial.pairwise_distance(pre_pos, post_pos)` | Use pairwise Euclidean distance between coordinate arrays. |
| `bp.spatial.Distance(layer_a, layer_b)` | Use a reusable distance expression between two layers. |

## Spatial probability kernels

Use these constructors to map distance or displacement to connection probability.

| Constructor | Use when |
|---|---|
| `bp.spatial.gaussian(x=bp.spatial.distance, mean=0.0, std=1.0)` | Use a Gaussian probability profile. |
| `bp.spatial.exponential(x=bp.spatial.distance, beta=1.0)` | Use an exponential probability profile. |
| `bp.spatial.gamma(x=bp.spatial.distance, kappa=1.0, theta=1.0)` | Use a gamma probability profile. |
| `bp.spatial.gabor(x=None, y=None, theta=0.0, gamma=1.0, std=1.0, lam=1.0, psi=0.0)` | Use a rectified Gabor profile over 2-D displacement. |
| `bp.spatial.gaussian2D(x=None, y=None, mean_x=0.0, mean_y=0.0, std_x=1.0, std_y=1.0, rho=0.0)` | Use a correlated bivariate-Gaussian profile. |

## Spatial masks

Use these constructors to restrict candidate targets to a 2-D or 3-D region.

| Constructor | Use when |
|---|---|
| `bp.spatial.circular(radius)` | Use a circular 2-D cutoff. |
| `bp.spatial.spherical(radius)` | Use a spherical 3-D cutoff. |
| `bp.spatial.box(lower_left, upper_right)` | Use an axis-aligned 2-D or 3-D box. |
| `bp.spatial.rectangular(lower_left, upper_right, azimuth_angle=0.0)` | Use a rotated 2-D rectangle. |
| `bp.spatial.doughnut(inner_radius, outer_radius)` | Use a 2-D annulus. |
| `bp.spatial.elliptical(major_axis, minor_axis, azimuth_angle=0.0, anchor=None)` | Use a rotated 2-D ellipse. |
| `bp.spatial.ellipsoidal(major_axis, minor_axis, polar_axis, azimuth_angle=0.0, polar_angle=0.0, anchor=None)` | Use a rotated 3-D ellipsoid. |

## Spatial connection rules

Use these constructors to combine a probability law with an optional mask before passing the rule to `Simulator.connect()`.

| Constructor | Use when |
|---|---|
| `bp.spatial.SpatialConnRule(p, mask=None, allow_autapses=True, _pre_pos=None, _post_pos=None)` | Use the concrete spatial rule type when positions are already bound. |
| `bp.spatial.spatial_pairwise_bernoulli(p, mask=None, allow_autapses=True)` | Use the public distance-dependent pairwise-Bernoulli builder. |

## Spatial node selection and introspection

Use these APIs to select nodes by geometry or inspect realized spatial targets.

| API | Use when |
|---|---|
| `bp.spatial.center_element(layer)` | Use the node nearest the layer centroid. |
| `bp.spatial.nearest_element(layer, locations, find_all=False)` | Use the node nearest each query location. |
| `bp.spatial.select_nodes_by_mask(layer, anchor, mask)` | Use local indices inside a mask anchored at a position. |
| `bp.spatial.target_nodes(sim, source, target)` | Use realized target indices for each source. |
| `bp.spatial.target_positions(sim, source, target)` | Use coordinates of each source node's realized targets. |

## Spatial export and visualization

Use these APIs to export layer data or visualize positions, realized edges, and probability fields.

| API | Use when |
|---|---|
| `bp.spatial.dump_layer_nodes(sim, pop, outname)` | Use to write local node indices and coordinates. |
| `bp.spatial.dump_layer_connections(sim, source, target, outname)` | Use to write realized edges, weights, delays, and displacement. |
| `bp.spatial.plot_layer(layer, fig=None, nodecolor="b", nodesize=20)` | Use to plot layer positions. |
| `bp.spatial.plot_targets(sim, src_node, target, fig=None, src_color="red", src_size=50, tgt_color="b", tgt_size=20)` | Use to highlight one source node's realized targets. |
| `bp.spatial.plot_sources(sim, source, tgt_node, fig=None, src_color="b", src_size=20, tgt_color="red", tgt_size=50)` | Use to highlight one target node's realized sources. |
| `bp.spatial.plot_probability_parameter(kernel, mask=None, extent=(-0.5, 0.5, -0.5, 0.5), shape=(100, 100), fig=None, cmap="Greys")` | Use to plot a 2-D probability field. |

## Spatial workflow

Use the same `Simulator.create()` and `Simulator.connect()` lifecycle after selecting the layout, kernel, mask, and spatial rule.

```python
positions = bp.spatial.grid([4, 3], extent=[2.0, 1.5])
source = sim.create(bp.iaf_psc_alpha, positions=positions)
target = sim.create(bp.iaf_psc_alpha, positions=positions)

rule = bp.spatial.spatial_pairwise_bernoulli(
    p=bp.spatial.gaussian(bp.spatial.distance, std=0.5),
    mask=bp.spatial.circular(3.0),
)
sim.connect(
    source,
    target,
    rule=rule,
    weight=1.0 * u.pA,
    delay=1.0 * u.ms,
    seed=0,
)
```

Spatial random draws differ between NEST and JAX even when the probability law matches. Validate the empirical probability-versus-distance curve distributionally, not edge by edge.

## Official sources

- https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/03-connect-network.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-network.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-spatial.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/spatial.html
