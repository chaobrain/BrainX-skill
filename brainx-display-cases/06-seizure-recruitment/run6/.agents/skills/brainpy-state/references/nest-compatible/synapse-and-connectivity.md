# Synapse and connectivity

Use this reference when selecting a NEST-compatible synapse or plasticity rule, choosing canonical connectivity, constructing a unitful synapse spec, or reading and writing realized edges. Use `references/nest-compatible/network-building.md` for the complete connection-rule and projection-class map, and `references/nest-compatible/divergence-and-parity.md` for STDP parameter relocation and numerical bands.

## Connection rules

Use these APIs to choose realized source-target pairs; choose the synapse spec separately to define what each realized edge does.

| API | Use when |
|---|---|
| `bp.all_to_all` | Use when every source must connect to every target, including the default generator fan-out. |
| `bp.one_to_one` | Use when source `i` must connect to target `i`; source and target views must have equal-size pairings. |
| `bp.fixed_indegree(k)` | Use when every target must receive exactly `k` randomly drawn presynaptic edges. |
| `bp.spatial.spatial_pairwise_bernoulli(p, mask=None, allow_autapses=True)` | Use when connection probability or eligibility depends on distance. |

Set `seed` to reproduce a BrainPy-State random realization. For `fixed_indegree`, set `allow_multapses=True` to match NEST's default and use `comm="sparse"` for large event fan-in. Open `references/nest-compatible/network-building.md` for the remaining rule constructors.

## Static synapses

Use these constructors when edge weights do not change through activity-dependent learning.

| Constructor | Use when |
|---|---|
| `bp.static_synapse(*args, **kwargs)` | Use the default fixed-weight, fixed-delay synapse spec. |
| `bp.static_synapse_hom_w(*args, **kwargs)` | Use one homogeneous weight shared across all realized connections. |
| `bp.bernoulli_synapse(*args, **kwargs)` | Use fixed structural edges with stochastic Bernoulli transmission. |
| `bp.cont_delay_synapse(*args, **kwargs)` | Use a continuous sub-timestep delay. |

Pass `synapse=None` to `Simulator.connect()` for the ordinary static event path; instantiate a static spec only when its specialized behavior is required.

## Gap junctions and special connections

Use these constructors for electrical, diffusion, rate, or astrocyte slow-inward-current coupling rather than ordinary static spike transmission.

| Constructor | Use when |
|---|---|
| `bp.gap_junction(*args, **kwargs)` | Use electrical synaptic coupling with a compatible gap-junction neuron. |
| `bp.diffusion_connection(*args, **kwargs)` | Use the NEST-compatible diffusion connection. |
| `bp.rate_connection_instantaneous(*args, **kwargs)` | Use rate coupling without a connection delay. |
| `bp.rate_connection_delayed(*args, **kwargs)` | Use delayed rate coupling. |
| `bp.sic_connection(*args, **kwargs)` | Use astrocyte-to-neuron slow inward current coupling. |

## Short-term plasticity

Use these constructors for transient, reversible efficacy changes on the timescale of individual spikes.

| Constructor | Use when |
|---|---|
| `bp.tsodyks_synapse(*args, **kwargs)` | Use the Tsodyks-Uziel-Markram short-term-plasticity rule. |
| `bp.tsodyks_synapse_hom(*args, **kwargs)` | Use the homogeneous shared-parameter Tsodyks form. |
| `bp.tsodyks2_synapse(*args, **kwargs)` | Use the two-variable Tsodyks form. |
| `bp.quantal_stp_synapse(*args, **kwargs)` | Use the quantal/binomial short-term-plasticity form. |

## Spike-timing-dependent plasticity

Use these constructors for long-term weight changes driven by pre-post spike timing; select the pairing, weight-dependence, triplet, hardware, or dopamine rule explicitly.

| Constructor | Use when |
|---|---|
| `bp.stdp_synapse(*args, **kwargs)` | Use pair-based STDP. |
| `bp.stdp_synapse_hom(*args, **kwargs)` | Use pair-based STDP with homogeneous shared parameters. |
| `bp.stdp_pl_synapse_hom(*args, **kwargs)` | Use power-law STDP. |
| `bp.stdp_facetshw_synapse_hom(*args, **kwargs)` | Use the FACETS/BrainScaleS hardware-emulating rule. |
| `bp.stdp_nn_pre_centered_synapse(*args, **kwargs)` | Use presynaptic-centered nearest-neighbour pairing. |
| `bp.stdp_nn_restr_synapse(*args, **kwargs)` | Use restricted symmetric nearest-neighbour pairing. |
| `bp.stdp_nn_symm_synapse(*args, **kwargs)` | Use symmetric nearest-neighbour pairing. |
| `bp.stdp_triplet_synapse(*args, **kwargs)` | Use triplet STDP. |
| `bp.stdp_dopamine_synapse(*args, **kwargs)` | Use dopamine-modulated STDP with a `volume_transmitter`. |

Open `references/nest-compatible/divergence-and-parity.md` before porting `tau_minus`, Clopath parameters, dopamine State, or a nearest-neighbour pairing rule.

## Voltage-based and specialized learning

Use these constructors when the weight update depends on voltage trajectories, dendritic prediction, inhibitory timing, or the Hill-Tononi vesicle pool.

| Constructor | Use when |
|---|---|
| `bp.clopath_synapse(*args, **kwargs)` | Use voltage-based Clopath plasticity with a compatible `_clopath` neuron. |
| `bp.jonke_synapse(*args, **kwargs)` | Use exponential weight-dependence STDP. |
| `bp.urbanczik_synapse(*args, **kwargs)` | Use dendritic prediction-error plasticity with `pp_cond_exp_mc_urbanczik`. |
| `bp.vogels_sprekeler_synapse(*args, **kwargs)` | Use symmetric inhibitory plasticity. |
| `bp.ht_synapse(*args, **kwargs)` | Use Hill-Tononi vesicle-pool depression. |

## Synapse specification

Use these `Simulator.connect()` arguments to attach a selected rule to realized edges.

| Argument | Use |
|---|---|
| `weight=<quantity>` | Use signed current such as `pA` for current-based event input; use voltage such as `mV` for the instantaneous jump of `iaf_psc_delta`. |
| `delay=<time quantity>` | Use for a homogeneous axonal delay. |
| `weight=[...] * unit` | Use for a multi-channel generator with one signed weight per channel. |
| `seed=<int>` | Use to reproduce the BrainPy-State random edge realization. |
| `allow_multapses=<bool>` | Use to permit or forbid repeated source-target edges under a random rule. |
| `comm="dense"` or `comm="sparse"` | Use to select the communication backend without changing the rule. |
| `synapse=<spec>` | Use to attach any constructor from the synapse families above. |

```python
projection = sim.connect(
    exc,
    exc + inh,
    rule=bp.fixed_indegree(80),
    weight=0.1 * u.pA,
    delay=1.5 * u.ms,
    seed=1,
    allow_multapses=True,
    comm="sparse",
)
```

**Invariant:** Do not copy one weight unit across neuron families. `iaf_psc_delta` consumes a voltage jump, while canonical alpha- and exponential-current events consume current.

## Plastic connection workflow

Use `synapse=` when the edge needs State or behavior beyond static weight and delay.

```python
plasticity = bp.tsodyks2_synapse(
    weight=250.0 * u.pA,
    U=0.67,
    tau_rec=450.0 * u.ms,
    tau_fac=0.0 * u.ms,
)
projection = sim.connect(spike_source, post, synapse=plasticity)
```

**Invariant:** A BrainPy-State `spike_generator` can drive a plastic edge directly. Do not add a `parrot_neuron` solely because the NEST version required one.

## Realized-connectivity inspection

Use these methods after construction to inspect or update the live edges selected by the rule.

| API | Use when |
|---|---|
| `sim.get_connections(source=None, target=None, synapse=None)` | Use to obtain an empty or populated lazy `SynapseCollection` filtered by source, target, or synapse name. |
| `connections.get(key)` | Use one key to read live `source`, `target`, `weight`, or `delay`, or a sequence to return several attributes as a dictionary; weights and delays retain units. |
| `connections.set(key, value)` | Use to write supported static `weight` or `delay` values; homogeneous, plastic, and other weight-evolving projections can reject the write. |

```python
connections = sim.get_connections(source=exc, target=inh)
edge_data = connections.get(["source", "target", "weight", "delay"])

assert edge_data["source"].shape == edge_data["target"].shape
assert edge_data["weight"].shape == edge_data["source"].shape
```

`SynapseCollection` re-reads live weights and delays, so a collection created before simulation reflects later plastic changes. Source and target indices are population-local, not global NEST node IDs.

## Official sources

- https://brainx.chaobrain.com/brainpy-state/apis/nest-synapses.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-plasticity.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html
