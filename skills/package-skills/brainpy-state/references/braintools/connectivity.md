# Braintools connectivity

Use this reference to choose and construct point-neuron or
multi-compartment connection patterns. It covers which pairs exist, their
indices, optional weights and delays, spatial structure, topology, and
compartment targeting. Use `braintools.init.DistanceModulated` only when an
existing connection's weight should be scaled by distance.

## Choose a connectivity family

| Family | Use when | Representative choice |
|---|---|---|
| Basic point-neuron | Connection probability or deterministic pairing is enough | `Random`, `AllToAll`, or `OneToOne` |
| Spatial point-neuron | Positions or spatial neighborhoods determine connection probability | `DistanceDependent`, `Gaussian`, `Ring`, or `Grid2d` |
| Topological | Degree, clustering, modularity, hierarchy, or core/periphery structure matters | `SmallWorld`, `ScaleFree`, `ModularRandom`, or `HierarchicalRandom` |
| Biological E/I | Presynaptic sign must follow Dale's principle | `ExcitatoryInhibitory` |
| Kernel based | Receptive-field geometry should follow an image-like kernel | `GaussianKernel`, `GaborKernel`, or `DoGKernel` |
| Multi-compartment | Synapses must target soma, axon, or dendritic compartments | `CompartmentSpecific` or an anatomical targeting subclass |

## Construction and results

Every connectivity object is a reusable pattern. Call it with population sizes
and any required positions or morphology metadata to obtain a
`ConnectionResult`.

| API | Description |
|---|---|
| `Connectivity(pre_size=None, post_size=None, seed=None)` | Subclass for a new general pattern. Implement `generate(...)`; use `weight_scale(factor)` or `delay_scale(factor)` to wrap a pattern. |
| `PointConnectivity` | Subclass for a new single-compartment pattern. |
| `MultiCompartmentConnectivity` | Subclass for a new morphology-aware pattern. |
| `ScaledConnectivity` | Returned when weights or delays are scaled. |
| `CompositeConnectivity` | Use when combining several connectivity patterns. |
| `ConnectionResult(pre_indices, post_indices, pre_size, post_size, ..., weights=None, delays=None, model_type='point', metadata=None)` | Inspect the generated edge list and optional annotations. `shape` is `(pre_size, post_size)` and `n_connections` is the number of edges. |
| `ConnectionResult.weight2dense()` / `weight2csr()` | Convert weights to a dense or CSR connectivity matrix. |
| `ConnectionResult.delay2matrix()` / `delay2csr()` | Convert delays to a dense or CSR connectivity matrix. |
| `ConnectionResult.get_distances()` | Compute distances for connected pairs when positions are available. |

```python
import brainunit as u
from braintools.conn import ExcitatoryInhibitory

pattern = ExcitatoryInhibitory(
    exc_ratio=0.8,
    exc_prob=0.1,
    inh_prob=0.2,
    exc_weight=1.0 * u.nS,
    inh_weight=-0.8 * u.nS,
)
result = pattern(pre_size=1000, post_size=1000)

assert result.shape == (1000, 1000)
assert result.pre_indices.shape == result.post_indices.shape
assert result.weights.shape[0] == result.n_connections
```

**Invariant:** a `ConnectionResult` edge is identified by matching entries in
`pre_indices` and `post_indices`. Keep any weights and delays aligned with that
edge order when converting or filtering.

## Basic and spatial point-neuron patterns

| API | Description |
|---|---|
| `Random(prob, allow_self_connections=False, weight=None, delay=None, **kwargs)` | Use for independent fixed-probability edges. Supply initializer objects or values for optional weights and delays. |
| `FixedProb(...)` | Use only as the alias for `Random`. |
| `AllToAll(...)` | Connect every allowed presynaptic/postsynaptic pair. |
| `OneToOne(...)` | Connect neuron `i` to neuron `i`; use only when the population indexing is aligned. |
| `DistanceDependent(distance_profile, weight=None, delay=None, allow_self_connections=False, **kwargs)` | Use when a `braintools.init.DistanceProfile` should determine connection probability from positions or precomputed distances. |
| `Gaussian(...)` | Use for Gaussian distance-dependent connection probability. |
| `Exponential(...)` | Use for exponentially decaying connection probability. |
| `Ring(...)` | Use for neurons arranged on a circular topology. |
| `Grid2d(connectivity='von_neumann', weight=None, delay=None, periodic=False, **kwargs)` | Use for immediate four-neighbor (`'von_neumann'`) or eight-neighbor (`'moore'`) connections on matching two-dimensional pre/post grids. Set `periodic=True` only for wraparound boundaries. |
| `RadialPatches(...)` | Use for several localized spatial connection patches. |
| `ClusteredRandom(...)` | Use for random connectivity enhanced within spatial clusters. |

```python
import brainunit as u
import numpy as np
from braintools.conn import DistanceDependent
from braintools.init import GaussianProfile, LogNormal

positions = np.random.default_rng(0).uniform(0, 1000, (500, 2)) * u.um
pattern = DistanceDependent(
    GaussianProfile(sigma=100 * u.um, max_distance=300 * u.um),
    weight=LogNormal(mean=1.0 * u.nS, std=0.5 * u.nS),
)
result = pattern(500, 500, positions, positions)
```

`DistanceDependent` consumes the profile's connection probability and samples
which pairs exist. In contrast, `braintools.init.DistanceModulated` multiplies
every supplied base weight by a profile value and never removes an edge.

## Export a `ConnectionResult` to BrainEvent CSR

Choose topology with BrainTools before choosing BrainEvent storage. Convert the
result's coordinate edge list to CSR when binary presynaptic events should drive
event-based communication.

| API | Description |
|---|---|
| `brainevent.coo2csr(row_ids, col_ids, *, shape)` | Convert `pre_indices` and `post_indices` to row-compressed structure; it returns `(indptr, indices, order)`, where `order` is the required edge-value permutation. |
| `brainevent.CSR((data, indices, indptr), shape=...)` | Store the reordered weights with presynaptic rows and postsynaptic columns for `BinaryArray @ CSR`. |
| `brainevent.BinaryArray(spikes) @ connectivity` | Communicate active presynaptic events and return one accumulated value per postsynaptic index. |

```python
import brainevent
import brainunit as u
from braintools.conn import Grid2d
from braintools.init import Constant

pattern = Grid2d(
    connectivity="von_neumann",
    weight=Constant(1.0 * u.nS),
    periodic=False,
)
result = pattern(pre_size=(10, 10), post_size=(10, 10))

indptr, indices, order = brainevent.coo2csr(
    result.pre_indices,
    result.post_indices,
    shape=result.shape,
)
connectivity = brainevent.CSR(
    (result.weights[order], indices, indptr),
    shape=result.shape,
)
postsynaptic_input = brainevent.BinaryArray(
    u.math.zeros(result.shape[0], dtype=bool)
) @ connectivity

assert postsynaptic_input.shape == (result.shape[1],)
```

Keep `result.weights[order]`: `coo2csr()` sorts edges by row and returns the
matching value permutation separately. If the topology has no generated
weights, supply an intentional scalar or edge-aligned weight array before
constructing `CSR`.

Open `skills/package-skills/brainevent/references/sparse-formats.md` when choosing among CSR,
CSC, dense, generated, or fixed-degree event representations after topology is
defined.

## Topological and biological patterns

| API | Description |
|---|---|
| `SmallWorld(...)` | Use for Watts-Strogatz small-world structure. |
| `ScaleFree(...)` | Use for Barabasi-Albert preferential attachment. |
| `Regular(...)` | Use when each neuron should have the same degree. |
| `ModularRandom(...)` | Use for random within-module and between-module probabilities. |
| `ModularGeneral(...)` | Use when intra- and inter-module structure each need their own `Connectivity` object. |
| `HierarchicalRandom(...)` | Use for multilevel modules with asymmetric feedforward and feedback connectivity. |
| `CorePeripheryRandom(...)` | Use for a dense core and sparse periphery. |
| `ExcitatoryInhibitory(exc_ratio=0.8, exc_prob=0.1, inh_prob=0.2, exc_weight=None, inh_weight=None, exc_delay=None, inh_delay=None, **kwargs)` | Use for separate excitatory and inhibitory presynaptic populations whose outgoing signs follow Dale's principle. |

## Kernel-based patterns

Use kernel patterns when source and target neurons occupy image-like spatial
grids and a local receptive-field operator defines edges.

| API | Description |
|---|---|
| `Conv2dKernel(...)` | Use a general convolutional kernel. |
| `GaussianKernel(...)` | Use a Gaussian receptive field. |
| `GaborKernel(...)` | Use an orientation-selective Gabor field. |
| `DoGKernel(...)` | Use a difference-of-Gaussians center-surround field. |
| `MexicanHat(...)` | Use a Laplacian-of-Gaussian field. |
| `SobelKernel(...)` | Use a Sobel orientation/edge kernel. |
| `LaplacianKernel(...)` | Use a Laplacian edge kernel. |
| `CustomKernel(...)` | Supply a user-defined kernel function. |

## Multi-compartment patterns

Use `SOMA`, `BASAL_DENDRITE`, `APICAL_DENDRITE`, and `AXON` as the documented
compartment identifiers. Choose the narrowest pattern that expresses the
anatomical constraint.

| Family | API | Description |
|---|---|---|
| General | `CompartmentSpecific(...)` | Define source and target compartment rules directly. |
| General | `AllToAllCompartments(...)` | Connect all allowed compartment pairs. |
| General | `CustomCompartment(...)` | Supply a custom compartment-selection function. |
| Anatomical | `SomaToDendrite(...)` | Target dendrites from somatic sources. |
| Anatomical | `AxonToSoma(...)` | Target soma from axons. |
| Anatomical | `DendriteToSoma(...)` | Target soma from dendrites. |
| Anatomical | `AxonToDendrite(target_dendrites=None, **kwargs)` | Target selected dendritic compartments from axons. |
| Anatomical | `DendriteToDendrite(...)` | Connect dendritic compartments. |
| Morphology | `ProximalTargeting(...)` | Prefer proximal dendritic compartments. |
| Morphology | `DistalTargeting(...)` | Prefer distal dendritic compartments. |
| Morphology | `BranchSpecific(...)` | Restrict targets to specified branches. |
| Morphology | `MorphologyDistance(...)` | Use morphology-aware distances. |
| Dendritic | `DendriticTree(...)` | Configure separate basal/apical probabilities. |
| Dendritic | `BasalDendriteTargeting(...)` | Restrict targets to basal dendrites. |
| Dendritic | `ApicalDendriteTargeting(...)` | Restrict targets to apical dendrites. |
| Dendritic | `DendriticIntegration(...)` | Express dendritic integration patterns. |
| Axonal | `AxonalProjection(...)` | Build topographic axonal projections. |
| Axonal | `AxonalBranching(...)` | Build branching axonal patterns. |
| Axonal | `AxonalArborization(...)` | Build axonal arborization patterns. |
| Axonal | `TopographicProjection(...)` | Preserve source/target topography. |
| Synaptic | `SynapticPlacement(...)` | Apply synapse-placement rules. |
| Synaptic | `SynapticClustering(...)` | Cluster synapses on target morphology. |

## Official source

- `https://brainx.chaobrain.com/braintools/apis/conn.html`
