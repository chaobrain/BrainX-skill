# Construct BrainSpike connectivity with Braintools

Use a Braintools connector when geometry or topology selects neuron pairs beyond the native Simulator rules. Keep pair selection, edge ordering, physical weights/delays, and event representation as separate decisions.

## Choose the connection owner

| API | Use |
|---|---|
| `brainspike.fixed_indegree(k)` / `pairwise_bernoulli(p)` | Prefer the Simulator's own rule when it already expresses the required network. |
| `braintools.conn.ExcitatoryInhibitory(...)` | Use when the topology must preserve declared excitatory/inhibitory source roles and associated signed efficacy. |
| `braintools.conn.DistanceDependent(...)` | Use when explicit positions determine pair selection; verify its required geometry inputs before constructing a spatial network. |
| `brainspike.explicit_edges(pre_indices, post_indices)` | Import selected pairs into the NEST Simulator; preserve population-local indices and paired edge order. |

## Generate an explicit paired edge list

```python
import braintools
import brainunit as u

pattern = braintools.conn.ExcitatoryInhibitory(
    exc_ratio=0.8, exc_prob=0.1, inh_prob=0.2,
    exc_weight=1.0*u.pA, inh_weight=-0.8*u.pA,
)
connections = pattern(pre_size=20, post_size=10)
assert connections.shape == (20, 10)
assert connections.pre_indices.shape == connections.post_indices.shape
assert connections.weights.shape[0] == connections.n_connections
```

Keep `pre_indices[i]`, `post_indices[i]`, `weights[i]`, and any `delays[i]` aligned through filtering, sorting, or compression. Preserve a connector seed through the documented interface when exact graph replay is required. Do not assume BrainState's RNG controls every host-side connector.

For a NEST graph, pass matching source/target views and an `explicit_edges` rule to `connect()`. Match the target model's weight unit and verify the required per-edge weight/delay form before passing arbitrary connector arrays: the local experiment establishes distinct delays on separate connections, not all per-edge array representations. A signed current weight does not define an inhibitory conductance model; use its receptor/reversal contract when conductance is represented.

For native fixed communication, open `brainevent` → `references/sparse-formats.md` to convert the paired edge list into CSR/CSC while preserving edge-value order. Use `references/native-composition.md` for conductance filtering and target update order. Do not replace the trainable ETP-aware projection with an arbitrary binary event operator and assume the same temporal gradients.

Open `references/nest-compatible/network-building.md` for Simulator rule, index, and projection semantics, and `references/braintools/parameter-initializer.md` when existing weights need distance modulation rather than new topology.

## Source

[BrainTools connectivity API](https://brainx.chaobrain.com/braintools/apis/conn.html). The connector example targets BrainTools 0.4.0.
