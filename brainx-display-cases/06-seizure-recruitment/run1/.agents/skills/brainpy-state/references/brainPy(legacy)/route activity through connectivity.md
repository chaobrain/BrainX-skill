# Route activity through connectivity

Use this reference when legacy `brainpy.math` code must expand presynaptic values to synapses or reduce edge values onto postsynaptic neurons. These operators execute activity routing over known indices; use `brainpy.connect` to construct the connectivity itself.

## Selection map

| Available data | Required result | Use |
|---|---|---|
| One scalar or one value per presynaptic neuron plus edge endpoint indices | One reduced value per postsynaptic neuron | A `pre2post_*` operator. |
| One value per presynaptic neuron plus `pre_ids` | One value per edge | `pre2syn(...)`. |
| One value per edge plus `post_ids` | Reduced values per postsynaptic neuron | A `syn2post_*` operator. |
| Boolean presynaptic events plus CSR pre-to-post structure | Event-driven weighted sums per postsynaptic neuron | `pre2post_event_sum(...)`. |

## Route presynaptic values directly to postsynaptic neurons

These operators gather values through `pre_ids` and reduce all incoming edges addressed by `post_ids`.

| API | Description |
|---|---|
| `pre2post_sum(pre_values, post_num, post_ids, pre_ids=None)` | Sum incoming presynaptic values for each postsynaptic neuron and return length `post_num`. |
| `pre2post_prod(pre_values, post_num, post_ids, pre_ids=None)` | Multiply incoming values into a zero-initialized postsynaptic array; see the warning below before using it for a true product reduction. |
| `pre2post_max(pre_values, post_num, post_ids, pre_ids=None)` | Reduce incoming values by maximum against a zero-initialized postsynaptic array. |
| `pre2post_min(pre_values, post_num, post_ids, pre_ids=None)` | Reduce incoming values by minimum against a zero-initialized postsynaptic array. |
| `pre2post_mean(pre_values, post_num, post_ids, pre_ids=None)` | Average incoming presynaptic values for each postsynaptic neuron. |

**Invariant:** pass `pre_ids` whenever `pre_values` is not scalar. Without it, BrainPy cannot determine which presynaptic value belongs to each edge.

**Zero-baseline warning:** the documented `pre2post_prod`, `pre2post_max`, and `pre2post_min` implementations start from zeros. Consequently, `pre2post_prod` keeps targeted outputs at zero, `pre2post_max` returns zero for an all-negative incoming group, and `pre2post_min` returns zero for an all-positive incoming group. Use `pre2syn(...)` followed by the corresponding `syn2post_*` segment reduction when those zero-baseline semantics are not intended.

For scalar `pre_values`, `pre2post_mean(...)` assigns that constant to targeted postsynaptic neurons and leaves untargeted neurons at zero; duplicate targets do not change the mean of identical values.

## Expand presynaptic values to edge values

Use `pre2syn(...)` when a synaptic rule must operate once per connection before postsynaptic reduction.

| API | Description |
|---|---|
| `pre2syn(pre_values, pre_ids)` | Gather `pre_values[pre_ids]` and return one value per synapse; a scalar input is repeated for every edge. |

The returned edge order is exactly the order of `pre_ids`. Preserve that order through any edge-local kinetics or weight computation so it still aligns with `post_ids`.

## Reduce edge values onto postsynaptic neurons

Use these after edge-local state, delays, kinetics, or weights have produced one value per synapse.

| API | Description |
|---|---|
| `syn2post_sum(syn_values, post_ids, post_num, indices_are_sorted=False)` | Sum edge values for each postsynaptic neuron. |
| `syn2post(syn_values, post_ids, post_num, indices_are_sorted=False)` | Alias of `syn2post_sum(...)`. |
| `syn2post_prod(syn_values, post_ids, post_num, indices_are_sorted=False)` | Multiply edge values within each postsynaptic segment. |
| `syn2post_max(syn_values, post_ids, post_num, indices_are_sorted=False)` | Take the maximum edge value within each postsynaptic segment. |
| `syn2post_min(syn_values, post_ids, post_num, indices_are_sorted=False)` | Take the minimum edge value within each postsynaptic segment. |
| `syn2post_mean(syn_values, post_ids, post_num, indices_are_sorted=False)` | Average edge values within each postsynaptic segment. |
| `syn2post_softmax(syn_values, post_ids, post_num, indices_are_sorted=False)` | Normalize edge values with a separate softmax over the edges entering each postsynaptic neuron. |

Set `indices_are_sorted=True` only when `post_ids` is known to be sorted, such as indices supplied in that order by the relevant BrainPy connector. Leave it `False` for arbitrary edge lists.

## Route boolean events through CSR connectivity

Use the event operator when presynaptic activity is boolean and connectivity is already in CSR pre-to-post form.

| API | Description |
|---|---|
| `pre2post_event_sum(events, pre2post, post_num, values=1.0)` | Sum weights from active presynaptic events; `pre2post` is `(post_ids, indptr)`, and `values` is either scalar or one value per stored edge. |
| `pre2post_csr_event_sum(events, pre2post, post_num, values=1.0)` | Alias of `pre2post_event_sum(...)`. |

**Invariant:** `events` must be boolean, and `indptr` must partition `post_ids` by presynaptic neuron. A coordinate edge list is not a valid replacement for this CSR tuple.

## Canonical workflow

Keep `pre_ids` and `post_ids` aligned as one edge list, expand only when edge-local computation is required, and reduce with the matching postsynaptic IDs:

```python
import brainpy.math as bm

pre_values = bm.asarray([2.0, 3.0, 5.0])
pre_ids = bm.asarray([0, 0, 1, 2])
post_ids = bm.asarray([0, 1, 1, 0])

syn_values = bm.pre2syn(pre_values, pre_ids)
post_values = bm.syn2post_sum(
    syn_values,
    post_ids,
    post_num=2,
)
direct_values = bm.pre2post_sum(
    pre_values,
    post_num=2,
    post_ids=post_ids,
    pre_ids=pre_ids,
)

assert syn_values.shape == (4,)
assert bm.allclose(post_values, bm.asarray([7.0, 5.0]))
assert bm.allclose(direct_values, post_values)
```

Use the direct route when no per-synapse computation is needed. Use `pre2syn(...)` followed by a `syn2post_*` reduction when the edge values must be transformed independently.

## Sources

- https://brainpy.readthedocs.io/apis/brainpy.math.pre_syn_post.html
