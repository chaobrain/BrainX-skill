# BrainState State-Graph Operations

Use this reference after the core `State`, `Module`, and transform rules in
`SKILL.md`. It covers the less-frequent graph operations needed to find,
extract, split, replace, or reconstruct state while retaining BrainState's
graph structure. All examples assume an existing graph node named `model`.

The graph engine flattens object graphs of `Node`s and `State`s into "a static
structure plus a dynamic state mapping, and back." A `GraphDef` is the static
structure. Graph nodes are hoisted into an index-keyed table, so sharing and
cycles are encoded by integer index rather than duplicated.

Source URL:
https://brainx.chaobrain.com/brainstate/apis/graph.html

## Find Nodes, Leaves, and States

Use traversal when the path is part of the decision. `iter_node()` yields
`(path, graph_node)` for every graph node; `iter_leaf()` yields `(path, value)`
for every leaf. The root node appears at path `()`.

```python
for path, node in brainstate.graph.iter_node(model):
    print(path, type(node).__name__)

for path, leaf in brainstate.graph.iter_leaf(model):
    print(path, type(leaf).__name__)
```

Use collection when the result should be a path-keyed mapping. `nodes()`
collects graph nodes as `FlattedDict` object(s); `states()` collects
`State` objects as `FlattedDict` object(s). Both accept optional filters.

```python
all_nodes = brainstate.graph.nodes(model)
all_states = brainstate.graph.states(model)
param_states = brainstate.graph.states(model, brainstate.ParamState)
params, remaining = brainstate.graph.states(
    model,
    brainstate.ParamState,
    ...,
)
```

A predicate filter has the exact shape below. `path` is a tuple of hashable,
comparable keys and `value` is the value at that path.

```python
def weight_path(path: tuple, value) -> bool:
    return path[-1:] == ("weight",)

weight_states = brainstate.graph.states(model, weight_path)
```

The filter DSL also accepts:

- `...` or `True`: match every value.
- `None` or `False`: match no values.
- A type: match instances of that type, or values whose `type` attribute is
  that type.
- A string: match a value whose string `tag` attribute equals that string.
- A tuple or list of filters: match any enclosed filter.

Source URL:
https://brainx.chaobrain.com/brainstate/how_to/inspect_and_edit_state_graph.html

## Extract State Objects or State PyTrees

Choose the extraction API by the representation required downstream:

- `states()` returns the graph's `State` objects, keyed by path.
- `treefy_states()` returns one or more treeified state mappings and no
  `GraphDef`. It is the lightweight extraction for optimization or
  checkpointing when reconstruction is not required.
- `pop_states()` removes and returns matching `State` objects. It is
  destructive and deduplicates matches by identity.

```python
# State objects keyed by their graph paths.
live_params = brainstate.graph.states(model, brainstate.ParamState)

# PyTree-compatible state data without carrying graph structure.
param_tree = brainstate.graph.treefy_states(model, brainstate.ParamState)
param_tree, other_tree = brainstate.graph.treefy_states(
    model,
    brainstate.ParamState,
    ...,
)

# Destructive extraction: matching State instances are removed from model.
removed_params = brainstate.graph.pop_states(model, brainstate.ParamState)
```

`treefy_states()` returns state trees keyed by state paths. Do not use it alone
when another process must rebuild the module; use `treefy_split()` and retain
the returned `GraphDef` instead.

Source URLs:
https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html
https://brainx.chaobrain.com/brainstate/apis/graph.html

## Split Structure from State

`treefy_split()` returns `(graph_def, state_tree1, state_tree2, ...)`. Each
filter adds one state mapping in the same order; `...` is the catch-all used for
remaining states. Use this operation when reconstruction or explicit JAX state
threading requires both structure and dynamic values.

```python
graph_def, params, other_states = brainstate.graph.treefy_split(
    model,
    brainstate.ParamState,
    ...,
)
```

When no partition is needed, the operation returns one state mapping:

```python
graph_def, all_state = brainstate.graph.treefy_split(model)
```

If only the static structure is needed, retrieve it directly:

```python
graph_def = brainstate.graph.graphdef(model)
```

Source URLs:
https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html
https://brainx.chaobrain.com/brainstate/how_to/inspect_and_edit_state_graph.html

## Replace State Values in Place

`update_states()` updates a graph node in place from one or more state mappings.
Extract the desired subset, produce a replacement mapping with the same paths,
then apply it. The documented JAX workflow maps arithmetic over the treeified
state data; the same representation can be written back with `update_states()`.

```python
import jax
import jax.numpy as jnp

params = brainstate.graph.treefy_states(model, brainstate.ParamState)
replacement_params = jax.tree.map(jnp.zeros_like, params)
brainstate.graph.update_states(model, replacement_params)
```

Multiple independently partitioned mappings may be applied together:

```python
brainstate.graph.update_states(model, replacement_params, replacement_counts)
```

This is state-value replacement, not structural replacement. `pop_states()`
removes matching states; `update_states()` updates states already located by
the supplied mapping. The assigned official pages do not document a generic
graph-node replacement API.

Source URLs:
https://brainx.chaobrain.com/brainstate/apis/graph.html
https://brainx.chaobrain.com/brainstate/how_to/inspect_and_edit_state_graph.html

## Reconstruct the Graph

Use `treefy_merge()` with the `GraphDef` and every state mapping returned by
the corresponding `treefy_split()` call. It reconstructs a new graph node;
the state mappings must be supplied in the compatible partitioning.

```python
rebuilt_model = brainstate.graph.treefy_merge(
    graph_def,
    params,
    other_states,
)
```

For the lower-level, unpartitioned representation, `flatten()` returns a
`(GraphDef, NestedDict)` pair and `unflatten()` reconstructs from that pair.
The decoder materializes states, creates node shells, then fills them, so graph
sharing and cycles resolve independently of fill order.

```python
flat_graph_def, nested_state = brainstate.graph.flatten(model)
rebuilt_model = brainstate.graph.unflatten(flat_graph_def, nested_state)
```

For graph nodes nested inside a larger pytree, `graph_to_tree()` converts the
container to a pure pytree and `tree_to_graph()` converts a pytree of
`NodeStates` back into graph nodes. `NodeStates` is the JAX pytree wrapper that
carries a `GraphDef` and one or more state mappings.

```python
tree_form = brainstate.graph.graph_to_tree(container_with_graph_nodes)
restored_container = brainstate.graph.tree_to_graph(tree_form)
```

Use `split_context` and `merge_context` only when splitting or merging multiple
graph nodes that share a reference index; those context managers preserve the
shared index across the related operations. For a deep copy of one graph,
`clone()` performs split/merge while preserving shared references.

Source URL:
https://brainx.chaobrain.com/brainstate/apis/graph.html

## Selection Rules

- Need paths plus graph objects: `iter_node()`, `iter_leaf()`, `nodes()`, or
  `states()`.
- Need only a parameter/state PyTree: `treefy_states()`.
- Need filtered state PyTrees and later reconstruction: `treefy_split()` plus
  `treefy_merge()`.
- Need one low-level graph IR and nested state mapping: `flatten()` plus
  `unflatten()`.
- Need to update an existing graph: `update_states()`.
- Need destructive removal: `pop_states()`.
- Need graph nodes to cross a larger pytree boundary: `graph_to_tree()` plus
  `tree_to_graph()`.
