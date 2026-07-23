# BrainState State Collections and Utility Toolkit

Use this reference for the collection-focused parts of BrainState's Utility Toolkit: managing mappings, nested configuration, dictionary conversion, immutable structured PyTrees, declarative filters, and readable PyTree containers. These utilities organize data; the core `State` and `Module` programming model remains in the parent `SKILL.md`.

Official page: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

## Managing Collections with `DictManager`

Source URL: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

`DictManager` extends the standard mapping interface with filters, splits, combination operators, and JAX PyTree support. The page demonstrates `subset(...)` for one matching collection, `split(...)` for matching and remaining collections, and `map_values(...)` for transforming the retained values.

```python
from brainstate.util import DictManager

modules = DictManager({
    'encoder': {'params': 32},
    'decoder': {'params': 45},
    'dropout': 0.1,
})

submods = modules.subset(dict)
dicts, remainder = modules.split(dict)
param_counts = submods.map_values(lambda layer: layer['params'])

print(submods)
print(dicts)
print(remainder)
print(param_counts)
```

The demonstrated results retain `encoder` and `decoder` in `submods` and `dicts`, put only `dropout` in `remainder`, and produce `DictManager({'encoder': 32, 'decoder': 45})` for `param_counts`.

## Configuration Access with `DotDict`

Source URL: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

`DotDict` lets nested dictionaries act like lightweight objects while preserving conversion back to standard dictionaries when needed.

```python
from brainstate.util import DotDict

config = DotDict({
    'model': {
        'layers': 4,
        'hidden': 256,
    },
    'training': {
        'lr': 3e-4,
        'scheduler': {'warmup_steps': 500},
    },
})

print(config.model.hidden)
config.training.dropout = 0.2
round_trip = config.to_dict()
```

The example reads `config.model.hidden` as `256`, adds `config.training.dropout`, and converts the complete nested configuration back with `to_dict()`.

## Merge, Flatten, and Unflatten Dictionaries

Source URL: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

`merge_dicts` performs optional recursive merges. `flatten_dict` and `unflatten_dict` convert between nested and dotted-key representations, which the page uses for logging or CLI overrides.

```python
from brainstate.util import flatten_dict, merge_dicts, unflatten_dict

base = {'optimizer': {'lr': 1e-3, 'beta1': 0.9}}
override = {'optimizer': {'lr': 5e-4}, 'seed': 1234}

merged = merge_dicts(base, override)
flat = flatten_dict(merged)
restored = unflatten_dict(flat)

print(merged)
print(flat)
print(restored)
```

The recursive merge preserves `optimizer.beta1`, overrides `optimizer.lr`, and adds `seed`. The demonstrated flattened form is:

```python
{
    'optimizer.lr': 0.0005,
    'optimizer.beta1': 0.9,
    'seed': 1234,
}
```

`unflatten_dict(flat)` restores the nested representation.

## Structured PyTrees and Frozen Mappings

Source URL: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

The `struct` submodule mirrors Flax-friendly data structures. Its `dataclass` decorator registers classes as PyTrees, while `FrozenDict` provides immutable mappings compatible with JAX transformations.

```python
import jax
import jax.numpy as jnp
from brainstate.util import struct


@struct.dataclass
class LayerConfig:
    weight: jax.Array
    bias: jax.Array
    name: str = struct.field(pytree_node=False, default='layer')


cfg = LayerConfig(
    weight=jnp.ones((2, 2)),
    bias=jnp.zeros(2),
)
cfg2 = cfg.replace(weight=jnp.full((2, 2), 3.0))
flat_leaves, _ = jax.tree_util.tree_flatten(cfg)

frozen = struct.freeze({'encoder': jnp.arange(3)})
unfrozen = struct.unfreeze(frozen)
```

In the page's result, `cfg.replace(...)` returns a configuration whose weight is filled with `3.0`; flattening `cfg` yields array leaves with shapes `(2, 2)` and `(2,)`, so the `name` field marked `pytree_node=False` is not a dynamic leaf. `struct.freeze(...)` returns a `FrozenDict`, and `struct.unfreeze(...)` returns the ordinary dictionary again.

## Filtering Nested Objects

Source URL: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

`brainstate.util.filter` turns declarative filters into callables. The page combines tag, type, and path checks when traversing a nested object tree; each predicate is called with `(path, value)`.

```python
from typing import Any

import jax.numpy as jnp
from brainstate.util import filter as util_filter


class Module:
    def __init__(self, tag: str | None, kind: str):
        self.tag = tag
        self.kind = kind
        self.params = jnp.arange(2)


model_tree = {
    'encoder': Module(tag='trainable', kind='linear'),
    'decoder': Module(tag='frozen', kind='linear'),
    'head': Module(tag='trainable', kind='mlp'),
}

tag_filter = util_filter.to_predicate('trainable')
type_filter = util_filter.OfType(Module)
combined = util_filter.All(
    type_filter,
    util_filter.WithTag('trainable'),
)


def collect(
    tree: dict[str, Any],
    predicate,
) -> dict[str, Any]:
    out = {}
    for key, value in tree.items():
        if predicate((key,), value):
            out[key] = value
    return out


trainable_modules = collect(model_tree, tag_filter)
both = collect(model_tree, combined)

print(tuple(trainable_modules.keys()))
print(tuple(both.keys()))
```

Both predicates select `('encoder', 'head')` in the demonstrated tree. Use `to_predicate(...)` to normalize the tag filter, `OfType(...)` for the type check, `WithTag(...)` for tag matching, and `All(...)` to require both conditions.

## Pretty PyTree Containers and Tuple-Path Flattening

Source URL: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

`NestedDict`, `FlattedDict`, and `PrettyList` provide readable representations plus PyTree semantics. The page uses them to explore checkpoints or log structured configurations.

```python
import jax.numpy as jnp
from brainstate.util import NestedDict, PrettyList, flat_mapping, nest_mapping

state = NestedDict({
    'encoder': {
        'weight': jnp.ones((2, 2)),
        'bias': jnp.zeros(2),
    },
    'decoder': {
        'weight': jnp.eye(2),
    },
})
print(state)

flat_state = flat_mapping(state)
print(list(flat_state.keys()))

round_trip = nest_mapping(flat_state)
print(round_trip == state)

history = PrettyList([
    {'loss': 0.8},
    {'loss': 0.42},
])
print(history)
```

The displayed flat keys are tuple paths:

```python
[
    ('encoder', 'weight'),
    ('encoder', 'bias'),
    ('decoder', 'weight'),
]
```

`nest_mapping(flat_state)` round-trips to the nested mapping (`True` in the page), while `NestedDict` and `PrettyList` print their nested contents in an indented, readable form. This tuple-path representation is the one demonstrated by `flat_mapping`; the dictionary utilities section above separately demonstrates dotted-string keys from `flatten_dict`.
