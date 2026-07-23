# BrainState Collective Operations

Use this reference when one operation must initialise, reset, batch, invoke a common method on, or restore stateful objects throughout a model without manually traversing its module graph. It assumes familiarity with `brainstate.nn` modules and states and basic JAX `vmap` usage; BrainState requires `brainunit`.

## Overview of the API

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

The official guide presents `brainstate.nn._collective_ops` through these public `brainstate.nn` utilities:

| Operation | API |
|---|---|
| Fix the execution order of methods | `brainstate.nn.call_order` |
| Call the same method on each model node | `brainstate.nn.call_all_fns`, `brainstate.nn.vmap_call_all_fns` |
| Initialise state variables everywhere | `brainstate.nn.init_all_states`, `brainstate.nn.vmap_init_all_states` |
| Reset existing states everywhere | `brainstate.nn.reset_all_states`, `brainstate.nn.vmap_reset_all_states` |
| Restore values keyed by absolute state paths | `brainstate.nn.assign_state_values` |

## Ordering Calls with `call_order`

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

By default, `call_all_fns` respects graph node order. When method interactions require explicit ordering, `call_order` attaches a `call_order` attribute to a method; lower levels run first.

```python
import brainstate


class EncoderDecoder(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = brainstate.nn.Linear((16,), (32,))
        self.decoder = brainstate.nn.Linear((32,), (16,))

    @brainstate.nn.call_order(0)
    def init_state(self):
        self.encoder.init_state()
        self.decoder.init_state()

    @brainstate.nn.call_order(1)
    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
```

The decorators make collective utilities honour this order while visiting child modules.

## Initialising Every Module

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

`init_all_states` walks the module graph and calls `init_state` on each node. Pass keyword arguments through to the lifecycle methods, exclude nodes with `node_to_exclude`, or retain the returned target for chaining.

```python
model = brainstate.nn.Sequential(
    brainstate.nn.Linear((10,), (32,)),
    brainstate.nn.GELU(),
    brainstate.nn.Dropout(prob=0.1),
)

brainstate.nn.init_all_states(model, batch_size=4)
brainstate.nn.init_all_states(
    model,
    node_to_exclude=brainstate.nn.Dropout,
)

model = brainstate.nn.init_all_states(model)
```

## Resetting State Between Sequences

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

For recurrent models, initialise once and use `reset_all_states` after a sequence to automate the reset pass across the entire module.

```python
rnn = brainstate.nn.ValinaRNNCell(num_in=8, num_out=16)
brainstate.nn.init_all_states(rnn, batch_size=2)

# ... run inference or training for one sequence ...

brainstate.nn.reset_all_states(rnn)
```

As with `init_all_states`, reset can exclude nodes or receive additional arguments. `call_order` still governs the pass, allowing buffers to reset before hidden states when required.

## Batched Initialisation with `vmap_*`

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

For independent ensemble or Monte-Carlo instances, the vectorised variants insert a leading axis and manage a separate random key for each copy.

```python
policy = brainstate.nn.Sequential(
    brainstate.nn.Linear((4,), (64,)),
    brainstate.nn.GELU(),
    brainstate.nn.Linear((64,), (2,)),
)

brainstate.nn.vmap_init_all_states(policy, axis_size=8)

# ... run the batched rollout ...

brainstate.nn.vmap_reset_all_states(policy, axis_size=8)
```

Pass a `state_to_exclude` filter to `vmap_init_all_states` when selected states, such as statistics buffers, must remain shared. Excluded states retain their original shape across the batch.

## Calling Arbitrary Methods Collectively

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

`call_all_fns` is the primitive behind the init and reset helpers. It can dispatch another common method only when each participating child module implements that method. The guide illustrates the required common-method shape with `log_stats`:

```python
import jax.numpy as jnp


class LoggingLayer(brainstate.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = brainstate.nn.Linear((size,), (size,))
        self.logged = []

    def init_state(self):
        self.linear.init_state()

    def log_stats(self):
        weight = self.linear.weight.value['weight']
        self.logged.append(jnp.mean(weight))


net = brainstate.nn.Sequential(
    LoggingLayer(size=8),
    LoggingLayer(size=8),
)

brainstate.nn.init_all_states(net)
for layer in net.layers:
    layer.log_stats()
```

The page identifies `brainstate.nn.vmap_call_all_fns` as the corresponding operation for `axis_size` independent instances and says it shares the interface and filter options. It does not provide a concrete `call_all_fns` or `vmap_call_all_fns` invocation signature, so consult API help rather than inferring argument order from this guide.

## Restoring States with `assign_state_values`

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

`assign_state_values` maps dictionary values back to state objects by absolute state path and returns mismatched keys as `(unexpected, missing)`. The guide constructs paths for dictionary-valued states by appending each inner key:

```python
autoencoder = brainstate.nn.Sequential(
    brainstate.nn.Linear((16,), (8,)),
    brainstate.nn.ReLU(),
    brainstate.nn.Linear((8,), (16,)),
)
brainstate.nn.init_all_states(autoencoder)

state_snapshot = {}
for path, state in autoencoder.states().items():
    if isinstance(state.value, dict):
        for key, value in state.value.items():
            state_snapshot[path + (key,)] = value
    else:
        state_snapshot[path] = state.value

# ... modify weights or states ...

unexpected, missing = brainstate.nn.assign_state_values(
    autoencoder,
    state_snapshot,
)
if unexpected or missing:
    raise ValueError(
        f'checkpoint mismatch: unexpected={unexpected}, missing={missing}'
    )
```

The guide's own output reports flattened paths as unexpected and their parent state paths as missing. Therefore, the example demonstrates why both return collections must be inspected; it does not demonstrate a mismatch-free restore.

## Putting It All Together

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

The final guide example combines the earlier operations in this order: construct a `ValinaRNNCell`, call `vmap_init_all_states(..., axis_size=4)`, build the same absolute-path snapshot shown above, run a time-stepped rollout, call `vmap_reset_all_states(..., axis_size=4)`, and pass the snapshot to `assign_state_values`. It repeats the same dictionary-flattening restoration pattern and likewise produces unexpected and missing keys, so retain the mismatch check when adapting that lifecycle.

## Best Practices

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

- Call `init_all_states` once after constructing a module.
- Decorate stateful methods with `call_order` when their interaction matters.
- Use `node_to_exclude` and `state_to_exclude` filters to fine-tune traversal.
- Inspect both return values from `assign_state_values` to catch mismatched checkpoints.
- Use vmapped helpers for ensembles while accounting for the added leading axis.
