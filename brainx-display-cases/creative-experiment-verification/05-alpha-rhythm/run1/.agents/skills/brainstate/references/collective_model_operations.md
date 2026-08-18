# BrainState collective operations

Use this reference when one operation must initialise, reset, batch, invoke a common method on, or restore stateful objects throughout a model without manually traversing its module graph. It assumes familiarity with `brainstate.nn` modules and states and basic JAX `vmap` usage; BrainState requires `brainunit`.

## Selection map

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

The official guide presents `brainstate.nn._collective_ops` through these public `brainstate.nn` utilities:

| Need | API | Key constraint |
|---|---|---|
| Fix the execution order of methods | `brainstate.nn.call_order` | Lower order values run first. |
| Call the same method on each model node | `brainstate.nn.call_all_fns`, `brainstate.nn.vmap_call_all_fns` | Filter nodes deliberately and verify the installed call signature. |
| Initialise state variables everywhere | `brainstate.nn.init_all_states`, `brainstate.nn.vmap_init_all_states` | Run after construction and before the first rollout. |
| Reset existing states everywhere | `brainstate.nn.reset_all_states`, `brainstate.nn.vmap_reset_all_states` | Reset at the intended sequence boundary; after vmapped reset, verify that every mapped dynamical State retains its leading lane axis. |
| Restore values keyed by absolute state paths | `brainstate.nn.assign_state_values` | Inspect both returned mismatch collections. |

## Ordering calls with `call_order`

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

## Initialising every module

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

## Resetting state between sequences

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

For recurrent models, initialise once and use `reset_all_states` after a sequence to automate the reset pass across the entire module.

```python
rnn = brainstate.nn.ValinaRNNCell(num_in=8, num_out=16)
brainstate.nn.init_all_states(rnn, batch_size=2)

# ... run inference or training for one sequence ...

brainstate.nn.reset_all_states(rnn)
```

As with `init_all_states`, reset can exclude nodes or receive additional arguments. `call_order` still governs the pass, allowing buffers to reset before hidden states when required.

## Batched lifecycle operations with `vmap_*`

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

API contract: https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.vmap_reset_all_states.html

For independent ensemble or Monte-Carlo instances, `vmap_init_all_states` inserts a leading axis and manages a separate random key for each copy. Pass a `state_to_exclude` filter when selected States, such as statistics buffers, must remain shared; excluded States retain their original shape.

```python
rnn = brainstate.nn.ValinaRNNCell(num_in=4, num_out=8)
axis_size = 8
brainstate.nn.vmap_init_all_states(rnn, axis_size=axis_size)

hidden_shapes = {
    path: state.value.shape
    for path, state in rnn.states(brainstate.HiddenState).items()
}
assert hidden_shapes
assert all(shape[0] == axis_size for shape in hidden_shapes.values())
```

The official `vmap_reset_all_states` contract intends to reset each lane independently, but the selected Module's `reset_state` implementation must still preserve the mapped State shape. Before selecting it for repeated rollouts, capture a direct-path snapshot, run one reset outside the transformed and timed path, and compare every mapped dynamical shape before and after. If any leading lane axis changes, restore the snapshot and use `assign_state_values` for subsequent exact resets instead of `vmap_reset_all_states`.

## Calling arbitrary methods collectively

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

## Restoring states with `assign_state_values`

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

`assign_state_values` maps values back to State objects by absolute State path and returns mismatched keys as `(unexpected, missing)`. Keep a dictionary-valued State intact under its State path; do not append its inner keys as if they were separate States.

```python
autoencoder = brainstate.nn.Sequential(
    brainstate.nn.Linear((16,), (8,)),
    brainstate.nn.ReLU(),
    brainstate.nn.Linear((8,), (16,)),
)
brainstate.nn.init_all_states(autoencoder)

state_snapshot = {
    path: state.value
    for path, state in autoencoder.states().items()
}

# ... modify weights or states ...

unexpected, missing = brainstate.nn.assign_state_values(
    autoencoder,
    state_snapshot,
)
if unexpected or missing:
    raise ValueError(
        f"checkpoint mismatch: unexpected={unexpected}, missing={missing}"
    )
```

Capture all States when exact whole-model restoration is required. A deliberately partial snapshot reports every omitted model State in `missing`; handle that result as an explicit partial-restore policy rather than ignoring it.

## Putting it all together

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

Use this order for a batched recurrent workflow: construct the model, call `vmap_init_all_states(..., axis_size=...)`, verify mapped dynamical shapes, capture all values under their existing State paths, run the transformed rollout, then either use a shape-verified `vmap_reset_all_states` or restore the exact snapshot with `assign_state_values`. Reject unexpected or missing paths before the next rollout.

## Best practices

Source: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

- Call `init_all_states` once after constructing a module.
- Decorate stateful methods with `call_order` when their interaction matters.
- Use `node_to_exclude` and `state_to_exclude` filters to fine-tune traversal.
- Inspect both return values from `assign_state_values` to catch mismatched checkpoints.
- Preserve dictionary-valued State under its existing absolute State path when taking a snapshot.
- Use vmapped helpers for ensembles while accounting for the added leading axis, and verify that reset preserves that axis for the selected Modules.
