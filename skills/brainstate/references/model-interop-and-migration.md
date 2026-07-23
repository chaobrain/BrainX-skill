# BrainState Model Interoperation and Migration

Use this reference after the main BrainState skill when a task must move registered weighted layers between BrainState, Flax NNX, Flax Linen, or Equinox, or translate a PyTorch workflow into BrainState equivalents. It assumes the main skill's `State`, `Module`, parameter-collection, randomness, and state-aware transformation model; this file covers only interoperability and migration deltas.

## Interop Contract and API Surface

Source: https://brainx.chaobrain.com/brainstate/apis/interop.html

`brainstate.interop` converts standard-layer models between `brainstate.nn` and `flax.nnx`, `flax.linen`, or `equinox`. The public functions construct an architecturally equivalent target model and transfer weights. For supported single layers and `Sequential` stacks, conversion guarantees numerical output equivalence; unsupported layers or structures raise informative errors instead of producing a silently wrong model.

Direction is encoded in each function name:

| Function | Exact role |
|---|---|
| `from_nnx(model, *, sample_input)` | Convert a `flax.nnx` model into an equivalent `brainstate.nn` model. |
| `to_nnx(model, *, rngs)` | Convert a `brainstate.nn` model into an equivalent `flax.nnx` model. |
| `from_linen(module, params, *, sample_input)` | Convert a `flax.linen` module and parameters into an equivalent `brainstate.nn` model. |
| `to_linen(model)` | Convert a `brainstate.nn` model into a Flax Linen `(module, params)` pair. |
| `from_equinox(model, *, sample_input)` | Convert an Equinox model into an equivalent `brainstate.nn` model. |
| `to_equinox(model, *, key)` | Convert a `brainstate.nn` model into an equivalent Equinox model. |

The layer-mapping registry drives conversion:

- `LayerMapping` describes a bidirectional conversion mapping for one layer type.
- `register_layer_mapping` registers or overrides a `LayerMapping` in both directions. Supply the to/from conversion functions to add a custom layer type; built-in mappings use the same mechanism.
- `supported_layers` lists the BrainState layer types with registered conversions.

Source: https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html

Inspect support before selecting an interop path:

```python
layers = interop.supported_layers()
for framework, names in layers.items():
    print(f'{framework:8} : {", ".join(names)}')
```

The official example reports:

```text
nnx      : BatchNorm1d, BatchNorm2d, BatchNorm3d, Conv1d, Conv2d, Conv3d, Dropout, Embedding, GroupNorm, LSTMCell, LayerNorm, Linear, RMSNorm
linen    : BatchNorm1d, BatchNorm2d, BatchNorm3d, Conv1d, Conv2d, Conv3d, Dropout, Embedding, GroupNorm, LSTMCell, LayerNorm, Linear, RMSNorm
equinox  : Conv1d, Conv2d, Conv3d, Dropout, Embedding, GroupNorm, LSTMCell, LayerNorm, Linear, RMSNorm
```

This support list is version-specific; call `supported_layers()` in the installed environment rather than assuming the rendered list is current.

### Convertible boundary

Source: https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html

Conversion operates on registered, weight-bearing layers and `Sequential` stacks of them.

- Activation functions are not layers in this conversion model. A weightless nonlinearity such as ReLU stays in the caller's forward code; conversion rebuilds the weighted structure.
- Custom forward logic is not convertible. Branching, skip connections, or hand-written control flow cannot be mechanically rebuilt; only single registered layers and `Sequential` stacks round-trip.

Use this exact convertible shape for the examples below:

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx

import brainstate
from brainstate import interop

brainstate.random.seed(0)


def make_model():
    return brainstate.nn.Sequential(
        brainstate.nn.Linear(4, 8),
        brainstate.nn.LayerNorm(8),
        brainstate.nn.Linear(8, 2),
    )


x = brainstate.random.randn(3, 4)
```

### Conversion failures

Source: https://brainx.chaobrain.com/brainstate/apis/interop.html

Handle the documented error hierarchy rather than catching a generic failure without diagnosis:

| Error | Documented condition |
|---|---|
| `InteropError` | Base class for all `brainstate.interop` errors. |
| `MissingDependencyError` | The optional Flax or Equinox dependency is not installed. |
| `UnmappedLayerError` | No conversion mapping is registered for a leaf layer type. |
| `UnsupportedLayerError` | A known layer type is deliberately unsupported in this version. |
| `UnsupportedStructureError` | A container's forward logic cannot be reconstructed. |
| `MissingShapeError` | A spatial layer such as Conv or BatchNorm is imported without a sample input. |
| `ConversionError` | Weight transfer fails because of a shape, dtype, or unit mismatch. |

## Flax NNX Round Trip

Source: https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html

`to_nnx` needs `nnx.Rngs` to construct foreign layers; their weights are then overwritten with converted values. `from_nnx` converts in the reverse direction. Bundle conversion with an output-equivalence check:

```python
model = make_model()
reference = model(x)

# BrainState -> NNX
nnx_model = interop.to_nnx(model, rngs=nnx.Rngs(0))
print('to_nnx output matches  :', bool(jnp.allclose(reference, nnx_model(x), atol=1e-5)))

# NNX -> BrainState
back = interop.from_nnx(nnx_model)
print('from_nnx output matches:', bool(jnp.allclose(reference, back(x), atol=1e-5)))
```

Expected result:

```text
to_nnx output matches  : True
from_nnx output matches: True
```

## Flax Linen Round Trip

Source: https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html

Linen separates the module definition from parameters, so `to_linen` returns `(module, params)` and execution uses `module.apply(params, x)`. `from_linen` requires both objects:

```python
model = make_model()
reference = model(x)

# BrainState -> Linen
linen_module, params = interop.to_linen(model)
print('to_linen output matches  :', bool(jnp.allclose(reference, linen_module.apply(params, x), atol=1e-5)))

# Linen -> BrainState
back = interop.from_linen(linen_module, params)
print('from_linen output matches:', bool(jnp.allclose(reference, back(x), atol=1e-5)))
```

Expected result:

```text
to_linen output matches  : True
from_linen output matches: True
```

## Equinox Round Trip

Source: https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html

`to_equinox` accepts an optional PRNG `key` for constructing foreign layers. The exported Equinox module operates on one example, so batch its call with `jax.vmap`; `from_equinox` returns a batched BrainState model:

```python
model = make_model()
reference = model(x)

# BrainState -> Equinox  (call per-example, so vmap over the batch)
eqx_model = interop.to_equinox(model, key=jax.random.PRNGKey(0))
eqx_out = jax.vmap(eqx_model)(x)
print('to_equinox output matches  :', bool(jnp.allclose(reference, eqx_out, atol=1e-5)))

# Equinox -> BrainState
back = interop.from_equinox(eqx_model)
print('from_equinox output matches:', bool(jnp.allclose(reference, back(x), atol=1e-5)))
```

Expected result:

```text
to_equinox output matches  : True
from_equinox output matches: True
```

## Import Spatial Layers with `sample_input`

Source: https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html

Importing a convolution or spatial normalization requires its input shape because BrainState materializes the layer input size up front. Pass one unbatched example or its shape through `sample_input` to `from_nnx`, `from_linen`, or `from_equinox`:

```python
conv = nnx.Conv(in_features=3, out_features=4, kernel_size=(3, 3), rngs=nnx.Rngs(0))
bst_conv = interop.from_nnx(conv, sample_input=(8, 8, 3))   # H, W, C (no batch dim)
image = brainstate.random.randn(2, 8, 8, 3)
print('converted conv output shape:', bst_conv(image).shape)
```

Expected result:

```text
converted conv output shape: (2, 8, 8, 4)
```

## PyTorch-to-BrainState Migration Deltas

Source: https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html

The official PyTorch route is a concept-by-concept, module-at-a-time port, not one of the `brainstate.interop` converters. Use the main BrainState skill for the underlying APIs and this mapping only to translate PyTorch responsibilities:

| PyTorch | BrainState migration target | Operational delta |
|---|---|---|
| `torch.Tensor` | `jax.Array` / `jnp.ndarray` | Use `jax.numpy` semantics. |
| `nn.Module` | `brainstate.nn.Module` | Replace the model container; define the relevant State attributes. |
| `nn.Parameter` | `brainstate.ParamState` | Retrieve trainable weights through `.states(brainstate.ParamState)`. |
| `autograd.grad` / `backward()` | `brainstate.transform.grad` | Explicitly select State or arguments that receive gradients. |
| `torch.optim` optimizers | `braintools.optim` (optional) | Register and update the selected State tree. |
| `torch.jit.script` / `torch.jit.trace` | `brainstate.transform.jit` | Use the BrainState transform for pure or stateful functions. |
| `state_dict()` / `load_state_dict()` | `brainstate.graph.treefy_states` / `brainstate.graph.update_states` | Serialize and restore State trees. |
| `torch.manual_seed` | `brainstate.random.seed` / `RandomState` | Use BrainState's JAX-key-backed random interface. |

### Port optimizer ownership

Source: https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html

PyTorch optimizers receive an implicit parameter list; `braintools.optim` operates on the explicitly selected State tree. The source's migration pattern is:

```python
import braintools.optim as optim

params = model.states(brainstate.ParamState)
optimizer = optim.SGD(lr=1e-1)
optimizer.register_trainable_weights(params)

# After brainstate.transform.grad returns grads for params:
optimizer.update(grads)
```

`params.to_flat()` mirrors the role of `state_dict()` when inspecting the parameter paths.

### Port `state_dict()` save and restore

Source: https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html

Bundle the `state_dict()` / `load_state_dict()` replacement as one round trip:

```python
state_tree = brainstate.graph.treefy_states(model)
print('stored keys:', list(state_tree.to_flat().keys()))

# Later (or in another process):
restored = LinearModel(1, 1)
brainstate.graph.update_states(restored, state_tree)
print('restored weight:', restored.weight.value)
```

For file-backed parameter persistence, the same source gives:

```python
import braintools.file

params = model.states(brainstate.ParamState)
braintools.file.msgpack_save('example.msgpack', params, verbose=False)
restored = braintools.file.msgpack_load('example.msgpack', target=params, verbose=False)
print('parameters saved and restored via msgpack')
```

The source assigns the return from `msgpack_load` to `restored`.

### Port gradients for model State plus an explicit argument

Source: https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html

PyTorch's manual `requires_grad` selection maps to explicit `grad_states` and `argnums`. This source example differentiates with respect to both the model parameter tree and `alpha`:

```python
scale = jnp.array(0.1)


def scaled_loss(alpha, inputs, targets):
    preds = model(inputs)
    mse = jnp.mean((preds - targets) ** 2)
    return mse + alpha * jnp.sum(model.weight.value ** 2)


(grads_state, alpha_grad), loss_val = grad(
    scaled_loss,
    grad_states=params,
    argnums=0,
    return_value=True,
)(scale, x_train, y_train)

print('loss:', float(loss_val))
print('grad w.r.t alpha:', float(alpha_grad))
for path, g in grads_state.items():
    print(path, g.shape)
```

Here `argnums=0` selects the explicit `alpha` argument while `grad_states=params` selects the model State tree; do not replace one with the other during a port.

## Source Pages

- `brainstate.interop` module: https://brainx.chaobrain.com/brainstate/apis/interop.html
- Interoperate with Flax and Equinox: https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html
- Migrating Concepts from PyTorch to BrainState: https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html
