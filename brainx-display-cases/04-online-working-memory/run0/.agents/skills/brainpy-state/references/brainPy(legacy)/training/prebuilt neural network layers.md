# Prebuilt neural network layers

Use this reference to choose and compose legacy `brainpy.dnn` activations, convolutions, connection operators, normalization, pooling, interoperability wrappers, and utility layers. These are `brainpy.dnn` dynamical-system layers, not `brainpy.state` or `brainstate.nn` modules.

## Selection map

| Family | Use when | Key constraint |
|---|---|---|
| Activations | A layer graph needs a stored nonlinear transformation. | Set the normalization dimension explicitly for softmax-family layers. |
| Convolutions | Local kernels operate over one, two, or three spatial dimensions. | Legacy BrainPy convolution inputs place channels last. |
| Dense and connectivity layers | Features or neural activity must be routed through dense, explicit sparse, masked, or implicit random connectivity. | Preserve event semantics and connection structure instead of replacing every operator with `Dense`. |
| Normalization | Activations must be normalized over batch, layer, group, or instance statistics. | Match the documented axes and expected tensor layout. |
| Pooling | Spatial values must be reduced locally or to a target shape. | Match the class suffix to the number of spatial dimensions and preserve `channel_axis`. |
| Flax interoperability | A Flax module and a legacy BrainPy dynamical system must cross framework boundaries. | Use the wrapper in the direction required by the owning runtime. |
| Utilities | A graph needs dropout, reshaping, or a function adapter. | Dropout's `prob` is the probability to keep a value and is active only in training mode. |

## Activation layers

Use an activation class when the operation belongs in a stored BrainPy layer graph. Use `Activation(...)` to wrap a callable that has no dedicated class.

| API | Choose it when |
|---|---|
| `Activation(activate_fun, name=None, mode=None, **kwargs)` | Wrap an arbitrary activation callable and retain its keyword arguments. |
| `Threshold(threshold, value, inplace=False)` | Replace values on the thresholded side with a fixed value. |
| `ReLU(inplace=False)` | Apply the standard rectified linear unit. |
| `RReLU(lower=0.125, upper=1 / 3, inplace=False)` | Randomize the negative slope within the supplied bounds. |
| `Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)` | Clamp through a hard-tanh response. |
| `ReLU6(inplace=False)` | Use ReLU capped at six. |
| `Sigmoid(name=None, mode=None)` | Apply the logistic sigmoid. |
| `Hardsigmoid(inplace=False)` | Use the piecewise-linear sigmoid approximation. |
| `Tanh(name=None, mode=None)` | Apply hyperbolic tangent. |
| `SiLU(inplace=False)` | Apply the sigmoid linear unit. |
| `Mish(inplace=False)` | Apply Mish. |
| `Hardswish(inplace=False)` | Apply hard-swish. |
| `ELU(alpha=1.0, inplace=False)` | Apply an exponential linear unit with the selected negative scale. |
| `CELU(alpha=1.0, inplace=False)` | Apply the continuously differentiable ELU form. |
| `SELU(inplace=False)` | Apply scaled ELU in a self-normalizing architecture. |
| `GLU(dim=-1)` | Split the selected dimension and gate one half with the other; the dimension size must be even. |
| `GELU(approximate=False)` | Apply GELU and optionally select its approximation. |
| `Hardshrink(lambd=0.5)` | Apply hard shrinkage around the configured threshold. |
| `LeakyReLU(negative_slope=0.01, inplace=False)` | Preserve a fixed slope for negative values. |
| `LogSigmoid(name=None, mode=None)` | Return log-sigmoid values. |
| `Softplus(beta=1, threshold=20.0)` | Apply a smooth positive rectifier with explicit beta and numerical threshold. |
| `Softshrink(lambd=0.5)` | Apply soft shrinkage around the configured threshold. |
| `PReLU(num_parameters=1, init=0.25, dtype=None)` | Learn one negative slope globally or one slope per input channel. |
| `Softsign(name=None, mode=None)` | Apply the soft-sign response. |
| `Tanhshrink(name=None, mode=None)` | Apply tanh shrinkage. |
| `Softmin(dim=None)` | Normalize so smaller inputs receive larger probabilities. |
| `Softmax(dim=None)` | Normalize values into probabilities along `dim`. |
| `Softmax2d(name=None, mode=None)` | Apply softmax over features at each spatial location. |
| `LogSoftmax(dim=None)` | Return log-probabilities normalized along `dim`. |

**Invariant:** set `dim` from the actual feature layout for `Softmin`, `Softmax`, and `LogSoftmax`. A call can succeed on the wrong dimension while producing the wrong probability semantics.

## Convolutional layers

Choose the suffix from the number of spatial dimensions. Inputs use channel-last layouts such as `[H, W, C]` or `[B, H, W, C]` for `Conv2d`.

| API | Constructor core and use |
|---|---|
| `Conv1d` | `Conv1d(in_channels, out_channels, kernel_size, stride=None, padding='SAME', lhs_dilation=1, rhs_dilation=1, groups=1, ...)` for one spatial dimension. |
| `Conv2d` | `Conv2d(in_channels, out_channels, kernel_size, stride=None, padding='SAME', lhs_dilation=1, rhs_dilation=1, groups=1, ...)` for two spatial dimensions. |
| `Conv3d` | `Conv3d(in_channels, out_channels, kernel_size, stride=None, padding='SAME', lhs_dilation=1, rhs_dilation=1, groups=1, ...)` for three spatial dimensions. |
| `ConvTranspose1d` | `ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding='SAME', ...)` for learned one-dimensional upsampling. |
| `ConvTranspose2d` | `ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding='SAME', ...)` for learned two-dimensional upsampling. |
| `ConvTranspose3d` | `ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding='SAME', ...)` for learned three-dimensional upsampling. |

`Conv1D`, `Conv2D`, and `Conv3D` are aliases of `Conv1d`, `Conv2d`, and `Conv3d`. Prefer the lowercase final suffix used by the canonical class names.

The standard convolution family also accepts weight and bias initializers and an optional weight mask. Preserve `groups`, padding, stride, and dilation because they change the operator rather than merely tuning performance.

## Dense and connectivity layers

Choose a dense feature transform only when connectivity structure is not itself part of the model.

| API | Description |
|---|---|
| `Dense(num_in, num_out, W_initializer=..., b_initializer=..., mode=None, name=None)` | Apply a learned dense transform over the last input dimension. |
| `Linear` | Alias of `Dense`. |
| `Identity(*args, **kwargs)` | Keep a graph position without changing the input. |
| `AllToAll(num_pre, num_post, weight, sharding=None, include_self=True, mode=None, name=None)` | Route activity through all-to-all synaptic weights and explicitly choose self-connections. |
| `OneToOne(num, weight, sharding=None, mode=None, name=None)` | Apply one independent weight per matched pre/post index. |
| `MaskedLinear(conn, weight, mask_fun=..., sharding=None, mode=None, name=None)` | Materialize a connector as a masked dense computation. |
| `CSRLinear(conn, weight, sharding=None, mode=None, name=None, method=None, transpose=True)` | Apply ordinary values through explicit CSR sparse connectivity. |
| `EventCSRLinear(conn, weight, sharding=None, mode=None, name=None, transpose=True)` | Apply spike or event input through explicit CSR sparse connectivity. |
| `JitFPHomoLinear(num_in, num_out, prob, weight, seed=None, ..., transpose=False, atomic=False)` | Generate fixed-probability connectivity during multiplication with one homogeneous weight. |
| `JitFPUniformLinear(num_in, num_out, prob, w_low, w_high, seed=None, ..., transpose=False, atomic=False)` | Generate fixed-probability connectivity with uniformly distributed weights. |
| `JitFPNormalLinear(num_in, num_out, prob, w_mu, w_sigma, seed=None, ..., transpose=False, atomic=False)` | Generate fixed-probability connectivity with normally distributed weights. |
| `EventJitFPHomoLinear(num_in, num_out, prob, weight, seed=None, ..., transpose=False, atomic=True)` | Use event input with implicit fixed-probability connectivity and one weight. |
| `EventJitFPUniformLinear(num_in, num_out, prob, w_low, w_high, seed=None, ..., transpose=False, atomic=True)` | Use event input with implicit fixed-probability connectivity and uniform weights. |
| `EventJitFPNormalLinear(num_in, num_out, prob, w_mu, w_sigma, seed=None, ..., transpose=False, atomic=True)` | Use event input with implicit fixed-probability connectivity and normal weights. |

**Invariant:** `MaskedLinear`, CSR layers, and JIT fixed-probability layers encode different storage and execution choices. Select from the known connectivity representation and input type; converting all of them to `Dense` changes memory use and may change event semantics.

## Normalization layers

Use batch normalization when batch statistics and running statistics are required; use layer, group, or instance normalization when the reduction domain follows feature structure instead.

| API | Description |
|---|---|
| `BatchNorm1d(num_features, axis=(0, 1), epsilon=1e-5, momentum=0.99, affine=True, ..., mode=None)` | Normalize `[B, L, C]` data while excluding the channel axis from the reduction. |
| `BatchNorm2d(num_features, axis=(0, 1, 2), epsilon=1e-5, momentum=0.99, affine=True, ..., mode=None)` | Normalize batch plus two spatial dimensions for channel-last data. |
| `BatchNorm3d(num_features, axis=(0, 1, 2, 3), epsilon=1e-5, momentum=0.99, affine=True, ..., mode=None)` | Normalize batch plus three spatial dimensions for channel-last data. |
| `LayerNorm(normalized_shape, epsilon=1e-5, ..., elementwise_affine=True, mode=None, name=None)` | Normalize the supplied trailing shape for each example. |
| `GroupNorm(num_groups, num_channels, epsilon=1e-5, affine=True, ..., mode=None, name=None)` | Partition channels into the requested number of groups. |
| `InstanceNorm(num_channels, epsilon=1e-5, affine=True, ..., mode=None, name=None)` | Normalize each instance independently over its non-channel dimensions. |

`BatchNorm1D`, `BatchNorm2D`, and `BatchNorm3D` are aliases of their lowercase-suffix counterparts.

Batch-normalization behavior depends on computation mode. Keep training and inference modes explicit when running the same layer in both workflows, and preserve any `axis_name` configuration used to aggregate statistics across mapped replicas.

## Pooling layers

Use local pooling for a fixed window and adaptive pooling when the output spatial shape is specified directly.

| API | Description |
|---|---|
| `MaxPool(kernel_size, stride=1, padding='VALID', channel_axis=None, mode=None, name=None)` | Take local maxima for a general window specification. |
| `MaxPool1d(kernel_size, stride=None, padding='VALID', channel_axis=-1, mode=None, name=None)` | Apply one-dimensional max pooling. |
| `MaxPool2d(kernel_size, stride=None, padding='VALID', channel_axis=-1, mode=None, name=None)` | Apply two-dimensional max pooling. |
| `MaxPool3d(kernel_size, stride=None, padding='VALID', channel_axis=-1, mode=None, name=None)` | Apply three-dimensional max pooling. |
| `MinPool(kernel_size, stride=1, padding='VALID', channel_axis=None, mode=None, name=None)` | Take local minima. |
| `AvgPool(kernel_size, stride=1, padding='VALID', channel_axis=None, mode=None, name=None)` | Take local averages for a general window specification. |
| `AvgPool1d(kernel_size, stride=1, padding='VALID', channel_axis=-1, mode=None, name=None)` | Apply one-dimensional average pooling. |
| `AvgPool2d(kernel_size, stride=1, padding='VALID', channel_axis=-1, mode=None, name=None)` | Apply two-dimensional average pooling. |
| `AvgPool3d(kernel_size, stride=1, padding='VALID', channel_axis=-1, mode=None, name=None)` | Apply three-dimensional average pooling. |
| `AdaptiveAvgPool1d(target_shape, channel_axis=-1, name=None, mode=None)` | Average-pool one spatial dimension to `target_shape`. |
| `AdaptiveAvgPool2d(target_shape, channel_axis=-1, name=None, mode=None)` | Average-pool two spatial dimensions to `target_shape`. |
| `AdaptiveAvgPool3d(target_shape, channel_axis=-1, name=None, mode=None)` | Average-pool three spatial dimensions to `target_shape`. |
| `AdaptiveMaxPool1d(target_shape, channel_axis=-1, name=None, mode=None)` | Max-pool one spatial dimension to `target_shape`. |
| `AdaptiveMaxPool2d(target_shape, channel_axis=-1, name=None, mode=None)` | Max-pool two spatial dimensions to `target_shape`. |
| `AdaptiveMaxPool3d(target_shape, channel_axis=-1, name=None, mode=None)` | Max-pool three spatial dimensions to `target_shape`. |

## Flax interoperability

Keep ownership clear when crossing the framework boundary.

| API | Description |
|---|---|
| `FromFlax(flax_module, *module_args, **module_kwargs)` | Wrap a Flax module as a legacy BrainPy `DynamicalSystem`. |
| `ToFlaxRNNCell(*args, **kwargs)` | Wrap a BrainPy `DynamicalSystem` as a Flax recurrent module. |
| `ToFlax` | Alias of `ToFlaxRNNCell`. |

## Utility layers

| API | Description |
|---|---|
| `Dropout(prob, mode=None, name=None)` | Keep each value with probability `prob`; during training, scale survivors by `1 / prob`, and outside training mode act as a no-op. |
| `Flatten(start_dim=0, end_dim=-1, name=None, mode=None)` | Flatten a contiguous dimension range; account for BrainPy batching mode when choosing `start_dim`. |
| `Unflatten(dim, sizes, mode=None, name=None)` | Expand one dimension into `sizes`, whose product must match the original dimension. |
| `FunAsLayer(fun, name=None, mode=None, **kwargs)` | Wrap an ordinary callable as a BrainPy layer and retain its keyword arguments. |

**Invariant:** the documented `Dropout.prob` is a keep probability, not a drop rate. Do not translate it directly from an API whose parameter means the probability to discard.

## Canonical composition

This example keeps the legacy channel-last layout explicit from convolution through pooling and dense readout:

```python
import brainpy as bp
import brainpy.math as bm

conv = bp.dnn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=(3, 3),
    padding="SAME",
)
activation = bp.dnn.ReLU()
pool = bp.dnn.MaxPool2d(kernel_size=2, stride=2, channel_axis=-1)

x = bm.ones((4, 28, 28, 1))
features = pool(activation(conv(x)))
flat = bm.reshape(features, (features.shape[0], -1))
readout = bp.dnn.Dense(num_in=flat.shape[-1], num_out=10)
logits = readout(flat)

assert features.shape == (4, 14, 14, 8)
assert logits.shape == (4, 10)
```

Use a connection-specific linear layer instead of `Dense` when the input represents neural events or when explicit sparse, masked, or implicit random connectivity is part of the model.

## Sources

- https://brainpy.readthedocs.io/apis/dnn.html
