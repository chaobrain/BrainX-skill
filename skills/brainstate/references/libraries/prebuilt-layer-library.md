# Prebuilt layer library

Use this reference to choose a prebuilt `brainstate.nn` layer family and identify
the constructor controls that change model behavior. Keep ordinary composition
in `SKILL.md`; open `../size-inference-variations.md` for output-size formulas,
`prebuilt-activation-library.md` for activations, and
`../brainstate/randomness-and-reproducibility.md` for stochastic replay.

## Selection map

| Family | Use when | Key constraint |
|---|---|---|
| Linear and connectivity | Features must be mixed densely, sparsely, by sign, one-to-one, all-to-all, or through low-rank adaptation. | Preserve the requested connectivity structure instead of replacing it with a dense matrix. |
| Convolution | Local kernels operate over one, two, or three spatial dimensions. | Match the class suffix and `channel_first` setting to the input layout. |
| Normalization | Activations or weights must be normalized by batch, layer, RMS, group, or weight statistics. | Choose reduction and feature axes from the actual tensor layout. |
| Pooling and reshaping | Spatial values must be reduced, restored from max indices, or rearranged. | Local pooling and adaptive pooling solve different output-size requirements. |
| Padding | A boundary condition must be explicit before another spatial operation. | Reflection, replication, circular, constant, and zero padding have different boundary semantics. |
| Dropout | Training requires elementwise, channelwise, alpha, or time-fixed stochastic masking. | Fitting mode and RNG state determine whether and how the mask is applied. |

The forms below show public arguments that affect selection. Open the generated
class page from the linked family catalog for version-specific initializer
objects, State subclasses, dtypes, names, and full defaults.

## Linear and connectivity layers

| API | Constructor core | Choose it when |
|---|---|---|
| `Linear` | `Linear(in_size, out_size, ...)` | Apply a standard dense linear transformation. |
| `ScaledWSLinear` | `ScaledWSLinear(in_size, out_size, ..., ws_gain=True, eps=1e-4)` | Standardize linear weights during the forward computation. |
| `SignedWLinear` | `SignedWLinear(in_size, out_size, ..., w_sign=None)` | Enforce a supplied sign pattern over absolute weights. |
| `SparseLinear` | `SparseLinear(spar_mat, ..., in_size=None)` | Use an existing sparse weight matrix. |
| `LoRA` | `LoRA(in_features, lora_rank, out_features, *, base_module=None, ...)` | Add a low-rank trainable update, optionally around a base Module. |
| `AllToAll` | `AllToAll(in_size, out_size, ..., include_self=True)` | Connect every input and output, optionally excluding self-connections. |
| `OneToOne` | `OneToOne(in_size, ...)` | Give each feature or unit its own scalar connection. |

**Invariant:** `SparseLinear`, `LoRA`, `AllToAll`, and `OneToOne` encode model
structure. Replacing them with `Linear` changes parameterization and often the
scientific meaning.

## Convolutional layers

Choose `1d`, `2d`, or `3d` from the number of spatial axes, excluding batch and
channel axes.

| Exact classes | Constructor family | Choose it when |
|---|---|---|
| `Conv1d`, `Conv2d`, `Conv3d` | `ConvNd(in_size, out_channels, kernel_size, stride=1, padding='SAME', lhs_dilation=1, rhs_dilation=1, groups=1, ..., channel_first=False)` | Apply a standard convolution. |
| `ScaledWSConv1d`, `ScaledWSConv2d`, `ScaledWSConv3d` | `ScaledWSConvNd(..., ws_gain=True, eps=1e-4, channel_first=False)` | Apply convolution with scaled weight standardization. |
| `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` | `ConvTransposeNd(in_size, out_channels, kernel_size, stride=1, padding='SAME', rhs_dilation=1, groups=1, ..., channel_first=False)` | Apply learned transposed-convolution upsampling. |

Do not pass `lhs_dilation` to the transposed-convolution family; it is not part
of its documented constructor. Open `../size-inference-variations.md` before
predicting shapes affected by stride, padding, dilation, or groups.

## Normalization layers

| API | Constructor core | Choose it when |
|---|---|---|
| `BatchNorm0d`, `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d` | `BatchNormNd(in_size, feature_axis=-1, *, track_running_stats=True, epsilon=1e-5, momentum=0.99, affine=True, ...)` | Normalize with batch statistics and optionally maintain running statistics. |
| `LayerNorm` | `LayerNorm(in_size, reduction_axes=-1, feature_axes=-1, *, epsilon=1e-6, use_bias=True, use_scale=True, ...)` | Normalize selected axes independently for each example. |
| `RMSNorm` | `RMSNorm(in_size, *, epsilon=1e-6, use_scale=True, reduction_axes=-1, feature_axes=-1, ...)` | Normalize by root mean square without a bias control. |
| `GroupNorm` | `GroupNorm(in_size, feature_axis=-1, num_groups=32, group_size=None, *, epsilon=1e-6, ...)` | Partition channels into groups; specify `num_groups` or `group_size` coherently. |
| `weight_standardization` | `weight_standardization(w, eps=1e-4, gain=None, out_axis=-1)` | Standardize a weight array directly rather than selecting a complete layer. |

Batch normalization has separate ranks because accepted layouts differ. Do not
choose a suffix from the total array rank without accounting for batch and
feature axes.

## Pooling and reshaping

| Exact APIs | Constructor family | Choose it when |
|---|---|---|
| `Flatten` | `Flatten(start_axis=0, end_axis=-1, in_size=None)` | Collapse a contiguous range of runtime axes. |
| `Unflatten` | `Unflatten(axis, sizes, name=None, in_size=None)` | Expand one axis into the supplied sizes. |
| `AvgPool1d`, `AvgPool2d`, `AvgPool3d` | `AvgPoolNd(kernel_size, stride=1, padding='VALID', channel_axis=-1, ...)` | Reduce local windows by their mean. |
| `MaxPool1d`, `MaxPool2d`, `MaxPool3d` | `MaxPoolNd(kernel_size, stride=None, padding='VALID', channel_axis=-1, return_indices=False, ...)` | Reduce local windows by their maximum; request indices when unpooling follows. |
| `LPPool1d`, `LPPool2d`, `LPPool3d` | `LPPoolNd(norm_type, kernel_size, stride=None, padding='VALID', channel_axis=-1, ...)` | Apply local Lp pooling with an explicit norm order. |
| `MaxUnpool1d`, `MaxUnpool2d`, `MaxUnpool3d` | `MaxUnpoolNd(kernel_size, stride=None, padding=0, channel_axis=-1, ...)` | Place pooled maxima back using compatible max-pool indices. |
| `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d` | `AdaptiveAvgPoolNd(target_size, channel_axis=-1, ...)` | Produce a fixed target size with average pooling. |
| `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d` | `AdaptiveMaxPoolNd(target_size, channel_axis=-1, ...)` | Produce a fixed target size with max pooling. |

**Invariant:** `MaxUnpool*` is only a partial inverse. Preserve indices from a
compatible `MaxPool*` call and do not expect non-maximal values to be recovered.

## Padding layers

Choose the boundary behavior first, then the `1d`, `2d`, or `3d` suffix.

| Exact class families | Constructor family | Boundary behavior |
|---|---|---|
| `ReflectionPad1d`, `ReflectionPad2d`, `ReflectionPad3d` | `ReflectionPadNd(padding, in_size=None, name=None)` | Mirror values at the boundary. |
| `ReplicationPad1d`, `ReplicationPad2d`, `ReplicationPad3d` | `ReplicationPadNd(padding, in_size=None, name=None)` | Repeat edge values. |
| `ZeroPad1d`, `ZeroPad2d`, `ZeroPad3d` | `ZeroPadNd(padding, in_size=None, name=None)` | Insert zeros. |
| `ConstantPad1d`, `ConstantPad2d`, `ConstantPad3d` | `ConstantPadNd(padding, value=0, in_size=None, name=None)` | Insert an explicit constant. |
| `CircularPad1d`, `CircularPad2d`, `CircularPad3d` | `CircularPadNd(padding, in_size=None, name=None)` | Wrap values from the opposite boundary. |

## Dropout layers

| API | Constructor core | Choose it when |
|---|---|---|
| `Dropout` | `Dropout(prob=0.5, broadcast_dims=(), name=None)` | Mask individual values, optionally sharing the mask across selected dimensions. |
| `Dropout1d`, `Dropout2d`, `Dropout3d` | `DropoutNd(prob=0.5, channel_axis=-1, name=None)` | Mask complete feature channels for the matching spatial rank. |
| `AlphaDropout` | `AlphaDropout(prob=0.5, name=None)` | Preserve self-normalizing-network statistics with elementwise alpha dropout. |
| `FeatureAlphaDropout` | `FeatureAlphaDropout(prob=0.5, channel_axis=-1, name=None)` | Apply alpha dropout to complete channels. |
| `DropoutFixed` | `DropoutFixed(in_size, prob=0.5, name=None)` | Reuse a mask along the time axis. |

Open `../brainstate/randomness-and-reproducibility.md` when dropout must use an
independent stream, replay exactly, map across an ensemble, or survive a
checkpoint.

## Representative composition

This example makes layout, spatial rank, normalization axis, and fixed output
size explicit:

```python
import brainstate.nn as nn

feature_extractor = nn.Sequential(
    nn.Conv2d(
        in_size=(28, 28, 1),
        out_channels=16,
        kernel_size=3,
        padding="SAME",
        channel_first=False,
    ),
    nn.GroupNorm(in_size=(28, 28, 16), feature_axis=-1, num_groups=4),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(target_size=(1, 1), channel_axis=-1),
)
```

Open `skills/brainstate/scripts/modern_cnn.py` for the full convolution,
normalization, pooling, dropout, and dense workflow.

## Official sources

- https://brainx.chaobrain.com/brainstate/apis/nn/linear.html
- https://brainx.chaobrain.com/brainstate/apis/nn/conv.html
- https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html
- https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html
- https://brainx.chaobrain.com/brainstate/apis/nn/padding.html
- https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html
