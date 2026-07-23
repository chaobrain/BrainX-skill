# BrainState Prebuilt Layer Library

Use this reference only to select a prebuilt `brainstate.nn` layer and recover its constructor form. The canonical layer example, `Module` composition, and `.desc()` workflow stay in `SKILL.md`. Route convolution, pooling, padding, and flatten output formulas to `../size-inference-variations.md`; route activation functions to `prebuilt-activation-library.md`; route dropout randomness, reproducibility, and fitting-mode work to `../brainstate/randomness-and-reproducibility.md`.

The constructor forms below preserve the documented public parameter order and defaults. To keep the catalog readable, these aliases compress only verbose initializer and State-type representations:

```text
P   = brainstate.ParamState
NP  = brainstate.nn._normalizations.NormalizationParamState
MS  = brainstate.BatchState
WK  = KaimingNormal(scale=2.0, mode='fan_in', in_axis=-2,
                    out_axis=-1, distribution='truncated_normal', ...)
WX  = XavierNormal(scale=1.0, mode='fan_avg', in_axis=-2,
                   out_axis=-1, distribution='truncated_normal', ...)
WL  = LecunNormal(scale=1.0, mode='fan_in', in_axis=-2,
                  out_axis=-1, distribution='truncated_normal', ...)
WN  = Normal(scale=1.0, mean=0.0, ...)
Z   = ZeroInit(unit=Unit("1"))
C0  = Constant(value=0.0)
C1  = Constant(value=1.0)
```

The `...` inside an initializer alias stands only for its rendered RNG and unit representation, not for omitted layer arguments.

## Linear And Connectivity Layers

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/linear.html

The official catalog includes standard dense, weight-standardized, signed-absolute-weight, sparse, low-rank-adaptation, all-to-all, and one-to-one forms.

| API | Constructor | Select when |
|---|---|---|
| `Linear` | `Linear(in_size, out_size, w_init=WK, b_init=Z, w_mask=None, name=None, param_type=P)` | A standard linear transformation is required. |
| `ScaledWSLinear` | `ScaledWSLinear(in_size, out_size, w_init=WK, b_init=Z, w_mask=None, ws_gain=True, eps=0.0001, name=None, param_type=P)` | The linear weights should use scaled weight standardization. |
| `SignedWLinear` | `SignedWLinear(in_size, out_size, w_init=WK, w_sign=None, name=None, param_type=P)` | The transformation should use signed absolute weights. |
| `SparseLinear` | `SparseLinear(spar_mat, b_init=None, in_size=None, name=None, param_type=P)` | The weight matrix is sparse. Supply the sparse matrix as `spar_mat`. |
| `LoRA` | `LoRA(in_features, lora_rank, out_features, *, base_module=None, kernel_init=WL, param_type=P, in_size=None)` | Low-rank adaptation is needed for parameter-efficient fine-tuning, optionally around `base_module`. |
| `AllToAll` | `AllToAll(in_size, out_size, w_init=WK, b_init=None, include_self=True, name=None, param_type=P)` | An all-to-all connection layer is required; use `include_self` to retain or exclude self-connections. |
| `OneToOne` | `OneToOne(in_size, w_init=WN, b_init=None, name=None, param_type=P)` | A one-to-one connection layer is required. |

Do not replace `SparseLinear`, `LoRA`, `AllToAll`, or `OneToOne` with a hand-written dense layer when the corresponding connection structure is the requirement.

## Convolutional Layers

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/conv.html

Choose the suffix by spatial rank: `1d` for one spatial dimension, `2d` for two, and `3d` for three. The catalog provides the same three ranks for standard convolution, scaled weight-standardized convolution, and transposed convolution.

### Standard convolution

Exact classes: `Conv1d`, `Conv2d`, `Conv3d`.

```text
ConvNd(
    in_size, out_channels, kernel_size,
    stride=1, padding='SAME', lhs_dilation=1, rhs_dilation=1, groups=1,
    w_init=WX, b_init=None, w_mask=None, channel_first=False,
    name=None, param_type=P,
)
```

Use the matching concrete class name in place of `ConvNd`. `channel_first=False` is the documented default; make layout choice explicit when selecting the non-default layout. Size outcomes for `stride`, `padding`, dilation, and groups belong in the size-inference reference.

### Scaled weight-standardized convolution

Exact classes: `ScaledWSConv1d`, `ScaledWSConv2d`, `ScaledWSConv3d`.

```text
ScaledWSConvNd(
    in_size, out_channels, kernel_size,
    stride=1, padding='SAME', lhs_dilation=1, rhs_dilation=1, groups=1,
    ws_gain=True, eps=0.0001,
    w_init=WX, b_init=None, w_mask=None, channel_first=False,
    name=None, param_type=P,
)
```

Select this family when convolution with weight standardization is required. `ws_gain` and `eps` are the family-specific constructor controls.

### Transposed convolution

Exact classes: `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`.

```text
ConvTransposeNd(
    in_size, out_channels, kernel_size,
    stride=1, padding='SAME', rhs_dilation=1, groups=1,
    w_init=WX, b_init=None, w_mask=None, channel_first=False,
    name=None, param_type=P,
)
```

Select this family for the catalog's transposed-convolution, or deconvolution, upsampling operation. Unlike the standard and scaled-WS forms, its documented constructor has no `lhs_dilation` argument.

## Normalization Layers

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html

This section inventories normalization APIs only. Activation-function selection remains in `prebuilt-activation-library.md`.

### Batch normalization ranks

Exact classes: `BatchNorm0d`, `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`.

```text
BatchNormNd(
    in_size, feature_axis=-1, *,
    track_running_stats=True, epsilon=1e-05, momentum=0.99, affine=True,
    bias_initializer=C0, scale_initializer=C1,
    axis_name=None, axis_index_groups=None, use_fast_variance=True,
    name=None, dtype=None, param_type=NP, mean_type=MS,
)
```

Choose `0d`, `1d`, `2d`, or `3d` to match the required batch-normalization rank. Running-statistics and affine behavior are constructor choices through `track_running_stats` and `affine`; do not silently change their documented defaults.

### Axis- and group-based normalization

| API | Constructor | Selection distinction |
|---|---|---|
| `LayerNorm` | `LayerNorm(in_size, reduction_axes=-1, feature_axes=-1, *, epsilon=1e-06, use_bias=True, use_scale=True, bias_init=Z, scale_init=C1, axis_name=None, axis_index_groups=None, use_fast_variance=True, dtype=None, param_type=NP)` | Layer normalization with explicit reduction and feature axes. |
| `RMSNorm` | `RMSNorm(in_size, *, epsilon=1e-06, dtype=None, use_scale=True, scale_init=C1, reduction_axes=-1, feature_axes=-1, axis_name=None, axis_index_groups=None, use_fast_variance=True, param_type=NP)` | Root Mean Square Layer Normalization; it exposes scale but no bias argument. |
| `GroupNorm` | `GroupNorm(in_size, feature_axis=-1, num_groups=32, group_size=None, *, epsilon=1e-06, dtype=None, use_bias=True, use_scale=True, bias_init=Z, scale_init=C1, reduction_axes=None, axis_name=None, axis_index_groups=None, use_fast_variance=True, param_type=NP)` | Group Normalization; choose the grouping with `num_groups` or `group_size`. |

The catalog also exposes the functional primitive:

```python
brainstate.nn.weight_standardization(w, eps=0.0001, gain=None, out_axis=-1)
```

Use it when standardizing a weight value directly; prefer the prebuilt `ScaledWSLinear` or `ScaledWSConv*` families when the whole layer already matches the task.

## Pooling And Reshaping

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html

The catalog covers shape conversion, local average/max/Lp pooling, max unpooling, and adaptive pooling. It does not own output-size formulas.

### Shape conversion

| API | Constructor | Select when |
|---|---|---|
| `Flatten` | `Flatten(start_axis=0, end_axis=-1, in_size=None)` | A contiguous range of axes should be flattened. |
| `Unflatten` | `Unflatten(axis, sizes, name=None, in_size=None)` | One axis should be expanded to `sizes`. |

### Local pooling

| Exact classes | Constructor family | Selection distinction |
|---|---|---|
| `AvgPool1d`, `AvgPool2d`, `AvgPool3d` | `AvgPoolNd(kernel_size, stride=1, padding='VALID', channel_axis=-1, name=None, in_size=None)` | Average pooling for the matching spatial rank. |
| `MaxPool1d`, `MaxPool2d`, `MaxPool3d` | `MaxPoolNd(kernel_size, stride=None, padding='VALID', channel_axis=-1, return_indices=False, name=None, in_size=None)` | Max pooling; request indices only through `return_indices`. |
| `LPPool1d`, `LPPool2d`, `LPPool3d` | `LPPoolNd(norm_type, kernel_size, stride=None, padding='VALID', channel_axis=-1, name=None, in_size=None)` | Power-average pooling with the required `norm_type`. |

Keep the default distinction exact: average pooling defaults to `stride=1`; max and Lp pooling default to `stride=None`.

### Max unpooling

Exact classes: `MaxUnpool1d`, `MaxUnpool2d`, `MaxUnpool3d`.

```text
MaxUnpoolNd(
    kernel_size, stride=None, padding=0,
    channel_axis=-1, name=None, in_size=None,
)
```

Select the matching rank when the partial inverse of the corresponding `MaxPoolNd` operation is required.

### Adaptive pooling

| Exact classes | Constructor family | Selection distinction |
|---|---|---|
| `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d` | `AdaptiveAvgPoolNd(target_size, channel_axis=-1, name=None, in_size=None)` | Adaptive average pooling to a fixed target size. |
| `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d` | `AdaptiveMaxPoolNd(target_size, channel_axis=-1, name=None, in_size=None)` | Adaptive max pooling to a fixed target size. |

## Padding Layers

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/padding.html

All five boundary-condition families are available in 1D, 2D, and 3D. Choose the family first, then the suffix matching the spatial rank.

| Exact classes | Constructor family | Boundary behavior |
|---|---|---|
| `ReflectionPad1d`, `ReflectionPad2d`, `ReflectionPad3d` | `ReflectionPadNd(padding, in_size=None, name=None)` | Reflect the input boundary. |
| `ReplicationPad1d`, `ReplicationPad2d`, `ReplicationPad3d` | `ReplicationPadNd(padding, in_size=None, name=None)` | Replicate the input boundary. |
| `ZeroPad1d`, `ZeroPad2d`, `ZeroPad3d` | `ZeroPadNd(padding, in_size=None, name=None)` | Pad with zeros. |
| `ConstantPad1d`, `ConstantPad2d`, `ConstantPad3d` | `ConstantPadNd(padding, value=0, in_size=None, name=None)` | Pad with `value`; the documented default is zero. |
| `CircularPad1d`, `CircularPad2d`, `CircularPad3d` | `CircularPadNd(padding, in_size=None, name=None)` | Wrap around with circular padding. |

Do not derive output dimensions here. Open `../size-inference-variations.md` when `padding` interacts with convolution, pooling, stride, dilation, or flattening.

## Dropout Layers

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html

The official catalog distinguishes elementwise dropout, channelwise spatial dropout, alpha-dropout variants, and a fixed mask along the time axis.

| API | Constructor | Select when |
|---|---|---|
| `Dropout` | `Dropout(prob=0.5, broadcast_dims=(), name=None)` | A subset of inputs should be ignored stochastically each training step; `broadcast_dims` controls mask broadcasting. |
| `Dropout1d` | `Dropout1d(prob=0.5, channel_axis=-1, name=None)` | Entire channels of 1D feature maps should be zeroed. |
| `Dropout2d` | `Dropout2d(prob=0.5, channel_axis=-1, name=None)` | Entire channels of 2D feature maps should be zeroed. |
| `Dropout3d` | `Dropout3d(prob=0.5, channel_axis=-1, name=None)` | Entire channels of 3D feature maps should be zeroed. |
| `AlphaDropout` | `AlphaDropout(prob=0.5, name=None)` | Alpha Dropout is required. |
| `FeatureAlphaDropout` | `FeatureAlphaDropout(prob=0.5, channel_axis=-1, name=None)` | Entire channels should be masked with Alpha Dropout properties. |
| `DropoutFixed` | `DropoutFixed(in_size, prob=0.5, name=None)` | A dropout mask fixed along the time axis is required. |

After selecting the class, route RNG streams, deterministic replay, mapped randomness, checkpoint behavior, or training/evaluation mode to `../brainstate/randomness-and-reproducibility.md`; those concerns are deliberately not duplicated here.

## Exact Official Sources

- https://brainx.chaobrain.com/brainstate/apis/nn/linear.html
- https://brainx.chaobrain.com/brainstate/apis/nn/conv.html
- https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html
- https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html
- https://brainx.chaobrain.com/brainstate/apis/nn/padding.html
- https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html
