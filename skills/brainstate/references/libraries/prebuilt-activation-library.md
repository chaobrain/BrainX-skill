# Prebuilt Activation Library

Use this reference to select an exact `brainstate.nn` activation symbol. The routed catalog provides symbol names and descriptions, but not constructor or callable signatures; do not invent argument lists from similarly named APIs in another library. It distinguishes stateful layer modules from pure functions, but leaves basic activation use and `Module` composition to `SKILL.md`.

Normalization layers are not present on this page. Select `BatchNorm*`, `LayerNorm`, `RMSNorm`, or `GroupNorm` from `prebuilt-layer-library.md`. The functional `standardize` entry below remains here because it is part of the activation catalog and is described as standardizing an array; it is not a normalization-layer API.

## Element-wise Layers

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/activation.html

The official page describes these as non-linear activation layers that operate element-wise on input tensors. Its catalog includes rectified linear units and variants, sigmoid functions, hyperbolic tangent, softmax for probability distributions, and specialized activations for specific architectures such as SELU, GELU, SiLU, and Mish.

| Exact symbol | Official catalog description |
|---|---|
| `Threshold` | Thresholds each element of the input Tensor. |
| `ReLU` | Applies the rectified linear unit function element-wise. |
| `RReLU` | Applies the randomized leaky rectified liner unit function, element-wise. |
| `Hardtanh` | Applies the HardTanh function element-wise. |
| `ReLU6` | Applies the element-wise function. |
| `Sigmoid` | Applies the element-wise function. |
| `Hardsigmoid` | Applies the Hardsigmoid function element-wise. |
| `Tanh` | Applies the Hyperbolic Tangent (Tanh) function element-wise. |
| `SiLU` | Applies the Sigmoid Linear Unit (SiLU) function, element-wise. |
| `Mish` | Applies the Mish function, element-wise. |
| `Hardswish` | Applies the Hardswish function, element-wise. |
| `ELU` | Applies the Exponential Linear Unit (ELU) function, element-wise. |
| `CELU` | Applies the element-wise function. |
| `SELU` | Applied element-wise. |
| `GLU` | Applies the gated linear unit function. |
| `GELU` | Applies the Gaussian Error Linear Units function. |
| `Hardshrink` | Applies the Hard Shrinkage (Hardshrink) function element-wise. |
| `LeakyReLU` | Applies the element-wise function. |
| `LogSigmoid` | Applies the element-wise function. |
| `Softplus` | Applies the Softplus function element-wise. |
| `Softshrink` | Applies the soft shrinkage function elementwise. |
| `PReLU` | Applies the element-wise function. |
| `Softsign` | Applies the element-wise function. |
| `Tanhshrink` | Applies the element-wise function. |
| `Softmin` | Applies the Softmin function to an n-dimensional input Tensor. |
| `Softmax` | Applies the Softmax function to an n-dimensional input Tensor. |
| `Softmax2d` | Applies SoftMax over features to each spatial location. |
| `LogSoftmax` | Applies the `log(Softmax(x))` function to an n-dimensional input Tensor. |
| `Identity` | A placeholder identity operator that is argument-insensitive. |
| `SpikeBitwise` | Bitwise addition for the spiking inputs. |

Selection cues supplied by the catalog:

- Use `Softmin`, `Softmax`, or `LogSoftmax` when the requested operation names that distribution transform; use `Softmax2d` when the request specifically calls for SoftMax over features at each spatial location.
- Use `Identity` when an argument-insensitive placeholder identity operator is required.
- Use `SpikeBitwise` only for the page's stated spiking-input bitwise addition role.
- Treat the remaining entries according to their exact named activation. The catalog does not expose parameters here, even for parameterized, thresholded, randomized, gated, or shrinkage variants.

## Functional Activations

Source URL: https://brainx.chaobrain.com/brainstate/apis/nn/activation.html

The official page describes these as functional, non-module activation functions for flexible composition. They are pure functions that may be used directly in `update()` methods or combined with JAX transformations, and provide activation behavior without state or module overhead.

| Exact symbol | Official catalog description |
|---|---|
| `tanh` | Hyperbolic tangent activation function. |
| `relu` | Rectified Linear Unit activation function. |
| `squareplus` | Squareplus activation function. |
| `softplus` | Softplus activation function. |
| `soft_sign` | Soft-sign activation function. |
| `sigmoid` | Sigmoid activation function. |
| `silu` | SiLU (Sigmoid Linear Unit) activation function. |
| `swish` | SiLU (Sigmoid Linear Unit) activation function. |
| `log_sigmoid` | Log-sigmoid activation function. |
| `elu` | Exponential Linear Unit activation function. |
| `leaky_relu` | Leaky Rectified Linear Unit activation function. |
| `hard_tanh` | Hard hyperbolic tangent activation function. |
| `celu` | Continuously-differentiable Exponential Linear Unit activation. |
| `selu` | Scaled Exponential Linear Unit activation. |
| `gelu` | Gaussian Error Linear Unit activation function. |
| `glu` | Gated Linear Unit activation function. |
| `logsumexp` | No description is supplied in the routed catalog. |
| `log_softmax` | Log-Softmax function. |
| `softmax` | Softmax activation function. |
| `standardize` | Standardize (normalize) an array. |
| `one_hot` | One-hot encode the given indices. |
| `relu6` | Rectified Linear Unit 6 activation function. |
| `hard_sigmoid` | Hard Sigmoid activation function. |
| `hard_silu` | Hard SiLU (Swish) activation function. |
| `hard_swish` | Hard SiLU (Swish) activation function. |
| `hard_shrink` | Hard shrinkage activation function. |
| `rrelu` | Randomized Leaky Rectified Linear Unit activation function. |
| `mish` | Mish activation function. |
| `soft_shrink` | Soft shrinkage activation function. |
| `prelu` | Parametric Rectified Linear Unit activation function. |
| `tanh_shrink` | Tanh shrink activation function. |
| `softmin` | Softmin activation function. |
| `sparse_plus` | Sparse plus activation function. |
| `sparse_sigmoid` | Sparse sigmoid activation function. |

Selection cues supplied by the catalog:

- Choose the lowercase functional form when the operation should remain a pure function inside `update()` or a JAX transformation rather than a layer module.
- `silu` and `swish` share the same catalog description; likewise, `hard_silu` and `hard_swish` share the same description.
- `standardize` is the page's functional array-standardization choice. Route stateful or configurable normalization-layer selection to `prebuilt-layer-library.md`.
- `one_hot` is an encoding utility listed in this catalog, not a nonlinear layer.
- The routed page gives no description or signature for `logsumexp`; use the exact symbol only when that operation is explicitly requested, and consult a version-matched generated API page before supplying arguments.

## Source

- https://brainx.chaobrain.com/brainstate/apis/nn/activation.html
