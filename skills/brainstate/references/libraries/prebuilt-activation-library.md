# Prebuilt activation library

Use this reference to choose an exact `brainstate.nn` activation symbol and decide
between a Module and a functional form. Open
`prebuilt-layer-library.md` for normalization layers; `standardize` remains here
because the activation API exports it as a pure array operation.

## Selection map

| Choice | Use when | Key constraint |
|---|---|---|
| Module class, such as `nn.ReLU()` | The activation must be stored in a Module graph, `Sequential`, or descriptor-based composition. | Instantiate the class; parameterized classes keep their configuration in the graph. |
| Functional form, such as `nn.relu(x)` | The activation is a stateless operation inside `update()` or another function. | Pass the array directly; do not instantiate the function. |
| `Softmax`, `LogSoftmax`, `Softmin`, or lowercase equivalents | The output must be normalized across a specified dimension. | Set the dimension or axis deliberately; the default may not match the feature dimension. |
| `standardize`, `one_hot`, or `logsumexp` | The task is array standardization, encoding, or stable log reduction rather than nonlinear activation. | These are utilities, not activation Modules. |
| `SpikeBitwise` | The input is a spiking representation requiring the catalog's bitwise-addition behavior. | Do not use it as a general numeric activation. |

## Module activations

Use a Module class when graph composition or constructor-held configuration
matters. The exact class names are case-sensitive.

| API | Choose it when | Decision-changing control |
|---|---|---|
| `Threshold` | Values on one side of a threshold must be replaced. | `Threshold(threshold, value)` |
| `ReLU` | Use the standard rectified linear unit. | `ReLU(name=None)` |
| `RReLU` | Use randomized negative slopes. | `RReLU(lower=0.125, upper=1 / 3)` |
| `Hardtanh` | Clamp with a hard-tanh response. | `Hardtanh(min_val=-1.0, max_val=1.0)` |
| `ReLU6` | Clamp ReLU output at six. | No selection-specific constructor control is documented. |
| `Sigmoid` | Map values through the logistic sigmoid. | No selection-specific constructor control is documented. |
| `Hardsigmoid` | Use the piecewise-linear sigmoid approximation. | No selection-specific constructor control is documented. |
| `Tanh` | Map values through hyperbolic tangent. | No selection-specific constructor control is documented. |
| `SiLU` | Use SiLU, also called swish. | No selection-specific constructor control is documented. |
| `Mish` | Use the Mish activation. | No selection-specific constructor control is documented. |
| `Hardswish` | Use the piecewise-linear hard-swish variant. | No selection-specific constructor control is documented. |
| `ELU` | Use exponential linear units. | `ELU(alpha=1.0)` |
| `CELU` | Use continuously differentiable ELU. | `CELU(alpha=1.0)` |
| `SELU` | Use scaled ELU in an architecture designed for self-normalization. | Pair with architecture-appropriate initialization and dropout rather than substituting it blindly. |
| `GLU` | Split and gate the input along a dimension. | `GLU(dim=-1)` requires a compatible size on the split dimension. |
| `GELU` | Use Gaussian error linear units. | Check the generated page when an approximation variant matters. |
| `Hardshrink` | Apply hard shrinkage. | Check the generated page for the threshold control. |
| `LeakyReLU` | Preserve a fixed negative slope. | `LeakyReLU(negative_slope=0.01)` |
| `LogSigmoid` | Return the log of sigmoid stably. | No selection-specific constructor control is documented. |
| `Softplus` | Use a smooth positive rectifier. | The current generated constructor is `Softplus(name=None)`; do not assume PyTorch's `beta` or `threshold` arguments. |
| `Softshrink` | Apply soft shrinkage. | Check the generated page for the threshold control. |
| `PReLU` | Learn the negative slope. | `PReLU(num_parameters=1, init=0.25, dtype=None)` creates trainable slope parameters. |
| `Softsign` | Use the soft-sign response. | No selection-specific constructor control is documented. |
| `Tanhshrink` | Return the tanh-shrink response. | No selection-specific constructor control is documented. |
| `Softmin` | Normalize with larger probability on smaller inputs. | Set the normalization dimension explicitly. |
| `Softmax` | Convert logits to normalized probabilities. | `Softmax(dim=None)`; pass the intended feature dimension. |
| `Softmax2d` | Normalize features independently at each spatial location. | Use only for the documented 2D spatial layout. |
| `LogSoftmax` | Return normalized log-probabilities. | Set the normalization dimension explicitly. |
| `Identity` | Keep a configurable graph slot without changing the input. | It is a placeholder, not an activation. |
| `SpikeBitwise` | Apply bitwise addition to spiking inputs. | Keep it within spiking-data workflows. |

## Functional activations

Use the lowercase form for stateless composition. Prefer the function whose name
matches the requested operation; aliases are shown explicitly.

| API | Distinguishing behavior |
|---|---|
| `tanh` | Hyperbolic tangent. |
| `relu` | Rectified linear unit. |
| `squareplus` | Smooth squareplus rectifier. |
| `softplus` | Smooth positive rectifier. |
| `soft_sign` | Soft-sign response. |
| `sigmoid` | Logistic sigmoid. |
| `silu`, `swish` | Equivalent SiLU/swish names. |
| `log_sigmoid` | Stable log-sigmoid. |
| `elu`, `celu`, `selu` | ELU-family functions; select by the requested activation contract. |
| `leaky_relu`, `rrelu`, `prelu` | Fixed, randomized, or supplied negative-slope behavior. |
| `hard_tanh`, `relu6`, `hard_sigmoid` | Piecewise-linear or clamped responses. |
| `hard_silu`, `hard_swish` | Equivalent hard-SiLU/hard-swish names. |
| `gelu`, `mish` | GELU or Mish response. |
| `glu` | Split-and-gate operation; preserve a compatible split-axis size. |
| `hard_shrink`, `soft_shrink`, `tanh_shrink` | Shrinkage-family functions. |
| `softmin`, `softmax`, `log_softmax` | Normalize across `axis`; choose probability, inverse-probability, or log-probability output. |
| `sparse_plus`, `sparse_sigmoid` | Sparse probability transformations. |
| `logsumexp` | Stable log of summed exponentials. |
| `standardize` | Standardize an array, optionally using supplied variance. |
| `one_hot` | Encode integer indices along a new class axis. |

## Canonical selection workflow

Use a Module in a registered layer pipeline and a function for an inline output
transformation:

```python
import brainstate
import brainstate.nn as nn

brainstate.random.seed(0)
feature_block = nn.Sequential(
    nn.Linear(in_size=(8,), out_size=(4,)),
    nn.ReLU(),
)

x = brainstate.random.randn(2, 8)
logits = feature_block(x)
probabilities = nn.softmax(logits, axis=-1)

assert probabilities.shape == logits.shape
```

**Invariant:** choose the normalization axis from the tensor layout. A valid
call on the wrong axis can silently produce the wrong probability semantics.

For any parameter not shown above, open the version-matched generated page
linked from the activation catalog instead of borrowing a signature from JAX,
PyTorch, or another library.

## Official source

- https://brainx.chaobrain.com/brainstate/apis/nn/activation.html
