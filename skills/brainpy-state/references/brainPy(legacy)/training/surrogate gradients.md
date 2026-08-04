# Surrogate gradients

Use this reference when legacy `brainpy.math.surrogate` code must emit discrete spikes in the forward pass while supplying a smooth derivative for gradient-based spiking-neural-network training. Keep the import and examples on the legacy BrainPy surface; do not replace them with `brainpy.state` APIs.

## Core contract

A surrogate spike function applies the Heaviside step in the forward pass and substitutes its configured surrogate derivative during backpropagation.

| API | Description |
|---|---|
| `Surrogate()` | Base object for reusable class-style surrogate functions; use a concrete subclass for normal workflows. |
| Class form, such as `Sigmoid(alpha=4.0)` | Store the derivative-shape parameters once and call the object repeatedly. |
| Functional form, such as `sigmoid(x, alpha=4.0)` | Apply one surrogate operation inline without storing configuration. |

**Invariant:** changing the surrogate family or its shape parameters changes the backward gradient, not the binary forward spike rule. Tune it as an optimization choice and verify gradients separately from spike outputs.

## Selection map

| Family | Use when | Main controls |
|---|---|---|
| Smooth sigmoid-like | A smooth, centered surrogate is required. | Usually `alpha`; `ERF` uses the same centered family interface. |
| Piecewise | A piecewise polynomial, exponential, or leaky response is required. | `alpha`, or the piecewise constants `c` and `w`. |
| Fourier or literature-specific | The training method specifies a named estimator. | Preserve the estimator-specific parameters instead of translating them to a generic slope. |
| ReLU-shaped | A rectangular, leaky, or tailed derivative is required. | Slope, width, and tail parameters. |
| Gaussian-shaped | A Gaussian or multi-Gaussian derivative is required. | Width, height, scale, and side-lobe parameters. |

## Smooth surrogate families

Choose between reusable class and inline functional forms without changing the mathematical family.

| Class form | Functional form | Distinguishing behavior |
|---|---|---|
| `Sigmoid(alpha=...)` | `sigmoid(x, alpha=...)` | Sigmoid-shaped surrogate gradient. |
| `SoftSign(alpha=...)` | `soft_sign(x, alpha=...)` | Soft-sign surrogate gradient. |
| `Arctan(alpha=...)` | `arctan(x, alpha=...)` | Arctangent surrogate gradient. |
| `NonzeroSignLog(alpha=...)` | `nonzero_sign_log(x, alpha=...)` | Nonzero-sign logarithmic surrogate gradient. |
| `ERF(alpha=...)` | `erf(x, alpha=...)` | Error-function surrogate gradient. |

## Piecewise surrogate families

Use these when the selected training formulation calls for a piecewise derivative.

| Class form | Functional form | Distinguishing behavior |
|---|---|---|
| `PiecewiseQuadratic(alpha=...)` | `piecewise_quadratic(x, alpha=...)` | Piecewise-quadratic surrogate gradient. |
| `PiecewiseExp(alpha=...)` | `piecewise_exp(x, alpha=...)` | Piecewise-exponential surrogate gradient. |
| `PiecewiseLeakyRelu(c=..., w=...)` | `piecewise_leaky_relu(x, c=..., w=...)` | Piecewise leaky-ReLU surrogate controlled by its constant and width. |

## Fourier and named estimators

Use a named estimator when reproducing a method that specifies it explicitly.

| Class form | Functional form | Distinguishing behavior |
|---|---|---|
| `SquarewaveFourierSeries(n=..., t_period=...)` | `squarewave_fourier_series(x, n=..., t_period=...)` | Square-wave Fourier-series surrogate. |
| `S2NN(alpha=..., beta=..., epsilon=...)` | `s2nn(x, alpha=..., beta=..., epsilon=...)` | S2NN surrogate spiking function. |
| `QPseudoSpike(alpha=...)` | `q_pseudo_spike(x, alpha=...)` | q-PseudoSpike surrogate function. |

## ReLU-shaped surrogate families

Use these when slope, width, or tail behavior must follow a ReLU-like estimator.

| Class form | Functional form | Distinguishing behavior |
|---|---|---|
| `LeakyRelu(alpha=..., beta=...)` | `leaky_relu(x, alpha=..., beta=...)` | Leaky-ReLU surrogate gradient. |
| `LogTailedRelu(alpha=...)` | `log_tailed_relu(x, alpha=...)` | Log-tailed ReLU surrogate gradient. |
| `ReluGrad(alpha=..., width=...)` | `relu_grad(x, alpha=..., width=...)` | Finite-width ReLU-gradient surrogate. |

## Gaussian and other gradient families

Use these when the chosen method specifies Gaussian lobes, inverse-square decay, or the SLAYER estimator.

| Class form | Functional form | Distinguishing behavior |
|---|---|---|
| `GaussianGrad(sigma=..., alpha=...)` | `gaussian_grad(x, sigma=..., alpha=...)` | Gaussian surrogate gradient. |
| `MultiGaussianGrad(h=..., s=..., sigma=..., scale=...)` | `multi_gaussian_grad(x, h=..., s=..., sigma=..., scale=...)` | Multi-Gaussian surrogate with configurable lobes and scale. |
| `InvSquareGrad(alpha=...)` | `inv_square_grad(x, alpha=...)` | Inverse-square surrogate gradient. |
| `SlayerGrad(alpha=...)` | `slayer_grad(x, alpha=...)` | SLAYER surrogate gradient. |

## Canonical workflow

Use the class form for a spike rule stored by a model and verify both the discrete forward result and the differentiable backward path:

```python
import brainpy.math as bm
import jax.numpy as jnp

spike_fn = bm.surrogate.Sigmoid(alpha=4.0)
x = jnp.asarray([-1.0, 0.0, 1.0])

class_spikes = spike_fn(x)
functional_spikes = bm.surrogate.sigmoid(x, alpha=4.0)
gradient = bm.grad(lambda value: bm.sum(spike_fn(value)))(x)

assert bm.allclose(class_spikes, jnp.asarray([0.0, 1.0, 1.0]))
assert bm.allclose(functional_spikes, class_spikes)
assert gradient.shape == x.shape
```

Do not validate a surrogate only by plotting its forward output; every family has the same step-like spike role. Inspect the gradient magnitude and support around the threshold because those determine the training signal.

## Sources

- https://brainpy.readthedocs.io/apis/brainpy.math.surrogate.html
