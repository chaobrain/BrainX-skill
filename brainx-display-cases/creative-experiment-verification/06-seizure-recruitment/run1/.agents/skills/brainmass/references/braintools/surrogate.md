# Braintools surrogate gradients

Use this reference to select a functional or reusable surrogate-gradient spike
operator for differentiable spiking neural networks. A surrogate emits a hard
spike in the forward pass and supplies a smooth derivative in the backward
pass.

For BrainMass, open this reference only when a custom model or task path
introduces a hard threshold or spike operator. Canonical neural-mass,
`Fitter`, and HORN workflows are differentiable without surrogate gradients.

## Choose an API style

| Style | Use when | Example |
|---|---|---|
| Class | A configured surrogate is stored on a neuron or layer and reused | `Sigmoid(alpha=4.0)` |
| Functional | One direct spike operation is clearer | `sigmoid(x, alpha=4.0)` |
| Custom class | The research method specifies a derivative absent from the library | Subclass `Surrogate` and implement `surrogate_grad`; optionally implement `surrogate_fun` for analysis. |

Corresponding class and functional forms have the same forward and backward
semantics.

## Choose a gradient family

| Family | Use when | Main tuning behavior |
|---|---|---|
| Sigmoid, soft-sign, arctangent, or ERF | A smooth bounded derivative is preferred | `alpha` controls slope or concentration. |
| Piecewise quadratic or exponential | A local, inexpensive piecewise derivative is preferred | `alpha` controls scale or sharpness. |
| ReLU-based | A simple or tailed piecewise derivative is required | Width, leak, and scale control support and tails. |
| Gaussian | Gradient should be localized around threshold | `sigma` controls support; scale controls magnitude. |
| Advanced published form | Reproducing a specified SNN method | Preserve the documented parameters and formulation. |

Start with `Sigmoid`, `PiecewiseQuadratic`, or `ReluGrad`, confirm finite
nonzero gradients, then vary one family or parameter at a time.

## Smooth bounded surrogates

| Functional API | Class API | Description |
|---|---|---|
| `sigmoid(x, alpha=4.0)` | `Sigmoid(alpha=4.0)` | Use a sigmoid-shaped backward derivative. |
| `soft_sign(x, alpha=...)` | `SoftSign(alpha=...)` | Use a soft-sign derivative with different tail decay. |
| `arctan(x, alpha=...)` | `Arctan(alpha=...)` | Use an arctangent-shaped smooth derivative. |
| `erf(x, alpha=...)` | `ERF(alpha=...)` | Use an error-function/Gaussian-CDF derivative. |

## Piecewise and ReLU-based surrogates

| Functional API | Class API | Description |
|---|---|---|
| `piecewise_quadratic(x, alpha=1.0)` | `PiecewiseQuadratic(alpha=1.0)` | Use a triangle-like piecewise-quadratic derivative. |
| `piecewise_exp(x, alpha=1.0)` | `PiecewiseExp(alpha=1.0)` | Use a piecewise-exponential derivative. |
| `piecewise_leaky_relu(x, c=..., alpha=...)` | `PiecewiseLeakyRelu(c=..., alpha=...)` | Use a piecewise leaky-ReLU derivative. |
| `leaky_relu(x, alpha=..., beta=...)` | `LeakyRelu(alpha=..., beta=...)` | Use a leaky-ReLU-shaped derivative. |
| `log_tailed_relu(x, alpha=...)` | `LogTailedRelu(alpha=...)` | Use a ReLU-like center with logarithmic tails. |
| `relu_grad(x, alpha=0.3, width=1.0)` | `ReluGrad(alpha=0.3, width=1.0)` | Use a constant local derivative within a configured threshold width. |

## Gaussian and inverse-square surrogates

| Functional API | Class API | Description |
|---|---|---|
| `gaussian_grad(x, sigma=0.5, alpha=0.5)` | `GaussianGrad(sigma=0.5, alpha=0.5)` | Use one Gaussian derivative peak around threshold. |
| `multi_gaussian_grad(x, h=0.15, s=6.0, sigma=0.5, scale=0.5)` | `MultiGaussianGrad(h=0.15, s=6.0, sigma=0.5, scale=0.5)` | Use the documented multi-Gaussian combination. |
| `inv_square_grad(x, alpha=...)` | `InvSquareGrad(alpha=...)` | Use an inverse-square derivative profile. |

## Advanced surrogates

| Functional API | Class API | Description |
|---|---|---|
| `squarewave_fourier_series(x, ...)` | `SquarewaveFourierSeries(...)` | Use a truncated Fourier approximation to a square wave. |
| `s2nn(x, alpha=4.0, beta=1.0, epsilon=1e-8)` | `S2NN(alpha=4.0, beta=1.0, epsilon=1e-8)` | Use the S2NN formulation. |
| `q_pseudo_spike(x, alpha=2.0)` | `QPseudoSpike(alpha=2.0)` | Use the q-pseudo-spike formulation. |
| `slayer_grad(x, alpha=1.0)` | `SlayerGrad(alpha=1.0)` | Use the SLAYER surrogate formulation. |
| `nonzero_sign_log(x, alpha=...)` | `NonzeroSignLog(alpha=...)` | Use nonzero sign with logarithmic damping. |

## Canonical neuron integration

Store the configured surrogate on the neuron and apply it to membrane state
relative to threshold.

```python
import brainstate as bst
import jax.numpy as jnp
from braintools.surrogate import Sigmoid


class LIFLayer(bst.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = bst.nn.Linear(in_size, out_size)
        self.v = bst.State(jnp.zeros(out_size))
        self.spike = Sigmoid(alpha=4.0)

    def __call__(self, x):
        self.v.value = 0.9 * self.v.value + self.linear(x)
        spikes = self.spike(self.v.value - 1.0)
        self.v.value = jnp.where(spikes > 0, 0.0, self.v.value)
        return spikes
```

**Invariant:** the forward value remains the discrete threshold event. The
surrogate changes only the derivative seen by automatic differentiation.

## Validate the chosen surrogate

Inspect representative gradients before a training run:

```python
import jax
import jax.numpy as jnp
from braintools.surrogate import Sigmoid

spike = Sigmoid(alpha=4.0)
x = jnp.linspace(-2.0, 2.0, 101)
gradient = jax.vmap(jax.grad(lambda value: spike(value)))(x)

assert jnp.all(jnp.isfinite(gradient))
assert jnp.any(gradient != 0)
```

If learning fails, first verify reset timing, loss reduction, gradient target,
and temporal rollout. Do not change the surrogate, optimizer, and loss
simultaneously.

## Custom surrogate

Subclass `Surrogate` only when the required derivative is not already
represented:

```python
import jax.numpy as jnp
from braintools.surrogate import Surrogate


class TanhSurrogate(Surrogate):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        return 0.5 * jnp.tanh(self.alpha * x) + 0.5

    def surrogate_grad(self, x):
        tanh_x = jnp.tanh(self.alpha * x)
        return 0.5 * self.alpha * (1.0 - tanh_x**2)
```

Keep `surrogate_grad` pure, shape preserving, JAX traceable, and finite near
threshold.

## Official source

- `https://brainx.chaobrain.com/braintools/apis/surrogate.html`
