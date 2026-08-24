# Braintools parameter initializers

Use this reference to initialize parameters, weights, delays, and spatial
profiles with unit-aware, reusable strategies. It covers distribution choice,
variance scaling, orthogonal methods, composition, and distance-based weight
modulation.

For BrainMass, use these initializers for model State, fitted starting values,
HORN weights, or distance-modulated network parameters when the consuming API
accepts a Braintools initialization specification.

## Choose an initializer

| Need | Use | Important constraint |
|---|---|---|
| Fixed value | `Constant` or `ZeroInit` | Preserve physical units in the value. |
| Bounded random values | `Uniform`, `TruncatedNormal`, or `Beta` | Choose bounds that match the parameter's domain. |
| Bell-shaped variability | `Normal` | Negative draws remain possible unless clipped. |
| Positive skew | `LogNormal`, `Gamma`, `Exponential`, or `Weibull` | Use only when the parameter domain and parameterization match the distribution. |
| Activation-aware matrix scale | Kaiming, Xavier, or LeCun variants | Match the initializer to the activation and fan mode. |
| Norm-preserving matrix | `Orthogonal` or `Identity` | Validate the required matrix or convolutional shape. |
| Heterogeneous population | `Mixture` or `Conditional` | Keep component units and output shapes compatible. |
| Spatial weight scaling | `DistanceModulated` | This scales existing weights; it does not sample connectivity. |

## Resolve initialization specifications

An `Initialization` object is a shape-aware callable. Use `param()` at APIs that
accept an initializer, scalar, array, or quantity through one boundary.

| API | Description |
|---|---|
| `Initialization` | Subclass when a new reusable initialization strategy is required. |
| `param(init, sizes, batch_size=None, allow_none=True, allow_scalar=True, **param_kwargs)` | Resolve an initializer or already concrete value into a parameter. It applies the requested shape policy and forwards keywords such as RNG or distances. |
| `Compose(*steps)` | Apply an initializer followed by one or more transformations. |

```python
import brainunit as u
import numpy as np
from braintools.init import Normal, param

rng = np.random.default_rng(0)
initializer = Normal(mean=0.5 * u.nS, std=0.1 * u.nS)
weights = param(initializer, sizes=100, rng=rng)

assert weights.shape == (100,)
assert weights.unit == u.nS
```

**Invariant:** pass an explicit RNG when reproducibility matters. Do not strip a
quantity's unit to make an initializer accept it.

## Basic distributions

| API | Description |
|---|---|
| `Constant(value)` | Fill the requested shape with one value. |
| `ZeroInit(unit=None)` | Fill the requested shape with zero, optionally carrying a unit. |
| `Uniform(low, high, unit=None)` | Sample uniformly between lower and upper bounds. |
| `Normal(mean, std, unit=None)` | Sample a Gaussian distribution. |
| `LogNormal(mean, std, unit=None)` | Sample a positive log-normal distribution. |
| `TruncatedNormal(mean, std, low, high, unit=None)` | Sample a normal distribution within explicit bounds. |
| `Gamma(...)` | Sample a gamma distribution for positive skewed values. |
| `Exponential(...)` | Sample an exponential distribution. |
| `Beta(...)` | Sample a beta distribution rescaled to a requested range. |
| `Weibull(...)` | Sample a Weibull distribution. |

Use explicit `unit=` only when the distribution parameters themselves do not
already establish the required unit.

## Variance scaling

Variance-scaling initializers derive scale from matrix fan dimensions. Use
`fan_in` to preserve forward variance, `fan_out` to preserve backward variance,
and `fan_avg` to balance both.

| API | Description |
|---|---|
| `VarianceScaling(...)` | Use for a custom scale, fan mode, and distribution combination. |
| `KaimingUniform(...)` | Use uniform Kaiming/He initialization for ReLU-family activations. |
| `KaimingNormal(scale=None, mode='fan_in', nonlinearity='relu', negative_slope=0.01, unit=None)` | Use normal Kaiming/He initialization and match `nonlinearity` plus `negative_slope` to the layer. |
| `XavierUniform(...)` | Use uniform Xavier/Glorot initialization for symmetric tanh/sigmoid-style activations. |
| `XavierNormal(...)` | Use normal Xavier/Glorot initialization. |
| `LecunUniform(...)` | Use uniform LeCun initialization for SELU-style self-normalizing layers. |
| `LecunNormal(...)` | Use normal LeCun initialization. |

```python
from braintools.init import KaimingNormal, XavierUniform

relu_weights = KaimingNormal(mode="fan_in")((256, 784), rng=rng)
tanh_weights = XavierUniform()((256, 784), rng=rng)
```

Confirm which axis the consuming layer treats as input before relying on fan
inference.

## Orthogonal methods

| API | Description |
|---|---|
| `Orthogonal(scale=1.0, unit=None)` | Use QR-based orthogonal initialization for recurrent or deep matrix weights. It supports non-square matrices. |
| `DeltaOrthogonal(...)` | Use for convolutional kernels whose central spatial slice should be orthogonal. |
| `Identity(scale=1.0, unit=None)` | Use when the initial transformation should be an optionally scaled identity. |

## Composite distributions

| API | Description |
|---|---|
| `Mixture(distributions, weights=None)` | Select among component distributions per parameter according to mixture weights. |
| `Conditional(condition_fn, true_dist, false_dist)` | Select one of two initializers from neuron indices or supplied properties. |
| `Scaled(initializer, scale_factor)` | Multiply another initializer's output. |
| `Clipped(initializer, min_val, max_val)` | Bound another initializer's output. |

```python
from braintools.init import Conditional, Normal

initializer = Conditional(
    condition_fn=lambda indices: indices < 800,
    true_dist=Normal(0.5 * u.nS, 0.1 * u.nS),
    false_dist=Normal(-0.3 * u.nS, 0.05 * u.nS),
)
weights = initializer(
    1000,
    neuron_indices=np.arange(1000),
    rng=rng,
)
```

## Distance-modulated weights

A `DistanceProfile` maps distances to both documented probability and
weight-scaling functions. `DistanceModulated` uses the scaling values only.

| API | Description |
|---|---|
| `DistanceModulated(base_dist, distance_profile)` | Return `base_weights * distance_profile(distances)` element-wise. Supply `distances` directly or `pre_positions` and `post_positions`. |
| `GaussianProfile(sigma, max_distance=None)` | Use a Gaussian distance profile. |
| `ExponentialProfile(decay_constant, max_distance=None)` | Use exponentially decaying distance dependence. |
| `LinearProfile(...)` | Use a linear distance profile. |
| `StepProfile(...)` | Use a hard distance threshold or step. |
| `PowerLawProfile(...)` | Use power-law distance dependence. |
| `MexicanHatProfile(...)` | Use a Ricker/Mexican-hat profile. |
| `DoGProfile(...)` | Use a difference-of-Gaussians profile. |
| `SigmoidProfile(...)` | Use a sigmoid transition with distance. |
| `BimodalProfile(...)` | Use two distance-dependent modes. |
| `LogisticProfile(...)` | Use logistic distance dependence. |
| `ComposedProfile(...)` | Combine two profiles arithmetically. |
| `ClipProfile(...)` / `profile.clip(...)` | Bound profile output. |
| `ApplyProfile(...)` / `profile.apply(func)` | Transform profile output with a function. |
| `PipeProfile(...)` / `profile \| func` | Chain profiles or transformations. |

```python
from braintools.init import DistanceModulated, GaussianProfile, Normal

profile = GaussianProfile(sigma=100 * u.um)
initializer = DistanceModulated(
    base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
    distance_profile=profile,
).clip(min_val=0.01 * u.nS)

distances = np.random.default_rng(1).uniform(0, 500, 1000) * u.um
weights = initializer(1000, distances=distances, rng=rng)
```

**Invariant:** a zero profile value creates a present but silent zero-weight
synapse. Use `braintools.conn.DistanceDependent` when distance must decide
whether an edge exists.

## Functional composition

All `Initialization` and `DistanceProfile` objects support arithmetic
composition. Initializers additionally support `.clip()`, `.add()`,
`.multiply()`, `.apply()`, and pipe composition with `|`.

Build and validate composition incrementally:

```python
base = Normal(0.5 * u.nS, 0.1 * u.nS)
bounded = base.clip(0.0 * u.nS, 1.0 * u.nS)
final = bounded * 2.0 + 0.1 * u.nS
```

Keep units compatible at every arithmetic step. Apply clipping through
composition; `DistanceModulated` does not accept a `min_weight` argument.

## Official source

- `https://brainx.chaobrain.com/braintools/apis/init.html`
