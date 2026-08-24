# Braintools data preprocessing

Use this reference to convert continuous values, event streams, or temporal
features into spike representations and to combine binary spike tensors. Use
`braintools.cogtask` for phase-structured behavioral trials and
`braintools.input` for injected-current waveforms.

## Choose an encoder

Choose the representation from the information the downstream model must
preserve.

| Encoder family | Use when | Important result |
|---|---|---|
| Latency | Magnitude should control time to first spike | Larger normalized values spike earlier. |
| Rate or Poisson | Magnitude should control spike frequency | Returns a time-major spike train whose empirical rate represents the input. |
| Population | A scalar should activate overlapping receptive fields | Expands each scalar across a neuron population. |
| Bernoulli | Each time bin should be sampled independently | Converts normalized values into per-bin spike probabilities. |
| Delta | Only meaningful temporal changes should emit spikes | Detects threshold-crossing differences between successive samples. |
| Step current | A spiking neuron should receive an analog current level | Returns current values rather than a binary spike code. |
| Spike count | The window must contain an exact input-dependent count | Places a bounded number of spikes randomly or regularly. |
| Temporal | A discrete value should select a precise timing pattern | Maps values to synchronized temporal templates. |
| Rank order | Relative feature ordering matters more than absolute scale | Emits higher-ranked features before lower-ranked features. |

## Spike encoders

All encoder objects are reusable callables. Pass the input as the first argument
and the number of encoded timesteps as `n_time=`.

| API | Description |
|---|---|
| `LatencyEncoder(min_val=None, max_val=None, method='log', threshold=0.01, clip=False, tau=1 * u.ms, normalize=False, first_spk_time=0 * u.ms, epsilon=1e-7)` | Use for time-to-first-spike coding. Normalize inputs when they are not already in `[0, 1]`; larger values produce earlier spikes. |
| `RateEncoder(gain=100.0, method='linear', min_rate=0.0, max_rate=None, normalize=False, min_val=None, max_val=None)` | Use when firing rate should vary with input intensity. Select the mapping with `method` and constrain rates deliberately. |
| `PoissonEncoder(time_window=1000.0, normalize=False, max_rate=100.0)` | Use when an input rate should generate Poisson-distributed spikes. Increase the observation window before judging whether empirical counts match the requested rate. |
| `PopulationEncoder(n_neurons, min_val=0.0, max_val=1.0, sigma=None, max_rate=100.0)` | Use to encode each scalar with overlapping population receptive fields. Set the represented range and population width explicitly. |
| `BernoulliEncoder(scale=1.0, normalize=True, min_val=None, max_val=None)` | Use for independent Bernoulli sampling at every time step. Ensure the resulting probabilities remain valid after scaling. |
| `DeltaEncoder(threshold=0.1, positive_only=False, absolute=False, normalize=True)` | Use for change detection in a temporal signal. Configure sign handling so negative changes are either retained, suppressed, or folded by absolute value. |
| `StepCurrentEncoder(current_scale=10.0, offset=0.0, normalize=True, min_val=None, max_val=None)` | Use to turn features into constant injected-current levels for integrate-and-fire neurons. The result is an analog current code, not a spike tensor. |
| `SpikeCountEncoder(max_spikes=10, distribution='random', normalize=True)` | Use when input magnitude should determine an exact bounded spike count. Choose random or regular placement with `distribution`. |
| `TemporalEncoder(n_patterns, pattern_length=10, jitter=0.1)` | Use when values select reusable temporal spike patterns. Set `jitter` according to the timing variability the model should tolerate. |
| `RankOrderEncoder(use_values=True, normalize=True, invert=False)` | Use when feature rank determines spike order. Use `invert=True` only when low-valued features must spike first. |

```python
import jax.numpy as jnp
from braintools import LatencyEncoder

values = jnp.array([0.02, 0.5, 1.0])
encoder = LatencyEncoder(method="linear", normalize=True)
spikes = encoder(values, n_time=5)

assert spikes.shape == (5, 3)
assert jnp.argmax(spikes[:, 2]) < jnp.argmax(spikes[:, 1])
assert jnp.argmax(spikes[:, 1]) < jnp.argmax(spikes[:, 0])
```

**Invariant:** treat the leading axis as encoded time. Validate normalization,
time-window length, and stochastic variability before comparing encoders.

## Spike operations

Use these element-wise operations only on boolean or binary-compatible spike
tensors with broadcast-compatible shapes.

| API | Description |
|---|---|
| `spike_bitwise_or(x, y)` | Return binary OR, implemented as `x + y - x * y`. |
| `spike_bitwise_and(x, y)` | Return binary AND, implemented as element-wise multiplication. |
| `spike_bitwise_iand(x, y)` | Return inverse AND, documented as `(NOT x) AND y`. |
| `spike_bitwise_not(x)` | Invert a binary spike tensor. |
| `spike_bitwise_xor(x, y)` | Return XOR between two spike tensors. |
| `spike_bitwise_ixor(x, y)` | Apply the documented inverse-XOR operation. |
| `spike_bitwise(x, y, op)` | Dispatch by `op` over `'or'`, `'and'`, `'iand'`, `'xor'`, and `'ixor'`. |

Do not use these helpers as fuzzy arithmetic on rates or arbitrary continuous
arrays; their formulas assume binary values.

## Official source

- `https://brainx.chaobrain.com/braintools/apis/braintools.html`
