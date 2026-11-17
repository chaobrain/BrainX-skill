# BrainTools losses and analysis

Use this reference to select supervised losses and neuroscience metrics for
classification, regression, spike trains, local field potentials, correlation,
connectivity, and pairwise comparison. Verify input orientation, units, and
reduction before treating any returned array as a scalar objective.

For BrainMass, prefer `brainmass.objectives` for standard `Fitter` time-series,
FC, and FCD comparisons. Use this reference for HORN supervised losses,
held-out metrics, or a custom fitting statistic not owned by
`brainmass.objectives`.

Use `braintools.metric` for training losses and vector comparisons. Use
`braintools.analysis` for neuroscience analysis. The analysis examples require
BrainTools 0.4.0 or newer.

## Choose a metric family

| Task | Use | Important constraint |
|---|---|---|
| Independent binary or multilabel classification | Sigmoid binary cross entropy or focal loss | Logits and labels must have broadcast-compatible class structure. |
| Mutually exclusive classes | Softmax cross entropy | Choose one-hot/probability labels or the integer-label variant explicitly. |
| Continuous fitting | Squared, absolute, Huber, log-cosh, or cosine losses | Check the function-specific reduction and reduce explicitly when a scalar is required. |
| Spike timing | Victor-Purpura or van Rossum distance | Inputs are spike-time trains, not dense spike matrices. |
| Population spiking | Firing-rate and synchrony metrics | Dense spike matrices are time-major unless the API says otherwise. |
| LFP spectrum or coupling | PSD, coherence, PAC, CSD, entropy, or phase coherence | For spectral routines with `dt`, pass the sample interval. CSD takes electrode spacing. |
| Functional connectivity | Correlation and connectivity functions | Time series use `(num_time, num_signals)`. |

## Classification losses (`braintools.metric`)

| API | Description |
|---|---|
| `sigmoid_binary_cross_entropy(logits, labels)` | Use for independent binary classes; returns element-wise sigmoid cross entropy. |
| `softmax_cross_entropy(logits, labels)` | Use for mutually exclusive classes represented by one-hot or probability labels. |
| `softmax_cross_entropy_with_integer_labels(logits, labels)` | Use the more direct integer-index form for mutually exclusive classes. |
| `hinge_loss(...)` | Use for binary margin classification. |
| `perceptron_loss(...)` | Use for the binary perceptron objective. |
| `multiclass_hinge_loss(...)` | Use for a multiclass margin objective. |
| `multiclass_perceptron_loss(...)` | Use for a multiclass perceptron objective. |
| `poly_loss_cross_entropy(...)` | Use when the specified method requires PolyLoss cross entropy. |
| `kl_divergence(log_predictions, targets)` | Compare target probabilities with predicted log probabilities. |
| `kl_divergence_with_log_targets(...)` | Use when both predicted and target probabilities are supplied in log space. |
| `convex_kl_divergence(...)` | Use when the convex KL variant is required. |
| `ctc_loss(...)` | Use for sequence alignment without frame-level labels. |
| `ctc_loss_with_forward_probs(...)` | Use when the CTC forward probabilities are also needed. |
| `sigmoid_focal_loss(...)` | Use for class imbalance among independent binary targets. |
| `nll_loss(...)` | Use with log-probability classification outputs. |

```python
import jax.numpy as jnp
from braintools.metric import (
    softmax_cross_entropy_with_integer_labels,
)

logits = jnp.array([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
])
labels = jnp.array([0, 1])

per_example = softmax_cross_entropy_with_integer_labels(logits, labels)
loss = jnp.mean(per_example)
assert loss.ndim == 0
```

Validate integer labels against `0 <= labels < logits.shape[-1]` before
computing the loss. Invalid indices can select a different class or return
`NaN` without raising.

## Regression and pairwise metrics (`braintools.metric`)

| API | Description |
|---|---|
| `squared_error(predictions, targets=None, axis=None, reduction='none')` | Use for element-wise squared error. Targets must match prediction shape; choose `reduction="mean"` for a scalar MSE with `axis=None`. |
| `absolute_error(...)` | Use for element-wise absolute error. |
| `l1_loss(logits, targets, reduction="mean")` | Average absolute error over non-batch axes first; use `reduction="none"` to keep one value per sample. |
| `l2_loss(...)` | Use for half the squared error. |
| `l2_norm(...)` | Return the L2 norm of prediction error. |
| `safe_norm(...)` | Use when a differentiable lower bound must prevent an unstable zero norm. |
| `huber_loss(predictions, targets=None, delta=1.0, axis=None, reduction='none')` | Use for quadratic small residuals and linear large residuals; tune `delta` in target units. |
| `log_cosh(...)` | Use for a smooth robust regression loss. |
| `cosine_similarity(...)` | Compare vector direction. |
| `cosine_distance(...)` | Return the corresponding cosine distance. |
| `pairwise_cosine_similarity(X, Y, ...)` | Return all pairwise cosine similarities between sample sets. |
| `pairwise_cosine_distance(X, Y, ...)` | Return all pairwise cosine distances between sample sets. |

Do not conflate element-wise loss with final optimization reduction. Preserve
batch and time axes until the scientific loss definition determines how to
reduce them.

## Correlation and functional connectivity (`braintools.analysis`)

| API | Description |
|---|---|
| `spike.cross_correlation(spikes, bin, dt=None, ...)` | Calculate a cross-correlation synchrony index from a time-major spike matrix. |
| `population.voltage_fluctuation(potentials, ...)` | Estimate synchrony from population voltage variance. |
| `stats.matrix_correlation(x, y)` | Correlate the strict upper triangles of two matching connectivity matrices. |
| `stats.weighted_correlation(x, y, w)` | Compute weighted Pearson correlation for two one-dimensional series. |
| `population.functional_connectivity(activities)` | Return the pairwise Pearson correlation matrix for `(num_time, num_signals)` data; NaNs are replaced with zero. |
| `population.functional_connectivity_dynamics(activities, window_size, step_size)` | Correlate the strict upper triangles of windowed FC matrices. Specify window size and step in samples. |

## Spike-train metrics

Pass physical time quantities on BrainTools 0.4. Bare `dt` values are in
seconds for `firing_rate`, `phase_locking_value`, and spectral functions.
Most other spike-matrix routines use milliseconds.

| API | Description |
|---|---|
| `spike.firing_rate(spikes, width, dt=None)` | Return a time series of population rate in Hz. Pass physical time quantities for `width` and `dt` to make units explicit. |
| `spike.victor_purpura_distance(spike_times_1, spike_times_2, cost_factor=1.0)` | Measure edit cost from insertions, deletions, and temporal shifts. |
| `spike.van_rossum_distance(spike_times_1, spike_times_2, tau=1.0, t_max=None)` | Compare exponentially filtered spike trains. |
| `spike.spike_train_synchrony(spike_matrix, window_size, dt=None)` | Average pairwise spike coincidences within a time window. Silent-neuron pairs are omitted; check activity before interpreting the score. |
| `spike.burst_synchrony_index(spike_matrix, burst_threshold=3, max_isi=100.0, dt=None)` | Measure co-occurring burst events. |
| `spike.phase_locking_value(spike_matrix, reference_freq, dt=None)` | Measure spike phase locking at a specified reference frequency; this score does not give a signed phase lag. |
| `spike.spike_time_tiling_coefficient(spike_matrix, dt=None, tau=5.0)` | Compute STTC within a temporal tolerance. |
| `spike.correlation_index(spike_matrix, window_size, dt=None)` | Compute a windowed spike-train correlation index. |

```python
import brainunit as u
import braintools
import jax.numpy as jnp

spikes = jnp.zeros((200, 5)).at[50:53, :].set(1.0)
rate = braintools.analysis.spike.firing_rate(spikes, width=10 * u.ms, dt=1 * u.ms)
synchrony = braintools.analysis.spike.spike_train_synchrony(
    spikes,
    window_size=5 * u.ms,
    dt=1 * u.ms,
)
```

Keep dense spike matrices and lists of spike times distinct. Distance metrics
consume event times; population rate and synchrony metrics consume matrices.
Use [`braintools.spike.SpikeTrains.from_dense`](https://brainx.chaobrain.com/braintools/apis/generated/braintools.spike.SpikeTrains.html#braintools.spike.SpikeTrains.from_dense)
to convert a regularly sampled matrix to spike trains; use
[`braintools.visualize.spike_raster`](https://brainx.chaobrain.com/braintools/apis/generated/braintools.visualize.spike_raster.html)
to draw a raster.

## Local field potential analysis

| API | Description |
|---|---|
| `field.unitary_LFP(times, spikes, spike_type, ...)` | Construct unitary LFP contributions from spikes. Supply times in milliseconds and choose `'exc'` or `'inh'`. |
| `spectral.power_spectral_density(lfp, dt, nperseg=None, noverlap=None, freq_range=None)` | Return one-sided Welch frequencies in Hz and PSD for `(time,)` or `(time, channels)` input. |
| `spectral.spectrogram(x, dt, nperseg, noverlap=None, window='hann', detrend='constant')` | Use when segment spectra or explicit window/detrending controls are required; returns frequencies, segment centers and PSD shaped `(frequency, segment, ...)`. |
| `spectral.coherence_analysis(lfp1, lfp2, dt, ...)` | Return frequency-resolved magnitude-squared coherence. |
| `spectral.phase_amplitude_coupling(lfp, dt, phase_freq_range, amplitude_freq_range, ...)` | Return Tort modulation index plus phase-bin details. |
| `spectral.theta_gamma_coupling(lfp, dt)` | Return theta-gamma coupling using standard bands. |
| `field.current_source_density(lfp_laminar, electrode_spacing, conductivity=1.0, axis=-1)` | Estimate laminar CSD using electrode spacing. Set `axis` explicitly when electrodes are not on the last axis. |
| `spectral.spectral_entropy(lfp, dt, ...)` | Return normalized spectral entropy. |
| `spectral.lfp_phase_coherence(lfp_signals, dt, freq_band=(8, 12))` | Compute pairwise phase coherence across channels. |

Use `power_spectral_density` when its estimator defaults match the protocol.
For a fixed Welch estimator requiring explicit window and detrending controls,
use [`spectral.spectrogram`](https://brainx.chaobrain.com/braintools/apis/generated/braintools.analysis.spectral.spectrogram.html)
and take `u.math.mean(segment_psd, axis=1)` over segments. Set `noverlap`
explicitly: its default is `nperseg // 8`, not 50%. `window='hann'` selects
a periodic Hann window; `detrend='constant'` removes each segment's mean.
For a Quantity signal with unit `U`, the segment PSD and its mean retain
`U**2 / Hz`. Preserve the frozen estimator when comparing implementations.

`power_spectral_density(..., freq_range=...)` uses boolean-mask indexing and
therefore has a data-dependent output length; that path is not JIT-compatible.
Compute the full spectrum inside JIT and filter outside when compilation is
required.

## Structured, ranking, and smoothing losses (`braintools.metric`)

| API | Description |
|---|---|
| `make_fenchel_young_loss(max_fun)` | Build a Fenchel-Young loss. `max_fun` must map one score vector to a scalar because it is vectorized with signature `(n)->()`. |
| `ranking_softmax_loss(scores, labels, ...)` | Use for list-wise learning-to-rank objectives. |
| `smooth_labels(labels, alpha=...)` | Apply label smoothing to one-hot targets. |

## Official sources

Open the [analysis API](https://brainx.chaobrain.com/braintools/apis/analysis.html)
for PSTH, interspike-interval statistics, or band power. Open the
[fitting API](https://brainx.chaobrain.com/braintools/apis/fit.html) when firing
rate, PSD, or FC is a fitting objective.

- [Training losses and deprecated alias routing](https://brainx.chaobrain.com/braintools/apis/metric.html)
