# Braintools metrics

Use this reference to select supervised losses and neuroscience metrics for
classification, regression, spike trains, local field potentials, correlation,
connectivity, and pairwise comparison. Verify input orientation, units, and
reduction before treating any returned array as a scalar objective.

## Choose a metric family

| Task | Use | Important constraint |
|---|---|---|
| Independent binary or multilabel classification | Sigmoid binary cross entropy or focal loss | Logits and labels must have broadcast-compatible class structure. |
| Mutually exclusive classes | Softmax cross entropy | Choose one-hot/probability labels or the integer-label variant explicitly. |
| Continuous fitting | Squared, absolute, Huber, log-cosh, or cosine losses | Set `reduction` rather than assuming a scalar result. |
| Spike timing | Victor-Purpura or van Rossum distance | Inputs are spike-time trains, not dense spike matrices. |
| Population spiking | Firing-rate and synchrony metrics | Dense spike matrices are time-major unless the API says otherwise. |
| LFP spectrum or coupling | PSD, coherence, PAC, CSD, entropy, or phase coherence | Pass the sample interval as `dt`, not sampling frequency as `fs`. |
| Functional connectivity | Correlation and connectivity functions | Time series use `(num_time, num_signals)`. |

## Classification losses

| API | Description |
|---|---|
| `sigmoid_binary_cross_entropy(logits, labels)` | Use for independent binary classes; returns element-wise sigmoid cross entropy. |
| `softmax_cross_entropy(logits, labels)` | Use for mutually exclusive classes represented by one-hot or probability labels. |
| `softmax_cross_entropy_with_integer_labels(logits, labels)` | Use the more direct integer-index form for mutually exclusive classes. Validate every label yourself. |
| `hinge_loss(...)` | Use for binary margin classification. |
| `perceptron_loss(...)` | Use for the binary perceptron objective. |
| `multiclass_hinge_loss(...)` | Use for a multiclass margin objective. |
| `multiclass_perceptron_loss(...)` | Use for a multiclass perceptron objective. |
| `poly_loss_cross_entropy(...)` | Use when the specified method requires PolyLoss cross entropy. |
| `kl_divergence(...)` | Compare probability distributions. |
| `kl_divergence_with_log_targets(...)` | Use when predictions and targets are already in log space. |
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

**Critical exception:** JAX gather silently clamps out-of-range integer labels
in `softmax_cross_entropy_with_integer_labels`. Validate
`0 <= labels < logits.shape[-1]`; the function does not raise for invalid
labels.

## Regression and pairwise metrics

| API | Description |
|---|---|
| `squared_error(predictions, targets=None, axis=None, reduction='none')` | Use for element-wise squared error; choose `mean` or another supported reduction explicitly when a scalar is required. |
| `absolute_error(...)` | Use for element-wise absolute error. |
| `l1_loss(...)` | Use for mean absolute-error behavior according to its axis/reduction options. |
| `l2_loss(...)` | Use for squared L2 loss. |
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

## Correlation and functional connectivity

| API | Description |
|---|---|
| `cross_correlation(spikes, bin, dt)` | Calculate a cross-correlation synchrony index from a time-major spike matrix. |
| `voltage_fluctuation(voltages)` | Estimate synchrony from population voltage variance. |
| `matrix_correlation(a, b)` | Correlate upper-triangular entries of two square matrices. |
| `weighted_correlation(x, y, weights)` | Compute weighted Pearson correlation for two one-dimensional series. |
| `functional_connectivity(activities)` | Return the pairwise Pearson correlation matrix for `(num_time, num_signals)` data; NaNs are replaced with zero. |
| `functional_connectivity_dynamics(activities, window_size, step_size)` | Compare sliding-window connectivity matrices to produce functional-connectivity dynamics. |

## Spike-train metrics

| API | Description |
|---|---|
| `raster_plot(spikes, times)` | Extract neuron indices and spike times from a dense spike matrix; this is a metric/data helper, not the visualization API. |
| `firing_rate(spikes, width, dt=None)` | Average across neurons and smooth population rate with a rectangular window of physical width. |
| `victor_purpura_distance(spike_times_1, spike_times_2, cost_factor=1.0)` | Measure edit cost from insertions, deletions, and temporal shifts. |
| `van_rossum_distance(spike_times_1, spike_times_2, tau=...)` | Compare exponentially filtered spike trains. |
| `spike_train_synchrony(spikes, window_size, dt=None)` | Compute SPIKE-style synchronization on a population matrix. |
| `burst_synchrony_index(spikes, dt=None, ...)` | Measure co-occurring burst events. |
| `phase_locking_value(spikes, reference_freq, dt=None, ...)` | Measure spike phase locking to a reference oscillation. |
| `spike_time_tiling_coefficient(spikes, dt=None, tau=...)` | Compute STTC within a temporal tolerance. |
| `correlation_index(spikes, window_size, dt=None)` | Compute a windowed spike-train correlation index. |

```python
import brainunit as u
from braintools.metric import firing_rate, spike_train_synchrony

# Shape: (num_time, num_neurons)
rate = firing_rate(spikes, width=10 * u.ms, dt=1 * u.ms)
synchrony = spike_train_synchrony(
    spikes,
    window_size=5 * u.ms,
    dt=1 * u.ms,
)
```

Keep dense spike matrices and lists of spike times distinct. Distance metrics
consume event times; population rate and synchrony metrics consume matrices.

## Local field potential analysis

| API | Description |
|---|---|
| `unitary_LFP(times, spikes, neuron_type, seed=None, ...)` | Construct unitary LFP contributions from spikes; use the documented `'exc'` or `'inh'` neuron type. |
| `power_spectral_density(lfp, dt, nperseg=None, noverlap=None, freq_range=None)` | Return one-sided Welch frequencies and PSD. |
| `coherence_analysis(signal1, signal2, dt, ...)` | Return frequency-resolved magnitude-squared coherence. |
| `phase_amplitude_coupling(lfp, dt, phase_freq_range, amplitude_freq_range, ...)` | Return Tort modulation index plus phase-bin details. |
| `theta_gamma_coupling(lfp, dt, ...)` | Return theta-gamma coupling using standard bands. |
| `current_source_density(lfp, electrode_spacing, axis=0, ...)` | Estimate laminar CSD along the electrode axis. |
| `spectral_entropy(lfp, dt, ...)` | Return normalized spectral entropy. |
| `lfp_phase_coherence(lfp, dt, freq_band, ...)` | Compute pairwise phase coherence across channels. |

`power_spectral_density(..., freq_range=...)` uses boolean-mask indexing and
therefore has a data-dependent output length; that path is not JIT-compatible.
Compute the full spectrum inside JIT and filter outside when compilation is
required.

## Structured, ranking, and smoothing losses

| API | Description |
|---|---|
| `make_fenchel_young_loss(max_fun)` | Build a Fenchel-Young loss. `max_fun` must map one score vector to a scalar because it is vectorized with signature `(n)->()`. |
| `ranking_softmax_loss(scores, labels, ...)` | Use for list-wise learning-to-rank objectives. |
| `smooth_labels(labels, alpha=...)` | Apply label smoothing to one-hot targets. |

## Official source

- `https://brainx.chaobrain.com/braintools/apis/metric.html`
