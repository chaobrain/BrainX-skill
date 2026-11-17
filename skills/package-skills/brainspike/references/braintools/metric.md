# Select BrainSpike losses and analysis metrics

Choose the observable and its time, neuron, and trial axes before reducing a loss. A valid analysis statistic is not automatically a differentiable training objective.

## Match the metric to the task

| API | Use |
|---|---|
| `braintools.metric.squared_error(predictions, targets, reduction="mean")` | Use for an aligned continuous target and scalar MSE; normalize physical residuals by a declared scale. |
| `braintools.metric.softmax_cross_entropy_with_integer_labels(logits, labels)` | Use for mutually exclusive classes from a trained readout; validate label range and reduce the returned example losses deliberately. |
| `braintools.analysis.spike.firing_rate(spikes, width=..., dt=...)` | Use for population-rate analysis; supply physical times and preserve the time-major binary spike matrix. |
| `braintools.analysis.spike.van_rossum_distance(...)` | Use for spike-time comparison; supply spike-time trains rather than passing a dense matrix as a list of event times. |

## Normalize a voltage objective

```python
import braintools
import brainunit as u

observed = u.math.asarray([[-65., -64.], [-63., -62.]])*u.mV
predicted = observed + 1.0*u.mV
residual = (predicted-observed)/(2.0*u.mV)
loss = braintools.metric.squared_error(residual, u.math.zeros_like(residual), reduction="mean")
assert loss.shape == ()
assert u.math.allclose(loss, 0.25)
```

Keep the normalization scale fixed across candidates and conditions. Align sampling, transient removal, recording units, trial masks, and reduction axes before comparing traces. Do not let broadcasting silently pool a trial or neuron axis. A scalar mean over all samples implies equal sample weighting; choose a different reduction when the scientific target requires it.

For native surrogate-gradient training, use the differentiable model/readout output described in `references/native-training.md`. For the verified NEST Simulator path, use the continuous multimeter trace described in `references/simulator-training.md`: hard recorded spike counts have zero gradient in the tested configuration. Spike timing, rate, and synchrony metrics remain useful for evaluation, but require separate gradient evidence before optimization.

Open `references/braintools/optimizer.md` after the scalar objective and reset boundary are fixed. Open `references/perturbation-experiment-validity.md` before interpreting silent populations or intervention-dependent metrics as scientific evidence.

## Sources

[BrainTools metric API](https://brainx.chaobrain.com/braintools/apis/metric.html), [analysis API](https://brainx.chaobrain.com/braintools/apis/analysis.html). The local contracts target BrainTools 0.4.0; preserve physical `dt` rather than relying on function-specific bare-time conventions.
