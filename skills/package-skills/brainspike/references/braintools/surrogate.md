# Choose BrainSpike surrogate gradients

A surrogate emits binary threshold spikes in the forward pass and supplies a selected derivative in the backward pass. Apply it to voltage relative to threshold on an explicit dimensionless scale.

## Choose and connect the surrogate

| API | Use |
|---|---|
| `braintools.surrogate.InvSquareGrad()` | Use for the verified custom LIF example; retain the scale and reset convention when comparing temporal gradients. |
| `braintools.surrogate.ReluGrad()` | Use when reproducing the documented NEST neuron's default surrogate configuration; check which output is actually exposed by the training path. |
| `brainspike.Neuron(..., spk_fun=surrogate)` | Store a reusable surrogate on a custom neuron and call `self.spk_fun` from its spike computation. |
| `braintools.surrogate.Surrogate` | Subclass only when the scientific method requires a derivative absent from documented built-ins. |

## Verify hard forward spikes and nonzero backward signal

```python
import braintools
import brainunit as u
import jax

surrogate = braintools.surrogate.InvSquareGrad()
relative_voltage = u.math.asarray([-0.1, 0.1])
spikes = surrogate(relative_voltage)
gradient = jax.grad(lambda x: u.math.sum(surrogate(x)))(relative_voltage)
assert u.math.all(spikes == u.math.asarray([0., 1.]))
assert u.math.all(u.math.isfinite(gradient))
assert u.math.all(gradient > 0)
```

In the bundled custom LIF, the surrogate argument is `(voltage-threshold)/(10*u.mV)`. The scale changes the effective derivative width in physical voltage; keep it fixed when comparing surrogate families. The custom update implements its soft reset explicitly, so changing a base-class option alone does not rewrite that equation.

Open `scripts/native_workflows.py` for the complete neuron and `references/native-training.md` for native BPTT/pp-prop. Verify finite nonzero gradients on the actual recurrent parameter path; a trainable readout bias can decrease loss while temporal credit remains broken.

A surrogate configured on a neuron does not guarantee that `SimulationResult.spikes()` preserves its derivative. The recorded hard-spike-count control in `references/simulator-training.md` has zero gradient. Use that reference for the separately verified continuous-voltage Simulator objective.

Do not compare a surrogate derivative with finite differences of hard threshold spikes and expect equality. Use finite differences for a smooth objective when its mathematical derivative is the intended oracle, or compare matched BPTT and online estimators under the same surrogate.

## Source

[BrainTools surrogate API](https://brainx.chaobrain.com/braintools/apis/surrogate.html). The verified custom-neuron pattern targets BrainTools 0.4.0.
