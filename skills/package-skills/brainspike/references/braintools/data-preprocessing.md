# Encode inputs for BrainSpike

Use a spike encoder when the scientific input representation requires binary events. Keep analog current protocols in `references/braintools/input-current.md`; encoding changes the representation and is not a generic normalization step.

## Choose the representation

| API | Use |
|---|---|
| `braintools.spike.LatencyEncoder(...)` | Use when feature magnitude should determine first-spike timing; set the physical timestep and represented window. |
| `braintools.spike.PoissonEncoder(...)` | Use when stochastic rate coding is part of the model; declare normalization, rate scale, RNG ownership, and observation window. |
| `braintools.spike.StepCurrentEncoder(...)` | Use for an analog current code; do not wrap its output as binary events. |

## Preserve the encoded time axis

```python
import brainstate
import braintools
import brainunit as u

values = u.math.asarray([0.02, 0.5, 1.0])
encoder = braintools.spike.LatencyEncoder(method="linear", normalize=True, tau=4*u.ms)
with brainstate.environ.context(dt=1*u.ms):
    spikes = encoder(values, n_time=5)
assert spikes.shape == (5, 3)
assert u.math.argmax(spikes[:, 2]) < u.math.argmax(spikes[:, 1])
assert u.math.argmax(spikes[:, 1]) < u.math.argmax(spikes[:, 0])
```

Match the encoder's feature/neuron axis to the receiving projection and keep trials distinct from encoded time. Fix normalization on training data only; do not re-fit it on validation trials. For stochastic encoders, compare empirical rates across repeated windows rather than expecting exact counts in one short sequence.

For native graphs, pass each time slice into the declared input layer through the State-aware sequence driver. A dimensionless spike tensor is not a physical current; use the model's explicit projection and current scale. The bundled `RecurrentSNN` has one external feature, so change its input dimension before supplying this three-feature example.

For a NEST graph, open `references/nest-compatible/devices.md` to express the desired spike times or generator rates through device APIs. An encoded tensor is not itself a `NodeView`. Verify the event-time/bin convention and initial-time handling before translating bins into a spike generator.

Open `references/braintools/metric.md` when evaluating encoded timing or rate accuracy, and `references/braintools/parameter-initializer.md` when the receiving projection requires deliberate weight scaling.

## Source

[BrainTools spike encoders](https://brainx.chaobrain.com/braintools/apis/spike.html). Use the BrainTools 0.4 namespace `braintools.spike` and `n_time=`.
