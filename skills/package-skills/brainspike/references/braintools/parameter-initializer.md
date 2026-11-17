# Initialize BrainSpike states and parameters

Use Braintools initializers to resolve a scientific value or distribution into the physical shape consumed by a neuron or projection. Resting voltage specifies dynamics; it does not substitute for an intentional initial membrane voltage.

## Choose the initializer

| API | Use |
|---|---|
| `braintools.init.Constant(value)` | Use for a known initial voltage or fixed parameter; preserve the physical unit of the supplied value. |
| `braintools.init.Normal(mean=..., std=...)` | Use for Gaussian heterogeneity; validate domain bounds because negative draws remain possible. |
| `braintools.init.param(initializer, sizes=..., rng=...)` | Resolve a distribution before graph construction with explicit shape and RNG ownership. |
| `braintools.init.XavierNormal()` | Use for a dimensionless native-layer matrix when fan scaling fits the chosen activation; check the layer's input/output orientation. |

## Freeze an initial membrane distribution

```python
import brainspike
import brainstate
import braintools
import brainunit as u
import numpy as np

initial_voltage = braintools.init.param(
    braintools.init.Normal(mean=-65.0*u.mV, std=1.0*u.mV),
    sizes=4, rng=np.random.default_rng(7),
)
sim = brainspike.Simulator(dt=0.1*u.ms)
neurons = sim.create(
    brainspike.iaf_psc_alpha, 4,
    V_initializer=braintools.init.Constant(initial_voltage),
)
assert initial_voltage.shape == (4,)
```

Sampling first and storing the sample in `Constant` preserves the same initial condition across independent `simulate()` evaluations. Resample deliberately when between-trial heterogeneity is part of the experiment. A BrainState seed alone does not replace an explicit NumPy RNG for a host-sampled initializer.

Keep physical connection weights separate from dimensionless native-layer weights. The bundled recurrent SNN multiplies a dimensionless projection output by `u.nA`; a NEST connection instead consumes the target model's required weight unit. Alpha/exponential PSC connections use current weights in the verified examples; delta connections use voltage jumps. Conductance-based targets require their own model-compatible contract.

For trainable values, initialize `ParamState` once and reset only runtime State at sequence boundaries. Open `references/native-training.md` for native graph resets or `references/simulator-training.md` for fixed-initial-condition candidate evaluation. Open `references/braintools/connectivity.md` when geometry selects edges; a distance-modulated initializer scales existing weights and does not generate topology.

## Source

[BrainTools initializer API](https://brainx.chaobrain.com/braintools/apis/init.html). The example targets BrainTools 0.4.0.
