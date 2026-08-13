# Legacy BrainPy built-in neuron models

Use this reference to select, initialize, run, and modify a built-in legacy `brainpy.dyn` point-neuron model. Use `customize neuron and synpase.md` when the required equations, state transitions, or spike rule are not represented by the built-in families.

## Choose the neuron family

Choose the simplest family that contains the state variables and firing behavior required by the scientific question; suffix variants change input/refractory handling without changing the base model family.

| Base family | Use when |
|---|---|
| `bp.dyn.Lif` | A one-state leaky integrate-and-fire model is sufficient. |
| `bp.dyn.ExpIF` | Spike initiation needs an exponential voltage term. |
| `bp.dyn.AdExIF` | Exponential spike initiation and an adaptation state are required. |
| `bp.dyn.QuaIF` | Spike initiation is represented by quadratic voltage dynamics. |
| `bp.dyn.AdQuaIF` | Quadratic voltage dynamics need an adaptation state. |
| `bp.dyn.Gif` | A generalized integrate-and-fire model needs dynamic currents or threshold adaptation. |
| `bp.dyn.Izhikevich` | A compact two-state model must reproduce varied firing patterns. |
| `bp.dyn.HH` | Full Hodgkin-Huxley sodium, potassium, and leak gating is required. |
| `bp.dyn.MorrisLecar` | Two-dimensional conductance dynamics are sufficient for excitability or oscillation studies. |
| `bp.dyn.WangBuzsakiHH` | The modified Hodgkin-Huxley formulation used for fast-spiking interneurons is required. |

### Choose a suffix variant

The reduced-neuron families expose systematic variants; select the behavior explicitly instead of treating the names as interchangeable aliases.

| Variant | Meaning |
|---|---|
| `Family` | Base model. |
| `FamilyLTC` | Liquid time-constant form of the base model. |
| `FamilyRef` | Base model with refractory-period state and `tau_ref`. |
| `FamilyRefLTC` | Liquid time-constant form with refractory-period state. |

These four forms are available for `Lif`, `ExpIF`, `AdExIF`, `QuaIF`, `AdQuaIF`, `Gif`, and `Izhikevich`. Conductance-based models expose `HH`/`HHLTC`, `MorrisLecar`/`MorrisLecarLTC`, and `WangBuzsakiHH`/`WangBuzsakiHHLTC`.

Use a `Ref` variant only when a refractory period is part of the model. Use the base variant when reset alone is sufficient.

## Set population shape

`size` defines the number or logical arrangement of neurons; `keep_size=True` preserves a multidimensional logical shape in state variables.

| Form | Result |
|---|---|
| `bp.dyn.HH(1)` | One neuron with one-dimensional state of length 1. |
| `bp.dyn.HH(10)` | A flat population of 10 neurons. |
| `bp.dyn.HH((10, 10), keep_size=True)` | A population whose state keeps shape `(10, 10)`. |
| `bp.dyn.HH((5, 4, 2), keep_size=True)` | A population whose state keeps shape `(5, 4, 2)`. |

Choose `keep_size=True` only when spatial or logical axes carry meaning. Flattened population state is simpler for ordinary connection operators.

## Initialize parameters and state

Built-in constructors accept shared scalars, per-neuron arrays, initializer objects, or callables for parameters and state initialization.

| Form | Use when |
|---|---|
| `gNa=120.0` | Every neuron shares one parameter value. |
| `gNa=bm.asarray([...])` | Each neuron has an explicit parameter value. The array must match the population shape. |
| `gNa=bp.init.Uniform(100.0, 140.0)` | Values should be sampled by a reusable initializer. |
| `gNa=lambda shape: bm.random.uniform(100.0, 140.0, shape)` | Initialization needs custom shape-dependent logic. |
| `V_initializer=bp.init.Uniform(-70.0, -60.0)` | State should start from a distribution. |

```python
import brainpy as bp
import brainpy.math as bm

neurons = bp.dyn.HH(
    size=3,
    gNa=bp.init.Uniform(min_val=110.0, max_val=130.0),
    V_initializer=bp.init.Uniform(min_val=-70.0, max_val=-60.0),
    method='exp_auto',
)

assert neurons.V.shape == (3,)
assert neurons.gNa.shape == (3,)
```

Use initializer arguments for initial state and ordinary model parameters for equation constants. Do not overwrite a state `Variable` with a plain array after construction.

## Run and monitor a built-in model

`DSRunner` supplies one input per step and records selected state variables; monitor names must match variables exposed by the chosen model.

| API | Description |
|---|---|
| `bp.DSRunner(model, monitors=[...], inputs=...)` | Bind the neuron group, monitor variables, and input policy. |
| `runner.run(duration)` | Run for a duration using configured inputs. |
| `runner.run(inputs=array)` | Consume a time-indexed input array; its leading length determines the rollout length. |
| `runner.mon[name]` | Return a `(time, population...)` monitor history. |

```python
import brainpy as bp
import brainpy.math as bm

bm.set_dt(0.1)

neurons = bp.dyn.MorrisLecar(
    3,
    V_initializer=bp.init.Uniform(-70.0, -60.0),
)
runner = bp.DSRunner(neurons, monitors=['V', 'W'])

inputs = bm.ones(int(100.0 / bm.get_dt())) * 100.0
runner.run(inputs=inputs)

assert runner.mon['V'].shape == (1000, 3)
assert runner.mon['W'].shape == (1000, 3)
```

Check the model API before choosing monitor names: reduced models commonly expose `V` and `spike`, while conductance-based models may expose gating variables such as `m`, `h`, `n`, or `W`.

## Change a parameter during simulation

Ordinary constructor parameters are static under object-oriented JIT; wrap a parameter in `bm.Variable` only when it must change between or during compiled runs.

```python
import brainpy as bp
import brainpy.math as bm

neurons = bp.dyn.HH(
    1,
    gNa=bm.Variable(bm.asarray([120.0])),
)
runner = bp.DSRunner(neurons, monitors=['V'])

current = bm.ones(int(50.0 / bm.get_dt())) * 6.0
runner.run(inputs=current)

neurons.gNa[:] = 50.0
runner.run(inputs=current, reset_state=True)
```

**Invariant:** Mark a changing parameter as `Variable` at construction. Replacing a static array or scalar after tracing does not reliably change compiled dynamics.

## Model-selection checks

- Use a reduced model for network-scale questions unless ion-channel gating is itself part of the hypothesis.
- Use adaptation families only when the adaptation state is needed in the analysis or fit.
- Use refractory variants only when the refractory interval is specified or scientifically relevant.
- Verify constructor defaults against the intended parameterization; class-family similarity does not make defaults interchangeable.
- Validate firing threshold, reset, initial voltage, integration method, and applied current before comparing models.
- Compare spike statistics and voltage ranges, not only whether the simulation completes.

## Routing

Open `customize neuron and synpase.md` when implementing new equations, reset logic, or state variables. Open `integrators.md` when selecting or configuring the ODE/SDE integrator. Open `brainpy legacy workflow.md` for networks, runner lifecycle, training, and analysis routing.

## Sources mirrored

- https://brainpy.readthedocs.io/apis/brainpy.dyn.neurons.html
- https://brainpy.readthedocs.io/tutorial_building/overview_of_dynamic_model.html
