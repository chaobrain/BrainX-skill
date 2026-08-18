# BrainCell custom ion channel authoring

Use this reference only after `references/ion-library.md` and `references/channel-library.md` confirm that no built-in or literature-derived mechanism fits. For a multicompartment task, establish the `Cell` workflow in `references/multicompartment/multicompartment-cell-workflow.md` before applying the authoring patterns here.

## Mental model

An ion owns reversal-potential and concentration information; a channel owns gating State and computes current from voltage plus the `IonInfo` supplied by its declared `root_type`.

Keep custom dynamics inside BrainCell's lifecycle:

`declare dependency and parameters -> allocate/reset State -> compute derivatives -> compute current -> integrate through the cell -> validate against a reference`

Do not replace this lifecycle with detached ODE functions or raw JAX arrays.

## Choose the channel extension level

| Use | Need |
|---|---|
| `braincell.channel._base.HH` | Use for independent Hodgkin-Huxley gates. Declare `Gate` objects, implement one recognized kinetic form per gate, and implement `current()`. |
| `braincell.channel._base.Markov` | Use for coupled probabilistic channel states. Declare transitions, one dependent state, a conservation total, transition-rate methods, and current from open-state probability. |
| `braincell.Channel` | Use for dynamics that HH and Markov templates cannot express. Implement State initialization/reset, derivatives, required lifecycle hooks, and current directly. |
| `register_channel` | Use when `mech.Channel(...)` must resolve the custom channel by name. Register a unique name when the defining module is imported. |

`braincell.channel._base` is a lower-level authoring module. Inspect the installed version before relying on it across releases.

## Define the channel contract

| API | Description |
|---|---|
| `root_type` | Set to the required ion class, ordered joint-ion dependency, or `HHTypedNeuron` for a root-cell channel; BrainCell uses it to validate attachment and supply the correct arguments. |
| `braincell.IonInfo` | Read `Ci`, `Co`, `E`, and `valence` from this value passed by the ion owner; do not reach into the parent ion object. |
| `init_state(V, *ion_info, batch_size=None)` | Allocate channel State for first use. |
| `reset_state(V, *ion_info, batch_size=None)` | Restore the channel's documented initial or steady-state condition without changing its declaration. |
| `compute_derivative(V, *ion_info)` | Write every integrable channel State derivative for the current voltage and ion information. |
| `current(V, *ion_info)` | Return the channel current with the same unit convention expected by the owning cell. |
| `pre_integral(V, *ion_info)` | Override only when work must occur before the solver advances State. |
| `post_integral(V, *ion_info)` | Override only when work must occur after the solver advances State. |

For a single-ion channel, the method argument order follows `root_type`. For a mixed-ion channel, it follows the ordered joint type exactly.

## Author an HH-template channel

Use `HH` when each gate follows either an `inf/tau` or `alpha/beta` first-order equation.

| API | Description |
|---|---|
| `Gate(name, power=..., phi=...)` | Declare one gating State, its conductance exponent, and an optional direct temperature factor. |
| `f_<gate>_inf(...)` and `f_<gate>_tau(...)` | Implement the steady-state and time-constant form; `tau` returns a plain number whose natural unit is milliseconds. |
| `f_<gate>_alpha(...)` and `f_<gate>_beta(...)` | Implement the forward/backward-rate form; rates return plain numbers whose natural unit is inverse milliseconds. |
| `conductance_factor(V, *ion_info)` | Use inside `current()` to obtain the product of all declared gate values raised to their configured powers. |

```python
import braincell
import brainunit as u
import jax.numpy as jnp
from braincell import IonInfo
from braincell.channel._base import Gate, HH
from braincell.mech import register_channel


@register_channel("DemoK")
class DemoKChannel(HH):
    root_type = braincell.ion.Potassium
    gates = (Gate("n", power=4, phi=2.0),)

    def __init__(
        self,
        size,
        g_max=1.0 * u.mS / u.cm**2,
    ):
        super().__init__(size=size)
        self.g_max = g_max

    def current(self, V, K: IonInfo):
        return (
            self.g_max
            * self.conductance_factor(V, K)
            * (K.E - V)
        )

    def f_n_inf(self, V, K: IonInfo):
        _ = K
        voltage_mV = V.to_decimal(u.mV)
        return 1.0 / (
            1.0 + u.math.exp(-(voltage_mV + 35.0) / 10.0)
        )

    def f_n_tau(self, V, K: IonInfo):
        _ = (V, K)
        return 5.0


V = jnp.array([-55.0]) * u.mV
k_info = IonInfo(
    Ci=jnp.array([54.4]) * u.mM,
    Co=jnp.array([2.5]) * u.mM,
    E=jnp.array([-90.0]) * u.mV,
    valence=1,
)

channel = DemoKChannel(size=1)
channel.init_state(V, k_info)
channel.n.value = jnp.array([0.2])
channel.compute_derivative(V, k_info)
current = channel.current(V, k_info)

assert channel.n.derivative.shape == (1,)
assert current.shape == (1,)
channel.n.derivative.to_decimal(u.Hz)
current.to_decimal(u.mA / u.cm**2)
```

For the `inf/tau` form, `HH` applies

`dx/dt = phi * (x_inf - x) / tau / ms`.

For the `alpha/beta` form, it applies

`dx/dt = phi * (alpha * (1 - x) - beta * x) / ms`.

Implement only one recognized form for each gate. Keep unit conversion at formula boundaries, such as `V.to_decimal(u.mV)`, and return the convention expected by the template.

## Use Markov or the lower-level channel interface

Use `Markov` when channel-state probabilities are coupled through a transition graph rather than independent gates.

| API | Description |
|---|---|
| `Transition(src, dst, forward, backward)` | Declare one bidirectional edge and the names of its rate methods. |
| `Markov.pairs` | Declare the complete transition graph. |
| `Markov.dependent_state` | Name the probability reconstructed from the conservation law instead of integrated independently. |
| `Markov.conserve` | Set the conserved total, normally `1.0` for state probabilities. |

Use `braincell.Channel` directly only when neither HH gates nor a conserved Markov graph represent the equations. In that path, allocate every `DiffEqState`, define a reproducible reset rule, write every derivative in `compute_derivative()`, and preserve units in `current()`. Leave `pre_integral()` and `post_integral()` as inherited no-ops unless the equations require an ordering-specific operation.

Do not implement all Markov probabilities as independent States when they must sum to one. Choose one dependent state and test conservation after stepping.

## Choose the ion extension level

| Use | Need |
|---|---|
| `braincell.ion._base.FixedIon` or a matching concrete ion base | Use for a custom ion with fixed `Ci`, `Co`, `E`, and `valence`. Preserve the ion/channel ownership contract. |
| `braincell.ion._base.InitNernstIon` | Use when concentrations stay fixed but initialization and reset must compute `E` from concentrations, valence, and temperature. |
| `braincell.ion._base.DynamicNernstIon` | Use when intracellular concentration is dynamic. Define its initializer, `derivative(Ci, V, total_current=...)`, and whether the total owned current is required. |
| `register_ion` | Use when `mech.Ion(...)` must resolve the custom ion by name. Register a unique name when the defining module is imported. |

`braincell.ion._base` is a lower-level authoring module. Inspect the installed version before relying on it across releases.

## Author a custom ion only when required

Use a custom ion when the missing behavior belongs to reversal potential or concentration dynamics, not when only the current equation is new.

1. Choose `FixedIon`, `InitNernstIon`, or `DynamicNernstIon` from the concentration behavior.
2. Define `Ci`, `Co`, `E`, `valence`, temperature, and initialization with BrainUnit quantities.
3. For a dynamic ion, implement `derivative(Ci, V, total_current=...)` and declare whether the total current of owned channels is required.
4. Package ion state through the inherited `IonInfo` path; do not pass the ion object directly into channels.
5. Register the concrete class only when string-based `mech.Ion(...)` declarations need it.

This custom calcium pool uses the owned channel current for influx and a saturating removal pump:

```python
import braincell
import braintools
import brainunit as u
import jax.numpy as jnp
from braincell.ion import Calcium
from braincell.ion._base import DynamicNernstIon

DEFAULT_TEMP = u.celsius2kelvin(36.0)


class DemoSaturatingCalcium(Calcium, DynamicNernstIon):
    uses_total_current = True

    def __init__(
        self,
        size,
        temp=DEFAULT_TEMP,
        depth=1.0 * u.um,
        pump_max=1.0e-4 * u.mM / u.ms,
        pump_half=1.0e-4 * u.mM,
        Co=None,
        Ci_initializer=2.4e-4 * u.mM,
    ):
        super().__init__(size=size)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )
        self.depth = braintools.init.param(
            depth, self.varshape, allow_none=False
        )
        self.pump_max = braintools.init.param(
            pump_max, self.varshape, allow_none=False
        )
        self.pump_half = braintools.init.param(
            pump_half, self.varshape, allow_none=False
        )

    def derivative(self, Ci, V, total_current=None):
        _ = V
        influx = total_current / (
            self.valence * u.faraday_constant * self.depth
        )
        influx = u.math.maximum(influx, u.math.zeros_like(influx))
        pump = self.pump_max * Ci / (Ci + self.pump_half)
        return influx - pump


V = jnp.array([-65.0]) * u.mV
initial_Ci = 2.4e-4 * u.mM

calcium = DemoSaturatingCalcium(size=1, Ci_initializer=initial_Ci)
calcium.add(
    ICaT=braincell.channel.CaT_HM1992(
        size=1,
        g_max=2.0 * u.mS / u.cm**2,
    )
)
calcium.init_state(V)
calcium.compute_derivative(V)

total_current = calcium.current(V, include_external=True)
expected = calcium.derivative(
    calcium.Ci.value,
    V,
    total_current=total_current,
)
assert u.math.allclose(
    calcium.Ci.derivative.to_decimal(u.mM / u.ms),
    expected.to_decimal(u.mM / u.ms),
)

E_before = calcium.E
calcium.Ci.value = jnp.array([1.0e-3]) * u.mM
assert not u.math.allclose(
    E_before.to_decimal(u.mV),
    calcium.E.to_decimal(u.mV),
)

calcium.reset_state(V)
assert u.math.allclose(
    calcium.Ci.value.to_decimal(u.mM),
    initial_Ci.to_decimal(u.mM),
)
```

Use `CalciumFixed`, `CalciumInitNernst`, and `CalciumDetailed` as the three concrete source patterns. Do not author a custom ion merely to change a channel's `g_max`, gating kinetics, or current formula.

## Validate the extension

Validate the mechanism at three levels:

| Level | Required checks |
|---|---|
| Formula | Compare steady state, time constants or transition rates, current sign, reversal behavior, and units against the governing equations. |
| Lifecycle | Exercise `init_state()`, `reset_state()`, derivative computation, one solver step, batching, and the expected `root_type` failure. |
| Model | Compare a voltage-clamp curve, analytic limit, published trace, or trusted simulator under matched parameters, temperature, units, `dt`, and solver. |

For an ohmic current written as `g * gates * (E - V)`, verify that current approaches zero at `V == E`. For an HH channel, perturb each gate away from steady state and verify that the derivative points back toward its expected equilibrium. For a Markov channel, verify non-negative probabilities and conservation.

Add a co-located test when the extension becomes repository code. Registration success alone does not validate kinetics.



## Sources

- Ions & Channels: https://brainx.chaobrain.com/braincell/concepts/ions_channels.html
- Channels tutorial: https://brainx.chaobrain.com/braincell/tutorials/channel.html
- Ions tutorial: https://brainx.chaobrain.com/braincell/tutorials/ion.html
- Extending BrainCell: https://brainx.chaobrain.com/braincell/developer/extending.html
