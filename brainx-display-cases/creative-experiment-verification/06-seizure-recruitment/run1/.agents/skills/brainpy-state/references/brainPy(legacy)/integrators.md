# Legacy BrainPy integrators

Use this child reference of `customize neuron and synpase.md` when a legacy
BrainPy neuron, synapse, or standalone differential equation needs a numerical
integrator. This reference covers `brainpy.odeint`, `brainpy.sdeint`,
`brainpy.fdeint`, and the legacy `brainpy.integrators` registry; it does not
cover `brainpy.state` or BrainCell solvers.

## Choose the equation family

Select the equation family before selecting a numerical method.

| API | Use when | Required model definition |
|---|---|---|
| `bp.odeint(f=None, method=None, var_type=None, dt=None, name=None, show_code=False, state_delays=None, neutral_delays=None, **kwargs)` | State evolves under deterministic ordinary differential equations. | `f` returns one derivative per state variable. |
| `bp.sdeint(f=None, g=None, method=None, dt=None, name=None, show_code=False, var_type=None, intg_type=None, wiener_type=None, state_delays=None)` | State includes stochastic diffusion. | `f` defines drift and `g` defines diffusion; select Ito versus Stratonovich semantics deliberately. |
| `bp.fdeint(alpha, num_memory, inits, f=None, method=None, dt=None, name=None)` | State follows a fractional differential equation with finite retained memory. | Supply derivative order `alpha`, memory length, and initial values. |
| `bp.JointEq(*eqs)` | Several first-order derivative functions form one coupled system. | Put equations in the same order as the state results expected from the integrator. |
| `bp.IntegratorRunner(target, inits=None, dt=None, monitors=None, dyn_vars=None, jit=True, numpy_mon_after_run=True, progress_bar=True, args=None, dyn_args=None, fun_monitors=None)` | A standalone integrator needs a simulation loop, initial values, dynamic arguments, and monitors. | The target must be a constructed legacy BrainPy integrator. |

## Construct a deterministic integrator

`bp.odeint` converts a derivative function into a callable one-step integrator;
the state arguments come first, `t` marks time, and remaining arguments are
model inputs or parameters.

```python
import brainpy as bp
import brainpy.math as bm

a = 0.7
b = 0.8
tau = 12.5


@bp.odeint(method='rk4', dt=0.04)
def integral(V, w, t, Iext):
    dV = V - V * V * V / 3.0 - w + Iext
    dw = (V + a - b * w) / tau
    return dV, dw


V = bm.asarray(0.0)
w = bm.asarray(0.0)
V, w = integral(V, w, 0.0, 0.5)
```

Return derivatives in state-argument order. When several derivative functions
are easier to maintain separately, combine them with `bp.JointEq` and construct
one integrator from the joint equation.

## Select an ODE method

Use an explicit Runge-Kutta method for a general deterministic baseline, an
adaptive method when local error control is required, and exponential Euler
when the model's form benefits from its automatic-differentiation construction.

### Explicit Runge-Kutta methods

| API | Description |
|---|---|
| `bp.integrators.ode.Euler(...)` | Use for the simplest first-order explicit baseline. |
| `bp.integrators.ode.MidPoint(...)` | Use for the explicit midpoint method. |
| `bp.integrators.ode.Heun2(...)` | Use for second-order Heun integration. |
| `bp.integrators.ode.Ralston2(...)` | Use for second-order Ralston integration. |
| `bp.integrators.ode.RK2(...)` | Use for a generic second-order Runge-Kutta method. |
| `bp.integrators.ode.RK3(...)` | Use for classical third-order Runge-Kutta integration. |
| `bp.integrators.ode.Heun3(...)` | Use for third-order Heun integration. |
| `bp.integrators.ode.Ralston3(...)` | Use for third-order Ralston integration. |
| `bp.integrators.ode.SSPRK3(...)` | Use when the third-order strong-stability-preserving method is intended. |
| `bp.integrators.ode.RK4(...)` | Use for the classical fourth-order fixed-step baseline. |
| `bp.integrators.ode.Ralston4(...)` | Use for fourth-order Ralston integration. |
| `bp.integrators.ode.RK4Rule38(...)` | Use for the fourth-order 3/8-rule method. |

### Adaptive and exponential methods

| API | Description |
|---|---|
| `bp.integrators.ode.RKF12(...)` | Use for the embedded Fehlberg RK1(2) method. |
| `bp.integrators.ode.RKF45(...)` | Use for Runge-Kutta-Fehlberg integration. |
| `bp.integrators.ode.DormandPrince(...)` | Use for the Dormand-Prince embedded pair. |
| `bp.integrators.ode.CashKarp(...)` | Use for the Cash-Karp embedded pair. |
| `bp.integrators.ode.BogackiShampine(...)` | Use for the Bogacki-Shampine embedded pair. |
| `bp.integrators.ode.HeunEuler(...)` | Use for the Heun-Euler embedded pair. |
| `bp.integrators.ode.ExponentialEuler(...)` | Use for exponential Euler built with automatic differentiation. |

Pass a documented method name such as `'rk4'` through `bp.odeint(method=...)`.
Do not derive registry strings from class names. Inspect
`bp.integrators.ode.get_supported_methods()` when the installed legacy release
must determine the accepted names.

## Select an SDE method

Match the numerical method to the stochastic-integral convention and Wiener
process structure declared through `bp.sdeint`.

| API | Description |
|---|---|
| `bp.integrators.sde.Euler(...)` | Use for Euler integration under the supported Ito or Stratonovich convention. |
| `bp.integrators.sde.Heun(...)` | Use for the Euler-Heun Stratonovich scheme. |
| `bp.integrators.sde.Milstein(...)` | Use for Milstein integration under the supported Ito or Stratonovich convention. |
| `bp.integrators.sde.MilsteinGradFree(...)` | Use when the derivative-free Milstein variant is required. |
| `bp.integrators.sde.ExponentialEuler(...)` | Use for first-order explicit exponential Euler SDE integration. |
| `bp.integrators.sde.SRK1W1(...)` | Use for the documented weak order-2 scalar-Wiener SRK method. |
| `bp.integrators.sde.SRK2W1(...)` | Use for the documented strong order-1.5 scalar-noise SRK method. |
| `bp.integrators.sde.KlPl(...)` | Use only when this registered scalar-Wiener scheme is explicitly required. |

`Euler` and `ExponentialEuler` exist in both ODE and SDE namespaces. Keep the
namespace visible in prose and code so the equation family cannot be confused.

## Select an FDE method

Choose the fractional derivative definition before choosing the implementation.

| API | Description |
|---|---|
| `bp.integrators.fde.CaputoEuler(...)` | Use for one-step Euler integration of a Caputo fractional equation. |
| `bp.integrators.fde.CaputoL1Schema(...)` | Use for the L1 approximation of a Caputo fractional derivative. |
| `bp.integrators.fde.GLShortMemory(...)` | Use for a Riemann-Liouville equation approximated by the Grunwald-Letnikov short-memory principle. |

The FDE constructor requires `num_memory`; treat that value as part of the model
and validation plan, not merely a runtime optimization.

## Configure and extend the registries

| API | Description |
|---|---|
| `bp.integrators.ode.set_default_odeint(method)` | Set the process-wide default ODE method. Prefer an explicit `method=` in reusable model code. |
| `bp.integrators.ode.get_default_odeint()` | Inspect the process-wide default ODE method. |
| `bp.integrators.sde.set_default_sdeint(method)` | Set the process-wide default SDE method. |
| `bp.integrators.sde.get_default_sdeint()` | Inspect the process-wide default SDE method. |
| `bp.integrators.fde.set_default_fdeint(method)` | Set the process-wide default FDE method. |
| `bp.integrators.fde.get_default_fdeint()` | Inspect the process-wide default FDE method. |
| `bp.integrators.ode.register_ode_integrator(name, integrator)` | Register a custom ODE integrator under a method name. |
| `bp.integrators.sde.register_sde_integrator(name, integrator)` | Register a custom SDE integrator under a method name. |
| `bp.integrators.fde.register_fde_integrator(name, integrator)` | Register a custom FDE integrator under a method name. |
| `bp.integrators.ode.get_supported_methods()` | Inspect registered ODE method names. |
| `bp.integrators.sde.get_supported_methods()` | Inspect registered SDE method names. |
| `bp.integrators.fde.get_supported_methods()` | Inspect registered FDE method names. |

Register a custom implementation only when no built-in family expresses the
required numerical method. Keep process-wide default changes out of imported
library modules because they alter later integrator construction globally.

## Run and verify a standalone equation

Use `IntegratorRunner` when no `DynamicalSystem` owns the integration loop.
Supply constant arguments through `args`, time-varying arrays through
`dyn_args`, and inspect the named histories in `runner.mon`.

```python
equations = bp.JointEq(
    lambda V, t, w, I: V - V**3 / 3.0 - w + I,
    lambda w, t, V, a, b: (V + a - b * w) / 12.5,
)
step = bp.odeint(equations, method='exp_auto')

runner = bp.IntegratorRunner(
    step,
    inits={'V': bm.zeros(4), 'w': bm.zeros(4)},
    monitors=['V', 'w'],
)
runner.run(
    100.0,
    args={'a': 0.7, 'b': 0.8},
    dyn_args={'I': bp.inputs.ramp_input(0.0, 1.0, 100.0)},
)

assert runner.mon.V.shape[0] == runner.mon.ts.shape[0]
```

Verify convergence by rerunning at a smaller `dt` and comparing the scientific
observable, not only whether the integrator executes. For stochastic equations,
also fix the random source and compare distributions or summary statistics.

## Official sources

- `https://brainpy.readthedocs.io/apis/integrators.html`
- Generated `brainpy.odeint`, `brainpy.sdeint`, `brainpy.fdeint`,
  `brainpy.JointEq`, and `brainpy.IntegratorRunner` pages linked by the official
  API navigation.
