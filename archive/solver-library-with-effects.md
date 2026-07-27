# BrainCell solver selection and numerical effects

Archived from `skills/brainstate/references/libraries/`. This BrainCell material
is not routed by the BrainState skill.

Use this reference when a BrainState-managed simulation contains BrainCell
differential-equation Modules and the task must select or compare their
integrator. BrainState owns State, environment, and transformed loop execution;
BrainCell owns the differential-equation protocol and solver registry. Route
cell construction, channels, ions, and morphology to the BrainCell skill.

## Selection map

| Solver family | Choose it when | Key constraint |
|---|---|---|
| `exp_euler` | Tightly coupled Hodgkin-Huxley-style states need the accurate exponential-Euler path used by the official comparison. | Prefer it for high-precision single-cell traces, then verify convergence in `dt`. |
| `ind_exp_euler` | A large vectorized point-neuron or network simulation can accept independent state updates for greater speed. | Rapid coupled changes may deviate from `exp_euler`; quantify the difference. |
| `rk4` and other explicit Runge-Kutta methods | The user requests a classical comparison or the system is demonstrably non-stiff at the chosen step. | Stability may require a smaller `dt` than exponential-Euler methods. |
| Implicit methods | The model requires implicit stepping. | Inspect the exact generated API and validate convergence before adopting the method. |
| Composite or cable methods | Multicompartment voltage coupling or cable structure determines the integration scheme. | Solver choice depends on morphology; route the model design to BrainCell. |

## Solver protocol and registry

| API | Use |
|---|---|
| `DiffEqState` | Represent an integrable state together with derivative information. |
| `DiffEqModule` | Expose the right-hand side over integrable states. Ordinary cell use relies on the implemented protocol rather than calling it manually. |
| `braincell.quad.get_integrator(method)` | Resolve a registered name or accepted callable to an integrator. |
| `braincell.quad.register_integrator(name, integrator)` | Add a custom integrator to the global registry. |
| `braincell.quad.all_integrators` | Inspect the names available in the installed BrainCell version. |

| Family | Registered names documented by the catalog |
|---|---|
| Exponential Euler | `exp_euler`, `ind_exp_euler`, `exp_exp_euler` |
| Explicit Runge-Kutta | `euler`, `midpoint`, `rk2`, `rk3`, `rk4`, `heun2`, `heun3`, `ssprk3`, `ralston2`, `ralston3`, `ralston4` |
| Implicit | `backward_euler`, `implicit_euler`, `implicit_exp_euler`, `implicit_rk4` |
| Composite and cable | `cn_exp_euler`, `cn_rk4`, `splitting`, `staggered`, `dhs_voltage` |

Check the installed registry instead of assuming every documented name exists:

```python
import braincell.quad as quad

available = sorted(quad.all_integrators)
```

## Canonical comparison workflow

Construct otherwise identical cells with explicit solvers, then run them under
the same unit-aware environment and inputs:

```python
class HH(braincell.SingleCompartment):
    def __init__(self, size, solver):
        super().__init__(size, V_th=20.0 * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))
        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))
        self.IL = braincell.channel.IL(
            size,
            E=-54.387 * u.mV,
            g_max=0.03 * (u.mS / u.cm**2),
        )


hh_coupled = HH(1, solver="exp_euler")
hh_independent = HH(1, solver="ind_exp_euler")

with brainstate.environ.context(dt=0.01 * u.ms):
    # Initialize and run both cells with identical initial State and input.
    ...
```

**Invariant:** hold the model, initial State, inputs, duration, and `dt` fixed
when comparing solvers. Then repeat at a smaller `dt`. A waveform difference is
a numerical effect until it remains stable under step-size refinement and the
modeling analysis supports a biological interpretation.

## Decision and failure rules

- Specify the solver name and the reason for choosing it.
- Use a unit-aware `dt`; do not pass bare `0.01` when the model expects time.
- Reduce `dt` before attributing instability to channel or cell biology.
- Do not default to `rk4` for stiff Hodgkin-Huxley gating without a stability
  or convergence check.
- Do not use cable-specific methods for a point-neuron model.
- Register a custom solver only when the built-in families cannot express the
  required numerical method; treat the registry mutation as process-wide.

## Official sources

- https://brainx.chaobrain.com/braincell/concepts/integration.html
- https://brainx.chaobrain.com/braincell/apis/integration.html
- https://brainx.chaobrain.com/braincell/examples/integration_methods.html
