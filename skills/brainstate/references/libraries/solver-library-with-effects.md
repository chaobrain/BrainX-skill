# BrainCell Solver Library With Effects

## Purpose

Catalog BrainCell integration and solver choices with their expected modeling consequences.

## Source Pages

* https://brainx.chaobrain.com/braincell/concepts/integration.html
* https://brainx.chaobrain.com/braincell/apis/integration.html
* https://brainx.chaobrain.com/braincell/examples/integration_methods.html

## Core Model

* `DiffEqState` is a state variable that carries its derivative.
* `DiffEqModule` exposes a right-hand side over integrable states.
* Cells implement this protocol, so solvers can advance them without user code touching the protocol in ordinary modeling.
* `braincell.quad` holds a registry of solvers selectable by name.
* `get_integrator` resolves an integrator from a string or callable; `register_integrator` adds a function to the global registry; `all_integrators` exposes registered names.

## Solver Families

| Family | Names / APIs | Use |
| --- | --- | --- |
| Exponential Euler | `exp_euler`, `ind_exp_euler`, `exp_exp_euler` | Good default family for HH-style gating; stable at practical point-neuron step sizes. |
| Explicit Runge-Kutta | `euler`, `midpoint`, `rk2`, `rk3`, `rk4`, `heun2`, `heun3`, `ssprk3`, `ralston2`, `ralston3`, `ralston4` | Useful for comparisons and non-stiff problems; may need tiny time steps for stiff HH dynamics. |
| Implicit | `backward_euler`, `implicit_euler`, `implicit_exp_euler`, `implicit_rk4` | Use when implicit stepping is required; inspect official API and examples first. |
| Composite / cable | `cn_exp_euler`, `cn_rk4`, `splitting`, `staggered`, `dhs_voltage` | Designed for multicompartment/cable coupling; delegate geometry tasks to `skills/braincell/SKILL.md`, which selects its multicompartment parent. |

List available names at runtime:

```python
import braincell.quad as quad

sorted(quad.all_integrators)
```

## Single-Compartment Defaults

* Use `exp_euler` for high-precision single-cell HH traces and electrophysiology-style examples.
* Use `ind_exp_euler` for large vectorized point-neuron or network simulations when speed matters.
* Use `rk4` when the user asks for classical Runge-Kutta comparison or explicitly requests it.
* Treat method-dependent waveform changes as numerical artifacts until proven otherwise.

## Solver Effects From Official HH Comparison

The official integration-methods example compares `exp_euler` and `ind_exp_euler` on HH dynamics:

* `exp_euler` produces smoother curves and more accurately fits tightly coupled gating variables and membrane potential.
* `ind_exp_euler` updates state variables independently, can deviate slightly during rapid changes, and is faster for large-scale simulations.
* For high-precision single-cell simulations, prefer `exp_euler`.
* For large-scale network simulations with thousands of neurons, prefer `ind_exp_euler` when the approximation is acceptable.

## Usage Pattern

```python
class HH(braincell.SingleCompartment):
    def __init__(self, size, solver="exp_euler"):
        super().__init__(size, V_th=20. * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))
        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))
        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))

hh_exp = HH(1, solver="exp_euler")
hh_ind = HH(1, solver="ind_exp_euler")
```

Always keep `dt` explicit in a BrainState environment context:

```python
with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, 100. * u.ms, brainstate.environ.get_dt())
    vs, spikes = brainstate.transform.for_loop(step, times)
```

## Common Mistakes -> Fix

* Choosing solver blindly -> name the solver and why it fits the task.
* Using too-large `dt` -> reduce `dt` before blaming model biology.
* Treating solver differences as biological findings -> compare methods and report as numerical effects.
* Using `rk4` as the default for stiff HH gating -> prefer exponential-Euler variants unless comparison requires RK.
* Using cable/composite solvers in a point-neuron task -> only use them when morphology/cable coupling exists.
* Forgetting units on `dt` -> use `0.01 * u.ms`, not `0.01`.
