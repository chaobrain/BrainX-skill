# BrainCell solver library with effects

Use this reference when selecting a registered BrainCell integrator, comparing solver-dependent traces, or deciding whether a numerical difference is stable enough to interpret. Keep `solver` and the unit-aware time step explicit.

## Mental model

BrainCell cells expose their integrable `DiffEqState` through the `DiffEqModule` protocol, while a solver from `braincell.quad` advances those states by one environment-defined `dt`.

Solver order does not by itself determine suitability. Hodgkin-Huxley gating mixes fast and slow time scales, so stability and coupling assumptions often matter more than formal order.

## Selection map

| Need | Start with | Numerical consequence |
|---|---|---|
| Accurate single-cell HH waveform | `exp_euler` | Couples the exponential-Euler update across interacting states; the official comparison reports smoother traces and better agreement during rapid changes than the independent variant. |
| Large vectorized point-cell simulation | `ind_exp_euler` | Linearizes each `DiffEqState` independently; it is faster but can deviate during rapid voltage-gate coupling. |
| Smooth non-stiff subsystem or explicit-method comparison | `rk4` or another explicit Runge-Kutta method | Higher order improves smooth-problem accuracy at fixed `dt`, but explicit stability can still fail on stiff neuron dynamics. |
| Very stiff or near-equilibrium dynamics | `backward_euler` or `implicit_exp_euler` | Implicit stepping trades a solve at each step for stronger stability. |
| Multicompartment cable | `staggered`, `cn_exp_euler`, or another documented cable scheme | These methods account for spatial voltage coupling; select with the multicompartment workflow and verify both temporal and CV convergence. |

Pass the solver name explicitly. The solver guide notes that `SingleCompartment` may otherwise default to `rk2`, which is not the normal starting choice for stiff HH dynamics.

## Inspect the installed registry

The registry is the source of truth for the current installation; static documentation may differ across releases.

| API | Description |
|---|---|
| `braincell.quad.get_integrator(name_or_callable)` | Use to resolve a registered string name or accept an integrator callable; an unknown name raises `ValueError`. |
| `braincell.quad.get_registry()` | Use to inspect registered names and metadata such as category, order, and description. |
| `braincell.quad.all_integrators` | Use as the backwards-compatible mapping view of the process-wide registry. |
| `braincell.quad.register_integrator(...)` | Use only when implementing a validated custom step function; it adds the function to the global registry. |

```python
import braincell.quad as quad


registry = quad.get_registry()
names = sorted(registry.names())
exp_entry = registry.entry("exp_euler")

assert "exp_euler" in names
assert callable(quad.get_integrator("exp_euler"))
print(exp_entry.category, exp_entry.order, exp_entry.description)
```

Inspect metadata rather than assuming every release exposes the same aliases or defaults.

## Registered solver families

### Exponential integrators

Use these for stiff gating dynamics or composite exponential updates.

| Solver name | Description |
|---|---|
| `exp_euler` | Coupled exponential Euler; use as the accuracy-oriented point-neuron starting choice. |
| `ind_exp_euler` | Per-state exponential Euler; use when its speed advantage justifies the independent-state approximation. |
| `exp_exp_euler` | Exponential cable update combined with exponential-Euler channel updates; use only for a compatible cell workflow. |

Do not confuse the `ind_exp_euler` solver with the `IndependentIntegration` mixin. The solver linearizes states independently; the mixin removes a subsystem from its parent's integration loop so it can use its own solver.

### Explicit Runge-Kutta integrators

Use these for method comparisons or smooth, non-stiff `DiffEqModule` systems. Reduce `dt` aggressively when applying them to HH dynamics.

| Solver name | Description |
|---|---|
| `euler` | First-order forward Euler. |
| `midpoint` | Explicit midpoint, also called modified Euler. |
| `rk2` | Generic second-order Runge-Kutta. |
| `heun2` | Heun's second-order Runge-Kutta. |
| `ralston2` | Ralston's second-order Runge-Kutta. |
| `rk3` | Classical third-order Runge-Kutta. |
| `heun3` | Heun's third-order Runge-Kutta. |
| `ssprk3` | Third-order strong-stability-preserving Runge-Kutta. |
| `ralston3` | Ralston's third-order Runge-Kutta. |
| `rk4` | Classical fourth-order Runge-Kutta. |
| `ralston4` | Ralston's fourth-order Runge-Kutta. |

Higher order reduces truncation error on smooth stable problems. It does not prevent divergence when `dt` lies outside the explicit method's stability region.

### Implicit and cable-aware integrators

| Solver name | Description |
|---|---|
| `backward_euler` | Linearized backward Euler for strongly stable first-order stepping. |
| `implicit_euler` | Implicit backward-Euler stepping. |
| `implicit_exp_euler` | Implicit voltage update with exponential-Euler channel updates. |
| `implicit_rk4` | Implicit voltage update with explicit RK4 channel updates. |
| `cn_exp_euler` | Crank-Nicolson voltage update with exponential-Euler channels. |
| `cn_rk4` | Crank-Nicolson voltage update with explicit RK4 channels. |
| `splitting` | Operator-splitting method for a multicompartment cell. |
| `staggered` | Staggered multicompartment cable update. |
| `dhs_voltage` | Specialized voltage-category scheme; inspect its installed registry metadata and use only in a documented compatible workflow. |

Open `references/multicompartment/multicompartment-cell-workflow.md` before choosing among cable solvers. Geometry, cable properties, and CV policy are part of the numerical problem.

## Compare solver effects

Run the same declaration and initial condition under each solver. Change only the solver first; then refine `dt` independently for each method.

```python
import braincell
import brainstate
import brainunit as u


class HH(braincell.SingleCompartment):
    def __init__(self, solver):
        super().__init__(size=1, solver=solver)

        self.na = braincell.ion.SodiumFixed(1, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(1))

        self.k = braincell.ion.PotassiumFixed(1, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(1))

        self.IL = braincell.channel.IL(
            1,
            E=-54.387 * u.mV,
            g_max=0.03 * u.mS / u.cm**2,
        )


def simulate(solver, dt):
    cell = HH(solver)
    cell.init_state()

    def step(t):
        with brainstate.environ.context(t=t):
            cell.update(10.0 * u.nA / u.cm**2)
        return cell.V.value

    with brainstate.environ.context(dt=dt):
        times = u.math.arange(0.0 * u.ms, 100.0 * u.ms, dt)
        voltage = brainstate.transform.for_loop(step, times)
    return times, voltage


times, v_exp = simulate("exp_euler", 0.1 * u.ms)
_, v_ind = simulate("ind_exp_euler", 0.1 * u.ms)

assert v_exp.shape == v_ind.shape
assert v_exp.shape[0] == times.shape[0]
```

The official HH comparison reports:

- `exp_euler` produces smoother curves and follows tightly coupled voltage-gate changes more accurately.
- `ind_exp_euler` can deviate slightly during rapid changes but is faster for large simulations.
- Both can generate spikes; method-dependent peak shape and timing remain numerical effects until convergence is shown.

Do not mix solver selection with a channel, current, initialization, or time-step change in the same comparison.

## Verify temporal and spatial convergence

Use an observable tied to the scientific question, not only visual trace similarity.

1. Run the selected solver at `dt` and `dt / 2`.
2. Compare peak voltage, spike time, latency, integral, or another declared observable.
3. Repeat with a second suitable solver when the result is sensitive or surprising.
4. For `Cell`, refine the CV policy as well as `dt`.
5. Accept the cheaper configuration only when both refinements change the observable by less than the stated tolerance.

If `rk4` produces NaN or extreme voltage while an exponential method remains bounded, reduce `dt` before diagnosing the biological model. If two converged methods disagree, inspect model equations, event handling, and coupling assumptions instead of choosing the preferred trace.

## Route advanced integration

Open the official advanced integration guide when a subsystem needs `IndependentIntegration`, when authoring a registered integrator, or when inspecting the currently unsupported stochastic `DiffEqState.diffusion` slot. Do not register a custom solver merely to rename or wrap a built-in method.

## Common failures

- Choosing a solver by formal order alone: account for stiffness and voltage-gate coupling.
- Leaving the constructor default implicit: pass `solver=...` so the numerical method is reviewable.
- Using a bare `dt`: provide a BrainUnit time quantity for cellular simulations.
- Treating one same-`dt` comparison as convergence: refine `dt` separately for each solver.
- Attributing a shifted spike or waveform to biology: hold the model fixed and test solver and time-step sensitivity.
- Selecting a cable solver for `SingleCompartment`: reserve spatial schemes for a compatible multicompartment model.
- Comparing `Cell` solvers without refining CVs: temporal convergence does not establish spatial convergence.

## Sources

- Integration concept: https://brainx.chaobrain.com/braincell/concepts/integration.html
- Integration API: https://brainx.chaobrain.com/braincell/apis/integration.html
- Choosing and using solvers: https://brainx.chaobrain.com/braincell/integration/solvers.html
- Advanced integration: https://brainx.chaobrain.com/braincell/integration/advanced.html
- Integration-method effects: https://brainx.chaobrain.com/braincell/examples/integration_methods.html
