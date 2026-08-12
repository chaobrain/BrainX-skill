# Legacy BrainPy analysis

Use this reference when a legacy BrainPy 2.x model needs low-dimensional phase-plane or bifurcation analysis, fast-slow analysis, or numerical fixed-point discovery in a high-dimensional state space. Keep model construction and simulation in `brainpy legacy workflow.md`.

## Choose the analyzer

Low-dimensional analyzers work from named model variables and parameters. `SlowPointFinder` instead optimizes candidate states and is the normal choice once plotting the full state space is no longer meaningful.

| API | Use when |
|---|---|
| `bp.analysis.PhasePlane1D(model, target_vars, ...)` | Inspect the vector field and fixed points of one state variable. |
| `bp.analysis.PhasePlane2D(model, target_vars, ...)` | Inspect nullclines, vector fields, fixed points, and trajectories of two state variables. |
| `bp.analysis.Bifurcation1D(model, target_pars, target_vars, ...)` | Continue equilibria of a one-variable system while sweeping one or two parameters. |
| `bp.analysis.Bifurcation2D(model, target_pars, target_vars, ...)` | Continue equilibria of a two-variable system and classify local stability while parameters change. |
| `bp.analysis.FastSlow1D(model, fast_vars, slow_vars, ...)` | Treat one variable as fast and one or more variables as slowly varying bifurcation parameters. |
| `bp.analysis.FastSlow2D(model, fast_vars, slow_vars, ...)` | Analyze a two-variable fast subsystem across one or more slow variables. |
| `bp.analysis.SlowPointFinder(f_cell, ...)` | Find fixed or slow points in a high-dimensional callable or `DynamicalSystem` from supplied candidate states. |

Enable 64-bit computation before constructing an analyzer when numerical root finding or stability classification is sensitive to precision:

```python
import brainpy as bp
import brainpy.math as bm

bm.enable_x64()
```

## Define variables, parameters, and resolution

The analyzer separates coordinates being solved, parameters being swept, and values held fixed; preserve that distinction so the resulting diagram represents the intended system.

| Argument | Description |
|---|---|
| `target_vars={'name': [low, high]}` | Give each analyzed state variable and its search interval. Names must match model or integrator arguments. |
| `target_pars={'name': [low, high]}` | Give each parameter to sweep in a bifurcation analysis. |
| `pars_update={'name': value}` | Hold model parameters or external inputs at fixed values during analysis. |
| `fixed_vars={'name': value}` | Hold non-target state variables fixed when reducing a larger system. |
| `resolutions=step` | Use one grid step for all target dimensions. |
| `resolutions={'name': step}` | Control grid density per variable or parameter; refine only regions that need it. |

Do not place a swept value in `pars_update`; a value is either varied through `target_pars` or fixed through `pars_update` for one analysis.

## Analyze a two-variable phase plane

Use `PhasePlane2D` when the decision depends on nullcline intersections, flow direction, stability, or trajectories in a two-state system.

| API | Description |
|---|---|
| `plot_nullcline(...)` | Compute and optionally plot the two zero-derivative curves. |
| `plot_vector_field(plot_method='streamplot', ...)` | Plot the local flow; use `plot_method='quiver'` for arrows instead of streamlines. |
| `plot_fixed_point(...)` | Find and classify fixed points from the analysis grid. |
| `plot_trajectory(initials, duration, ...)` | Integrate trajectories from explicit initial states on the phase plane. |
| `show_figure()` | Display accumulated analysis plots when individual plot calls use `show=False`. |

```python
import brainpy as bp
import brainpy.math as bm

bm.enable_x64()

model = bp.neurons.FHN(1)
plane = bp.analysis.PhasePlane2D(
    model=model,
    target_vars={'V': [-3.0, 3.0], 'w': [-3.0, 3.0]},
    pars_update={'I_ext': 0.8},
    resolutions={'V': 0.01, 'w': 0.01},
)
plane.plot_nullcline(x_style={'fmt': '-'}, y_style={'fmt': '-'})
plane.plot_vector_field()
plane.plot_fixed_point()
plane.plot_trajectory(
    initials={'V': [-2.0], 'w': [0.0]},
    duration=100.0,
    show=True,
)
```

**Invariant:** `target_vars` keys must identify the variables whose derivatives define the phase plane. A visually plausible plot with the wrong variable or parameter names analyzes a different system.

## Continue equilibria across parameters

Use a bifurcation analyzer when the question is how equilibria or their stability change as a parameter varies.

```python
import brainpy as bp
import brainpy.math as bm

bm.enable_x64()

model = bp.dyn.ExpIF(1, delta_T=1.0)
bifurcation = bp.analysis.Bifurcation1D(
    model=model,
    target_vars={'V': [-70.0, -55.0]},
    target_pars={'I': [0.0, 6.0]},
    resolutions={'I': 0.01},
)
bifurcation.plot_bifurcation(show=True)
```

Use a dictionary of per-parameter resolutions when a narrow transition region needs a finer grid. Use `num_rank` on two-dimensional bifurcation plots to increase the number of initial candidates only after verifying the variable and parameter ranges.

## Find high-dimensional fixed or slow points

`SlowPointFinder` minimizes state change rather than drawing a state-space grid; candidate coverage therefore determines which fixed points can be found.

| API | Description |
|---|---|
| `SlowPointFinder(f_cell, f_type=..., target_vars=..., inputs=...)` | Bind a callable or model, optionally restrict the optimized variables, and hold explicit model inputs fixed. |
| `find_fps_with_gd_method(candidates, tolerance=..., num_batch=..., num_opt=...)` | Optimize batches of candidate states with gradient descent. |
| `find_fps_with_opt_solver(candidates, opt_solver='BFGS')` | Use a named optimization solver instead of the gradient-descent path. |
| `filter_loss(tolerance)` | Remove candidates whose fixed-point loss remains above the threshold. |
| `keep_unique(tolerance)` | Remove fixed points closer than the uniqueness tolerance. |
| `compute_jacobians(points, plot=False, ...)` | Linearize the dynamics at retained points and return their Jacobians. |

```python
# Given a constructed legacy BrainPy model with Variables V and w:
finder = bp.analysis.SlowPointFinder(
    f_cell=model,
    target_vars={'V': model.V, 'w': model.w},
    inputs=[model.Iext, fixed_input],
)
finder.find_fps_with_gd_method(
    candidates={
        'V': bm.random.normal(0.0, 2.0, (1000, model.num)),
        'w': bm.random.normal(0.0, 1.0, (1000, model.num)),
    },
    tolerance=1e-5,
    num_batch=200,
)
finder.filter_loss(1e-5)
finder.keep_unique(1e-3)
jacobians = finder.compute_jacobians(finder.fixed_points)
```

Treat the last example as a composition pattern: `model`, `model.Iext`, and `fixed_input` must come from the actual model. Sample candidates around states reached in simulation when random candidates do not cover the relevant attractor basin.

## Common failures

- Enable x64 before creating analyzers, not after an analysis has already been traced.
- Do not infer exhaustive fixed-point coverage from one candidate distribution.
- Filter by loss before interpreting or deduplicating optimized points.
- Use `f_type='continuous'` when a raw callable returns continuous-time derivatives; verify the callable convention before optimization.
- Compare analysis predictions with a `DSRunner` trajectory before treating a fixed point or bifurcation branch as scientifically meaningful.

## Sources mirrored

- https://brainpy.readthedocs.io/apis/analysis.html
- https://brainpy.readthedocs.io/tutorial_analysis/lowdim_analysis.html
- https://brainpy.readthedocs.io/tutorial_analysis/highdim_analysis.html
- https://brainpy.readthedocs.io/tutorial_analysis/decision_making_model.html
