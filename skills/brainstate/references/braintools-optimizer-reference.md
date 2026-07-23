# Braintools Optimizer Selection Reference

Use this reference to select a `braintools.optim` optimizer, learning-rate scheduler, Optax bridge, or external optimizer wrapper. It is a selection adapter, not a BrainState training-loop guide: return to `skills/brainstate/SKILL.md` for the canonical State-aware training structure. Advanced full-training workflows are outside the supplied router.

**Source URL:** https://brainx.chaobrain.com/braintools/optim/index.html

## Route From The Official Optimization Index

The index states: "Optimization guides highlight practical solvers for tuning models and experiments. Compare gradient-free Nevergrad strategies with SciPy-based routines and learn when to apply each."

Use its five routes as follows, without importing their unshown details into this reference:

| Official index route | Selection cue available on the index |
|---|---|
| Tutorial 1: `NevergradOptimizer` Tutorial | Gradient-free Nevergrad strategy |
| Tutorial 2: `ScipyOptimizer` Tutorial | SciPy-based routine |
| Tutorial 3: Getting Started with `optax` Optimizers | Optax optimizer route |
| Tutorial 4: Learning Rate Scheduling Strategies | Learning-rate scheduler selection |
| Tutorial 5: Advanced Optimizers and Techniques | Optimizer choices beyond the basic route |

**Source URL:** https://brainx.chaobrain.com/braintools/optim/index.html

## Optimizer And State Types

The index exposes the `braintools.optim` module and lists `Optimizer`, `OptaxOptimizer`, and `OptimState` before its concrete optimizer classes. For class selection, preserve the module and symbol names exactly:

```python
import braintools.optim

optimizer_type = braintools.optim.Adam
```

Concrete optimizer classes, in index order:

- `SGD`
- `Momentum`
- `MomentumNesterov`
- `Adam`
- `AdamW`
- `Adagrad`
- `Adadelta`
- `RMSprop`
- `Adamax`
- `Nadam`
- `RAdam`
- `Lamb`
- `Lars`
- `Lookahead`
- `Yogi`
- `LBFGS`
- `Rprop`
- `Adafactor`
- `AdaBelief`
- `Lion`
- `SM3`
- `Novograd`
- `Fromage`
- `SOFO`
- `SOFOScan`

The index does not provide constructor signatures, update semantics, or comparative algorithm guidance. Select the requested class here; do not infer arguments or substitute a training workflow from this catalog.

**Source URL:** https://brainx.chaobrain.com/braintools/optim/index.html

## Learning-Rate Schedulers

The scheduler base and concrete scheduler symbols are listed together. Select the exact class first, then return to the main BrainState skill for the surrounding canonical training structure.

```python
import braintools.optim

scheduler_type = braintools.optim.ExponentialDecayLR
```

Scheduler classes, in index order:

- `LRScheduler`
- `StepLR`
- `MultiStepLR`
- `ConstantLR`
- `LinearLR`
- `ExponentialLR`
- `PolynomialLR`
- `ExponentialDecayLR`
- `CosineAnnealingLR`
- `CosineAnnealingWarmRestarts`
- `WarmupCosineSchedule`
- `CyclicLR`
- `OneCycleLR`
- `ReduceLROnPlateau`
- `WarmupScheduler`
- `PiecewiseConstantSchedule`
- `ChainedScheduler`
- `SequentialLR`

**Source URL:** https://brainx.chaobrain.com/braintools/optim/index.html

## Optax Bridge

The index pairs a "Getting Started with `optax` Optimizers" route with the exported `OptaxOptimizer` symbol. Select that bridge only when the task explicitly calls for the Optax route through Braintools:

```python
import braintools.optim

optax_bridge_type = braintools.optim.OptaxOptimizer
```

The index gives no adapter constructor or Optax transformation example, so this reference does not invent one.

**Source URL:** https://brainx.chaobrain.com/braintools/optim/index.html

## SciPy And Nevergrad Wrappers

The index distinguishes "gradient-free Nevergrad strategies" from "SciPy-based routines" and lists the wrapper symbols `NevergradOptimizer` and `ScipyOptimizer`.

```python
import braintools.optim

gradient_free_wrapper_type = braintools.optim.NevergradOptimizer
scipy_wrapper_type = braintools.optim.ScipyOptimizer
```

- Select `NevergradOptimizer` for the index's gradient-free Nevergrad route.
- Select `ScipyOptimizer` for the index's SciPy-based route.
- Do not assume that either wrapper follows the ordinary BrainState gradient-step interface; the index does not state their call contracts.

**Source URL:** https://brainx.chaobrain.com/braintools/optim/index.html

## Integration Boundary

After choosing a symbol, route back to the main BrainState skill for its canonical State-aware training structure. Advanced full-training workflows remain outside this supplied router. Training operations are deliberately not repeated here because the optimization index supplies names and routes, not the BrainState training structure.

**Source URL:** https://brainx.chaobrain.com/braintools/optim/index.html
