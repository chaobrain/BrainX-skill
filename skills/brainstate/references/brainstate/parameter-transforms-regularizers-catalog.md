# BrainState Parameter Transforms and Regularizers Catalog

This is the exhaustive selection reference for `brainstate.nn` parameter transforms and regularizers. Open it only from `references/brainstate/parameter-constraints-regularization.md`, after that parent has established the operational `nn.Param` workflow. Return to the parent for `ParamState` versus `nn.Param`, `nn.Const`, forward computation, optimizer targeting, and loss integration.

These are parameter-value transforms, not execution transforms such as `brainstate.transform.jit`, `grad`, or `vmap`.

## Transform contract

`nn.Transform` defines `forward()`, `inverse()`, and optional `log_abs_det_jacobian()` operations between stored and model-space values. `nn.Param` initializes its storage with `t.inverse(value)`, returns `t.forward(param.val)` from `param.value()`, and applies `t.inverse(value)` in `param.set_value()`.

The official API describes the family as bijective parameter transformations. The implementation also includes projection-like transforms such as `ClipT`, `ReluT`, and `UnitVectorT` whose forward operations are not one-to-one. Prefer a smooth bijection when unconstrained optimization and reversible change-of-variables semantics matter.

Official sources:

- https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html
- https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.Transform.html
- https://github.com/chaobrain/brainstate/blob/main/brainstate/nn/_transform.py

## Transform selection by target domain

The call forms below come from the current official `_transform.py` constructors. Check the installed-version generated class page before depending on a signature across BrainState releases.

### Identity, rescaling, and reparameterization

| Need | Class and current call form | Selection distinction |
|---|---|---|
| No constraint | `IdentityT()` | Default transform; stored and model values coincide. |
| Linear scale and shift | `AffineT(scale, shift)` | Applies a reversible affine reparameterization. |
| Box-Cox-style power map | `PowerT(lmbda=0.5)` | Select for the documented power transformation rather than a domain guarantee. |

### Positive, negative, and half-line values

| Need | Class and current call form | Selection distinction |
|---|---|---|
| Smooth value above a lower bound | `SoftplusT(lower)` | Maps the real line to `(lower, infinity)` with a softplus forward map. |
| Smooth value below an upper bound | `NegSoftplusT(upper)` | Maps the real line to `(-infinity, upper)` with a negative-softplus forward map. |
| Exponential value above a lower bound | `ExpT(lower)` | Maps the real line to `(lower, infinity)` with an exponential forward map. |
| Logarithmic forward reparameterization | `LogT(lower)` | `forward(x)` maps `(lower, infinity)` to the real line; its direction differs from `ExpT`. |
| ReLU lower bound | `ReluT(lower_bound=0.0)` | Projection-like lower bound; negative inputs collapse and the map is not one-to-one. |
| Strictly positive value | `PositiveT()` | Maps model values to `(0, infinity)`. |
| Strictly negative value | `NegativeT()` | Maps model values to `(-infinity, 0)`. |

`SoftplusT(lower)` is the canonical smooth positive-domain choice. Do not replace it with `ReluT` or clipping when the workflow needs gradients through a reversible unconstrained parameterization.

### Bounded values

| Need | Class and current call form | Selection distinction |
|---|---|---|
| Smooth open interval | `SigmoidT(lower, upper)` | Maps the real line to `(lower, upper)` and approaches, but does not cross, either bound. |
| Tanh-based open interval | `TanhT(lower, upper)` | Uses tanh for a bounded range. |
| Softsign-based open interval | `SoftsignT(lower, upper)` | Uses softsign for a bounded range. |
| Adjustable sigmoid sharpness | `ScaledSigmoidT(lower, upper, beta=1.0)` | Adds a `beta` sharpness/temperature control to the bounded sigmoid map. |
| Hard bounded projection | `ClipT(lower, upper)` | Clips in both `forward()` and `inverse()`; it is not a smooth bijection at or beyond the bounds. |

### Structured and composite values

| Need | Class and current call form | Selection distinction |
|---|---|---|
| Monotonically increasing entries | `OrderedT()` | Produces ordered output values. |
| Probability simplex | `SimplexT()` | Stick-breaking map to non-negative entries summing to one. An unconstrained final axis of length `n` produces a simplex axis of length `n + 1`. |
| Unit L2 norm | `UnitVectorT()` | Normalizes the input vector; the forward map is projection-like and its inverse returns the supplied value. |
| Sequential composition | `ChainT(*transforms)` | Applies transforms left to right in the supplied order and inverses in reverse order. |
| Selective transformation | `MaskedT(mask, transform, safe_value=1.0)` | Applies one transform only at Boolean-mask positions. |

Use the documented composition order:

```python
chained = nn.ChainT(
    nn.AffineT(scale=2.0, shift=1.0),
    nn.SoftplusT(lower=0.0),
)
value = chained.forward(jnp.array(0.0))
```

`AffineT` runs first and `SoftplusT` runs second.

Official sources:

- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html
- https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html
- https://github.com/chaobrain/brainstate/blob/main/brainstate/nn/_transform.py

## Regularization contract

`nn.Regularization` defines three operations:

- `loss(value)` returns the scalar penalty added to an objective.
- `sample_init(shape)` samples an initial value from the implied prior.
- `reset_value()` returns the value used by `nn.Param.reset_to_prior()`.

`nn.Param.reg_loss()` returns zero for a fixed parameter or a parameter without `reg=`; otherwise it evaluates `reg.loss(param.value())`. Attaching `reg=` never adds the result to the training objective automatically.

`nn.ChainedReg(*regularizations, weight=1.0, fit_hyper=False)` sums its component losses and multiplies the result by the overall weight. Its `sample_init()` and `reset_value()` use the first component.

Official sources:

- https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html
- https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.Regularization.html
- https://brainx.chaobrain.com/brainstate/apis/generated/brainstate.nn.ChainedReg.html
- https://github.com/chaobrain/brainstate/blob/main/brainstate/nn/_regularization.py

## Classical and structural regularizer selection

| Modeling intent | Official API |
|---|---|
| Abstract regularization contract | `Regularization` |
| L1/Lasso sparsity | `L1Reg` |
| L2/Ridge shrinkage | `L2Reg` |
| Combined L1 and L2 | `ElasticNetReg` |
| Robust penalty | `HuberReg` |
| Group sparsity | `GroupLassoReg` |
| Total variation or local smoothness | `TotalVariationReg` |
| Soft maximum-norm pressure | `MaxNormReg` |
| Entropy pressure | `EntropyReg` |
| Orthogonal structure | `OrthogonalReg` |
| Spectral-norm structure | `SpectralNormReg` |
| Several penalties on one value | `ChainedReg` |

Use the generated class page for exact axes, reduction behavior, weights, and initialization arguments; these differ by regularizer and should not be inferred from neighboring classes.

Official source: https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html

## Prior-distribution regularizer selection

Prior regularizers encode distributional assumptions for Bayesian-inspired parameter estimation. Match both the scientific assumption and the prior's support. A regularizer contributes a preference through the loss; attach a transform separately when the parameter must remain in that support by construction.

| Parameter assumption | Official API |
|---|---|
| Gaussian prior | `GaussianReg` |
| Student's t prior | `StudentTReg` |
| Cauchy prior | `CauchyReg` |
| Bounded uniform prior | `UniformReg` |
| Value in `[0, 1]` under a Beta prior | `BetaReg` |
| Positive value under a log-normal prior | `LogNormalReg` |
| Positive value under an exponential prior | `ExponentialReg` |
| Positive value under a Gamma prior | `GammaReg` |
| Variance under an inverse-Gamma prior | `InverseGammaReg` |
| Scale-invariant log-uniform/Jeffreys prior | `LogUniformReg` |
| Strong sparsity with heavy tails | `HorseshoeReg` |
| Variable selection | `SpikeAndSlabReg` |
| Probability simplex | `DirichletReg` |

Official source: https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html

## Interaction safeguards

- Keep transform direction explicit: `nn.Param.value()` calls `forward()`, while initialization and `set_value()` call `inverse()`.
- Do not treat `LogT` and `ExpT` as interchangeable; their documented forward directions are opposite.
- Do not treat a projection-like transform as a smooth bijective change of variables.
- Do not assume `SimplexT` preserves the final-axis length.
- Apply `ChainT` in listed order and invert it in reverse order.
- Use `MaskedT` for Boolean-mask selection instead of hand-rolling selective transformation.
- Current source makes transformed-value caching explicit: `param.cache()` populates it, writes invalidate it, and `param.value()` otherwise recomputes without populating it.
- `reset_to_prior()` uses `reg.reset_value()`; use `reg.sample_init(shape)` separately when a random prior draw is required.
- Return to `parameter-constraints-regularization.md` for objective integration and complete workflow code.

## Mirror source URLs

- https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html
- https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html
- https://github.com/chaobrain/brainstate/blob/main/brainstate/nn/_param.py
- https://github.com/chaobrain/brainstate/blob/main/brainstate/nn/_transform.py
- https://github.com/chaobrain/brainstate/blob/main/brainstate/nn/_regularization.py
