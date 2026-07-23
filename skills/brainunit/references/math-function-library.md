# Math Function Library

Use this reference after the basic `brainunit.math` workflow is known. Route a function by what it requires from input units and what it does to output units. Array creation and purely structural manipulation are owned by `array-creation.md` and `array-mechanics.md`.

## Contents

- [Selection map](#selection-map)
- [Activation functions](#activation-functions)
- [Dimensionless evaluation](#dimensionless-evaluation)
- [Unit-changing operations](#unit-changing-operations)
- [Unit-preserving operations](#unit-preserving-operations)
- [Unit-removing operations](#unit-removing-operations)
- [Reductions and contractions](#reductions-and-contractions)
- [Einstein operations](#einstein-operations)
- [Source-backed gotchas](#source-backed-gotchas)



## Activation Functions

Activation functions do not have one shared unit rule. `relu()` and `leaky_relu()` preserve physical units; sigmoid-, exponential-, and gating-style functions evaluate dimensionless magnitudes and return plain arrays.

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `relu(x)` | Need to zero negative physical values without changing their physical kind. | Apply `max(x, 0)` element-wise and preserve the input unit. | `u.math.relu(u.math.array([-1., 2.]) * u.meter)`<br>`# Quantity([0., 2.], "m")` |
| `sigmoid(x, unit_to_scale=None)` | Need a smooth gate in `(0, 1)` from a dimensionless value or a physical value expressed relative to a chosen scale. | Evaluate the logistic function and return a plain array. | `u.math.sigmoid(u.math.array([-1., 0., 1.]) * u.mV, unit_to_scale=u.mV)`<br>`# Array([0.2689414, 0.5, 0.7310586])` |
| `softplus(x, unit_to_scale=None)` | Need a smooth approximation to ReLU whose output is dimensionless. | Evaluate `log(1 + exp(x))` after optional unit scaling. | `u.math.softplus(u.math.array([0., 1.]) * u.mV, unit_to_scale=u.mV)`<br>`# Array([0.6931472, 1.3132616])` |
| `silu(x, unit_to_scale=None)`<br>`swish(x, unit_to_scale=None)` | Need a smooth self-gated activation; `swish` is the same activation family. | Multiply the scaled dimensionless input by its sigmoid and return a plain array. | `u.math.silu(u.math.array([0., 1.]) * u.mV, unit_to_scale=u.mV)`<br>`# Array([0., 0.7310586])` |
| `glu(x, axis=-1, unit_to_scale=None)` | Need a learned-style gate encoded by splitting one tensor axis in half. | Split an even-length axis into value and sigmoid-gate halves; the output is half as long on that axis. | `x = u.math.array([[1., 2., 3., 4.]]); u.math.glu(x)`<br>`# Array([[0.9525741, 1.9640275]]); shape (1, 2)` |

Key rules:

- `unit_to_scale` converts a compatible physical input to a dimensionless magnitude before evaluation; it does not attach that unit to the output.
- `glu()` requires the selected axis length to be divisible by two.
- `silu()` and `swish()` are aliases at the activation-family level; do not create separate decision paths for them.

### Quick Reference

| API | Description and result |
|---|---|
| `relu6(x, unit_to_scale=None)` | Use when clipping a ReLU response at 6 after optional unit scaling. Returns a plain array. |
| `sparse_plus(x, unit_to_scale=None)` | Use when applying the sparse-plus activation to a dimensionless or explicitly scaled value. Returns a plain array. |
| `sparse_sigmoid(x, unit_to_scale=None)` | Use when applying the sparse-sigmoid activation to a dimensionless or explicitly scaled value. Returns a plain array. |
| `soft_sign(x, unit_to_scale=None)` | Use when smoothly mapping a dimensionless or explicitly scaled value through the soft-sign function. Returns a plain array. |
| `log_sigmoid(x, unit_to_scale=None)` | Use when computing log-sigmoid values from dimensionless or explicitly scaled input. Returns a plain array. |
| `leaky_relu(x, negative_slope=0.01)` | Use when retaining a configurable slope for negative physical values. Returns a quantity with the input unit. |
| `hard_sigmoid(x, unit_to_scale=None)` | Use when a piecewise-linear sigmoid approximation is sufficient. Returns a plain array after optional unit scaling. |
| `hard_silu(x, unit_to_scale=None)` | Use when a piecewise-linear SiLU approximation is sufficient. Returns a plain array after optional unit scaling. |
| `hard_swish` | Use when selecting the documented alias of `hard_silu`. Returns the same plain-array result as `hard_silu`. |
| `hard_tanh(x, unit_to_scale=None)` | Use when clipping a dimensionless or explicitly scaled value to the hard-tanh range. Returns a plain array. |
| `elu(x, alpha=1.0, unit_to_scale=None)` | Use when applying an exponential linear unit with configurable negative saturation. Returns a plain array. |
| `celu(x, alpha=1.0, unit_to_scale=None)` | Use when the exponential linear activation must be continuously differentiable. Returns a plain array. |
| `selu(x, unit_to_scale=None)` | Use when applying the scaled exponential linear unit to dimensionless or explicitly scaled input. Returns a plain array. |
| `gelu(x, approximate=True, unit_to_scale=None)` | Use when applying a Gaussian error linear unit with exact or approximate evaluation. Returns a plain array. |
| `squareplus(x, b=4, unit_to_scale=None)` | Use when applying a square-root-based smooth ReLU alternative with configurable `b`. Returns a plain array. |
| `mish(x, unit_to_scale=None)` | Use when applying the Mish activation to dimensionless or explicitly scaled input. Returns a plain array. |

## Dimensionless Evaluation

These functions evaluate mathematical operations whose arguments must be dimensionless.

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `exp(x, unit_to_scale=None, **kwargs)` | Need exponential growth from a dimensionless ratio, including a physical value divided by an explicit scale. | Return `e ** x` after optional unit scaling. | `u.math.exp(u.math.array([0., 1.]) * u.mV, unit_to_scale=u.mV)`<br>`# Array([1., 2.7182817])` |
| `log(x, unit_to_scale=None, **kwargs)` | Need a natural logarithm of a dimensionless ratio. | Return the natural logarithm after optional unit scaling. | `u.math.log(u.math.array([1., 2.7182817]) * u.mV, unit_to_scale=u.mV)`<br>`# Array([0., 1.])` |
| `hypot(x, y, unit_to_scale=None, **kwargs)` | Need Euclidean magnitude from two compatible components but want a dimensionless result relative to a scale. | Compute `sqrt(x**2 + y**2)` on scaled magnitudes. | `u.math.hypot(3. * u.meter, 4. * u.meter, unit_to_scale=u.meter)`<br>`# Array(5.); plain, not meters` |
| `arctan2(x, y, unit_to_scale=None, **kwargs)` | Need a quadrant-aware angle from two compatible components. | Compute the signed angle from scaled dimensionless magnitudes. | `u.math.arctan2(1. * u.mV, 1. * u.mV, unit_to_scale=u.mV)`<br>`# Array(0.7853982); radians as a plain value` |
| `corrcoef(x, y=None, rowvar=True, unit_to_scale=None, **kwargs)` | Need unit-independent linear correlation between variables. | Return Pearson correlation coefficients after optional unit scaling. | `x = u.math.array([[1., 2., 3.], [2., 4., 6.]]) * u.mV; u.math.corrcoef(x, unit_to_scale=u.mV)`<br>`# Array([[1., 1.], [1., 1.]]); shape (2, 2)` |
| `cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, unit_to_scale=None, **kwargs)` | Need covariance of values after expressing them relative to a chosen physical scale. | Estimate covariance on dimensionless scaled data. | `x = u.math.array([1., 2., 3.]) * u.mV; u.math.cov(x, unit_to_scale=u.mV)`<br>`# Array(1.); dimensionless sample covariance` |

Do not invent arguments for `**kwargs`; forward only backend arguments documented for the active BrainUnit/backend version.

For `corrcoef()` and `cov()`, `rowvar=True` treats each row as a variable and columns as observations; use `rowvar=False` when variables are stored in columns. `unit_to_scale` controls the dimensionless magnitudes evaluated, so it also fixes the numerical scale of `cov()`.

### Quick Reference

| API | Description and result |
|---|---|
| `exprel(x, **kwargs)` | Use when evaluating `(exp(x) - 1) / x` accurately near zero. Requires dimensionless input and returns a plain array. |
| `exp2(x, unit_to_scale=None, **kwargs)` | Use when computing a base-2 exponential of a dimensionless or explicitly scaled value. Returns a plain array. |
| `expm1(x, unit_to_scale=None, **kwargs)` | Use when computing `exp(x) - 1` accurately near zero from dimensionless or explicitly scaled input. Returns a plain array. |
| `log2(x, unit_to_scale=None, **kwargs)` | Use when computing a base-2 logarithm of a dimensionless or explicitly scaled value. Returns a plain array. |
| `log10(x, unit_to_scale=None, **kwargs)` | Use when computing a base-10 logarithm of a dimensionless or explicitly scaled value. Returns a plain array. |
| `log1p(x, unit_to_scale=None, **kwargs)` | Use when computing `log(1 + x)` accurately near zero from dimensionless or explicitly scaled input. Returns a plain array. |
| `arccos(x, unit_to_scale=None, **kwargs)` | Use when computing inverse cosine from dimensionless or explicitly scaled input. Returns a plain angle array. |
| `arccosh(x, unit_to_scale=None, **kwargs)` | Use when computing inverse hyperbolic cosine from dimensionless or explicitly scaled input. Returns a plain array. |
| `arcsin(x, unit_to_scale=None, **kwargs)` | Use when computing inverse sine from dimensionless or explicitly scaled input. Returns a plain angle array. |
| `arcsinh(x, unit_to_scale=None, **kwargs)` | Use when computing inverse hyperbolic sine from dimensionless or explicitly scaled input. Returns a plain array. |
| `arctan(x, unit_to_scale=None, **kwargs)` | Use when computing inverse tangent from dimensionless or explicitly scaled input. Returns a plain angle array. |
| `arctanh(x, unit_to_scale=None, **kwargs)` | Use when computing inverse hyperbolic tangent from dimensionless or explicitly scaled input. Returns a plain array. |
| `cos(x, unit_to_scale=None, **kwargs)` | Use when computing cosine of a dimensionless or explicitly scaled value. Returns a plain array. |
| `cosh(x, unit_to_scale=None, **kwargs)` | Use when computing hyperbolic cosine of a dimensionless or explicitly scaled value. Returns a plain array. |
| `sin(x, unit_to_scale=None, **kwargs)` | Use when computing sine of a dimensionless or explicitly scaled value. Returns a plain array. |
| `sinc(x, unit_to_scale=None, **kwargs)` | Use when computing normalized `sin(pi*x) / (pi*x)` from dimensionless or explicitly scaled input. Returns a plain array. |
| `sinh(x, unit_to_scale=None, **kwargs)` | Use when computing hyperbolic sine of a dimensionless or explicitly scaled value. Returns a plain array. |
| `tan(x, unit_to_scale=None, **kwargs)` | Use when computing tangent of a dimensionless or explicitly scaled value. Returns a plain array. |
| `tanh(x, unit_to_scale=None, **kwargs)` | Use when computing hyperbolic tangent of a dimensionless or explicitly scaled value. Returns a plain array. |
| `deg2rad(x, unit_to_scale=None, **kwargs)` | Use when converting numeric degrees to radians after optional unit scaling. Returns a plain angle array. |
| `rad2deg(x, unit_to_scale=None, **kwargs)` | Use when converting numeric radians to degrees after optional unit scaling. Returns a plain angle array. |
| `degrees(x, unit_to_scale=None, **kwargs)` | Use when selecting the documented alias of `rad2deg`. Returns a plain angle array in degrees. |
| `radians(x, unit_to_scale=None, **kwargs)` | Use when selecting the documented alias of `deg2rad`. Returns a plain angle array in radians. |
| `angle(x, unit_to_scale=None, **kwargs)` | Use when extracting the phase of a complex argument after optional unit scaling. Returns a plain angle array. |
| `frexp(x, unit_to_scale=None, **kwargs)` | Use when decomposing dimensionless or explicitly scaled values into base-2 parts. Returns `(mantissa, exponent)` as two plain arrays. |
| `ldexp(x, y, unit_to_scale=None, **kwargs)` | Use when reconstructing values as `x * 2**y` after optional scaling of `x`. Returns a plain array. |
| `logaddexp(x, y, unit_to_scale=None, **kwargs)` | Use when stably computing the natural logarithm of summed exponentials. Returns a plain array from dimensionless or explicitly scaled operands. |
| `logaddexp2(x, y, unit_to_scale=None, **kwargs)` | Use when stably computing the base-2 logarithm of summed base-2 exponentials. Returns a plain array. |
| `correlate(a, v, mode='valid', *, precision=None, preferred_element_type=None, unit_to_scale=None, **kwargs)` | Use when cross-correlating two one-dimensional sequences. Returns unit `a.unit * v.unit` by default, or a plain array when `unit_to_scale` is supplied. |
| `set_exprel_order(order)` | Use when trading computation for Taylor-series accuracy in `exprel()` near zero. Accepts orders 2 through 20 and returns `None`. |
| `bitwise_not(x, **kwargs)` | Use when flipping every bit of dimensionless integer input. Returns a plain integer array. |
| `invert(x, **kwargs)` | Use when selecting the bitwise-inversion alias of `bitwise_not`. Returns a plain integer array. |
| `bitwise_and(x, y, **kwargs)` | Use when applying element-wise bitwise AND to dimensionless integer operands. Returns a plain integer array. |
| `bitwise_or(x, y, **kwargs)` | Use when applying element-wise bitwise OR to dimensionless integer operands. Returns a plain integer array. |
| `bitwise_xor(x, y, **kwargs)` | Use when applying element-wise bitwise XOR to dimensionless integer operands. Returns a plain integer array. |
| `left_shift(x, y, **kwargs)` | Use when shifting dimensionless integer bits left by `y`. Returns a plain integer array. |
| `right_shift(x, y, **kwargs)` | Use when shifting dimensionless integer bits right by `y`. Returns a plain integer array. |

## Unit-Changing Operations

These functions change physical dimensions because they invert, multiply, divide, power, integrate, or contract values.

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `reciprocal(x, **kwargs)` | Need the multiplicative inverse of a physical value. | Invert the value and its unit. | `u.math.reciprocal(2. * u.second)`<br>`# Quantity(0.5, "s^-1")` |
| `multiply(x, y, **kwargs)` | Need element-wise products with broadcasted operands. | Multiply values and input units element-wise. | `u.math.multiply(2. * u.meter, 3. * u.second)`<br>`# Quantity(6., "m s")` |
| `divide(x, y, **kwargs)` | Need an element-wise physical ratio. | Divide values and input units element-wise. | `u.math.divide(6. * u.meter, 2. * u.second)`<br>`# Quantity(3., "m / s")` |
| `power(x, y, **kwargs)` | Need element-wise exponentiation with a dimensionless exponent. | Raise each value and its unit to `y`. | `u.math.power(2. * u.meter, 3)`<br>`# Quantity(8., "m^3")` |
| `square(x, **kwargs)` | Need the element-wise square of a physical value. | Square the value and input unit. | `u.math.square(3. * u.meter)`<br>`# Quantity(9., "m^2")` |
| `sqrt(x, **kwargs)` | Need a principal square root with the corresponding root unit. | Take the square root of the value and unit. | `u.math.sqrt(9. * (u.meter ** 2))`<br>`# Quantity(3., "m")` |
| `prod(x, axis=None, dtype=None, keepdims=False, initial=None, where=None)` | Need to multiply all elements along an axis rather than add them. | Multiply selected elements and raise the unit to the number reduced. | `x = u.math.array([2., 3.]) * u.meter; u.math.prod(x)`<br>`# Quantity(6., "m^2")` |
| `var(a, axis=None, dtype=None, ddof=0, keepdims=False, *, where=None, **kwargs)` | Need spread expressed in squared physical units. | Compute variance; the output unit is the square of the input unit. | `x = u.math.array([1., 2., 3.]) * u.meter; u.math.var(x)`<br>`# Quantity(0.6666667, "m^2")` |
| `dot(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | Need a vector dot product or general backend dot semantics. | Contract the dot axes and multiply operand units. | `a = u.math.array([1., 2.]) * u.meter; b = u.math.array([3., 4.]) * u.second; u.math.dot(a, b)`<br>`# Quantity(11., "m s"); scalar` |
| `matmul(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | Need conventional matrix or batched matrix multiplication. | Contract matrix axes, broadcast batch axes, and multiply units. | `a = u.math.array([[1., 2.]]) * u.meter; b = u.math.array([[3.], [4.]]) * u.second; u.math.matmul(a, b)`<br>`# Quantity([[11.]], "m s"); shape (1, 1)` |
| `tensordot(a, b, axes=2, precision=None, preferred_element_type=None, **kwargs)` | Need to choose exactly which tensor axes are contracted. | Sum products over `axes` and multiply operand units. | `a = u.math.array([1., 2.]) * u.meter; b = u.math.array([3., 4.]) * u.second; u.math.tensordot(a, b, axes=1)`<br>`# Quantity(11., "m s"); scalar` |
| `matrix_power(a, n, **kwargs)` | Need an integer power of a square matrix. | Repeatedly multiply the matrix; the output unit is `a.unit ** n`. | `a = u.math.array([[2., 0.], [0., 3.]]) * u.meter; u.math.matrix_power(a, 2)`<br>`# Quantity([[4., 0.], [0., 9.]], "m^2")` |

Key rules:

- `power()` requires a dimensionless exponent.
- `prod()` uses `x.unit ** n`, where `n` is the number of elements reduced. A unitful `initial` raises `TypeError`.
- `dot()`, `matmul()`, and `tensordot()` all multiply operand units; choose among them by contraction semantics, not unit behavior.

| API | Description and result |
|---|---|
| `cbrt(x, **kwargs)` | Use when taking element-wise cube roots of physical values. Returns a quantity with unit `x.unit ** (1/3)`. |
| `float_power(x, y, **kwargs)` | Use when power results must be inexact with at least float64 precision. Returns a quantity with unit `x.unit ** y`, and `y` must be dimensionless. |
| `cumprod(x, axis=None, dtype=None, **kwargs)` | Use when accumulating products along an axis. Returns cumulative values whose unit power grows with the number of factors. |
| `cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None, **kwargs)` | Use when computing vector cross products with configurable vector axes. Returns a quantity with unit `a.unit * b.unit`. |
| `floor_divide(x, y, **kwargs)` | Use when element-wise quotient values must be floored. Same-dimension inputs return plain integers; mixed dimensions return the base quotient unit. |
| `divmod(x, y, **kwargs)` | Use when both floored quotient and remainder are needed. Returns a tuple whose quotient follows `floor_divide` and whose remainder carries the dividend's physical kind. |
| `convolve(a, v, mode='full', *, precision=None, preferred_element_type=None, **kwargs)` | Use when computing discrete linear convolution of one-dimensional sequences. Returns a quantity with unit `a.unit * v.unit`. |
| `trapezoid(y, x=None, dx=1.0, axis=-1)` | Use when integrating sampled values with the composite trapezoidal rule. Returns unit `y.unit * x.unit`, or `y.unit * dx.unit` when `x` is omitted. |
| `multi_dot(arrays, *, precision=None, **kwargs)` | Use when multiplying a chain of matrices with an efficient contraction order. Returns a quantity whose unit is the product of all operand units. |
| `vdot(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | Use when flattening two inputs and conjugating the first before their dot product. Returns unit `a.unit * b.unit`. |
| `vecdot(a, b, /, *, axis=-1, precision=None, preferred_element_type=None, **kwargs)` | Use when computing conjugate batched vector products along a chosen axis. Returns unit `a.unit * b.unit` with that axis removed. |
| `inner(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | Use when contracting the last axes of two inputs. Returns a quantity with unit `a.unit * b.unit`. |
| `outer(a, b, out=None, **kwargs)` | Use when forming every pairwise product of flattened inputs. Returns an `(a.size, b.size)` quantity with unit `a.unit * b.unit`. |
| `kron(a, b, **kwargs)` | Use when forming a Kronecker product. Returns a quantity with combined shape and unit `a.unit * b.unit`. |
| `true_divide(x, y, **kwargs)` | Use when selecting the true-division alias of `divide`. Returns a quantity with unit `x.unit / y.unit`. |
| `product` | Use when selecting the documented alias of `prod`. Returns the same product and unit-exponent behavior as `prod`. |
| `cumproduct` | Use when selecting the documented alias of `cumprod`. Returns the same cumulative values and unit powers as `cumprod`. |
| `nanprod(x, axis=None, dtype=None, keepdims=False, initial=None, where=None, **kwargs)` | Use when multiplying along an axis while treating NaNs as one. Returns a quantity whose unit is raised to the number of reduced elements. |
| `nancumprod(x, axis=None, dtype=None, **kwargs)` | Use when accumulating products while treating NaNs as one. Returns cumulative values with corresponding cumulative unit powers. |
| `nanvar(x, axis=None, dtype=None, ddof=0, keepdims=False, where=None, **kwargs)` | Use when computing variance while ignoring NaNs. Returns a quantity whose unit is the square of the input unit. |

## Unit-Preserving Operations

These functions return values of the same physical kind as their primary input. Bounds, branches, interpolation coordinates, and replacement values must still be compatible with the quantities they constrain.

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `where(condition, x=None, y=None, /, *, size=None, fill_value=None, **kwargs)` | Need element-wise selection between two unit-compatible branches. | Choose from `x` where true and `y` otherwise, preserving the branch unit. | `a = u.math.array([1., 2.]) * u.meter; u.math.where(u.math.array([True, False]), a, 0. * u.meter)`<br>`# Quantity([1., 0.], "m")` |
| `clip(a, a_min, a_max, **kwargs)` | Need to enforce lower and upper physical bounds. | Clamp values to compatible bounds and preserve `a`'s unit. | `a = u.math.array([0., 2., 4.]) * u.meter; u.math.clip(a, 1. * u.meter, 3. * u.meter)`<br>`# Quantity([1., 2., 3.], "m")` |
| `interp(x, xp, fp, left=None, right=None, period=None, **kwargs)` | Need one-dimensional interpolation with physical coordinates and physical results. | Interpolate along coordinates compatible with `xp`; `fp` determines the output unit. | `xp = u.math.array([0., 1.]) * u.second; fp = u.math.array([0., 10.]) * u.meter; u.math.interp(0.5 * u.second, xp, fp)`<br>`# Quantity(5., "m")` |
| `histogram(x, bins=10, range=None, weights=None, density=None, **kwargs)` | Need counts over physical bins while retaining physical bin edges. | Return histogram values plus bin edges compatible with `x`. | `x = u.math.array([1., 2., 3.]) * u.second; bins = u.math.array([0., 2., 4.]) * u.second; u.math.histogram(x, bins=bins)`<br>`# (Array([1, 2]), Quantity([0., 2., 4.], "s"))` |
| `nan_to_num(x, nan=None, posinf=None, neginf=None, **kwargs)` | Need to replace non-finite physical values without dropping units. | Replace NaN/infinities with compatible values and preserve `x`'s unit. | `x = u.math.array([float('nan'), 1.]) * u.meter; u.math.nan_to_num(x, nan=0. * u.meter)`<br>`# Quantity([0., 1.], "m")` |
| `percentile(a, q, axis=None, method='linear', keepdims=False, **kwargs)`<br>`quantile(a, q, axis=None, method='linear', keepdims=False, **kwargs)` | Need percentile or quantile summaries while preserving physical kind. | Use percentile scale `0..100` or quantile scale `0..1`; both preserve the input unit. | `x = u.math.array([0., 10.]) * u.meter; (u.math.percentile(x, 50), u.math.quantile(x, 0.5))`<br>`# (Quantity(5., "m"), Quantity(5., "m"))` |
| `promote_dtypes(*args, **kwargs)` | Need several inputs to share a numeric dtype without changing their units. | Promote mantissa dtypes to the most precise input type while retaining each input shape and unit. | `a = u.math.array([1]) * u.second; b = u.math.array([2.5]) * u.second; pa, pb = u.math.promote_dtypes(a, b); (pa.unit, pb.unit, pa.dtype == pb.dtype)`<br>`# (u.second, u.second, True)` |

Key rules:

- With three arguments, `where()` requires `condition`, `x`, and `y` to broadcast together and requires compatible branch values. With only `condition`, prefer `nonzero()`.
- `interp()` requires increasing `xp`; `x`, `xp`, and any `period` must be coordinate-compatible, while `left` and `right` must match `fp`.
- `histogram()` returns plain counts when unweighted; quantity weights or `density=True` change histogram-value semantics, while physical bin edges remain compatible with `x`.

### Quick Reference

| API | Description and result |
|---|---|
| `sort(a, axis=-1, *, kind=None, order=None, stable=True, descending=False, **kwargs)` | Use when ordering physical values along an axis. Returns a same-shape quantity with the input unit. |
| `max(a, axis=None, keepdims=False, initial=None, where=None, **kwargs)` | Use when selecting maximum physical values over chosen axes. Returns a quantity with the input unit. |
| `min(a, axis=None, keepdims=False, initial=None, where=None, **kwargs)` | Use when selecting minimum physical values over chosen axes. Returns a quantity with the input unit. |
| `amax` | Use when selecting the documented alias of `max`. Returns the same maximum values with the input unit. |
| `amin` | Use when selecting the documented alias of `min`. Returns the same minimum values with the input unit. |
| `ptp(x, axis=None, keepdims=False, **kwargs)` | Use when measuring peak-to-peak range as maximum minus minimum. Returns a quantity with the input unit. |
| `sum(x, axis=None, dtype=None, keepdims=False, initial=None, where=None)` | Use when computing additive totals over chosen axes. Returns a quantity with the input unit. |
| `cumsum(x, axis=None, dtype=None, **kwargs)` | Use when computing cumulative additive totals along an axis. Returns a same-shape quantity with the input unit. |
| `median(x, axis=None, overwrite_input=False, keepdims=False, **kwargs)` | Use when computing median physical values over chosen axes. Returns a quantity with the input unit. |
| `average(x, axis=None, weights=None, returned=False, keepdims=False, **kwargs)` | Use when computing a weighted physical mean. Returns the average with the input unit and, with `returned=True`, also the plain sum of weights. |
| `mean(x, axis=None, dtype=None, keepdims=False, *, where=None, **kwargs)` | Use when computing arithmetic means over chosen axes. Returns a quantity with the input unit. |
| `std(x, axis=None, dtype=None, ddof=0, keepdims=False, *, where=None, **kwargs)` | Use when measuring spread in the original physical kind. Returns a quantity with the input unit. |
| `real(x, **kwargs)` | Use when extracting real components of complex physical values. Returns a quantity with the input unit. |
| `imag(x, **kwargs)` | Use when extracting imaginary components of complex physical values. Returns a quantity with the input unit. |
| `conj(x, **kwargs)` | Use when conjugating complex physical values. Returns a quantity with the input unit. |
| `negative(x, **kwargs)` | Use when negating physical values element-wise. Returns a quantity with the input unit. |
| `positive(x, **kwargs)` | Use when applying unary plus to physical values. Returns a quantity with the input unit. |
| `abs(x, **kwargs)` | Use when taking element-wise magnitudes of physical values. Returns a quantity with the input unit. |
| `fabs(x, **kwargs)` | Use when taking floating-point absolute values. Returns a quantity with the input unit. |
| `round(x, decimals=0, **kwargs)` | Use when rounding physical mantissas to a chosen number of decimals. Returns a quantity with the input unit. |
| `rint(x, **kwargs)` | Use when rounding physical mantissas to nearest integers. Returns a quantity with the input unit. |
| `floor(x, **kwargs)` | Use when rounding physical mantissas down. Returns a quantity with the input unit. |
| `ceil(x, **kwargs)` | Use when rounding physical mantissas up. Returns a quantity with the input unit. |
| `trunc(x, **kwargs)` | Use when discarding fractional mantissa parts. Returns a quantity with the input unit. |
| `fix(x, **kwargs)` | Use when rounding physical mantissas toward zero. Returns a quantity with the input unit. |
| `add(x, y, **kwargs)` | Use when adding broadcast-compatible physical values. Returns a quantity in the operands' common physical kind. |
| `subtract(x, y, **kwargs)` | Use when subtracting broadcast-compatible physical values. Returns a quantity in the operands' common physical kind. |
| `maximum(x1, x2, **kwargs)` | Use when selecting element-wise maxima from compatible physical values. Returns a quantity with the shared physical unit. |
| `minimum(x1, x2, **kwargs)` | Use when selecting element-wise minima from compatible physical values. Returns a quantity with the shared physical unit. |
| `fmax(x1, x2, **kwargs)` | Use when selecting element-wise maxima while ignoring NaNs. Returns a quantity with the shared physical unit. |
| `fmin(x1, x2, **kwargs)` | Use when selecting element-wise minima while ignoring NaNs. Returns a quantity with the shared physical unit. |
| `ediff1d(x, to_end=None, to_begin=None, **kwargs)` | Use when computing consecutive differences on flattened physical data. Returns a one-dimensional quantity with the input unit. |
| `diff(x, n=1, axis=-1, prepend=None, append=None, **kwargs)` | Use when computing repeated consecutive differences along an axis. Returns a quantity with the input unit and a shortened selected axis. |
| `modf(x, **kwargs)` | Use when splitting physical mantissas into fractional and integral parts. Returns two quantities with the input unit. |
| `fmod(x1, x2, **kwargs)` | Use when computing truncation-based remainders of compatible physical values. Returns a quantity with the shared physical unit. |
| `mod(x1, x2, **kwargs)` | Use when computing element-wise modulus of compatible physical values. Returns a quantity with the shared physical unit. |
| `remainder(x, y, **kwargs)` | Use when computing floor-division remainders of compatible physical values. Returns a quantity in the dividend's physical kind. |
| `copysign(x1, x2, **kwargs)` | Use when copying signs between compatible physical values. Returns a quantity with the shared unit. |
| `nextafter(x, y, **kwargs)` | Use when finding the next representable physical value from `x` toward compatible `y`. Returns a quantity with `x`'s unit. |
| `lcm(x1, x2, **kwargs)` | Use when computing element-wise least common multiples of compatible integer quantities. Returns a quantity with the shared unit. |
| `gcd(x1, x2, **kwargs)` | Use when computing element-wise greatest common divisors of compatible integer quantities. Returns a quantity with the shared unit. |
| `trace(a, offset=0, axis1=0, axis2=1, dtype=None, **kwargs)` | Use when summing a selected matrix diagonal. Returns a quantity with the input unit and removes the two traced axes. |
| `select(condlist, choicelist, default=0, **kwargs)` | Use when choosing among ordered condition branches. Returns a quantity when the choices resolve to a compatible physical kind. |
| `conjugate(x, **kwargs)` | Use when selecting the documented alias of `conj`. Returns a quantity with the input unit. |
| `absolute(x, **kwargs)` | Use when selecting the documented alias of `abs`. Returns a quantity with the input unit. |
| `around(x, decimals=0, **kwargs)` | Use when selecting the documented alias of `round`. Returns a quantity with the input unit. |
| `nancumsum(x, axis=None, dtype=None, **kwargs)` | Use when accumulating sums while treating NaNs as zero. Returns a same-shape quantity with the input unit. |
| `nansum(x, axis=None, dtype=None, keepdims=False, initial=None, where=None, **kwargs)` | Use when summing physical values while ignoring NaNs. Returns a quantity with the input unit. |
| `nanmin(x, axis=None, keepdims=False, initial=None, where=None, **kwargs)` | Use when selecting minimum physical values while ignoring NaNs. Returns a quantity with the input unit. |
| `nanmax(x, axis=None, keepdims=False, initial=None, where=None, **kwargs)` | Use when selecting maximum physical values while ignoring NaNs. Returns a quantity with the input unit. |
| `nanmedian(x, axis=None, overwrite_input=False, keepdims=False, **kwargs)` | Use when computing median physical values while ignoring NaNs. Returns a quantity with the input unit. |
| `nanmean(x, axis=None, dtype=None, keepdims=False, *, where=None, **kwargs)` | Use when computing arithmetic means while ignoring NaNs. Returns a quantity with the input unit. |
| `nanstd(x, axis=None, dtype=None, ddof=0, keepdims=False, *, where=None, **kwargs)` | Use when computing standard deviation while ignoring NaNs. Returns a quantity with the input unit. |
| `nanpercentile(a, q, axis=None, method='linear', keepdims=False, **kwargs)` | Use when computing percentiles while ignoring NaNs. Returns a quantity with the input unit. |
| `nanquantile(a, q, axis=None, method='linear', keepdims=False, **kwargs)` | Use when computing quantiles while ignoring NaNs. Returns a quantity with the input unit. |
| `rot90(m, k=1, axes=(0, 1), **kwargs)` | Use when rotating physical array data by quarter turns in an axis plane. Returns a quantity with the input unit and rotated shape. |
| `intersect1d(ar1, ar2, assume_unique=False, return_indices=False, **kwargs)` | Use when finding sorted unique values common to compatible inputs. Returns a quantity with the common unit and optional plain index arrays. |
| `compress(condition, a, axis=None, *, size=None, fill_value=None, **kwargs)` | Use when selecting slices with a boolean condition. Returns a quantity with `a`'s unit and a selection-dependent axis length. |
| `extract(condition, arr, *, size=None, fill_value=None, **kwargs)` | Use when extracting elements that satisfy a boolean condition. Returns a quantity with `arr`'s unit and optional fixed output size. |
| `take(a, indices, axis=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None, **kwargs)` | Use when gathering indexed values along an axis. Returns the same array type and unit as `a`. |
| `choose(a, choices, mode='raise', **kwargs)` | Use when an integer index array selects among broadcast-compatible choices. Returns a quantity when the choices carry a common unit. |
| `unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=False, size=None, fill_value=None, **kwargs)` | Use when deduplicating physical values. Returns sorted unique values with the input unit plus requested plain index/count arrays. |
| `gather(input, dim, index, **kwargs)` | Use when gathering values along `dim` with an index array. Returns a quantity with the input unit and the index shape. |

## Unit-Removing Operations

These functions return booleans, indices, counts, signs, or inspection metadata. Some accept physical input and remove its unit because the result is not a physical magnitude. Logical reductions such as `all()` and `any()` are different: their inputs must already be dimensionless.


| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `equal(x, y, *args, **kwargs)` | Need exact element-wise equality between compatible physical values. | Convert two quantities to compatible units and return booleans. | `u.math.equal(1. * u.meter, 100. * u.cm)`<br>`# Array(True); boolean, therefore unitless` |
| `isclose(x, y, rtol=None, atol=None, equal_nan=False, **kwargs)` | Need element-wise tolerance comparison of physical values. | Compare compatible values and return one boolean per element. | `u.math.isclose(1. * u.meter, 100.1 * u.cm, atol=2. * u.mm)`<br>`# Array(True)` |
| `allclose(x, y, rtol=None, atol=None, equal_nan=False, **kwargs)` | Need one boolean stating whether all physical values are close. | Reduce element-wise tolerance comparisons to one boolean. | `a = u.math.array([1., 2.]) * u.meter; b = u.math.array([100., 200.1]) * u.cm; u.math.allclose(a, b, atol=2. * u.mm)`<br>`# Array(True)` |
| `searchsorted(a, v, side='left', sorter=None, *, method='scan', **kwargs)` | Need insertion positions for compatible physical values in sorted data. | Convert `v` to `a`'s unit and return integer insertion indices. | `a = u.math.array([1., 2., 3.]) * u.second; u.math.searchsorted(a, 2500. * u.ms)`<br>`# Array(2); index, therefore unitless` |
| `argmax(a, axis=None, **kwargs)` | Need the position of the largest physical value. | Strip units for ordering and return an index. | `x = u.math.array([1., 3., 2.]) * u.meter; u.math.argmax(x)`<br>`# Array(1); index` |
| `nonzero(a, *, size=None, fill_value=None, **kwargs)` | Need coordinates of nonzero physical values. | Strip units and return one index array per dimension. | `x = u.math.array([0., 2., 0.]) * u.meter; u.math.nonzero(x)`<br>`# (Array([1]),); tuple of index arrays` |
| `count_nonzero(a, axis=None, keepdims=None, **kwargs)` | Need the number of nonzero physical values. | Strip units and return an integer count. | `x = u.math.array([0., 2., 3.]) * u.meter; u.math.count_nonzero(x)`<br>`# Array(2); count` |
| `all(x, axis=None, keepdims=False, where=None, **kwargs)` | Need to know whether every dimensionless element is truthy. | Perform logical AND reduction on dimensionless input. | `u.math.all(u.math.array([True, True, False]))`<br>`# Array(False)` |
| `any(x, axis=None, keepdims=False, where=None, **kwargs)` | Need to know whether any dimensionless element is truthy. | Perform logical OR reduction on dimensionless input. | `u.math.any(u.math.array([False, False, True]))`<br>`# Array(True)` |

Key rules:

- `equal()` converts `y` to `x`'s unit when both are quantities; it raises `TypeError` when only one operand has units.
- `isclose()` and `allclose()` return booleans. Match `atol` to the compared physical dimension.
- `all()` and `any()` reject physical input with `TypeError`; they do not interpret nonzero physical magnitudes as truth values.
- `size` on `nonzero()` fixes result length for compiled code; `fill_value` supplies padding indices, not physical values.

### Quick Reference

| API | Description and result |
|---|---|
| `not_equal(x, y, *args, **kwargs)` | Use when testing element-wise inequality of compatible physical values. Returns a plain boolean array. |
| `greater(x, y, *args, **kwargs)` | Use when testing whether compatible physical values in `x` exceed `y`. Returns a plain boolean array. |
| `greater_equal(x, y, *args, **kwargs)` | Use when testing whether compatible physical values in `x` meet or exceed `y`. Returns a plain boolean array. |
| `less(x, y, *args, **kwargs)` | Use when testing whether compatible physical values in `x` are below `y`. Returns a plain boolean array. |
| `less_equal(x, y, *args, **kwargs)` | Use when testing whether compatible physical values in `x` are at most `y`. Returns a plain boolean array. |
| `array_equal(x, y, *args, **kwargs)` | Use when requiring two compatible arrays to have identical shapes and values. Returns one plain boolean. |
| `logical_not(x, **kwargs)` | Use when negating dimensionless truth values element-wise. Returns a plain boolean array. |
| `logical_and(x, y, *args, **kwargs)` | Use when combining dimensionless truth values with element-wise AND. Returns a plain boolean array. |
| `logical_or(x, y, *args, **kwargs)` | Use when combining dimensionless truth values with element-wise OR. Returns a plain boolean array. |
| `logical_xor(x, y, *args, **kwargs)` | Use when combining dimensionless truth values with element-wise XOR. Returns a plain boolean array. |
| `argsort(a, axis=-1, *, kind=None, order=None, stable=True, descending=False, **kwargs)` | Use when finding the permutation that orders physical values along an axis. Returns a plain integer index array with `a`'s shape. |
| `argmin(a, axis=None, keepdims=None, **kwargs)` | Use when locating minimum physical values over an axis. Returns plain integer indices, optionally retaining the reduced axis. |
| `argwhere(a, *, size=None, fill_value=None, **kwargs)` | Use when finding coordinates of nonzero physical values. Returns a plain integer array with one coordinate row per match. |
| `flatnonzero(a, *, size=None, fill_value=None, **kwargs)` | Use when finding nonzero positions in flattened physical data. Returns a one-dimensional plain index array. |
| `digitize(x, bins, right=False, **kwargs)` | Use when assigning compatible physical values to monotonic bins. Returns a plain integer array with the same shape as `x`. |
| `heaviside(x1, x2, **kwargs)` | Use when mapping physical values to a Heaviside step with dimensionless zero value `x2`. Returns a dimensionless array. |
| `signbit(x, **kwargs)` | Use when testing whether physical values have their sign bit set. Returns a plain boolean array. |
| `sign(x, **kwargs)` | Use when extracting only the sign of physical values. Returns a plain dimensionless array. |
| `nanargmax(a, axis=None, keepdims=False, **kwargs)` | Use when locating maximum physical values while ignoring NaNs. Returns plain integer indices. |
| `nanargmin(a, axis=None, keepdims=False, **kwargs)` | Use when locating minimum physical values while ignoring NaNs. Returns plain integer indices. |
| `alltrue` | Use when selecting the documented alias of `all`. Requires dimensionless input and returns a plain boolean result. |
| `sometrue` | Use when selecting the documented alias of `any`. Requires dimensionless input and returns a plain boolean result. |
| `iscomplexobj(x, **kwargs)` | Use when inspecting whether a value or array has a complex type. Returns one plain Python boolean. |
| `get_promote_dtypes(*args, **kwargs)` | Use when determining the common promoted dtype for several inputs. Returns dtype metadata rather than a quantity. |
| `diag_indices_from(arr, **kwargs)` | Use when obtaining indices for the main diagonal of an array. Returns a tuple of plain integer index arrays. |
| `bincount(x, weights=None, minlength=0, *, length=None, **kwargs)` | Use when counting occurrences of dimensionless non-negative integers. Returns a plain count or weighted-sum array indexed by input value. |

## Reductions And Contractions

Choose a reduction by the physical question, then choose axes and shape retention.

| Physical question | Exact signature | Unit behavior | Axis and shape rule |
|---|---|---|---|
| What is the additive total? | `sum(x, axis=None, dtype=None, keepdims=False, initial=None, where=None)` | Preserves the input unit. | `axis=None` reduces all axes; an integer or tuple selects axes. `keepdims=True` leaves reduced axes at size one. |
| What is the arithmetic center? | `mean(x, axis=None, dtype=None, keepdims=False, *, where=None, **kwargs)` | Preserves the input unit. | Same reduction-axis and `keepdims` behavior as `sum()`. |
| What is the spread in the original physical kind? | `std(x, axis=None, dtype=None, ddof=0, keepdims=False, *, where=None, **kwargs)` | Preserves the input unit. | `axis` selects populations; `keepdims=True` supports broadcasting back to `x`. |
| What is the squared spread? | `var(a, axis=None, dtype=None, ddof=0, keepdims=False, *, where=None, **kwargs)` | Squares the input unit. | Axis and `keepdims` follow `std()`; `ddof` changes the divisor `N - ddof`. |
| What is the multiplicative accumulation? | `prod(x, axis=None, dtype=None, keepdims=False, initial=None, where=None)` | Raises the input unit to the number of reduced factors. | `axis` determines both output shape and unit exponent; `keepdims` retains reduced axes at size one. |
| What is the dot product under backend dot rules? | `dot(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | Multiplies operand units. | Contraction axes follow `dot()` rank rules; there is no `axis` or `keepdims` argument. |
| What is the matrix or batched matrix product? | `matmul(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | Multiplies operand units. | Contracts matrix inner dimensions and broadcasts batch dimensions. |
| What tensor axes should be contracted explicitly? | `tensordot(a, b, axes=2, precision=None, preferred_element_type=None, **kwargs)` | Multiplies operand units. | Integer `N` contracts the last `N` axes of `a` with the first `N` of `b`; axis sequences choose explicit pairs. |

Choice rules:

- Use `sum()`, not `prod()`, for additive totals. `prod()` changes the physical dimension as the factor count changes.
- Use `std()` when the result must remain directly comparable with the input; use `var()` when squared spread is the intended quantity.
- Use `dot()` for vector dot/general rank-dependent dot semantics, `matmul()` for matrix and batched matrix multiplication, and `tensordot()` for explicit contraction axes.
- For reducers, `axis=None` flattens conceptually, an integer removes one axis, a tuple reduces several axes, and `keepdims=True` retains each reduced axis with length one.

The original worked example, extended to cover all eight choices:

```python
import brainunit as u

x = u.math.asarray([1.0, 2.0, 3.0], unit=u.meter)

total = u.math.sum(x)
# Expected: 6 m.
center = u.math.mean(x)
# Expected: 2 m.
spread = u.math.std(x)
# Expected: sqrt(2/3) m, approximately 0.8165 m.
spread2 = u.math.var(x)
# Expected: 2/3 m^2, approximately 0.6667 m^2.
product = u.math.prod(x)
# Expected: 6 m^3.
area_terms = u.math.square(x)
# Expected: [1, 4, 9] m^2.

a = u.math.array([1.0, 2.0]) * u.meter
b = u.math.array([3.0, 4.0]) * u.second
inner = u.math.dot(a, b)
# Expected: scalar 11 m s.
explicit_inner = u.math.tensordot(a, b, axes=1)
# Expected: scalar 11 m s.

matrix = u.math.array([[1.0, 2.0]]) * u.meter
column = u.math.array([[3.0], [4.0]]) * u.second
matrix_product = u.math.matmul(matrix, column)
# Expected: shape (1, 1), value [[11]] m s.
```

## Einstein Operations

### `einreduce`

| Exact signature | Use when | Pattern, reduction, shape, and unit |
|---|---|---|
| `einreduce(x, pattern, reduction, **axes_lengths)` | Need a readable named-axis reduction combined with splitting, merging, or reordering axes. | Read `pattern` as `input axes -> output axes`. An input axis absent from the output is reduced. Parentheses compose or decompose axes; `axes_lengths` resolves named factors. The output shape follows the right side. `mean`, `min`, `max`, and `sum` preserve the input unit; `prod` raises it according to the number of reduced factors. |

```python
import brainunit as u

x = u.math.ones((1, 4, 4, 1)) * u.mV
pooled = u.math.einreduce(
    x,
    'b (h h2) (w w2) c -> b h w c',
    'mean',
    h2=2,
    w2=2,
)
# Expected: shape (1, 2, 2, 1), every value 1 mV.
```

Here `h2` and `w2` are absent from the output, so each `2 x 2` block is reduced. Changing `mean` to `sum` keeps millivolts but changes values; changing it to `prod` makes each block's result carry `mV ** 4`.

### `einsum`

| Exact signature | Use when | Subscripts, reduction, shape, and unit |
|---|---|---|
| `einsum(subscripts, /, *operands, optimize='optimal', precision=None, preferred_element_type=None)` | Need a general inner product, matrix multiplication, outer product, reduction, or axis reordering expressed in Einstein notation. | Each operand maps to the labels before `->`; labels on the right determine output-axis order. Repeated labels across operands are multiplied element-wise, and labels omitted from the output are summed. Output shape is formed from retained labels. Contractions multiply operand units; additive reduction alone preserves the operand unit. |

Vector inner product: `i` is repeated and omitted from the output, so it is contracted to a scalar.

```python
import brainunit as u

x = u.math.array([1.0, 2.0]) * u.meter
y = u.math.array([3.0, 4.0]) * u.second
inner = u.math.einsum('i,i->', x, y)
# Expected: scalar 11 m s.
```

Matrix multiplication: `j` is repeated and omitted, while `i` and `k` remain in output order.

```python
import brainunit as u

a = u.math.array([[1.0, 2.0], [3.0, 4.0]]) * u.meter
b = u.math.array([[3.0], [4.0]]) * u.second
product = u.math.einsum('ij,jk->ik', a, b)
# Expected: shape (2, 1), values [[11], [25]] m s.
```

Use `einrearrange()` and `einrepeat()` through `array-mechanics.md` when element count is preserved or increased rather than reduced.

## Source-Backed Gotchas

- `all()` and `any()` require dimensionless input and raise `TypeError` for physical units.
- `power()` requires a dimensionless exponent.
- `prod()` changes the unit exponent according to the number of factors; it is not interchangeable with additive reductions.
- `where(condition, x, y)` requires `x`, `y`, and `condition` to be broadcastable; use unit-compatible branches such as `a` and `0 * u.meter`.
- `searchsorted()` converts quantity `v` to the unit of quantity `a` before searching.
- `isclose()` and `allclose()` return booleans, not quantities; use a dimension-compatible absolute tolerance.
- Exact generated signatures include `**kwargs` for backend forwarding. Pass only arguments documented by the active backend; arbitrary keyword values are not valid.

## Sources Mirrored

- https://brainunit.readthedocs.io/apis/brainunit.math.html
- https://brainunit.readthedocs.io/unit_operations/einstein_operations.html
- Generated Level A pages under `https://brainunit.readthedocs.io/apis/generated/brainunit.math.<function>.html`
- https://brainunit.readthedocs.io/apis/generated/brainunit.math.einreduce.html
- https://brainunit.readthedocs.io/apis/generated/brainunit.math.einsum.html
