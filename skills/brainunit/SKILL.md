---
name: brainunit
description: Enforce BrainUnit physical-quantity, dimensional, typing, conversion, unit-aware math, JAX-transformation, and external-library boundary safety. Use when Codex works with BrainUnit or BrainCell values involving voltage, current, time, conductance, capacitance, length, concentration, temperature, physical constants, unit errors, dimensional mismatches, or suspicious bare numbers.
---

## Purpose and boundary

Use BrainUnit to keep physical meaning attached to numerical values throughout scientific computations. Keep the canonical path here; open only the narrow reference needed for specialized constructors, array mechanics, function semantics, custom units, extended typing, prefixes, or constants.

## Underlying design of brainunit

A `Quantity` stores a numerical `mantissa` and a `unit`; the unit's conversion magnitude is `unit.factor * unit.base ** unit.scale`, so the numerical scaling of the represented value follows `mantissa * unit.factor * unit.base ** unit.scale`, while `unit.dim` supplies its physical dimension. Prefix-only units normally have a factor of one, so `5 * u.ms` corresponds to `5 * 1 * 10**-3 * u.second = 0.005 * u.second`. Use this model to diagnose prefix, scale, factor, or conversion behavior.

Note: This is underlying knowledge, not the ordinary user-facing coding model. In normal code, keep quantities intact and use public unit arithmetic and conversion APIs instead of manipulating `.mantissa`, `unit.factor`, `unit.base`, or `unit.scale` directly.

## Relations between quantity, unit, and dimension

| Concept | Meaning | Example for `20 * u.ms` |
| --- | --- | --- |
| **Quantity** | The complete physical value | `20 ms` |
| **Unit** | The measurement scale used to express it | `ms` |
| **Dimension** | The physical category or structure | time (`T`) |

A `Dimension` describes the physical kind of a value. Internally, it is an immutable tuple of exponents over the seven SI base dimensions. A `Unit` is a particular measurement scale for a dimension. A `Quantity` combines a numerical value with a unit.

A quantity therefore inherits its dimension from its unit. For example, meters and kilometers are different units with different scales, but both have the dimension of length, so they can be converted into each other. Meters and seconds have different dimensions, so they are not convertible.

## Relations between saiunit and brainunit

`saiunit` is the general scientific-computing implementation and can dispatch quantity mantissas across supported backends such as NumPy, JAX, CuPy, PyTorch, Dask, and ndonnx. `brainunit` is the neuroscience-facing namespace tailored to brain dynamics and the JAX-centered BrainX ecosystem; since BrainUnit's integration into SaiUnit, it reimports SaiUnit data structures and functions. Use only  `brainunit` for BrainX and brain-modeling code.

## Other core concepts

- `brainunit` provides physical units and a unit-aware mathematical system for BrainX brain-dynamics workflows, with JAX integration for transformations and accelerated execution.
- BrainUnit combines unit names with prefixes and appends a number for predefined squared and cubed forms. Examples include `msiemens`, `siemens2`, and `usiemens3`. Import `brainunit as u` instead of guessing a name or case-sensitive prefix.

### Common neuroscience units

Use BrainUnit's predefined unit objects instead of bare scaling factors:

- Time: `u.second`, `u.ms`, `u.us`.
- Voltage: `u.volt`, `u.mV`.
- Current: `u.amp`, `u.mA`, `u.uA`, `u.nA`, `u.pA`.
- Conductance: `u.siemens`, `u.mS`, `u.uS`, `u.nS`.
- Capacitance: `u.farad`, `u.uF`, `u.nF`, `u.pF`.
- Resistance: `u.ohm`, `u.kohm`, `u.Mohm`.
- Frequency: `u.Hz`, `u.kHz`.

## Brainunit API modules

| Module | Primary responsibility |
| --- | --- |
| `brainunit` | Top-level access to `Quantity`, `Unit`, `Dimension`, mismatch errors, compatibility checks, validation decorators, backend selection, temperature conversion, and custom-array integration. |
| `brainunit.typing` | `QuantityLike`, `UnitLike`, `DimensionLike`, runtime-checkable physical types and aliases, and `validate_units` for annotation-driven call validation. |
| `brainunit.autograd` | Unit-aware `grad`, `value_and_grad`, `vector_grad`, Jacobian transforms, and Hessians whose result units follow the differentiated function and arguments. |
| `brainunit.math` | Unit-aware array creation, activations, Einstein operations, and NumPy-style mathematics grouped by whether functions require unitless input, keep units, change units, or remove units. |
| `brainunit.linalg` | Unit-aware matrix products, decompositions, solvers, and norms, with explicit semantics for operations that keep, change, or remove units. |
| `brainunit.lax` | Low-level `jax.lax` wrappers for unit-aware array creation, elementwise operations, slicing, scattering, convolution, and linear algebra. |
| `brainunit.fft` | Unit-aware FFT, inverse FFT, frequency-bin, and shift operations that preserve or derive units according to the transform. |
| `brainunit.sparse` | Unit-bearing `SparseMatrix`, `CSR`, `CSC`, and `COO` structures plus dense-to-sparse, sparse-to-dense, and format-conversion operations. |
| `brainunit.constants` | Predefined physical constants as quantities with canonical values, dimensions, and units instead of bare numbers. |

## Canonical imports

Use these imports for the bundled patterns below.

```python
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
```

## Create quantities

A `Quantity` is a numeric value plus a physical unit. Attach units at creation and retain them through code.

```python
# Scalars and derived units.
speed = 10.0 * u.meter / u.second
# 10. m / s

# JAX array via multiplication shorthand.
jax_voltages = jnp.array([1.0, 2.5, 3.7]) * u.mV
# [1.  2.5 3.70000005] mV

# NumPy array via direct construction.
numpy_currents = u.Quantity(np.array([0.1, 0.2, 0.3]), unit=u.nA)
# [0.1 0.2 0.3] nA

# A string is accepted as the unit argument and parsed as a Unit.
string_unit_voltages = u.Quantity([1.0, 2.5, 3.7], "mV")
# [1.  2.5 3.7] mV
```

Direct JAX and NumPy array mantissas retain their respective backends. In the string form, only the unit is a string; the mantissa must remain numeric.

## Inspect quantities

Inspect quantity metadata without discarding its physical meaning.

```python
q = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.volt
print(q.mantissa, q.unit, q.dim, q.shape, q.dtype)
# Expected: numeric 2x2 mantissa, volt unit/dimension, shape (2, 2); dtype follows JAX configuration.
```

Do not infer a unit from a parameter name. Inspect `.unit` and `.dim` when debugging; `.mantissa` exposes the current stored scale without converting it.

## Compute with dimensions

BrainUnit represents units through seven irreducible SI dimensions: length, mass, time, electric current, temperature, amount of substance, and luminous intensity. Units are tracked automatically: addition and subtraction require matching dimensions and align compatible scales, while multiplication, division, powers, and supported functions combine dimension exponents. Treat an incompatibility as a model error rather than stripping units.

```python
t1 = 500.0 * u.ms
t2 = 1.5 * u.second
elapsed = t1 + t2
# 2000. ms

work = (10.0 * u.newton) * (3.0 * u.meter)
# 30. J
average_speed = (100.0 * u.meter) / (10.0 * u.second)
# 10. m / s

try:
    invalid = 5.0 * u.meter + 3.0 * u.second
except Exception as error:
    print("dimension error:", error)
# Expected: prints a dimension-mismatch error
```

## Convert at explicit boundaries

Unit conversion changes the mantissa and unit used to represent a value, but it does not change the underlying physical quantity.

Use `in_unit(target)` to rescale while retaining a `Quantity`. Use `to_decimal(target)` only when an external API requires raw numbers, and make its expected unit explicit in the variable or parameter name.

```python
distance = 2.5 * u.kmeter
distance_m = distance.in_unit(u.meter)
# 2500. m
distance_m_raw = distance.to_decimal(u.meter)
# 2500.0
```

Do not substitute `.mantissa` for conversion.

## Unit-aware math

Use `u.math` where unit semantics matter: functions may preserve units, change them, require dimensionless input, or return indices or booleans.

```python
data = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]) * u.newton
total = u.math.sum(data)
# 30. N
mean = u.math.mean(data)
# 6. N
length = u.math.sqrt(4.0 * u.meter2)
# 2. m
ordered = u.math.sort(jnp.array([3.0, 1.0, 2.0]) * u.volt)
# [1. 2. 3.] V
```

## Constants

Do not confuse a unit constant with a quantity constant. A unit constant such as `u.minute` is a `Unit`: it defines a dimension and conversion scale but has no mantissa or physical amount until combined with a number. A value from `brainunit.constants` such as `u.constants.minute` or `u.constants.boltzmann` is already a `Quantity` with both a numerical mantissa and a unit.

```python
minute_unit = u.minute
assert isinstance(minute_unit, u.Unit)

one_minute = u.constants.minute
assert isinstance(one_minute, u.Quantity)
# 60. s

five_minutes = 5 * minute_unit
# 5. min

avogadro = u.constants.avogadro
# 6.02214076e+23 1 / mol
boltzmann = u.constants.boltzmann
# 1.380649e-23 J / K
elementary_charge = u.constants.elementary_charge
# 1.60217663e-19 C
```

## Normalize inputs and create ranges

Use `u.math.asarray()` to normalize plain data and quantities. Without `unit`, it returns an array for plain data or preserves the input unit; with `unit=target`, it returns a `Quantity` converted to `target`. Incompatible quantity inputs raise `UnitMismatchError`. `u.math.array` is an alias.

Use `u.math.arange()` to create half-open intervals `[start, stop)`, returning an array for plain arguments or a `Quantity` for unit-bearing arguments. Unit-bearing `start`, `stop`, and `step` must share one unit. At least one of `start` or `stop` is required.

```python
plain = u.math.asarray([1, 2, 3])
# Array([1, 2, 3], dtype=int32)
inferred = u.math.asarray([1 * u.second, 2 * u.second])
# Expected: a Quantity with mantissa [1, 2] and unit second.
seconds = u.math.asarray([1000 * u.ms, 2000 * u.ms], unit=u.second)
# Quantity([1. 2.], "s")

indices = u.math.arange(5)
# Array([0, 1, 2, 3, 4], dtype=int32)
times = u.math.arange(0.0 * u.ms, 10.0 * u.ms, 0.1 * u.ms)
# Expected: 100 values from 0.0 through 9.9 ms; array precision follows JAX.
```

Do not strip units before normalization or attach a unit to an unchecked plain range afterward.

## Type and validate quantity boundaries

BrainUnit typing expresses whether an interface accepts convertible values, a particular unit dimension, or a named physical type. `QuantityLike` includes plain numbers, NumPy/JAX arrays, and existing quantities; it does not itself require a unit. `UnitLike` accepts a `Unit`, a unit string, or `None`. Use `u.Quantity[unit]` for a unit-derived dimension and `u.Quantity["physical type"]` for a named dimension.

Annotations alone describe the contract. Apply `@u.typing.validate_units` when calls must be checked at runtime; by default it accepts dimensionally compatible scales, while `strict=True` requires an exact unit for unit-based annotations.

```python
def normalize(values: u.typing.QuantityLike, unit: u.typing.UnitLike = None):
    return u.math.asarray(values, unit=unit)


@u.typing.validate_units
def travel_time(
    distance: u.Quantity["length"],
    speed: u.Quantity["speed"],
) -> u.Quantity["time"]:
    return distance / speed


duration = travel_time(100.0 * u.meter, 5.0 * u.meter / u.second)
# Expected: 20 seconds.
```

Use `@u.typing.validate_units` when `Quantity[...]` annotations define the runtime contract. Open the typing reference for additional aliases, runtime type helpers, and strict-validation details.

If `brainunit.typing` is unavailable, upgrade BrainUnit with its matched SaiUnit dependency; do not mix validators and `Quantity` types from different releases.

## Transform and validate unit-aware functions

BrainUnit integrates with automatic differentiation, JIT compilation, vectorization, and parallel computation. Its strict physical-unit type checking and dimensional inference perform unit conversion and analysis at compilation time in compiled workflows; eager invalid operations raise when evaluated. Use `jax.jit` and `jax.vmap` with quantities, `u.autograd.grad` for unit-aware derivatives, and `@u.check_units` at scientific function boundaries.

```python
@jax.jit
def kinetic_energy(m, v):
    return 0.5 * m * v**2


energy = kinetic_energy(2.0 * u.kilogram, 3.0 * u.meter / u.second)
# 9. J
velocities = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.meter / u.second
energies = jax.vmap(lambda v: kinetic_energy(2.0 * u.kilogram, v))(velocities)
# [ 1.  4.  9. 16. 25.] J

denergy_dv = u.autograd.grad(
    lambda v: 0.5 * (2.0 * u.kilogram) * v**2
)
momentum = denergy_dv(3.0 * u.meter / u.second)
# 6. kg * m / s


@u.check_units(v=u.meter / u.second, t=u.second)
def displacement(v, t):
    return v * t


distance_traveled = displacement(
    10.0 * u.meter / u.second,
    5.0 * u.second,
)
# 50. m
```

## Convert Celsius at affine boundaries

BrainUnit stores temperature quantities in kelvin. `u.celsius2kelvin(x)` accepts a plain scalar or array and returns a kelvin `Quantity` using `K = C + 273.15`; `u.kelvin2celsius(q)` requires a temperature `Quantity` and returns a plain scalar or array using `C = K - 273.15`.

```python
kelvin = u.celsius2kelvin(jnp.array([0.0, 25.0]))
# Expected: [273.15, 298.15] K; array precision follows JAX.
celsius = u.kelvin2celsius(kelvin)
# Expected: a plain array [0.0, 25.0], subject to floating-point precision.
delta_t = 10.0 * u.kelvin
# 10. K
```

The first function rejects quantities, and the second rejects plain or non-temperature inputs. Apply the Celsius offset only to absolute temperatures, not differences; `u.constants.zero_celsius` is the explicit `273.15 K` offset for formulas that need it.

## Reference routing

| Reference | Open when |
|---|---|
| `references/quantity-inspection-and-conversion.md` | Inspecting compatibility, dimensions, conversions, decomposition, formatting, or raw-value boundaries beyond the canonical pattern. |
| `references/array-creation.md` | Creating specialized ranges, grids, filled or template-shaped arrays, matrices, triangular indices, or tree-shaped arrays beyond `asarray` and `arange`. |
| `references/array-mechanics.md` | Indexing, functional updates, reshaping, broadcasting, joining, splitting, repeating, backend conversion, or named-axis rearrangement. |
| `references/math-function-library.md` | Selecting functions by dimensionless-input, unit-preserving, unit-changing, reduction, contraction, comparison, boolean, or index-returning semantics. |
| `references/unit-structure-and-definition.md` | Inspecting unit structure, comparing scale and dimension, composing units, or defining named, derived, or scaled custom units. |
| `references/typing-and-runtime-validation.md` | Selecting core type aliases, physical-type aliases, runtime type helpers, or strict annotation-driven validation. |
| `references/prefix-library.md` | Looking up predefined SI base or derived units, generated unit names, prefix symbols, and prefix scales. |
| `references/physical-constant-library.md` | Looking up predefined constant names, values, dimensions, and canonical units. |
