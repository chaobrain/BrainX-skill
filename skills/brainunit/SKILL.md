---
name: brainunit
description: Enforce BrainUnit physical-quantity, dimensional, typing, conversion, unit-aware math, JAX-transformation, and external-library boundary safety. Use when Codex works with BrainUnit or BrainCell values involving voltage, current, time, conductance, capacitance, length, concentration, temperature, physical constants, unit errors, dimensional mismatches, or suspicious bare numbers.
---

# BrainUnit Quantity Safety

## Purpose And Boundary

Use BrainUnit to keep physical meaning attached to numerical values throughout scientific computations. Keep the canonical path here; open only the narrow reference needed for specialized constructors, array mechanics, function semantics, custom units, extended typing, prefixes, or constants.

Underlying representation of Brainunit: a `Quantity` stores a numerical `mantissa` and a `unit`, and its represented value follows `mantissa * unit.base ** unit.scale * dimension`. Thus `5 * u.ms` corresponds to `5 * 10**-3 * u.second = 0.005 * u.second`. Use this model to diagnose prefix, scale, or conversion behavior
Note: This is debugging knowledge, not the ordinary user-facing coding model. In normal code, keep quantities intact and use public unit arithmetic and conversion APIs instead of manipulating `.mantissa`, `unit.base`, or `unit.scale` directly.

## Core Model

- `brainunit` provides physical units and a unit-aware mathematical system in JAX for general AI-driven scientific computing.
- BrainUnit combines unit names with prefixes and appends a number for predefined squared and cubed forms. Examples include `msiemens`, `siemens2`, and `usiemens3`. Import `brainunit as u` instead of guessing a name or case-sensitive prefix.

## Canonical Imports

Use these imports for the bundled patterns below.

```python
import brainunit as u
import jax
import jax.numpy as jnp
from brainunit import constants
from brainunit.typing import QuantityLike, UnitLike, validate_units
```

## Create And Inspect Quantities

A `Quantity` is a numeric value plus a physical unit. Attach units at creation and retain them through code.

```python
# Scalars, arrays, and direct construction.
mass = 5.0 * u.kilogram
# 5. kg
speed = 10.0 * u.meter / u.second
# 10. m / s
voltages = jnp.array([1.0, 2.5, 3.7]) * u.mV
# [1.  2.5 3.70000005] mV
current = u.Quantity(jnp.array([0.1, 0.2, 0.3]), unit=u.ampere)
# [0.1 0.2 0.30000001] A

# Inspect without discarding physical metadata.
q = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.volt
print(q.mantissa, q.unit, q.dim, q.shape, q.dtype)
# Expected: numeric 2x2 mantissa, volt unit/dimension, shape (2, 2); dtype follows JAX configuration.
```

Do not infer a unit from a parameter name. Inspect `.unit` and `.dim` when debugging; `.mantissa` exposes the current stored scale without converting it.

## Compute With Dimensions

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

## Convert At Explicit Boundaries

Use `in_unit(target)` to rescale while retaining a `Quantity`. Use `to_decimal(target)` only when an external API requires raw numbers, and make its expected unit explicit in the variable or parameter name.

```python
distance = 2.5 * u.kmeter
distance_m = distance.in_unit(u.meter)
# 2500. m
distance_m_raw = distance.to_decimal(u.meter)
# 2500.0
```

Do not substitute `.mantissa` for conversion.

## Use Unit-Aware Math And Constants

Use `u.math` where unit semantics matter: functions may preserve units, change them, require dimensionless input, or return indices or booleans. Import predefined physical constants as quantities rather than recreating bare values.

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

avogadro = constants.avogadro
# 6.02214076e+23 1 / mol
boltzmann = constants.boltzmann
# 1.380649e-23 J / K
elementary_charge = constants.elementary_charge
# 1.60217663e-19 C
```

## Normalize Inputs And Create Ranges

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

## Type And Validate Quantity Boundaries

BrainUnit typing expresses whether an interface accepts convertible values, a particular unit dimension, or a named physical type. `QuantityLike` includes plain numbers, NumPy/JAX arrays, and existing quantities; it does not itself require a unit. `UnitLike` accepts a `Unit`, a unit string, or `None`. Use `u.Quantity[unit]` for a unit-derived dimension and `u.Quantity["physical type"]` for a named dimension.

Annotations alone describe the contract. Apply `@validate_units` when calls must be checked at runtime; by default it accepts dimensionally compatible scales, while `strict=True` requires an exact unit for unit-based annotations.

```python
def normalize(values: QuantityLike, unit: UnitLike = None):
    return u.math.asarray(values, unit=unit)


@validate_units
def travel_time(
    distance: u.Quantity["length"],
    speed: u.Quantity["speed"],
) -> u.Quantity["time"]:
    return distance / speed


duration = travel_time(100.0 * u.meter, 5.0 * u.meter / u.second)
# Expected: 20 seconds.
```

Use `@validate_units` when `Quantity[...]` annotations define the runtime contract. Open the typing reference for additional aliases, runtime type helpers, and strict-validation details.

If `brainunit.typing` is unavailable, upgrade BrainUnit with its matched SaiUnit dependency; do not mix validators and `Quantity` types from different releases.

## Transform And Validate Unit-Aware Functions

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

## Convert Celsius At Affine Boundaries

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

## Reference Routing

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

## Official Sources

- https://brainx.chaobrain.com/brainunit/
- https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html
- https://brainx.chaobrain.com/brainunit/physical_units/quantity.html
- https://brainx.chaobrain.com/brainunit/physical_units/standard_units.html
- https://brainx.chaobrain.com/brainunit/physical_units/temperature.html
- https://brainx.chaobrain.com/brainunit/unit_operations/array_creation.html
- https://brainx.chaobrain.com/brainunit/apis/brainunit.typing.html
- https://brainx.chaobrain.com/braincell/concepts/units.html
