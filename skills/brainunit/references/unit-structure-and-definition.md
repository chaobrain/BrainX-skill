# Unit Structure And Definition

Use this reference to inspect scale encoding, distinguish dimension from magnitude, compare units, combine existing units, or define a named, factored, or prefixed custom unit.

## Contents

- [Unit anatomy](#unit-anatomy)
- [Comparison methods](#comparison-methods)
- [Combining units](#combining-units)
- [Defining units](#defining-units)
- [Source-backed gotchas](#source-backed-gotchas)

## Unit Anatomy

The official API represents a unit with dimension, scale, base, factor, and naming metadata:

```text
Unit(dim=None, scale=0, base=10.0, factor=1.0,
     name=None, dispname=None, is_fullname=True, display_parts=None)
```

| Property | Official meaning |
|---|---|
| `unit.dim` | Physical dimensions |
| `unit.scale` | Exponent applied to `base`; `mvolt.scale == -3` |
| `unit.base` | Base of the scale exponent; SI prefixes use `10` |
| `unit.factor` | Additional conversion factor, such as `4.184` for calorie relative to joule |
| `unit.magnitude` | `factor * base ** scale` |
| `unit.name` | Full name such as `'volt'` or `'mvolt'` |
| `unit.dispname` | Display symbol such as `'V'` or `'mV'` |
| `unit.is_unitless` | Dimensionless with scale `0` and factor `1.0` |
| `unit.should_display_unit` | True for physical units and named dimensionless units such as radians |

`unit.factorless()` returns an equivalent unit with factor `1`; when a registered unit with the same dimension and scale exists, that standard unit may be returned. `unit.reverse()` returns the reciprocal unit.

## Comparison Methods

| Exact signature | Comparison |
|---|---|
| `has_same_dim(other)` | Same physical dimensions |
| `has_same_base(other)` | Same scale base |
| `has_same_magnitude(other)` | Same `scale`, `base`, and `factor` components |

`has_same_magnitude()` is component-wise. A unit represented with `scale=3` and one represented with `factor=1000` can have the same computed magnitude but compare unequal.

```python
import brainunit as u

u.volt.has_same_dim(u.mvolt)          # True
u.volt.has_same_dim(u.amp)            # False
u.mvolt.has_same_magnitude(u.mamp)    # True
u.mvolt.has_same_magnitude(u.volt)    # False
```

## Combining Units

Units and quantities use normal Python numeric operators. The official tutorial constructs volt from component units and compares it to the predefined unit:

```python
import brainunit as u

volt = u.meter2 * u.kilogram / (u.second3 * u.ampere)
assert volt == u.volt
# Expected: assertion passes; the derived unit is Unit("V").
```

Use `unit.reverse()` when reciprocal intent should be explicit:

```python
frequency = u.second.reverse()
# Unit("Hz")
inverse_length = u.metre.reverse()
# Unit("1 / m")
```

## Defining Units

Exact constructors:

```text
Unit.create(dim, name, dispname, scale=0, base=10.0, factor=1.0)
Unit.create_scaled_unit(baseunit, scalefactor)
```

Define a named unit from an existing dimension when a non-prefix conversion factor is needed:

```python
import brainunit as u

calorie = u.Unit.create(
    u.joule.dim,
    'calorie',
    'cal',
    factor=4.184,
)
# Unit("cal")
```

Define a fundamental or compound unit through `get_or_create_dimension()`:

```python
metre = u.Unit.create(
    u.get_or_create_dimension(m=1),
    'metre',
    'm',
    base=10.0,
)
# Unit("m")

volt = u.Unit.create(
    u.get_or_create_dimension(m=2, kg=1, s=-3, A=-1),
    'volt',
    'V',
)
# Unit("V")
```

Create a supported prefixed scale from a base unit:

```python
kilometre = u.Unit.create_scaled_unit(metre, 'k')
# Unit("km")
ratio = 1 * kilometre / (1 * metre)
# Expected: 1000.0, a plain dimensionless value.
```

See `prefix-library.md` for the complete prefix symbol and exponent table.

## Source-Backed Gotchas

- `Unit.create()` supports only base `10`; other values raise `ValueError`.
- Use `factor` for conversions such as calorie-to-joule; use `scale` for powers of `base`.
- `is_unitless` requires dimensionless dimension, zero scale, and factor `1.0`; a scaled dimensionless unit is not unitless by this definition.
- `factorless()` intentionally avoids blindly replacing every alias, because that could rename units such as becquerel to hertz or steradian to radian.
- `create_scaled_unit(baseunit, scalefactor)` takes a prefix string such as `'u'` or `'k'`, not a numeric multiplier.

## Sources Mirrored

- https://brainunit.readthedocs.io/apis/generated/brainunit.Unit.html
- https://brainunit.readthedocs.io/advanced_tutorials/combining_and_defining.html
