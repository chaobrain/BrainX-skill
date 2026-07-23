# Quantity Inspection And Conversion

Use this reference after the basic `Quantity` workflow is known. Open it when diagnosing scale or dimension issues, choosing an extraction boundary, formatting a value, or converting across compatible SI and non-SI units.

## Contents

- [Inspection map](#inspection-map)
- [Conversion and extraction choices](#conversion-and-extraction-choices)
- [Compatibility probes](#compatibility-probes)
- [Boundary patterns](#boundary-patterns)
- [Source-backed gotchas](#source-backed-gotchas)

## Inspection Map

| API | Official meaning | Use when |
|---|---|---|
| `q.mantissa` | Raw numerical data without the unit | Inspecting the value in its current unit scale |
| `q.magnitude` | Alias for `q.mantissa` | Interoperating with code that uses magnitude terminology |
| `q.unit` | Attached `Unit`, including physical dimension and scale | Scale-sensitive inspection |
| `q.dim` | Physical dimension, independent of scale | Comparing metres with kilometres or volts with millivolts |
| `q.has_same_unit(other)` | Checks identical physical dimensions | Preflighting a conversion to another `Quantity` or `Unit` |
| `q.is_unitless` | True when the quantity has no physical unit | Guarding scalar-only boundaries |
| `u.split_mantissa_unit(q)` | Returns `(mantissa, unit)` | Decomposing and later reconstructing a quantity |

Despite its name, `Quantity.has_same_unit(other)` is documented as a dimension check: values in `mV` and `V` pass because their physical dimensions match.

## Conversion And Extraction Choices

| Signature | Return | Behavior |
|---|---|---|
| `q.to(new_unit)` | `Quantity` | Rescales the mantissa and attaches a compatible target unit |
| `q.in_unit(unit, err_msg=None)` | `Quantity` | Behaves identically to `to()`; retained for API compatibility |
| `q.to_decimal(unit=Unit('1'))` | Plain number or array | Expresses the value in `unit` and removes the `Quantity` wrapper |
| `q.factorless()` | `Quantity` | Folds the unit factor into the mantissa and returns an equivalent factor-1 quantity |
| `q.item(*args)` | Scalar `Quantity` | Extracts one element without discarding its unit |
| `q.tolist()` | Nested list of scalar `Quantity` objects | Converts array structure while preserving units at every leaf |
| `q.repr_in_unit(precision=None)` | `str` | Formats the value in its current unit |
| `u.display_in_unit(q, unit)` | Display string | Displays a quantity in a selected compatible unit |

Use `to_decimal()` only at a boundary that requires bare numerical data. Use `item()` or `tolist()` when scalar or Python-container structure is needed but physical units must remain attached.

```python
import brainunit as u
import jax.numpy as jnp

q = jnp.array([1.0, 2.0, 3.0]) * u.mV

q_volts = q.to(u.volt)
# Quantity([0.001 0.002 0.003], "V")
raw_volts = q.to_decimal(u.volt)
# Array([0.001, 0.002, 0.003], dtype=float32)
first = q.item(0)
# Expected: scalar Quantity 1 mV.
parts = q.tolist()
# Expected: [1 mV, 2 mV, 3 mV] as scalar Quantity leaves.
mantissa, unit = u.split_mantissa_unit(q)
# Expected: mantissa [1, 2, 3] and unit mV.
rebuilt = mantissa * unit
# Expected: equivalent to q, [1, 2, 3] mV.
```

## Compatibility Probes

The conversion tutorial uses these package-level checks:

```python
import brainunit as u

u.have_same_dim(u.meter, u.kmeter)                         # True
u.have_same_dim(u.meter, u.second)                         # False
u.have_same_dim(u.mV, u.volt)                              # True
u.have_same_dim(
    u.newton,
    u.kilogram * u.meter / u.second**2,
)                                                           # True
u.is_dimensionless((5.0 * u.meter) / (2.0 * u.meter))      # True
```

An incompatible target raises an error rather than silently reinterpreting the mantissa.

## Boundary Patterns

Convert to the unit expected by an external system before removing the wrapper:

```python
voltage = jnp.array([100.0, 200.0, 300.0]) * u.mV
payload_in_volts = voltage.to_decimal(u.volt)
# Array([0.1, 0.2, 0.3], dtype=float32)
```

Conversions are not limited to SI prefixes. The official tutorial demonstrates `mile` to `meter`, `kmeter`, and `foot`; `atmosphere` to `pascal`, `bar`, and `mmHg`; and `joule` to `calorie`, `electron_volt`, and `erg`.

## Source-Backed Gotchas

- Dividing quantities with the same dimension produces a dimensionless result; depending on the input, the result is a plain numeric scalar or array rather than a unit-bearing `Quantity`.
- `float()` and `int()` accept dimensionless values. Direct `float()` on a unit-bearing `Quantity` raises `TypeError`; convert with `to_decimal(target_unit)` first.
- `q.mantissa` is expressed in the current unit scale. It is not automatically converted to a preferred external unit.
- `q.to_decimal()` validates that its argument is a `Unit` and raises on a dimension mismatch.
- `q.in_unit()` accepts `err_msg` for a custom dimension-mismatch message; `q.to()` does not expose that parameter.

## Sources Mirrored

- https://brainunit.readthedocs.io/physical_units/conversion.html
- https://brainunit.readthedocs.io/apis/generated/brainunit.Quantity.html
