# Typing And Runtime Validation

Use this reference when a BrainUnit interface needs precise annotations, reusable physical-type aliases, runtime `isinstance` checks, or annotation-driven unit validation. The canonical `QuantityLike`, `UnitLike`, `Quantity[...]`, and `validate_units` workflow remains in the parent skill.

## Annotation Selection

| Form | Expresses | Use when |
|---|---|---|
| `QuantityLike` | A plain number, NumPy/JAX array, or existing `Quantity` | An adapter accepts values that will be normalized later |
| `UnitLike` | A `Unit`, unit string, or `None` | A parameter selects an optional target unit |
| `DimensionLike` | A `Dimension` or physical-type string | A parameter accepts either representation of a dimension |
| `u.Quantity[u.meter]` | A `Quantity` compatible with the meter dimension | A concrete unit makes the expected dimension clearest |
| `u.Quantity["length"]` | A `Quantity` with the named physical dimension | Scale is intentionally unspecified |
| `HAS_UNIT` | Any `Quantity`, regardless of dimension | Only the presence of a unit-aware value matters |
| `LENGTH`, `TIME`, `SPEED`, and other pre-built aliases | A reusable named physical dimension | Repeated signatures should share a short domain term |

`QuantityLike` is intentionally broad and does not require a unit. Normalize it with `u.math.asarray()` before relying on array or unit properties.

```python
import brainunit as u
from brainunit.typing import QuantityLike, UnitLike


def normalize(values: QuantityLike, unit: UnitLike = None):
    return u.math.asarray(values, unit=unit)
```

## Array And Structural Aliases

`brainunit.typing` re-exports these aliases for signatures shared with JAX and NumPy code.

| Alias | Meaning |
|---|---|
| `Array` | `jax.Array` when JAX is installed; a non-instantiable sentinel type on the no-JAX path |
| `ArrayLike` | `jax.Array`, `numpy.ndarray`, `numpy.number`, or `numpy.bool_`; excludes bare Python scalars |
| `ScalarOrArrayLike` | `ArrayLike` plus Python `bool`, `int`, `float`, and `complex` |
| `DTypeLike` | A NumPy/JAX-compatible dtype specifier |
| `Shape` | `Sequence[int]` |
| `Axis` | One integer axis |
| `Axes` | One integer axis or a sequence of integer axes |
| `PyTree` | An opaque arbitrarily nested PyTree alias |

Prefer `ArrayLike` when code reads `.shape`, `.ndim`, or `.dtype`; use `ScalarOrArrayLike` only when bare Python scalars are also valid.

The unit-specific core aliases are:

- `QuantityLike = int | float | complex | numpy.number | numpy.ndarray | jax.Array | Quantity`
- `UnitLike = Unit | str | None`; `None` means unitless or no requested target unit.
- `DimensionLike = Dimension | str`; strings use a known physical-type name.

## Unit And Physical-Type Annotations

`Quantity[item]` accepts a `Unit` or a physical-type string. Both forms create a type usable in annotations and runtime checks.

```python
def kinetic_energy(
    mass: u.Quantity[u.kilogram],
    speed: u.Quantity[u.meter / u.second],
) -> u.Quantity[u.joule]:
    return 0.5 * mass * speed**2


def travel_time(
    distance: u.Quantity["length"],
    speed: u.Quantity["speed"],
) -> u.Quantity["time"]:
    return distance / speed
```

For a unit item, the runtime type checks physical dimension rather than exact scale. For example, `u.Quantity[u.meter]` accepts a kilometer quantity unless `@validate_units(strict=True)` requests an exact unit match. A physical-type string carries no scale, so strict validation cannot make it scale-specific.

Unknown physical-type strings raise `ValueError`; non-`Unit`, non-string items inside `Quantity[...]` raise `TypeError`.

## Physical-Type Names

The seven base dimensions and their accepted names are:

| Dimension | Accepted names |
|---|---|
| Length | `length` |
| Mass | `mass` |
| Time | `time` |
| Electric current | `current`, `electric current` |
| Temperature | `temperature` |
| Amount of substance | `substance`, `amount of substance` |
| Luminous intensity | `luminosity`, `luminous intensity` |
| Dimensionless | `dimensionless` |

Derived and compound physical-type strings are:

- `frequency`, `force`, `energy`, `power`, `pressure`, and `charge`
- `voltage` or `electric potential`
- `resistance`, `capacitance`, `conductance`, `magnetic flux`, `magnetic field`, and `inductance`
- `speed` or `velocity`, `acceleration`, `area`, `volume`, and `density`
- `momentum`, `angular velocity`, and `torque`

Names are normalized with surrounding whitespace removed and case lowered before lookup.

## Pre-Built Physical-Type Aliases

Use pre-built aliases when the physical type is common and a short annotation improves a repeated signature.

| Group | Aliases |
|---|---|
| General | `HAS_UNIT`, `DIMENSIONLESS_TYPE` |
| Base dimensions | `LENGTH`, `MASS`, `TIME`, `CURRENT`, `TEMPERATURE`, `SUBSTANCE`, `LUMINOSITY` |
| Mechanics and thermodynamics | `FREQUENCY`, `FORCE`, `ENERGY`, `POWER`, `PRESSURE`, `SPEED`, `ACCELERATION`, `AREA`, `VOLUME`, `DENSITY` |
| Electromagnetism | `CHARGE`, `VOLTAGE`, `RESISTANCE`, `CAPACITANCE`, `CONDUCTANCE`, `MAGNETIC_FLUX`, `MAGNETIC_FIELD`, `INDUCTANCE` |

```python
from brainunit.typing import LENGTH, SPEED, TIME


def displacement(speed: SPEED, duration: TIME) -> LENGTH:
    return speed * duration
```

The accepted string catalog is larger than the pre-built alias catalog. Use `Quantity["momentum"]`, `Quantity["angular velocity"]`, or `Quantity["torque"]` when no named alias is exported.

## Runtime Type Helpers

`PhysicalType(name)` returns a runtime-checkable type whose `isinstance` behavior compares a quantity's dimension. `is_physical_type(obj)` reports whether an object is a type created by `PhysicalType`.

`quantity_type(item)` returns the same kind of runtime-checkable quantity type for a `Unit` or physical-type string. Prefer it in `isinstance` when a static analyzer flags parameterized checks such as `isinstance(x, u.Quantity["length"])`.

```python
from brainunit.typing import PhysicalType, is_physical_type, quantity_type

distance = 2.0 * u.kmeter
length_type = quantity_type("length")
meter_type = quantity_type(u.meter)
speed_type = PhysicalType("speed")

assert isinstance(distance, length_type)
assert isinstance(distance, meter_type)
assert not isinstance(distance, speed_type)
assert is_physical_type(speed_type)
# Expected: all four assertions pass.
```

All three positive checks are dimension-based. `quantity_type(u.meter)` does not require the stored scale to be exactly meters.

## Annotation-Driven Runtime Validation

`validate_units` inspects `Quantity[...]` argument annotations and validates each constrained argument on every call. It also recognizes legacy `typing.Annotated[Quantity, unit_or_physical_type]` metadata.

```python
from brainunit.typing import validate_units


@validate_units
def rescale_length(value: u.Quantity[u.meter]):
    return value.in_unit(u.meter)


compatible = rescale_length(2.0 * u.kmeter)
# Expected: 2000 m; compatible dimensions are accepted by default.


@validate_units(strict=True)
def require_meters(value: u.Quantity[u.meter]):
    return value


exact = require_meters(2.0 * u.meter)
# Expected: 2 m; strict validation accepts the exact meter unit.
```

Validation behavior:

- With `strict=False`, the default, a unit annotation requires compatible dimensions and permits a different scale.
- With `strict=True`, a unit annotation requires matching dimension and exact unit magnitude/scale.
- A physical-type annotation validates dimension only under either strictness setting because it specifies no scale.
- A constrained argument must be a `Quantity`; a bare number or array raises `TypeError`.
- A dimension mismatch raises `DimensionMismatchError`; a strict exact-unit mismatch raises `UnitMismatchError` in the current implementation.
- `None` arguments are skipped.
- Only arguments are validated. A return annotation documents the result but is not checked by the decorator.
- If a function has no recognized unit constraints, decoration leaves the function unchanged.

Use `@validate_units` when annotations should own the runtime contract. Use `@u.check_units(...)` when the existing keyword-driven declaration is clearer. Do not stack them by default.

## Common Gotchas

- Do not use `QuantityLike` to claim an argument is unit-bearing; the alias includes plain numbers and arrays.
- Do not assume `u.Quantity[u.meter]` means exact meters during `isinstance` or default validation; it means the meter dimension.
- Do not expect annotations alone to reject an invalid call; apply `validate_units` or another explicit runtime boundary.
- Do not rely on a return annotation for runtime validation; validate the returned quantity separately when that boundary is critical.
- The current BrainUnit compatibility package re-exports `saiunit.typing`, so generated documentation may display the underlying `saiunit` module or exception names. BrainUnit-facing code in this bundle uses the public `brainunit` namespace.
- Some released BrainUnit wheels may lag the official site and lack `brainunit.typing`. Upgrade BrainUnit together with its exact SaiUnit dependency; do not mix a legacy `brainunit.Quantity` with `saiunit.typing` objects from another release.

## Sources Mirrored

- https://brainx.chaobrain.com/brainunit/apis/brainunit.typing.html
- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.PhysicalType.html
- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.is_physical_type.html
- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.quantity_type.html
- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.QuantityLike.html
- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.UnitLike.html
- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.DimensionLike.html
- https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.validate_units.html
- https://brainx.chaobrain.com/brainunit/_modules/saiunit/typing.html
- https://github.com/chaobrain/saiunit/blob/main/brainunit/brainunit/typing.py
- https://github.com/chaobrain/saiunit/blob/main/saiunit/_typing.py
