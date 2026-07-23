# Prefix Library

Use this reference to look up supported SI prefix symbols and scales, inspect the official base/derived unit spellings shown by the standard-units page, or create a prefixed custom unit.

## Contents

- [Supported prefixes](#supported-prefixes)
- [Base-unit spelling lookup](#base-unit-spelling-lookup)
- [Derived-unit lookup](#derived-unit-lookup)
- [Creating scaled units](#creating-scaled-units)
- [Source-backed gotchas](#source-backed-gotchas)

## Supported Prefixes

| Symbol | Prefix | Value |
|---|---|---:|
| `Y` | yotta | `1e24` |
| `Z` | zetta | `1e21` |
| `E` | exa | `1e18` |
| `P` | peta | `1e15` |
| `T` | tera | `1e12` |
| `G` | giga | `1e9` |
| `M` | mega | `1e6` |
| `k` | kilo | `1e3` |
| `h` | hecto | `1e2` |
| `da` | deka/deca | `1e1` |
| `d` | deci | `1e-1` |
| `c` | centi | `1e-2` |
| `m` | milli | `1e-3` |
| `u` | micro | `1e-6` |
| `n` | nano | `1e-9` |
| `p` | pico | `1e-12` |
| `f` | femto | `1e-15` |
| `a` | atto | `1e-18` |
| `z` | zepto | `1e-21` |
| `y` | yocto | `1e-24` |

BrainUnit uses ASCII `u` for the micro prefix in API names and `create_scaled_unit()` calls.

## Base-Unit Spelling Lookup

The standard-units page creates these named objects:

| Local variable | `Unit.create()` name | Display symbol | Dimension key |
|---|---|---|---|
| `metre` | `"meter"` | `"m"` | `m=1` |
| `kilogram` | `"kilogram"` | `"kg"` | `kg=1` |
| `second` | `"second"` | `"s"` | `s=1` |
| `ampere` | `"ampere"` | `"A"` | `A=1` |
| `kelvin` | `"kelvin"` | `"K"` | `K=1` |
| `mole` | `"mole"` | `"mol"` | `mol=1` |
| `candle` | `"candle"` | `"cd"` | `candle=1` |

This is a spelling and symbol lookup. The conceptual role of the fundamental dimensions remains in the main skill.

## Derived-Unit Lookup

The official page constructs these derived named units with `Unit.create(get_or_create_dimension(...), name, symbol)`:

| Name | Symbol | Dimension exponents shown by the source |
|---|---|---|
| `newton` | `N` | `m=1, kg=1, s=-2` |
| `pascal` | `Pa` | `m=-1, kg=1, s=-2` |
| `joule` | `J` | `m=2, kg=1, s=-2` |
| `watt` | `W` | `m=2, kg=1, s=-3` |
| `coulomb` | `C` | `s=1, A=1` |
| `volt` | `V` | `m=2, kg=1, s=-3, A=-1` |
| `farad` | `F` | `m=-2, kg=-1, s=4, A=2` |
| `ohm` | `ohm` | `m=2, kg=1, s=-3, A=-2` |
| `siemens` | `S` | `m=-2, kg=-1, s=3, A=2` |
| `weber` | `Wb` | `m=2, kg=1, s=-2, A=-1` |
| `tesla` | `T` | `kg=1, s=-2, A=-1` |
| `henry` | `H` | `m=2, kg=1, s=-2, A=-2` |
| `lux` | `lx` | `m=-2, cd=1` |
| `gray` | `Gy` | `m=2, s=-2` |
| `sievert` | `Sv` | `m=2, s=-2` |
| `katal` | `kat` | `s=-1, mol=1` |

## Creating Scaled Units

The page constructs every supported prefix variant of a local `metre` object with `Unit.create_scaled_unit()`:

```python
from brainunit import Unit, get_or_create_dimension

metre = Unit.create(get_or_create_dimension(m=1), "meter", "m")
# Unit("m")

kilometre = Unit.create_scaled_unit(metre, "k")
# Unit("km")
millimetre = Unit.create_scaled_unit(metre, "m")
# Unit("mm")
micrometre = Unit.create_scaled_unit(metre, "u")
# Unit("um")
```

Inspect the resulting `Unit.scale` when validating a custom prefix:

```python
assert kilometre.scale == 3
assert millimetre.scale == -3
assert micrometre.scale == -6
# Expected: all assertions pass.
```

## Source-Backed Gotchas

- Prefix symbols are case-sensitive: `M` is mega while `m` is milli.
- `da` is the two-character deka/deca prefix.
- `u` is the supported ASCII spelling for micro.
- The page says BrainUnit provides almost all units with prefixes. Do not infer that every conceivable unit-prefix combination is predefined without checking the package namespace.
- The source uses `metre` as a local variable but assigns the full unit name `"meter"`; distinguish Python variable spelling from `Unit.name`.

## Sources Mirrored

- https://brainx.chaobrain.com/brainunit/physical_units/standard_units.html
