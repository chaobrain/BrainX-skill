# Physical Constant Library

Use this reference to select the exact predefined constant name, inspect its displayed unit, and avoid replacing a unit-aware constant with a bare numeric literal.

## Import

```python
from brainunit import constants
```

Each listed object is displayed by the official page as a `Quantity`.

## Constant Lookup

| API | Official role | Displayed value and unit |
|---|---|---|
| `constants.avogadro` | Avogadro number | `Quantity(6.02214076e+23, "1 / mol")` |
| `constants.boltzmann` | Boltzmann constant | `Quantity(1.380649e-23, "J / K")` |
| `constants.electric` | Electric constant / vacuum permittivity | `Quantity(8.85418782e-12, "F / m")` |
| `constants.electron_mass` | Electron mass | `Quantity(9.10938371e-31, "kg")` |
| `constants.elementary_charge` | Elementary charge magnitude | `Quantity(1.60217663e-19, "C")` |
| `constants.faraday` | Faraday constant | `Quantity(96485.33212331, "C / mol")` |
| `constants.gas` | Universal gas constant | `Quantity(8.31446262, "J / (K * mol)")` |
| `constants.magnetic` | Magnetic constant / vacuum permeability | `Quantity(1.25663706e-06, "N / A^2")` |
| `constants.molar_mass` | Molar mass constant | `Quantity(0.001, "kg / mol")` |
| `constants.zero_celsius` | Zero Celsius expressed in kelvin | `Quantity(273.15, "K")` |

The values and units mirror the constants page. Use the constant object in calculations so BrainUnit retains its physical dimensions.

## Selection Cues

- Statistical mechanics energy per temperature: `constants.boltzmann`.
- Per-mole ideal-gas calculations: `constants.gas`.
- Particle count per mole: `constants.avogadro`.
- Charge per mole in electrochemistry: `constants.faraday`.
- Single elementary-charge magnitude: `constants.elementary_charge`.
- Electromagnetic vacuum properties: `constants.electric` and `constants.magnetic`.
- Celsius offset handling: `constants.zero_celsius`; see `Convert Celsius At Affine Boundaries` in the parent skill for the affine conversion functions.

## Unit-Aware Use

```python
import brainunit as u

R = u.constants.gas
# Quantity(8.31446262, "J / (K * mol)")
T = u.celsius2kelvin(25.0)
# 298.15 K
V = 0.0224 * u.meter**3
pressure = R * T / V
# Expected: approximately 110667.73 kg / (m * s^2 * mol).
```

This reference is a lookup catalog. Physical-constant arithmetic and the basic constants workflow remain in the main skill.

## Sources Mirrored

- https://brainx.chaobrain.com/brainunit/physical_units/constants.html
