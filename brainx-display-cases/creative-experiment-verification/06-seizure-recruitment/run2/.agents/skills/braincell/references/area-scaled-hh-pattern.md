# Area-Scaled HH Pattern

Use this reference when a single-compartment HH model provides cell area, total capacitance, or conductance densities that must be converted to total conductance and total capacitance.

## Rule

Keep `C`, `g_max`, and injected `I` all density-based by default. Only multiply by area when converting the entire model consistently to total capacitance, total conductance, and total current.

## Pattern

```python
import braincell
import brainunit as u


area = (20000. * u.um ** 2).in_unit(u.cm ** 2)
Cm = (1. * u.uF / u.cm ** 2) * area


class AreaHH(braincell.SingleCompartment):
    def __init__(self, size, solver="ind_exp_euler"):
        super().__init__(size, C=Cm, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(
            INa=braincell.channel.Na_TM1991(
                size,
                g_max=(100. * u.mS / u.cm ** 2) * area,
                V_sh=-63. * u.mV,
            )
        )

        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(
            IK=braincell.channel.K_TM1991(
                size,
                g_max=(30. * u.mS / u.cm ** 2) * area,
                V_sh=-63. * u.mV,
            )
        )

        self.IL = braincell.channel.IL(
            size,
            E=-60. * u.mV,
            g_max=(5. * u.nS / u.cm ** 2) * area,
        )
```

## Checks

- Density conductance times area should become conductance.
- Capacitance density times area should become capacitance.
- If conductances and capacitance are total quantities, injected current should also be a total current rather than a current density.
- If the model remains density-based, do not multiply only one channel by area.
