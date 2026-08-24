"""BrainCell implementation of a phenomenological C. elegans muscle HH cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp

import braincell
import braincell.channel._base
import brainstate
import braintools
import brainunit as u


CONDUCTANCE_UNIT = u.mS / u.cm**2
CAPACITANCE_UNIT = u.uF / u.cm**2
CELL_AREA = 2000.0 * u.um**2


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: u.Unit
    lower: float
    upper: float
    initial: float
    destination: str


PARAMETER_SPECS = (
    ParameterSpec("g_shk1", CONDUCTANCE_UNIT, 0.05, 5.00, 1.000, "k.SHK1.g_max"),
    ParameterSpec("g_egl19", CONDUCTANCE_UNIT, 0.001, 0.15, 0.025, "ca.EGL19.g_max"),
    ParameterSpec("g_slo2", CONDUCTANCE_UNIT, 0.001, 0.20, 0.030, "kca.SLO2.g_max"),
    ParameterSpec("g_kr", CONDUCTANCE_UNIT, 0.05, 5.00, 1.000, "k.Kr.g_max"),
    ParameterSpec("g_na", CONDUCTANCE_UNIT, 0.05, 5.00, 1.000, "na.Na.g_max"),
    ParameterSpec("g_leak", CONDUCTANCE_UNIT, 0.0001, 0.03, 0.003, "IL.g_max"),
    ParameterSpec("capacitance", CAPACITANCE_UNIT, 0.50, 2.50, 1.00, "C"),
)


def _sigmoid_activation(voltage_mV, half_mV, slope_mV):
    return 1.0 / (1.0 + u.math.exp(-(voltage_mV - half_mV) / slope_mV))


def _sigmoid_inactivation(voltage_mV, half_mV, slope_mV):
    return 1.0 / (1.0 + u.math.exp((voltage_mV - half_mV) / slope_mV))


class SHK1(braincell.channel._base.HH):
    """Fast Shaker-like potassium current with activation and inactivation."""

    root_type = braincell.ion.Potassium
    gates = (
        braincell.channel._base.Gate("m", power=3),
        braincell.channel._base.Gate("h", power=1),
    )

    def __init__(self, size, g_max=0.06 * CONDUCTANCE_UNIT):
        super().__init__(size=size)
        self.g_max = braintools.init.param(g_max, self.varshape)

    def current(self, V, potassium):
        return self.g_max * self.conductance_factor(V, potassium) * (potassium.E - V)

    def f_m_inf(self, V, potassium):
        del potassium
        return _sigmoid_activation(V.to_decimal(u.mV), -8.0, 7.5)

    def f_m_tau(self, V, potassium):
        del potassium
        vm = V.to_decimal(u.mV)
        return 2.0 + 5.0 / (1.0 + u.math.exp((vm + 5.0) / 10.0))

    def f_h_inf(self, V, potassium):
        del potassium
        return _sigmoid_inactivation(V.to_decimal(u.mV), -24.0, 7.0)

    def f_h_tau(self, V, potassium):
        del (V, potassium)
        return 55.0


class EGL19(braincell.channel._base.HH):
    """Slowly inactivating L-type calcium current that drives the action potential."""

    root_type = braincell.ion.Calcium
    gates = (
        braincell.channel._base.Gate("m", power=2),
        braincell.channel._base.Gate("h", power=1),
    )

    def __init__(self, size, g_max=0.025 * CONDUCTANCE_UNIT):
        super().__init__(size=size)
        self.g_max = braintools.init.param(g_max, self.varshape)

    def current(self, V, calcium):
        return self.g_max * self.conductance_factor(V, calcium) * (calcium.E - V)

    def f_m_inf(self, V, calcium):
        del calcium
        return _sigmoid_activation(V.to_decimal(u.mV), -12.0, 6.5)

    def f_m_tau(self, V, calcium):
        del calcium
        vm = V.to_decimal(u.mV)
        return 1.5 + 2.5 / (1.0 + u.math.exp((vm + 10.0) / 8.0))

    def f_h_inf(self, V, calcium):
        del (V, calcium)
        return 1.0

    def f_h_tau(self, V, calcium):
        del (V, calcium)
        return 1.0


class Kr(braincell.channel._base.HH):
    """Delayed rectifier potassium current."""

    root_type = braincell.ion.Potassium
    gates = (braincell.channel._base.Gate("n", power=4),)

    def __init__(self, size, g_max=0.04 * CONDUCTANCE_UNIT):
        super().__init__(size=size)
        self.g_max = braintools.init.param(g_max, self.varshape)

    def current(self, V, potassium):
        return self.g_max * self.conductance_factor(V, potassium) * (potassium.E - V)

    def f_n_inf(self, V, potassium):
        del potassium
        return _sigmoid_activation(V.to_decimal(u.mV), -10.0, 9.0)

    def f_n_tau(self, V, potassium):
        del potassium
        vm = V.to_decimal(u.mV)
        return 8.0 + 18.0 / (1.0 + u.math.exp((vm + 15.0) / 10.0))


class Na(braincell.channel._base.HH):
    """Small fast sodium current retained from the requested model."""

    root_type = braincell.ion.Sodium
    gates = (
        braincell.channel._base.Gate("m", power=3),
        braincell.channel._base.Gate("h", power=1),
    )

    def __init__(self, size, g_max=0.008 * CONDUCTANCE_UNIT):
        super().__init__(size=size)
        self.g_max = braintools.init.param(g_max, self.varshape)

    def current(self, V, sodium):
        return self.g_max * self.conductance_factor(V, sodium) * (sodium.E - V)

    def f_m_inf(self, V, sodium):
        del sodium
        return _sigmoid_activation(V.to_decimal(u.mV), -10.0, 5.5)

    def f_m_tau(self, V, sodium):
        del (V, sodium)
        return 0.7

    def f_h_inf(self, V, sodium):
        del sodium
        return _sigmoid_inactivation(V.to_decimal(u.mV), -22.0, 6.0)

    def f_h_tau(self, V, sodium):
        del (V, sodium)
        return 5.0


class MuscleCell(braincell.SingleCompartment):
    """One isopotential body-wall muscle cell with six named currents."""

    def __init__(self, size, params: Mapping[str, u.Quantity], initial_v: u.Quantity):
        super().__init__(
            size=size,
            C=params["capacitance"],
            V_initializer=braintools.init.Constant(initial_v),
            V_th=0.0 * u.mV,
            solver="rk4",
        )
        self.k = braincell.ion.PotassiumFixed(size, E=-75.0 * u.mV)
        self.k.add(SHK1=SHK1(size, g_max=params["g_shk1"]))
        self.k.add(Kr=Kr(size, g_max=params["g_kr"]))

        self.na = braincell.ion.SodiumFixed(size, E=55.0 * u.mV)
        self.na.add(Na=Na(size, g_max=params["g_na"]))

        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5.0e-5 * u.mM,
            tau=80.0 * u.ms,
            d=0.8 * u.um,
        )
        self.ca.add(EGL19=EGL19(size, g_max=params["g_egl19"]))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(
            SLO2=braincell.channel.AHP_De1994(size, g_max=params["g_slo2"])
        )

        self.IL = braincell.channel.IL(
            size,
            E=-31.0 * u.mV,
            g_max=params["g_leak"],
        )


def decode_parameters(values) -> dict[str, u.Quantity]:
    values = jnp.asarray(values)
    if values.shape[-1] != len(PARAMETER_SPECS):
        raise ValueError(f"Expected {len(PARAMETER_SPECS)} parameters, got {values.shape}.")
    return {spec.name: values[..., i] * spec.unit for i, spec in enumerate(PARAMETER_SPECS)}


def initial_parameter_vector():
    return jnp.asarray([spec.initial for spec in PARAMETER_SPECS])


def parameter_bounds():
    return [(spec.lower, spec.upper) for spec in PARAMETER_SPECS]


def simulate(
    parameter_values,
    current: u.Quantity,
    initial_v: u.Quantity,
    dt: u.Quantity = 0.1 * u.ms,
    return_states: bool = False,
):
    """Run one BrainCell rollout, optionally retaining all dynamic biological State."""
    parameter_values = jnp.asarray(parameter_values)
    params = decode_parameters(parameter_values)
    candidate_shape = parameter_values.shape[:-1]
    scalar_candidate = not candidate_shape
    size = 1 if scalar_candidate else candidate_shape
    cell = MuscleCell(size=size, params=params, initial_v=initial_v)
    current_density = current / CELL_AREA
    if scalar_candidate:
        current_density = u.math.expand_dims(current_density, axis=-1)
    else:
        current_density = u.math.broadcast_to(
            u.math.expand_dims(current_density, axis=-1),
            (current.shape[0],) + candidate_shape,
        )
    with brainstate.environ.context(dt=dt):
        cell.init_state()
        indices = jnp.arange(current.shape[0])

        def step(index, input_current):
            with brainstate.environ.context(i=index, t=index * dt):
                cell.update(input_current)
            if not return_states:
                return cell.V.value
            return {
                "voltage": cell.V.value,
                "shk1_m": cell.k.SHK1.m.value,
                "shk1_h": cell.k.SHK1.h.value,
                "egl19_m": cell.ca.EGL19.m.value,
                "egl19_h": cell.ca.EGL19.h.value,
                "kr_n": cell.k.Kr.n.value,
                "na_m": cell.na.Na.m.value,
                "na_h": cell.na.Na.h.value,
                "calcium_i": cell.ca.Ci.value,
            }

        output = brainstate.transform.for_loop(step, indices, current_density)
    if return_states:
        return {
            name: values[:, 0] if scalar_candidate else values
            for name, values in output.items()
        }
    return output[:, 0] if scalar_candidate else output
