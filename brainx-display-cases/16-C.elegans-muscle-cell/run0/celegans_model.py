"""BrainCell HH surrogate for a C. elegans body-wall muscle cell."""

from __future__ import annotations

from dataclasses import dataclass

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
from braincell.channel._base import Gate, HH


DT = 0.05 * u.ms
DURATION = 500.0 * u.ms
STIMULUS_START = 50.0 * u.ms
STIMULUS_END = 250.0 * u.ms
VOLTAGE_OFFSET_MV = 36.0


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower_nS: float
    upper_nS: float
    initial_nS: float


PARAMETER_SPECS = (
    ParameterSpec("g_na", 30.0, 900.0, 350.0),
    ParameterSpec("g_kr", 10.0, 300.0, 60.0),
    ParameterSpec("g_shk1", 0.01, 150.0, 1.0),
    ParameterSpec("g_egl19", 0.01, 80.0, 1.0),
    ParameterSpec("g_slo2", 0.01, 150.0, 0.1),
    ParameterSpec("g_leak", 0.05, 10.0, 0.75),
)


class _ParameterizedHH(HH):
    def __init__(self, size, g_max):
        super().__init__(size=size)
        self.g_max = brainstate.ParamState(g_max)


class Sodium(_ParameterizedHH):
    root_type = braincell.HHTypedNeuron
    gates = (Gate("m", power=3, phi=0.25), Gate("h", power=1, phi=0.25))

    def current(self, V):
        return self.g_max.value * self.conductance_factor(V) * (28.0 * u.mV - V)

    def f_m_alpha(self, V):
        shifted = V.to_decimal(u.mV) - VOLTAGE_OFFSET_MV
        return 1.0 / u.math.exprel(-(shifted + 40.0) / 10.0)

    def f_m_beta(self, V):
        shifted = V.to_decimal(u.mV) - VOLTAGE_OFFSET_MV
        return 4.0 * u.math.exp(-(shifted + 65.0) / 18.0)

    def f_h_alpha(self, V):
        shifted = V.to_decimal(u.mV) - VOLTAGE_OFFSET_MV
        return 0.07 * u.math.exp(-(shifted + 65.0) / 20.0)

    def f_h_beta(self, V):
        shifted = V.to_decimal(u.mV) - VOLTAGE_OFFSET_MV
        return 1.0 / (1.0 + u.math.exp(-(shifted + 35.0) / 10.0))


class DelayedRectifierKr(_ParameterizedHH):
    root_type = braincell.HHTypedNeuron
    gates = (Gate("n", power=4, phi=0.25),)

    def current(self, V):
        return self.g_max.value * self.conductance_factor(V) * (-41.0 * u.mV - V)

    def f_n_alpha(self, V):
        shifted = V.to_decimal(u.mV) - VOLTAGE_OFFSET_MV
        return 0.1 / u.math.exprel(-(shifted + 55.0) / 10.0)

    def f_n_beta(self, V):
        shifted = V.to_decimal(u.mV) - VOLTAGE_OFFSET_MV
        return 0.125 * u.math.exp(-(shifted + 65.0) / 80.0)


class SHK1(_ParameterizedHH):
    root_type = braincell.HHTypedNeuron
    gates = (Gate("a", power=3), Gate("b", power=1))

    def current(self, V):
        return self.g_max.value * self.conductance_factor(V) * (-41.0 * u.mV - V)

    def f_a_inf(self, V):
        return 1.0 / (1.0 + u.math.exp(-(V.to_decimal(u.mV) + 20.0) / 6.0))

    def f_a_tau(self, V):
        return 8.0 + 0.0 * V.to_decimal(u.mV)

    def f_b_inf(self, V):
        return 1.0 / (1.0 + u.math.exp((V.to_decimal(u.mV) + 25.0) / 6.0))

    def f_b_tau(self, V):
        return 80.0 + 0.0 * V.to_decimal(u.mV)


class EGL19(_ParameterizedHH):
    root_type = braincell.HHTypedNeuron
    gates = (Gate("c", power=2), Gate("d", power=1))

    def current(self, V):
        return self.g_max.value * self.conductance_factor(V) * (60.0 * u.mV - V)

    def f_c_inf(self, V):
        return 1.0 / (1.0 + u.math.exp(-(V.to_decimal(u.mV) + 15.0) / 5.0))

    def f_c_tau(self, V):
        return 5.0 + 0.0 * V.to_decimal(u.mV)

    def f_d_inf(self, V):
        return 1.0 / (1.0 + u.math.exp((V.to_decimal(u.mV) + 10.0) / 6.0))

    def f_d_tau(self, V):
        return 60.0 + 0.0 * V.to_decimal(u.mV)


class SLO2(_ParameterizedHH):
    root_type = braincell.HHTypedNeuron
    gates = (Gate("q", power=1),)

    def current(self, V):
        return self.g_max.value * self.conductance_factor(V) * (-41.0 * u.mV - V)

    def f_q_inf(self, V):
        return 1.0 / (1.0 + u.math.exp(-(V.to_decimal(u.mV) + 5.0) / 7.0))

    def f_q_tau(self, V):
        return 100.0 + 0.0 * V.to_decimal(u.mV)


class Leak(braincell.Channel):
    root_type = braincell.HHTypedNeuron

    def __init__(self, size, g_max):
        super().__init__(size=size)
        self.g_max = brainstate.ParamState(g_max)

    def current(self, V):
        return self.g_max.value * (-18.387 * u.mV - V)

    def compute_derivative(self, V):
        _ = V


class CElegansMuscle(braincell.SingleCompartment):
    """Single-compartment model with the six currents named in the prompt."""

    def __init__(self, initial_voltage, parameters_nS=None, solver="ind_exp_euler"):
        values = initial_parameter_vector() if parameters_nS is None else parameters_nS
        super().__init__(
            size=1,
            V_initializer=braintools.init.Constant(initial_voltage),
            V_th=-10.0 * u.mV,
            C=2.5 * u.pF,
            solver=solver,
        )
        self.Na = Sodium(1, values[0] * u.nS)
        self.Kr = DelayedRectifierKr(1, values[1] * u.nS)
        self.SHK1 = SHK1(1, values[2] * u.nS)
        self.EGL19 = EGL19(1, values[3] * u.nS)
        self.SLO2 = SLO2(1, values[4] * u.nS)
        self.Leak = Leak(1, values[5] * u.nS)


def initial_parameter_vector():
    return jnp.asarray([spec.initial_nS for spec in PARAMETER_SPECS])


def parameter_bounds():
    return {
        spec.name: (spec.lower_nS, spec.upper_nS) for spec in PARAMETER_SPECS
    }


def parameter_dict(vector):
    return {spec.name: float(vector[i]) for i, spec in enumerate(PARAMETER_SPECS)}


def apply_parameter_vector(cell, vector):
    cell.Na.g_max.value = vector[0] * u.nS
    cell.Kr.g_max.value = vector[1] * u.nS
    cell.SHK1.g_max.value = vector[2] * u.nS
    cell.EGL19.g_max.value = vector[3] * u.nS
    cell.SLO2.g_max.value = vector[4] * u.nS
    cell.Leak.g_max.value = vector[5] * u.nS


def initialize_gates_at_steady_state(cell):
    V = cell.V.value
    cell.Na.m.value = cell.Na.f_m_alpha(V) / (
        cell.Na.f_m_alpha(V) + cell.Na.f_m_beta(V)
    )
    cell.Na.h.value = cell.Na.f_h_alpha(V) / (
        cell.Na.f_h_alpha(V) + cell.Na.f_h_beta(V)
    )
    cell.Kr.n.value = cell.Kr.f_n_alpha(V) / (
        cell.Kr.f_n_alpha(V) + cell.Kr.f_n_beta(V)
    )
    cell.SHK1.a.value = cell.SHK1.f_a_inf(V)
    cell.SHK1.b.value = cell.SHK1.f_b_inf(V)
    cell.EGL19.c.value = cell.EGL19.f_c_inf(V)
    cell.EGL19.d.value = cell.EGL19.f_d_inf(V)
    cell.SLO2.q.value = cell.SLO2.f_q_inf(V)


def reset_runtime_state(cell):
    cell.reset_state()
    initialize_gates_at_steady_state(cell)


def current_protocol(current_pA, dt=DT):
    times = u.math.arange(0.0 * u.ms, DURATION, dt)
    current = u.math.where(
        (times >= STIMULUS_START) & (times < STIMULUS_END),
        current_pA * u.pA,
        0.0 * u.pA,
    )
    return times, current


def rollout(cell, current, dt=DT, record_currents=False):
    indices = jnp.arange(current.shape[0])

    def step(index, input_current):
        with brainstate.environ.context(t=index * dt, i=index):
            cell.update(input_current)
        voltage = cell.V.value
        if not record_currents:
            return voltage
        return (
            voltage,
            cell.Na.current(voltage),
            cell.Kr.current(voltage),
            cell.SHK1.current(voltage),
            cell.EGL19.current(voltage),
            cell.SLO2.current(voltage),
            cell.Leak.current(voltage),
        )

    with brainstate.environ.context(dt=dt):
        return brainstate.transform.for_loop(step, indices, current)


def simulate(current_pA, initial_voltage_mV, parameters_nS=None, dt=DT, solver="ind_exp_euler", record_currents=False):
    cell = CElegansMuscle(
        initial_voltage=initial_voltage_mV * u.mV,
        parameters_nS=parameters_nS,
        solver=solver,
    )
    cell.init_state()
    initialize_gates_at_steady_state(cell)
    _, current = current_protocol(current_pA, dt=dt)
    return rollout(cell, current, dt=dt, record_currents=record_currents)
