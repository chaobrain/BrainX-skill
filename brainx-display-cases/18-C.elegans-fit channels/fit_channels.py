"""Fit BrainCell HH models for C. elegans SHK-1 and EGL-19 channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from braincell.channel._base import Gate, HH
from igor2 import packed
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
K_FILE = ROOT / "Fig1C D I-V K currents.pxp"
CA_FILE = ROOT / "Fig. 3A I-V Ca currents.pxp"
RESULTS = Path(os.environ.get("BRAINX_CHANNEL_RESULTS", ROOT / "results"))
V_HOLD_MV = -60.0
E_K_MV = -30.0
E_CA_MV = 60.0
SHK_VOLTAGES_MV = np.arange(0.0, 101.0, 20.0)
EGL_VOLTAGES_MV = np.arange(-20.0, 41.0, 10.0)
PARAMETER_ORDER = (
    "g_max_nS",
    "v_half_m_mV",
    "k_m_mV",
    "tau_m_min_ms",
    "tau_m_amp_ms",
    "v_tau_m_mV",
    "k_tau_m_mV",
    "h_min",
    "h_max",
    "v_half_h_mV",
    "k_h_mV",
    "tau_h_ms",
)


@dataclass(frozen=True)
class ClampData:
    channel: str
    time_ms: np.ndarray
    step_time_ms: np.ndarray
    voltages_mV: np.ndarray
    currents_pA: np.ndarray
    source_waves: tuple[str, ...]
    step_start_ms: float
    step_end_ms: float


@dataclass(frozen=True)
class ShkParameters:
    g_max_nS: float
    e_rev_mV: float
    n_v_offset_mV: float
    n_slope_mV: float
    tau_min_ms: float
    tau_amp_ms: float
    tau_slope_mV: float


@dataclass(frozen=True)
class EglParameters:
    g_max_nS: float
    e_rev_mV: float
    v_half_m_mV: float
    k_m_mV: float
    tau_m_min_ms: float
    tau_m_amp_ms: float
    v_tau_m_mV: float
    k_tau_m_mV: float
    h_min: float
    h_max: float
    v_half_h_mV: float
    k_h_mV: float
    tau_h_ms: float


def _wave(root: dict, number: int) -> np.ndarray:
    record = root[f"wave{number}".encode()]
    return np.asarray(record.wave["wave"]["wData"], dtype=float)


def _packed_root(path: Path) -> dict:
    return packed.load(str(path))[1]["root"]


def _step_bounds(time_ms: np.ndarray, command_mV: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(time_ms) & np.isfinite(command_mV)
    finite_time = time_ms[finite]
    finite_command = command_mV[finite]
    changes = np.flatnonzero(np.diff(finite_command) != 0.0)
    if changes.size != 2:
        raise ValueError(f"expected exactly two command transitions, got {changes.size}")
    return float(finite_time[changes[0] + 1]), float(finite_time[changes[1] + 1])


def _baseline_correct(
    time_ms: np.ndarray,
    currents_pA: np.ndarray,
    start_ms: float,
    end_ms: float,
    transient_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_mask = (time_ms >= 2.0) & (time_ms < 10.0)
    baselines = np.mean(currents_pA[:, baseline_mask], axis=1, keepdims=True)
    corrected = currents_pA - baselines
    fit_mask = (
        (time_ms >= start_ms + transient_ms)
        & (time_ms < end_ms)
        & np.all(np.isfinite(corrected), axis=0)
    )
    return time_ms[fit_mask] - start_ms, corrected[:, fit_mask]


def load_shk_data() -> ClampData:
    root = _packed_root(K_FILE)
    time_ms = _wave(root, 86)
    commands = np.stack([_wave(root, i) for i in range(167, 179)])
    processed_difference = np.stack([_wave(root, i) for i in range(154, 166)])
    start_ms, end_ms = _step_bounds(time_ms, commands[-1])
    if not np.isclose(end_ms - start_ms, 100.0, atol=0.11):
        raise ValueError("SHK-1 command step is not 100 ms")
    selected = np.arange(6, 12)
    step_time, currents = _baseline_correct(
        time_ms,
        processed_difference[selected],
        start_ms,
        end_ms,
        transient_ms=0.5,
    )
    observed_voltages = np.array([np.nanmax(commands[i]) for i in selected])
    np.testing.assert_allclose(observed_voltages, SHK_VOLTAGES_MV)
    return ClampData(
        channel="SHK-1",
        time_ms=time_ms,
        step_time_ms=step_time,
        voltages_mV=observed_voltages,
        currents_pA=currents,
        source_waves=tuple(f"wave{i}" for i in range(160, 166)),
        step_start_ms=start_ms,
        step_end_ms=end_ms,
    )


def load_egl_data() -> ClampData:
    root = _packed_root(CA_FILE)
    time_ms = _wave(root, 0)
    commands = np.stack([_wave(root, i) for i in range(34, 45)])
    wt_currents = np.stack([_wave(root, i) for i in range(12, 23)])
    start_ms, end_ms = _step_bounds(time_ms, commands[-1])
    if not np.isclose(end_ms - start_ms, 100.0, atol=0.03):
        raise ValueError("EGL-19 command step is not 100 ms")
    selected = np.arange(4, 11)
    step_time, currents = _baseline_correct(
        time_ms,
        wt_currents[selected],
        start_ms,
        end_ms,
        transient_ms=1.0,
    )
    observed_voltages = np.array([np.max(commands[i]) for i in selected])
    np.testing.assert_allclose(observed_voltages, EGL_VOLTAGES_MV)
    return ClampData(
        channel="EGL-19",
        time_ms=time_ms,
        step_time_ms=step_time,
        voltages_mV=observed_voltages,
        currents_pA=currents,
        source_waves=tuple(f"wave{i}" for i in range(16, 23)),
        step_start_ms=start_ms,
        step_end_ms=end_ms,
    )


def shk_n_inf(voltage_mV: np.ndarray, p: ShkParameters) -> np.ndarray:
    return 0.5 * (np.tanh((voltage_mV + p.n_v_offset_mV) / p.n_slope_mV) + 1.0)


def shk_tau_n(voltage_mV: np.ndarray, p: ShkParameters) -> np.ndarray:
    return p.tau_min_ms + p.tau_amp_ms / (
        1.0 + np.exp(voltage_mV / p.tau_slope_mV)
    )


def simulate_shk(
    step_time_ms: np.ndarray,
    voltages_mV: np.ndarray,
    p: ShkParameters,
) -> np.ndarray:
    voltages = np.asarray(voltages_mV)[:, None]
    times = np.asarray(step_time_ms)[None, :]
    n0 = shk_n_inf(np.asarray([V_HOLD_MV]), p)[0]
    n_ss = shk_n_inf(voltages, p)
    tau = shk_tau_n(voltages, p)
    n = n_ss + (n0 - n_ss) * np.exp(-times / tau)
    hold = p.g_max_nS * n0**4 * (V_HOLD_MV - p.e_rev_mV)
    return p.g_max_nS * n**4 * (voltages - p.e_rev_mV) - hold


def _fit_shk_trace(time_ms: np.ndarray, current_pA: np.ndarray) -> tuple[float, float]:
    scale = max(50.0, float(np.percentile(np.abs(current_pA), 95)))

    def residual(theta):
        amplitude, tau = theta
        predicted = amplitude * (1.0 - np.exp(-time_ms / tau)) ** 4
        return (predicted - current_pA) / scale

    amplitude0 = max(1.0, float(np.mean(current_pA[-100:])))
    best = least_squares(
        residual,
        x0=np.array([amplitude0, 10.0]),
        bounds=(np.array([0.0, 0.1]), np.array([10000.0, 300.0])),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=4000,
    )
    return float(best.x[0]), float(best.x[1])


def fit_shk(data: ClampData) -> tuple[ShkParameters, dict[str, np.ndarray]]:
    local = np.array(
        [_fit_shk_trace(data.step_time_ms, trace) for trace in data.currents_pA]
    )
    amplitudes, tau_points = local.T
    conductance = amplitudes / (data.voltages_mV - E_K_MV)
    n_points = np.clip((conductance / np.max(conductance)) ** 0.25, 0.0, 1.0)

    def n_residual(theta):
        offset, slope = theta
        pred = 0.5 * (np.tanh((data.voltages_mV + offset) / slope) + 1.0)
        return pred - n_points

    n_fit = least_squares(
        n_residual,
        x0=np.array([15.2, 36.22]),
        bounds=(np.array([-100.0, 1.0]), np.array([150.0, 200.0])),
    )

    def tau_residual(theta):
        tau_min, tau_amp, slope = theta
        pred = tau_min + tau_amp / (
            1.0 + np.exp(data.voltages_mV / slope)
        )
        return (pred - tau_points) / np.maximum(tau_points, 1.0)

    tau_fit = least_squares(
        tau_residual,
        x0=np.array([0.8, 10.0, 20.0]),
        bounds=(
            np.array([0.05, 0.0, 1.0]),
            np.array([100.0, 1000.0, 200.0]),
        ),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=10000,
    )
    provisional = ShkParameters(
        g_max_nS=float(np.max(conductance)),
        e_rev_mV=E_K_MV,
        n_v_offset_mV=float(n_fit.x[0]),
        n_slope_mV=float(n_fit.x[1]),
        tau_min_ms=float(tau_fit.x[0]),
        tau_amp_ms=float(tau_fit.x[1]),
        tau_slope_mV=float(tau_fit.x[2]),
    )
    factors = shk_n_inf(data.voltages_mV, provisional) ** 4 * (
        data.voltages_mV - E_K_MV
    )
    g_max = float(np.dot(factors, amplitudes) / np.dot(factors, factors))
    params = ShkParameters(**{**asdict(provisional), "g_max_nS": g_max})
    return params, {
        "amplitude_pA": amplitudes,
        "n_inf": n_points,
        "tau_n_ms": tau_points,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _decode_egl(theta: np.ndarray) -> EglParameters:
    values = dict(zip(PARAMETER_ORDER, np.asarray(theta, dtype=float), strict=True))
    return EglParameters(e_rev_mV=E_CA_MV, **values)


def egl_gate_functions(
    voltage_mV: np.ndarray, p: EglParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voltage = np.asarray(voltage_mV, dtype=float)
    m_inf = _sigmoid((voltage - p.v_half_m_mV) / p.k_m_mV)
    tau_m = p.tau_m_min_ms + p.tau_m_amp_ms / (
        np.exp(np.clip(-(voltage - p.v_tau_m_mV) / p.k_tau_m_mV, -60.0, 60.0))
        + np.exp(np.clip((voltage - p.v_tau_m_mV) / p.k_tau_m_mV, -60.0, 60.0))
    )
    h_inf = p.h_min + (p.h_max - p.h_min) * _sigmoid(
        -(voltage - p.v_half_h_mV) / p.k_h_mV
    )
    tau_h = np.full_like(voltage, p.tau_h_ms)
    return m_inf, tau_m, h_inf, tau_h


def simulate_egl(
    step_time_ms: np.ndarray,
    voltages_mV: np.ndarray,
    p: EglParameters,
) -> np.ndarray:
    voltages = np.asarray(voltages_mV, dtype=float)[:, None]
    times = np.asarray(step_time_ms, dtype=float)[None, :]
    m_inf, tau_m, h_inf, tau_h = egl_gate_functions(voltages, p)
    m0, _, h0, _ = egl_gate_functions(np.asarray([V_HOLD_MV]), p)
    m = m_inf + (m0[0] - m_inf) * np.exp(-times / tau_m)
    h = h_inf + (h0[0] - h_inf) * np.exp(-times / tau_h)
    hold = p.g_max_nS * m0[0] ** 2 * h0[0] * (V_HOLD_MV - p.e_rev_mV)
    return p.g_max_nS * m**2 * h * (voltages - p.e_rev_mV) - hold


def fit_egl(data: ClampData) -> EglParameters:
    trace_scale = np.maximum(25.0, np.percentile(np.abs(data.currents_pA), 95, axis=1))

    def residual(theta):
        prediction = simulate_egl(data.step_time_ms, data.voltages_mV, _decode_egl(theta))
        return ((prediction - data.currents_pA) / trace_scale[:, None]).ravel()

    lower = np.array([1.0, -50.0, 1.0, 0.05, 0.0, -50.0, 1.0, 0.0, 0.2, -50.0, 0.5, 2.0])
    upper = np.array([80.0, 30.0, 40.0, 10.0, 30.0, 40.0, 60.0, 0.8, 1.0, 30.0, 40.0, 150.0])
    starts = [
        np.array([19.8, -8.0, 8.6, 0.4, 0.7, -5.0, 15.0, 0.28, 0.70, -11.0, 2.0, 30.0]),
        np.array([15.0, -5.0, 10.0, 0.5, 2.0, -5.0, 20.0, 0.20, 0.80, -5.0, 5.0, 40.0]),
        np.array([25.0, -10.0, 6.0, 0.2, 5.0, 0.0, 12.0, 0.30, 0.90, -15.0, 8.0, 20.0]),
    ]
    fits = [
        least_squares(
            residual,
            x0=start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.08,
            max_nfev=3000,
        )
        for start in starts
    ]
    finite = [fit for fit in fits if np.isfinite(fit.cost)]
    if not finite:
        raise RuntimeError("all EGL-19 fitting starts failed")
    return _decode_egl(min(finite, key=lambda fit: fit.cost).x)


def extract_egl_points(data: ClampData, p: EglParameters) -> dict[str, np.ndarray]:
    global_values = np.stack(egl_gate_functions(data.voltages_mV, p), axis=1)
    m0, _, h0, _ = egl_gate_functions(np.asarray([V_HOLD_MV]), p)
    hold = p.g_max_nS * m0[0] ** 2 * h0[0] * (V_HOLD_MV - p.e_rev_mV)
    trace_scale = np.maximum(25.0, np.percentile(np.abs(data.currents_pA), 95, axis=1))
    points = []
    for voltage, current, initial, scale in zip(
        data.voltages_mV, data.currents_pA, global_values, trace_scale, strict=True
    ):
        def residual(theta):
            m_inf, tau_m = theta
            _, _, h_inf, tau_h = initial
            m = m_inf + (m0[0] - m_inf) * np.exp(-data.step_time_ms / tau_m)
            h = h_inf + (h0[0] - h_inf) * np.exp(-data.step_time_ms / tau_h)
            pred = p.g_max_nS * m**2 * h * (voltage - p.e_rev_mV) - hold
            return (pred - current) / scale

        start = np.clip(initial[:2], [0.01, 0.05], [0.99, 60.0])
        result = least_squares(
            residual,
            x0=start,
            bounds=([0.005, 0.05], [0.995, 60.0]),
            loss="soft_l1",
            f_scale=0.08,
            max_nfev=2000,
        )
        points.append(result.x)
    points_array = np.asarray(points)
    return {
        "m_inf": points_array[:, 0],
        "tau_m_ms": points_array[:, 1],
    }


class SHK1Channel(HH):
    """BrainCell SHK-1 delayed-rectifier channel with fitted n^4 kinetics."""

    root_type = braincell.ion.Potassium
    gates = (Gate("n", power=4),)

    def __init__(self, size, params: ShkParameters):
        super().__init__(size=size)
        self.params = params
        self.g_max = params.g_max_nS * u.nS

    def f_n_inf(self, voltage, potassium):
        del potassium
        value = voltage.to_decimal(u.mV)
        p = self.params
        return 0.5 * (u.math.tanh((value + p.n_v_offset_mV) / p.n_slope_mV) + 1.0)

    def f_n_tau(self, voltage, potassium):
        del potassium
        value = voltage.to_decimal(u.mV)
        p = self.params
        return p.tau_min_ms + p.tau_amp_ms / (
            1.0 + u.math.exp(value / p.tau_slope_mV)
        )

    def current(self, voltage, potassium):
        return self.g_max * self.conductance_factor(voltage, potassium) * (
            potassium.E - voltage
        )


class EGL19Channel(HH):
    """BrainCell EGL-19 L-type calcium channel with fitted m^2 h kinetics."""

    root_type = braincell.ion.Calcium
    gates = (Gate("m", power=2), Gate("h", power=1))

    def __init__(self, size, params: EglParameters):
        super().__init__(size=size)
        self.params = params
        self.g_max = params.g_max_nS * u.nS

    def f_m_inf(self, voltage, calcium):
        del calcium
        p = self.params
        value = voltage.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(value - p.v_half_m_mV) / p.k_m_mV))

    def f_m_tau(self, voltage, calcium):
        del calcium
        p = self.params
        value = voltage.to_decimal(u.mV)
        x = (value - p.v_tau_m_mV) / p.k_tau_m_mV
        return p.tau_m_min_ms + p.tau_m_amp_ms / (u.math.exp(-x) + u.math.exp(x))

    def f_h_inf(self, voltage, calcium):
        del calcium
        p = self.params
        value = voltage.to_decimal(u.mV)
        return p.h_min + (p.h_max - p.h_min) / (
            1.0 + u.math.exp((value - p.v_half_h_mV) / p.k_h_mV)
        )

    def f_h_tau(self, voltage, calcium):
        del voltage, calcium
        return self.params.tau_h_ms

    def current(self, voltage, calcium):
        return self.g_max * self.conductance_factor(voltage, calcium) * (
            calcium.E - voltage
        )


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    residual = predicted - observed
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    span = np.maximum(np.ptp(observed, axis=1), 1.0)
    correlation = []
    for obs, pred in zip(observed, predicted, strict=True):
        correlation.append(float(np.corrcoef(obs, pred)[0, 1]))
    return {
        "rmse_pA": rmse.tolist(),
        "nrmse_by_range": (rmse / span).tolist(),
        "correlation": correlation,
        "aggregate_rmse_pA": float(np.sqrt(np.mean(residual**2))),
    }


def run_nominal_recovery(shk_data: ClampData, egl_data: ClampData) -> dict:
    shk_truth = ShkParameters(
        g_max_nS=30.0,
        e_rev_mV=E_K_MV,
        n_v_offset_mV=14.0,
        n_slope_mV=31.0,
        tau_min_ms=1.0,
        tau_amp_ms=10.0,
        tau_slope_mV=20.0,
    )
    shk_synthetic = ClampData(
        **{
            **asdict(shk_data),
            "currents_pA": simulate_shk(
                shk_data.step_time_ms, shk_data.voltages_mV, shk_truth
            ),
        }
    )
    shk_recovered, _ = fit_shk(shk_synthetic)
    shk_prediction = simulate_shk(
        shk_data.step_time_ms, shk_data.voltages_mV, shk_recovered
    )

    egl_truth = EglParameters(
        g_max_nS=19.8,
        e_rev_mV=E_CA_MV,
        v_half_m_mV=-8.0,
        k_m_mV=8.6,
        tau_m_min_ms=0.4,
        tau_m_amp_ms=0.7,
        v_tau_m_mV=-5.0,
        k_tau_m_mV=15.0,
        h_min=0.28,
        h_max=0.70,
        v_half_h_mV=-11.0,
        k_h_mV=2.0,
        tau_h_ms=30.0,
    )
    egl_synthetic = ClampData(
        **{
            **asdict(egl_data),
            "currents_pA": simulate_egl(
                egl_data.step_time_ms, egl_data.voltages_mV, egl_truth
            ),
        }
    )
    egl_recovered = fit_egl(egl_synthetic)
    egl_prediction = simulate_egl(
        egl_data.step_time_ms, egl_data.voltages_mV, egl_recovered
    )

    return {
        "design": "one deterministic nominal truth per channel, exact observed protocol and objective",
        "SHK-1": {
            "truth": asdict(shk_truth),
            "recovered": asdict(shk_recovered),
            "trace_rmse_pA": float(
                np.sqrt(np.mean((shk_prediction - shk_synthetic.currents_pA) ** 2))
            ),
        },
        "EGL-19": {
            "truth": asdict(egl_truth),
            "recovered": asdict(egl_recovered),
            "trace_rmse_pA": float(
                np.sqrt(np.mean((egl_prediction - egl_synthetic.currents_pA) ** 2))
            ),
        },
        "classification": "mechanics-only recovery; insufficient for unique biological parameter interpretation",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_results(
    shk_data: ClampData,
    shk_params: ShkParameters,
    shk_points: dict[str, np.ndarray],
    egl_data: ClampData,
    egl_params: EglParameters,
    egl_points: dict[str, np.ndarray],
    recovery: dict,
) -> dict:
    RESULTS.mkdir(exist_ok=True)
    shk_prediction = simulate_shk(shk_data.step_time_ms, shk_data.voltages_mV, shk_params)
    egl_prediction = simulate_egl(egl_data.step_time_ms, egl_data.voltages_mV, egl_params)
    report = {
        "protocol": {
            "holding_potential_mV": V_HOLD_MV,
            "step_duration_ms": 100.0,
            "shk_voltages_mV": shk_data.voltages_mV.tolist(),
            "egl_voltages_mV": egl_data.voltages_mV.tolist(),
        },
        "observation_model": {
            "SHK-1": "processed Igor WT-minus-shk-1(lf), baseline corrected, first 0.5 ms excluded",
            "EGL-19": "isolated WT calcium current, baseline corrected, first 1.0 ms excluded",
        },
        "parameters": {
            "SHK-1": asdict(shk_params),
            "EGL-19": asdict(egl_params),
        },
        "metrics": {
            "SHK-1": _metrics(shk_data.currents_pA, shk_prediction),
            "EGL-19": _metrics(egl_data.currents_pA, egl_prediction),
        },
        "nominal_recovery": recovery,
        "data_provenance": {
            K_FILE.name: _sha256(K_FILE),
            CA_FILE.name: _sha256(CA_FILE),
            "SHK-1_waves": list(shk_data.source_waves),
            "EGL-19_waves": list(egl_data.source_waves),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": 20260825,
        },
        "claim_boundary": "Population-average phenomenological fits over the measured voltage ranges; no unique single-cell parameter claim.",
    }
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        RESULTS / "fit_data.npz",
        shk_time_ms=shk_data.step_time_ms,
        shk_voltages_mV=shk_data.voltages_mV,
        shk_observed_pA=shk_data.currents_pA,
        shk_predicted_pA=shk_prediction,
        shk_n_inf_points=shk_points["n_inf"],
        shk_tau_n_points_ms=shk_points["tau_n_ms"],
        egl_time_ms=egl_data.step_time_ms,
        egl_voltages_mV=egl_data.voltages_mV,
        egl_observed_pA=egl_data.currents_pA,
        egl_predicted_pA=egl_prediction,
        egl_m_inf_points=egl_points["m_inf"],
        egl_tau_m_points_ms=egl_points["tau_m_ms"],
    )
    return report


def plot_results(
    shk_data: ClampData,
    shk_params: ShkParameters,
    shk_points: dict[str, np.ndarray],
    egl_data: ClampData,
    egl_params: EglParameters,
    egl_points: dict[str, np.ndarray],
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = plt.get_cmap("viridis")

    shk_pred = simulate_shk(shk_data.step_time_ms, shk_data.voltages_mV, shk_params)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.7), sharex=True, sharey=True)
    for idx, (axis, voltage) in enumerate(zip(axes.flat, shk_data.voltages_mV, strict=True)):
        axis.plot(shk_data.step_time_ms, shk_data.currents_pA[idx], color="#d98279", lw=1.1, label="Experiment")
        axis.plot(shk_data.step_time_ms, shk_pred[idx], color="#1769aa", lw=1.8, label="HH fit")
        axis.set_title(f"{voltage:.0f} mV")
        axis.set_xlim(0, 100)
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Time after voltage step (ms)")
    fig.supylabel("SHK-1 outward current (pA)")
    fig.tight_layout()
    fig.savefig(RESULTS / "shk1_current_fits.png", dpi=220)
    plt.close(fig)

    voltage_grid = np.linspace(-70.0, 110.0, 500)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].plot(voltage_grid, shk_n_inf(voltage_grid, shk_params), color="#1769aa", lw=2, label="Fitted function")
    axes[0].scatter(shk_data.voltages_mV, shk_points["n_inf"], color="#c4473d", s=34, zorder=3, label="Trace estimates")
    axes[0].set(xlabel="Voltage (mV)", ylabel="Steady-state activation, n_inf", ylim=(-0.03, 1.03))
    axes[0].legend(frameon=False)
    axes[1].plot(voltage_grid, shk_tau_n(voltage_grid, shk_params), color="#1769aa", lw=2)
    axes[1].scatter(shk_data.voltages_mV, shk_points["tau_n_ms"], color="#c4473d", s=34, zorder=3)
    axes[1].set(xlabel="Voltage (mV)", ylabel="Activation time constant, tau_n (ms)")
    fig.tight_layout()
    fig.savefig(RESULTS / "shk1_gating.png", dpi=220)
    plt.close(fig)

    egl_pred = simulate_egl(egl_data.step_time_ms, egl_data.voltages_mV, egl_params)
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), sharex=True, sharey=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for idx, voltage in enumerate(egl_data.voltages_mV):
        axis = axes.flat[idx]
        axis.set_visible(True)
        color = colors(idx / max(1, len(egl_data.voltages_mV) - 1))
        axis.plot(egl_data.step_time_ms, egl_data.currents_pA[idx], color="#d98279", lw=0.9, label="Experiment")
        axis.plot(egl_data.step_time_ms, egl_pred[idx], color=color, lw=1.8, label="HH fit")
        axis.set_title(f"{voltage:.0f} mV")
        axis.set_xlim(0, 100)
    axes.flat[0].legend(frameon=False)
    fig.supxlabel("Time after voltage step (ms)")
    fig.supylabel("EGL-19 calcium current (pA)")
    fig.tight_layout()
    fig.savefig(RESULTS / "egl19_current_fits.png", dpi=220)
    plt.close(fig)

    voltage_grid = np.linspace(-60.0, 50.0, 500)
    m_inf, tau_m, _, _ = egl_gate_functions(voltage_grid, egl_params)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(voltage_grid, m_inf, color="#1769aa", lw=2, label="Fitted function")
    axes[0].scatter(egl_data.voltages_mV, egl_points["m_inf"], color="#c4473d", s=34, marker="o", label="Trace estimates")
    axes[0].set(xlabel="Voltage (mV)", ylabel="Steady-state activation, m_inf", ylim=(-0.03, 1.03))
    axes[0].legend(frameon=False)
    axes[1].plot(voltage_grid, tau_m, color="#1769aa", lw=2)
    axes[1].scatter(egl_data.voltages_mV, egl_points["tau_m_ms"], color="#c4473d", s=34, marker="o")
    axes[1].set(xlabel="Voltage (mV)", ylabel="Activation time constant, tau_m (ms)")
    fig.tight_layout()
    fig.savefig(RESULTS / "egl19_gating.png", dpi=220)
    plt.close(fig)


def run(make_plots: bool = True) -> dict:
    started = time.perf_counter()
    brainstate.random.seed(20260825)
    shk_data = load_shk_data()
    egl_data = load_egl_data()
    recovery = run_nominal_recovery(shk_data, egl_data)
    shk_params, shk_points = fit_shk(shk_data)
    egl_params = fit_egl(egl_data)
    egl_points = extract_egl_points(egl_data, egl_params)
    report = save_results(
        shk_data,
        shk_params,
        shk_points,
        egl_data,
        egl_params,
        egl_points,
        recovery,
    )
    if make_plots:
        plot_results(shk_data, shk_params, shk_points, egl_data, egl_params, egl_points)
    report["runtime_seconds"] = time.perf_counter() - started
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="fit and save numerical artifacts only")
    args = parser.parse_args()
    report = run(make_plots=not args.no_plots)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
