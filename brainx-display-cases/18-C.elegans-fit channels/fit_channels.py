"""Fit BrainCell HH models for C. elegans SHK-1 and EGL-19 channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from braincell.channel._base import Gate, HH
from igor2 import packed


ROOT = Path(__file__).resolve().parent
K_FILE = ROOT / "Fig1C D I-V K currents.pxp"
CA_FILE = ROOT / "Fig. 3A I-V Ca currents.pxp"
RESULTS = Path(os.environ.get("BRAINX_CHANNEL_RESULTS", ROOT / "results"))
V_HOLD_MV = -60.0
SHK_VOLTAGES_MV = np.arange(0.0, 101.0, 20.0)
EGL_VOLTAGES_MV = np.arange(-20.0, 41.0, 10.0)
SHK_POWER = 2
EGL_POWER = 4
SHK_PARAMETER_ORDER = (
    "g_max_nS",
    "e_rev_mV",
    "v_half_n_mV",
    "k_n_mV",
    "tau_n_min_ms",
    "tau_n_amp_ms",
    "k_tau_n_mV",
)
EGL_PARAMETER_ORDER = (
    "g_max_nS",
    "e_rev_mV",
    "v_half_m_mV",
    "k_m_mV",
    "tau_m_min_ms",
    "tau_m_amp_ms",
    "v_tau_m_mV",
    "k_tau_m_mV",
)
SHK_LOWER = np.array([1.0, -150.0, -60.0, 2.0, 0.05, 0.0, 5.0])
SHK_UPPER = np.array([100.0, -1.0, 40.0, 50.0, 20.0, 50.0, 100.0])
EGL_LOWER = np.array([1.0, 40.1, -40.0, 2.0, 0.05, 0.0, -40.0, 2.0])
EGL_UPPER = np.array([100.0, 100.0, 20.0, 30.0, 20.0, 50.0, 40.0, 40.0])
HUBER_DELTA = 0.08
MAX_ITERATIONS = 1200
FIT_INTERVAL_MS = 0.5
OPTIMIZER_SEEDS = (20260902, 20260903, 20260904, 20260905, 20260906, 20260907)


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
    baseline_noise_pA: np.ndarray


@dataclass(frozen=True)
class ShkParameters:
    g_max_nS: float
    e_rev_mV: float
    v_half_n_mV: float
    k_n_mV: float
    tau_n_min_ms: float
    tau_n_amp_ms: float
    k_tau_n_mV: float


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


def _wave(root: dict, number: int) -> np.ndarray:
    record = root[f"wave{number}".encode()]
    return np.asarray(record.wave["wave"]["wData"], dtype=float)


def _packed_root(path: Path) -> dict:
    return packed.load(str(path))[1]["root"]


def verify_packed_metadata(path: Path) -> dict:
    """Verify wave identities from the packed experiment's recreation record."""
    records, _ = packed.load(str(path))
    record = next(
        (item for item in records if type(item).__name__ == "RecreationRecord"),
        None,
    )
    if record is None:
        raise ValueError(f"{path.name} has no Igor recreation record")
    text = record.text.decode("latin1") if isinstance(record.text, bytes) else record.text
    if path == K_FILE:
        wave_contract = {
            "time": [86],
            "WT": list(range(87, 99)),
            "shk-1 mutant": list(range(113, 125)),
            "commands": list(range(167, 179)),
        }
        required = {
            "WT traces": (
                "Display /W=(1084.5,112.25,1779.75,746)/L=a1/B=b1 wave87,wave88,wave89,wave90,wave91 vs wave86",
                "AppendToGraph/L=a1/B=b1 wave92,wave93,wave94,wave95,wave96,wave97,wave98 vs wave86",
            ),
            "shk-1 mutant traces": (
                "AppendToGraph/L=a3/B=b3 wave113,wave114,wave115,wave116,wave117,wave118,wave119,wave120 vs wave86",
                "AppendToGraph/L=a3/B=b3 wave121,wave122,wave123,wave124 vs wave86",
                'DrawText 0.700753917858146,0.0204081632653061,"shk-1(lf)"',
            ),
            "command traces": (
                "AppendToGraph/L=a7/B=b7 wave167,wave168,wave169,wave170,wave171,wave172,wave173,wave174 vs wave86",
                "AppendToGraph/L=a7/B=b7 wave175,wave176,wave177,wave178 vs wave86",
            ),
        }
    elif path == CA_FILE:
        wave_contract = {
            "time": [0],
            "WT": list(range(1, 12)),
            "n582": list(range(12, 23)),
            "ad1006": list(range(23, 34)),
            "commands": list(range(34, 45)),
        }
        required = {
            "WT traces": (
                "Display /W=(408,101.75,1279.5,752)/L=a1/B=b1 wave1,wave2,wave3,wave4,wave5 vs wave0",
                "AppendToGraph/L=a1/B=b1 wave6 vs wave0",
                "AppendToGraph/L=a1/B=b1 wave7,wave8,wave9,wave10,wave11 vs wave0",
                'DrawText 0.0860918628432913,0.0784922052186231,"wild type"',
            ),
            "n582 traces": (
                "AppendToGraph/L=a2/B=b2 wave12,wave13,wave14,wave15,wave16,wave17,wave18,wave19 vs wave0",
                "AppendToGraph/L=a2/B=b2 wave20,wave21,wave22 vs wave0",
                'DrawText 0.436022706550068,0.0784922052186231,"n582"',
            ),
            "ad1006 traces": (
                "AppendToGraph/L=a3/B=b3 wave23,wave24,wave25,wave26,wave27,wave28,wave29,wave30 vs wave0",
                "AppendToGraph/L=a3/B=b3 wave31,wave32,wave33 vs wave0",
                'DrawText 0.762440410561134,0.0784922052186231,"ad1006"',
            ),
            "command traces": (
                "AppendToGraph/L=a4/B=b4 wave34,wave35,wave36,wave37,wave38,wave39,wave40,wave41 vs wave0",
                "AppendToGraph/L=a4/B=b4 wave42,wave43,wave44 vs wave0",
            ),
        }
    else:
        raise ValueError(f"no metadata contract is defined for {path.name}")
    missing = [label for label, snippets in required.items() if not all(s in text for s in snippets)]
    if missing:
        raise ValueError(f"packed metadata verification failed for {path.name}: {missing}")
    return {
        "file": path.name,
        "recreation_record_sha256": hashlib.sha256(text.encode("latin1")).hexdigest(),
        "verified_groups": list(required),
        "verified_wave_contract": wave_contract,
    }


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


def _fit_samples(data: ClampData) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic grid for fitting while preserving full traces for tests."""
    spacing = float(np.median(np.diff(data.step_time_ms)))
    stride = max(1, int(round(FIT_INTERVAL_MS / spacing)))
    indices = np.arange(0, data.step_time_ms.size, stride)
    return data.step_time_ms[indices], data.currents_pA[:, indices]


def load_shk_data() -> ClampData:
    verify_packed_metadata(K_FILE)
    root = _packed_root(K_FILE)
    time_ms = _wave(root, 86)
    commands = np.stack([_wave(root, i) for i in range(167, 179)])
    wt_currents = np.stack([_wave(root, i) for i in range(87, 99)])
    mutant_currents = np.stack([_wave(root, i) for i in range(113, 125)])
    start_ms, end_ms = _step_bounds(time_ms, commands[-1])
    if not np.isclose(end_ms - start_ms, 100.0, atol=0.11):
        raise ValueError("SHK-1 command step is not 100 ms")
    selected = np.arange(6, 12)
    difference = (wt_currents - mutant_currents)[selected]
    baseline_mask = (time_ms >= 2.0) & (time_ms < 10.0)
    baseline_noise = np.std(
        difference[:, baseline_mask]
        - np.mean(difference[:, baseline_mask], axis=1, keepdims=True),
        axis=1,
    )
    step_time, currents = _baseline_correct(
        time_ms,
        difference,
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
        source_waves=tuple(
            f"wave{wt}-wave{mutant}"
            for wt, mutant in zip(range(93, 99), range(119, 125), strict=True)
        ),
        step_start_ms=start_ms,
        step_end_ms=end_ms,
        baseline_noise_pA=baseline_noise,
    )


def load_egl_data() -> ClampData:
    verify_packed_metadata(CA_FILE)
    root = _packed_root(CA_FILE)
    time_ms = _wave(root, 0)
    commands = np.stack([_wave(root, i) for i in range(34, 45)])
    wt_currents = np.stack([_wave(root, i) for i in range(1, 12)])
    start_ms, end_ms = _step_bounds(time_ms, commands[-1])
    if not np.isclose(end_ms - start_ms, 100.0, atol=0.03):
        raise ValueError("EGL-19 command step is not 100 ms")
    selected = np.arange(4, 11)
    selected_currents = wt_currents[selected]
    baseline_mask = (time_ms >= 2.0) & (time_ms < 10.0)
    baseline_noise = np.std(
        selected_currents[:, baseline_mask]
        - np.mean(selected_currents[:, baseline_mask], axis=1, keepdims=True),
        axis=1,
    )
    step_time, currents = _baseline_correct(
        time_ms,
        selected_currents,
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
        source_waves=tuple(f"wave{i}" for i in range(5, 12)),
        step_start_ms=start_ms,
        step_end_ms=end_ms,
        baseline_noise_pA=baseline_noise,
    )


def load_egl_mutant_controls() -> dict[str, np.ndarray]:
    root = _packed_root(CA_FILE)
    time_ms = _wave(root, 0)
    commands = np.stack([_wave(root, i) for i in range(34, 45)])
    start_ms, end_ms = _step_bounds(time_ms, commands[-1])
    selected = np.arange(4, 11)
    wt = np.stack([_wave(root, i) for i in range(1, 12)])
    controls = {}
    for name, wave_ids in {
        "n582": range(12, 23),
        "ad1006": range(23, 34),
    }.items():
        mutant = np.stack([_wave(root, i) for i in wave_ids])
        _, difference = _baseline_correct(
            time_ms,
            (wt - mutant)[selected],
            start_ms,
            end_ms,
            transient_ms=1.0,
        )
        controls[name] = difference
    return controls


def shk_n_inf(voltage_mV: np.ndarray, p: ShkParameters) -> np.ndarray:
    return _sigmoid((np.asarray(voltage_mV) - p.v_half_n_mV) / p.k_n_mV)


def shk_tau_n(voltage_mV: np.ndarray, p: ShkParameters) -> np.ndarray:
    exponent = np.clip(-np.asarray(voltage_mV) / p.k_tau_n_mV, -60.0, 60.0)
    return p.tau_n_min_ms + p.tau_n_amp_ms * np.exp(exponent)


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
    hold = p.g_max_nS * n0**SHK_POWER * (V_HOLD_MV - p.e_rev_mV)
    return p.g_max_nS * n**SHK_POWER * (voltages - p.e_rev_mV) - hold


def _fit_activation_family(
    time_ms: np.ndarray,
    currents_pA: np.ndarray,
    power: int,
    amplitude_bounds: tuple[float, float],
    max_iterations: int = 300,
) -> tuple[np.ndarray, np.ndarray, dict]:
    count = currents_pA.shape[0]
    amp_lo, amp_hi = amplitude_bounds
    if amp_lo >= 0.0:
        amplitude_lower = np.zeros(count)
        amplitude_upper = np.maximum(1.0, 1.5 * np.max(currents_pA, axis=1))
    elif amp_hi <= 0.0:
        amplitude_lower = np.minimum(-1.0, 1.5 * np.min(currents_pA, axis=1))
        amplitude_upper = np.zeros(count)
    else:
        amplitude_lower = np.full(count, amp_lo)
        amplitude_upper = np.full(count, amp_hi)
    lower = np.concatenate([amplitude_lower, np.full(count, 0.05)])
    upper = np.concatenate([amplitude_upper, np.full(count, 100.0)])
    time = jnp.asarray(time_ms)[None, :]
    currents = jnp.asarray(currents_pA)
    scale = jnp.asarray(
        np.maximum(25.0, np.percentile(np.abs(currents_pA), 95, axis=1))
    )

    def loss_from_parameters(theta):
        amplitude = theta[:count, None]
        tau = theta[count:, None]
        prediction = amplitude * (1.0 - jnp.exp(-time / tau)) ** power
        return _braincell_huber_loss(prediction, currents, scale)

    fits = [
        _fit_braintools(
            loss_from_parameters,
            lower,
            upper,
            seed + 100 * power,
            max_iterations=max_iterations,
        )
        for seed in OPTIMIZER_SEEDS
    ]
    finite = [fit for fit in fits if np.isfinite(fit[1]["final_loss"])]
    converged = [fit for fit in finite if fit[1]["success"]]
    values, diagnostics = min(converged or finite, key=lambda fit: fit[1]["final_loss"])
    diagnostics = {**diagnostics, "candidates": [item for _, item in fits]}
    diagnostics["amplitude_pA"] = values[:count].tolist()
    diagnostics["tau_activation_ms"] = values[count:].tolist()
    return values[:count], values[count:], diagnostics


def _fit_braintools(
    loss_from_parameters,
    lower: np.ndarray,
    upper: np.ndarray,
    seed: int,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[np.ndarray, dict]:
    @brainstate.transform.jit
    def objective(*parameters):
        loss = loss_from_parameters(jnp.asarray(parameters))
        return u.math.where(u.math.isfinite(loss), loss, 1.0e12)

    np.random.seed(seed)
    rng_state = np.random.get_state()
    sampled_start = np.asarray(
        [np.random.uniform(float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True)]
    )
    np.random.set_state(rng_state)
    initial_loss = float(objective(*sampled_start))
    objective_history = [initial_loss]

    def record_iteration(parameters):
        objective_history.append(float(objective(*parameters)))

    optimizer = braintools.optim.ScipyOptimizer(
        objective,
        bounds=list(zip(lower, upper, strict=True)),
        method="L-BFGS-B",
        callback=record_iteration,
        options={"maxiter": max_iterations, "ftol": 1.0e-10, "gtol": 1.0e-6},
    )
    result = optimizer.minimize(n_iter=1)
    values = np.asarray([float(value) for value in result.x])
    return values, {
        "seed": seed,
        "initial_parameters": sampled_start.tolist(),
        "initial_loss": initial_loss,
        "final_loss": float(result.fun),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "final_parameters": values.tolist(),
        "objective_history": objective_history,
        "active_bounds": [
            name
            for name, value, lo, hi in zip(range(len(values)), values, lower, upper, strict=True)
            if np.isclose(value, lo, rtol=0.0, atol=1.0e-4 * max(1.0, abs(lo)))
            or np.isclose(value, hi, rtol=0.0, atol=1.0e-4 * max(1.0, abs(hi)))
        ],
    }


def _braincell_huber_loss(prediction, observed, trace_scale):
    return braintools.metric.huber_loss(
        prediction / trace_scale[:, None],
        observed / trace_scale[:, None],
        delta=HUBER_DELTA,
        reduction="mean",
    )


def _per_voltage_huber(prediction, observed, trace_scale):
    return np.asarray(
        [
            braintools.metric.huber_loss(
                jnp.asarray(pred) / scale,
                jnp.asarray(target) / scale,
                delta=HUBER_DELTA,
                reduction="mean",
            )
            for pred, target, scale in zip(
                prediction, observed, trace_scale, strict=True
            )
        ],
        dtype=float,
    )


def simulate_shk_braincell(step_time_ms, voltages_mV, p: ShkParameters):
    time = jnp.asarray(step_time_ms)[None, :]
    voltage_values = jnp.broadcast_to(
        jnp.asarray(voltages_mV)[:, None],
        (len(voltages_mV), len(step_time_ms)),
    )
    voltage = voltage_values * u.mV
    hold_voltage = jnp.full_like(voltage_values, V_HOLD_MV) * u.mV
    potassium = braincell.IonInfo(
        Ci=jnp.ones_like(voltage_values) * u.mM,
        Co=jnp.ones_like(voltage_values) * u.mM,
        E=jnp.full_like(voltage_values, p.e_rev_mV) * u.mV,
        valence=1,
    )
    channel = SHK1Channel(voltage_values.shape, p)
    channel.init_state(hold_voltage, potassium)
    channel.reset_state(hold_voltage, potassium)
    hold_current = -channel.current(hold_voltage, potassium)
    n0 = channel.n.value
    n_inf = channel.f_n_inf(voltage, potassium)
    tau_n = channel.f_n_tau(voltage, potassium)
    channel.n.value = n_inf + (n0 - n_inf) * jnp.exp(-time / tau_n)
    return (-channel.current(voltage, potassium) - hold_current).to_decimal(u.pA)


def _decode_shk(theta: np.ndarray) -> ShkParameters:
    return ShkParameters(
        **dict(zip(SHK_PARAMETER_ORDER, theta, strict=True))
    )


def fit_shk(
    data: ClampData,
    compute_structure: bool = True,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[ShkParameters, dict[str, np.ndarray], dict]:
    fit_time, fit_currents = _fit_samples(data)
    structural_scores = {}
    structural_diagnostics = {}
    if compute_structure:
        selected_local = None
        for power in range(1, 6):
            amplitudes, tau_points, local_diagnostics = _fit_activation_family(
                fit_time,
                fit_currents,
                power,
                (0.0, 10000.0),
                max_iterations=min(400, max_iterations),
            )
            structural_scores[str(power)] = local_diagnostics["final_loss"]
            structural_diagnostics[str(power)] = local_diagnostics
            if power == SHK_POWER:
                selected_local = (amplitudes, tau_points)
        assert selected_local is not None
        amplitudes, tau_points = selected_local
    else:
        amplitudes = np.full(len(data.voltages_mV), np.nan)
        tau_points = np.full(len(data.voltages_mV), np.nan)
    observed = jnp.asarray(fit_currents)
    trace_scale = jnp.asarray(
        np.maximum(50.0, np.percentile(np.abs(fit_currents), 95, axis=1))
    )
    def loss_from_parameters(theta):
        prediction = simulate_shk_braincell(
            fit_time, data.voltages_mV, _decode_shk(theta)
        )
        return _braincell_huber_loss(prediction, observed, trace_scale)

    fits = []
    for seed in OPTIMIZER_SEEDS:
        values, diagnostics = _fit_braintools(
            loss_from_parameters,
            SHK_LOWER,
            SHK_UPPER,
            seed,
            max_iterations=max_iterations,
        )
        diagnostics["active_bounds"] = [
            SHK_PARAMETER_ORDER[index] for index in diagnostics["active_bounds"]
        ]
        diagnostics["parameters"] = asdict(_decode_shk(values))
        diagnostics["per_voltage_final_loss"] = _per_voltage_huber(
            np.asarray(simulate_shk_braincell(fit_time, data.voltages_mV, _decode_shk(values))),
            fit_currents,
            np.asarray(trace_scale),
        ).tolist()
        fits.append((values, diagnostics))
    finite = [fit for fit in fits if np.isfinite(fit[1]["final_loss"])]
    if not finite:
        raise RuntimeError("all SHK-1 fitting starts failed")
    converged = [fit for fit in finite if fit[1]["success"]]
    best = min(converged or finite, key=lambda fit: fit[1]["final_loss"])
    params = _decode_shk(best[0])
    denominator = params.g_max_nS * (data.voltages_mV - params.e_rev_mV)
    n_unclipped = np.maximum(amplitudes / denominator, 0.0) ** (1.0 / SHK_POWER)
    n_points = np.clip(n_unclipped, 0.0, 1.0)
    return params, {
        "amplitude_pA": amplitudes,
        "n_inf": n_points,
        "n_inf_unclipped": n_unclipped,
        "tau_n_ms": tau_points,
    }, {
        "activation_power_scores": structural_scores,
        "selected_activation_power": SHK_POWER,
        "selected_seed": best[1]["seed"],
        "physical_bounds": dict(
            zip(
                SHK_PARAMETER_ORDER,
                zip(SHK_LOWER.tolist(), SHK_UPPER.tolist(), strict=True),
                strict=True,
            )
        ),
        "structural_fit_diagnostics": structural_diagnostics,
        "starts": [
            diagnostics for _, diagnostics in fits
        ],
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _decode_egl(theta: np.ndarray) -> EglParameters:
    return EglParameters(
        **dict(zip(EGL_PARAMETER_ORDER, theta, strict=True))
    )


def egl_gate_functions(
    voltage_mV: np.ndarray, p: EglParameters
) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.asarray(voltage_mV, dtype=float)
    m_inf = _sigmoid((voltage - p.v_half_m_mV) / p.k_m_mV)
    tau_m = p.tau_m_min_ms + p.tau_m_amp_ms / (
        np.exp(np.clip(-(voltage - p.v_tau_m_mV) / p.k_tau_m_mV, -60.0, 60.0))
        + np.exp(np.clip((voltage - p.v_tau_m_mV) / p.k_tau_m_mV, -60.0, 60.0))
    )
    return m_inf, tau_m


def simulate_egl(
    step_time_ms: np.ndarray,
    voltages_mV: np.ndarray,
    p: EglParameters,
) -> np.ndarray:
    voltages = np.asarray(voltages_mV, dtype=float)[:, None]
    times = np.asarray(step_time_ms, dtype=float)[None, :]
    m_inf, tau_m = egl_gate_functions(voltages, p)
    m0, _ = egl_gate_functions(np.asarray([V_HOLD_MV]), p)
    m = m_inf + (m0[0] - m_inf) * np.exp(-times / tau_m)
    hold = p.g_max_nS * m0[0] ** EGL_POWER * (V_HOLD_MV - p.e_rev_mV)
    return p.g_max_nS * m**EGL_POWER * (voltages - p.e_rev_mV) - hold


def simulate_egl_braincell(step_time_ms, voltages_mV, p: EglParameters):
    time = jnp.asarray(step_time_ms)[None, :]
    voltage_values = jnp.broadcast_to(
        jnp.asarray(voltages_mV)[:, None],
        (len(voltages_mV), len(step_time_ms)),
    )
    voltage = voltage_values * u.mV
    hold_voltage = jnp.full_like(voltage_values, V_HOLD_MV) * u.mV
    calcium = braincell.IonInfo(
        Ci=jnp.ones_like(voltage_values) * u.mM,
        Co=jnp.ones_like(voltage_values) * u.mM,
        E=jnp.full_like(voltage_values, p.e_rev_mV) * u.mV,
        valence=2,
    )
    channel = EGL19Channel(voltage_values.shape, p)
    channel.init_state(hold_voltage, calcium)
    channel.reset_state(hold_voltage, calcium)
    hold_current = -channel.current(hold_voltage, calcium)
    m0 = channel.m.value
    m_inf = channel.f_m_inf(voltage, calcium)
    tau_m = channel.f_m_tau(voltage, calcium)
    channel.m.value = m_inf + (m0 - m_inf) * jnp.exp(-time / tau_m)
    return (-channel.current(voltage, calcium) - hold_current).to_decimal(u.pA)


def _fit_inactivation_family(
    time_ms: np.ndarray,
    currents_pA: np.ndarray,
    power: int,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[np.ndarray | None, dict]:
    count = currents_pA.shape[0]
    amplitude_lower = np.minimum(-1.0, 1.5 * np.min(currents_pA, axis=1))
    lower = np.concatenate(
        [amplitude_lower, np.full(count, 0.05), np.zeros(count), np.full(count, 2.0)]
    )
    upper = np.concatenate(
        [np.zeros(count), np.full(count, 100.0), np.ones(count), np.full(count, 300.0)]
    )
    time = jnp.asarray(time_ms)[None, :]
    current = jnp.asarray(currents_pA)
    scale = jnp.asarray(
        np.maximum(25.0, np.percentile(np.abs(currents_pA), 95, axis=1))
    )

    def loss_from_parameters(theta):
        amplitude = theta[:count, None]
        tau_m = theta[count : 2 * count, None]
        h_ss = theta[2 * count : 3 * count, None]
        tau_h = theta[3 * count :, None]
        prediction = amplitude * (1.0 - jnp.exp(-time / tau_m)) ** power
        prediction *= h_ss + (1.0 - h_ss) * jnp.exp(-time / tau_h)
        return _braincell_huber_loss(prediction, current, scale)

    fits = [
        _fit_braintools(
            loss_from_parameters,
            lower,
            upper,
            seed + 900,
            max_iterations=max_iterations,
        )
        for seed in OPTIMIZER_SEEDS
    ]
    finite = [fit for fit in fits if np.isfinite(fit[1]["final_loss"])]
    converged = [fit for fit in finite if fit[1]["success"]]
    if not converged:
        return None, {
            "comparison_status": "inconclusive: no candidate terminated successfully",
            "candidates": [item for _, item in fits],
        }
    values, diagnostics = min(converged, key=lambda fit: fit[1]["final_loss"])
    diagnostics = {**diagnostics, "candidates": [item for _, item in fits]}
    diagnostics["amplitude_pA"] = values[:count].tolist()
    diagnostics["tau_m_ms"] = values[count : 2 * count].tolist()
    diagnostics["h_steady"] = values[2 * count : 3 * count].tolist()
    diagnostics["tau_h_ms"] = values[3 * count :].tolist()
    return values, diagnostics


def fit_egl(
    data: ClampData,
    compute_structure: bool = True,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[EglParameters, dict]:
    fit_time, fit_currents = _fit_samples(data)
    observed = jnp.asarray(fit_currents)
    trace_scale = jnp.asarray(
        np.maximum(25.0, np.percentile(np.abs(fit_currents), 95, axis=1))
    )
    def loss_from_parameters(theta):
        prediction = simulate_egl_braincell(
            fit_time, data.voltages_mV, _decode_egl(theta)
        )
        return _braincell_huber_loss(prediction, observed, trace_scale)

    fits = []
    for seed in OPTIMIZER_SEEDS:
        values, diagnostics = _fit_braintools(
            loss_from_parameters,
            EGL_LOWER,
            EGL_UPPER,
            seed,
            max_iterations=max_iterations,
        )
        diagnostics["active_bounds"] = [
            EGL_PARAMETER_ORDER[index] for index in diagnostics["active_bounds"]
        ]
        diagnostics["parameters"] = asdict(_decode_egl(values))
        diagnostics["per_voltage_final_loss"] = _per_voltage_huber(
            np.asarray(simulate_egl_braincell(fit_time, data.voltages_mV, _decode_egl(values))),
            fit_currents,
            np.asarray(trace_scale),
        ).tolist()
        fits.append((values, diagnostics))
    finite = [fit for fit in fits if np.isfinite(fit[1]["final_loss"])]
    if not finite:
        raise RuntimeError("all EGL-19 fitting starts failed")
    converged = [fit for fit in finite if fit[1]["success"]]
    best = min(converged or finite, key=lambda fit: fit[1]["final_loss"])
    structural_scores = {}
    structural_diagnostics = {}
    inactivation_scores = {}
    inactivation_diagnostics = {}
    if compute_structure:
        selected_activation_values = None
        for power in range(1, 5):
            local_amplitude, local_tau, local_diagnostics = _fit_activation_family(
                fit_time,
                fit_currents,
                power,
                (-5000.0, 0.0),
                max_iterations=min(400, max_iterations),
            )
            structural_scores[str(power)] = local_diagnostics["final_loss"]
            structural_diagnostics[str(power)] = local_diagnostics
            if power == EGL_POWER:
                selected_activation_values = (local_amplitude, local_tau)
        inactivation_values, local_diagnostics = _fit_inactivation_family(
            fit_time,
            fit_currents,
            EGL_POWER,
            max_iterations=max_iterations,
        )
        inactivation_diagnostics[str(EGL_POWER)] = local_diagnostics
        if inactivation_values is None:
            improvement = None
            comparison_status = local_diagnostics["comparison_status"]
            architecture_evidence = None
        else:
            inactivation_scores[str(EGL_POWER)] = local_diagnostics["final_loss"]
            activation_loss = structural_scores[str(EGL_POWER)]
            inactivation_loss = inactivation_scores[str(EGL_POWER)]
            improvement = max(
                0.0, (activation_loss - inactivation_loss) / activation_loss
            )
            comparison_status = "closed: at least one m4h candidate terminated successfully"
            assert selected_activation_values is not None
            count = len(data.voltages_mV)
            local_amplitude, local_tau = selected_activation_values
            activation_prediction = local_amplitude[:, None] * (
                1.0 - np.exp(-fit_time[None, :] / local_tau[:, None])
            ) ** EGL_POWER
            h_amplitude = inactivation_values[:count]
            h_tau_m = inactivation_values[count : 2 * count]
            h_steady = inactivation_values[2 * count : 3 * count]
            h_tau = inactivation_values[3 * count :]
            inactivation_prediction = h_amplitude[:, None] * (
                1.0 - np.exp(-fit_time[None, :] / h_tau_m[:, None])
            ) ** EGL_POWER
            inactivation_prediction *= h_steady[:, None] + (
                1.0 - h_steady[:, None]
            ) * np.exp(-fit_time[None, :] / h_tau[:, None])
            sample_count = fit_currents.size
            activation_rss = float(np.sum((activation_prediction - fit_currents) ** 2))
            inactivation_rss = float(
                np.sum((inactivation_prediction - fit_currents) ** 2)
            )
            activation_bic = sample_count * np.log(activation_rss / sample_count) + (
                2 * count
            ) * np.log(sample_count)
            inactivation_bic = sample_count * np.log(
                inactivation_rss / sample_count
            ) + (4 * count) * np.log(sample_count)
            architecture_evidence = {
                "sample_count": sample_count,
                "activation_parameter_count": 2 * count,
                "activation_plus_inactivation_parameter_count": 4 * count,
                "activation_rss_pA2": activation_rss,
                "activation_plus_inactivation_rss_pA2": inactivation_rss,
                "activation_bic": float(activation_bic),
                "activation_plus_inactivation_bic": float(inactivation_bic),
                "delta_bic_m4h_minus_m4": float(inactivation_bic - activation_bic),
            }
    else:
        improvement = None
        comparison_status = "not run"
        architecture_evidence = None
    if architecture_evidence is None:
        selection_reason = "No closed penalized architecture comparison is available."
    elif architecture_evidence["delta_bic_m4h_minus_m4"] > 0.0:
        selection_reason = "BIC favors m4 after penalizing the two added local parameters per trace."
    else:
        selection_reason = "BIC favors m4h; an activation-only selection is not supported."
    return _decode_egl(best[0]), {
        "activation_power_scores": structural_scores,
        "activation_plus_inactivation_scores": inactivation_scores,
        "selected_activation_power": EGL_POWER,
        "selected_seed": best[1]["seed"],
        "physical_bounds": dict(
            zip(
                EGL_PARAMETER_ORDER,
                zip(EGL_LOWER.tolist(), EGL_UPPER.tolist(), strict=True),
                strict=True,
            )
        ),
        "implemented_gate_structure": "activation-only m4",
        "architecture_comparison_status": comparison_status,
        "architecture_evidence": architecture_evidence,
        "m4h_relative_objective_improvement": improvement,
        "selection_reason": selection_reason,
        "structure_boundary": "The minimal m4 model is implemented; infer whether inactivation is supported only when the m4h comparison status is closed.",
        "structural_fit_diagnostics": structural_diagnostics,
        "inactivation_fit_diagnostics": inactivation_diagnostics,
        "starts": [
            diagnostics for _, diagnostics in fits
        ],
    }


def extract_egl_points(data: ClampData, p: EglParameters) -> dict[str, np.ndarray]:
    fit_time, fit_currents = _fit_samples(data)
    global_values = np.stack(egl_gate_functions(data.voltages_mV, p), axis=1)
    m0, _ = egl_gate_functions(np.asarray([V_HOLD_MV]), p)
    hold = p.g_max_nS * m0[0] ** EGL_POWER * (V_HOLD_MV - p.e_rev_mV)
    trace_scale = np.maximum(25.0, np.percentile(np.abs(fit_currents), 95, axis=1))
    count = len(data.voltages_mV)
    lower = np.concatenate([np.full(count, 0.005), np.full(count, 0.05)])
    upper = np.concatenate([np.full(count, 0.995), np.full(count, 60.0)])
    time = jnp.asarray(fit_time)[None, :]
    voltage = jnp.asarray(data.voltages_mV)[:, None]
    observed = jnp.asarray(fit_currents)
    scale = jnp.asarray(trace_scale)

    def loss_from_parameters(theta):
        m_inf = theta[:count, None]
        tau_m = theta[count:, None]
        m = m_inf + (m0[0] - m_inf) * jnp.exp(-time / tau_m)
        prediction = p.g_max_nS * m**EGL_POWER * (voltage - p.e_rev_mV) - hold
        return _braincell_huber_loss(prediction, observed, scale)

    fits = [
        _fit_braintools(
            loss_from_parameters,
            lower,
            upper,
            seed + 1900,
            max_iterations=800,
        )
        for seed in OPTIMIZER_SEEDS
    ]
    finite = [fit for fit in fits if np.isfinite(fit[1]["final_loss"])]
    converged = [fit for fit in finite if fit[1]["success"]]
    if not converged:
        raise RuntimeError("all EGL-19 gate-point fits failed to terminate successfully")
    points_array, diagnostics = min(converged, key=lambda fit: fit[1]["final_loss"])
    diagnostics = {**diagnostics, "candidates": [item for _, item in fits]}
    return {
        "m_inf": points_array[:count],
        "tau_m_ms": points_array[count:],
        "fit_diagnostics": diagnostics,
    }


class SHK1Channel(HH):
    """BrainCell SHK-1 channel with data-selected n^2 kinetics."""

    root_type = braincell.ion.Potassium
    gates = (Gate("n", power=SHK_POWER),)

    def __init__(self, size, params: ShkParameters):
        super().__init__(size=size)
        self.params = params
        self.g_max = params.g_max_nS * u.nS

    def f_n_inf(self, voltage, potassium):
        del potassium
        value = voltage.to_decimal(u.mV)
        p = self.params
        return 1.0 / (1.0 + u.math.exp(-(value - p.v_half_n_mV) / p.k_n_mV))

    def f_n_tau(self, voltage, potassium):
        del potassium
        value = voltage.to_decimal(u.mV)
        p = self.params
        return p.tau_n_min_ms + p.tau_n_amp_ms * u.math.exp(
            -value / p.k_tau_n_mV
        )

    def current(self, voltage, potassium):
        return self.g_max * self.conductance_factor(voltage, potassium) * (
            potassium.E - voltage
        )


class EGL19Channel(HH):
    """BrainCell EGL-19 channel with data-selected m^4 kinetics."""

    root_type = braincell.ion.Calcium
    gates = (Gate("m", power=EGL_POWER),)

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

    def current(self, voltage, calcium):
        return self.g_max * self.conductance_factor(voltage, calcium) * (
            calcium.E - voltage
        )


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    residual = predicted - observed
    rmse = np.asarray(
        [
            np.sqrt(
                braintools.metric.squared_error(
                    jnp.asarray(pred), jnp.asarray(obs), reduction="mean"
                )
            )
            for obs, pred in zip(observed, predicted, strict=True)
        ]
    )
    span = np.maximum(np.ptp(observed, axis=1), 1.0)
    correlation = []
    for obs, pred in zip(observed, predicted, strict=True):
        if np.std(obs) == 0.0 or np.std(pred) == 0.0:
            correlation.append(None)
        else:
            correlation.append(float(np.corrcoef(obs, pred)[0, 1]))
    return {
        "rmse_pA": rmse.tolist(),
        "nrmse_by_range": (rmse / span).tolist(),
        "correlation": correlation,
        "aggregate_rmse_pA": float(
            np.sqrt(
                braintools.metric.squared_error(
                    jnp.asarray(predicted), jnp.asarray(observed), reduction="mean"
                )
            )
        ),
    }


def _subset_data(data: ClampData, indices: np.ndarray) -> ClampData:
    return replace(
        data,
        voltages_mV=data.voltages_mV[indices],
        currents_pA=data.currents_pA[indices],
        source_waves=tuple(data.source_waves[index] for index in indices),
        baseline_noise_pA=data.baseline_noise_pA[indices],
    )


def run_leave_one_voltage_out(
    shk_data: ClampData, egl_data: ClampData
) -> tuple[dict, dict[str, np.ndarray]]:
    results = {}
    raw_arrays = {}
    for channel, data, fit, simulate in (
        ("SHK-1", shk_data, fit_shk, simulate_shk_braincell),
        ("EGL-19", egl_data, fit_egl, simulate_egl_braincell),
    ):
        channel_results = []
        for held_index, voltage in enumerate(data.voltages_mV):
            train_indices = np.delete(np.arange(len(data.voltages_mV)), held_index)
            train = _subset_data(data, train_indices)
            if channel == "SHK-1":
                parameters, _, diagnostics = fit(
                    train,
                    compute_structure=False,
                    max_iterations=MAX_ITERATIONS,
                )
            else:
                parameters, diagnostics = fit(
                    train,
                    compute_structure=False,
                    max_iterations=MAX_ITERATIONS,
                )
            prediction = np.asarray(
                simulate(data.step_time_ms, np.asarray([voltage]), parameters)
            )
            observation = data.currents_pA[held_index : held_index + 1]
            metrics = _metrics(observation, prediction)
            key = f"lovo_{channel.lower().replace('-', '').replace(' ', '')}_{held_index}"
            raw_arrays[f"{key}_observed_pA"] = observation
            raw_arrays[f"{key}_predicted_pA"] = prediction
            raw_arrays[f"{key}_residual_pA"] = prediction - observation
            channel_results.append(
                {
                    "held_voltage_mV": float(voltage),
                    "metrics": metrics,
                    "parameters": asdict(parameters),
                    "selected_training_objective": next(
                        start["final_loss"]
                        for start in diagnostics["starts"]
                        if start["seed"] == diagnostics["selected_seed"]
                    ),
                    "fit_diagnostics": diagnostics,
                }
            )
        results[channel] = channel_results
    return {
        "design": "Leave one command voltage out; retain the locked gate structure, BrainCell objective, physical bounds, optimizer seeds, restart count, budget, and candidate-selection rule.",
        "results": results,
    }, raw_arrays


def _domain_truths(lower: np.ndarray, upper: np.ndarray, count: int = 5) -> np.ndarray:
    levels = np.linspace(0.1, 0.9, count)
    fractions = np.stack(
        [np.roll(levels, 2 * parameter_index) for parameter_index in range(len(lower))],
        axis=1,
    )
    return lower + fractions * (upper - lower)


def run_recovery_domain(
    shk_data: ClampData, egl_data: ClampData
) -> tuple[dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260902)
    output = {}
    raw_arrays = {}
    configurations = (
        (
            "SHK-1",
            shk_data,
            ShkParameters,
            SHK_LOWER,
            SHK_UPPER,
            fit_shk,
            simulate_shk_braincell,
            SHK_PARAMETER_ORDER,
        ),
        (
            "EGL-19",
            egl_data,
            EglParameters,
            EGL_LOWER,
            EGL_UPPER,
            fit_egl,
            simulate_egl_braincell,
            EGL_PARAMETER_ORDER,
        ),
    )
    for channel, data, parameter_type, lower, upper, fit, simulate, names in configurations:
        cases = []
        for truth_values in _domain_truths(lower, upper):
            truth = parameter_type(**dict(zip(names, truth_values, strict=True)))
            clean = np.asarray(simulate(data.step_time_ms, data.voltages_mV, truth))
            noise = rng.normal(0.0, data.baseline_noise_pA[:, None], clean.shape)
            synthetic = replace(data, currents_pA=clean + noise)
            if channel == "SHK-1":
                recovered, _, diagnostics = fit(
                    synthetic,
                    compute_structure=False,
                    max_iterations=MAX_ITERATIONS,
                )
            else:
                recovered, diagnostics = fit(
                    synthetic,
                    compute_structure=False,
                    max_iterations=MAX_ITERATIONS,
                )
            recovered_values = np.asarray(list(asdict(recovered).values()))
            prediction = np.asarray(
                simulate(data.step_time_ms, data.voltages_mV, recovered)
            )
            case_index = len(cases)
            key = f"recovery_{channel.lower().replace('-', '').replace(' ', '')}_{case_index}"
            raw_arrays[f"{key}_truth_pA"] = clean
            raw_arrays[f"{key}_observed_pA"] = synthetic.currents_pA
            raw_arrays[f"{key}_predicted_pA"] = prediction
            raw_arrays[f"{key}_residual_pA"] = prediction - synthetic.currents_pA
            cases.append(
                {
                    "truth": asdict(truth),
                    "recovered": asdict(recovered),
                    "absolute_error_as_fraction_of_bound_range": dict(
                        zip(
                            names,
                            (np.abs(recovered_values - truth_values) / (upper - lower)).tolist(),
                            strict=True,
                        )
                    ),
                    "trace_metrics_against_noisy_data": _metrics(
                        synthetic.currents_pA, prediction
                    ),
                    "selected_active_bounds": next(
                        start["active_bounds"]
                        for start in diagnostics["starts"]
                        if start["seed"] == diagnostics["selected_seed"]
                    ),
                    "fit_diagnostics": diagnostics,
                }
            )
        parameter_errors = {
            name: [case["absolute_error_as_fraction_of_bound_range"][name] for case in cases]
            for name in names
        }
        output[channel] = {
            "cases": cases,
            "median_parameter_error_fraction": {
                name: float(np.median(errors)) for name, errors in parameter_errors.items()
            },
            "maximum_parameter_error_fraction": {
                name: float(np.max(errors)) for name, errors in parameter_errors.items()
            },
        }
    return {
        "design": "Five deterministic space-filling interior truths per channel, observed protocol, baseline-noise perturbations, and the locked production pipeline.",
        "results": output,
        "claim_boundary": "Recovery diagnoses protocol-level identifiability; trace prediction can remain accurate when individual HH parameters are not uniquely recovered.",
    }, raw_arrays


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
    diagnostics: dict,
    egl_controls: dict[str, np.ndarray],
    recovery: dict,
    cross_validation: dict,
    validation_arrays: dict[str, np.ndarray],
) -> dict:
    RESULTS.mkdir(exist_ok=True)
    shk_prediction = np.asarray(
        simulate_shk_braincell(shk_data.step_time_ms, shk_data.voltages_mV, shk_params)
    )
    egl_prediction = np.asarray(
        simulate_egl_braincell(egl_data.step_time_ms, egl_data.voltages_mV, egl_params)
    )
    report = {
        "protocol": {
            "holding_potential_mV": V_HOLD_MV,
            "step_duration_ms": 100.0,
            "shk_voltages_mV": shk_data.voltages_mV.tolist(),
            "egl_voltages_mV": egl_data.voltages_mV.tolist(),
        },
        "observation_model": {
            "SHK-1": "direct Igor WT-minus-shk-1(lf), baseline corrected, first 0.5 ms excluded",
            "EGL-19": "Igor WT calcium family, baseline corrected, first 1.0 ms excluded",
        },
        "parameters": {
            "SHK-1": asdict(shk_params),
            "EGL-19": asdict(egl_params),
        },
        "gating_trace_estimates": {
            "SHK-1": {
                key: np.asarray(value).tolist() for key, value in shk_points.items()
            },
            "EGL-19": {
                key: np.asarray(value).tolist()
                for key, value in egl_points.items()
                if key != "fit_diagnostics"
            },
        },
        "metrics": {
            "SHK-1": {
                **_metrics(shk_data.currents_pA, shk_prediction),
                "zero_current_baseline": _metrics(
                    shk_data.currents_pA, np.zeros_like(shk_data.currents_pA)
                ),
                "prestep_noise_sd_pA": shk_data.baseline_noise_pA.tolist(),
            },
            "EGL-19": {
                **_metrics(egl_data.currents_pA, egl_prediction),
                "zero_current_baseline": _metrics(
                    egl_data.currents_pA, np.zeros_like(egl_data.currents_pA)
                ),
                "prestep_noise_sd_pA": egl_data.baseline_noise_pA.tolist(),
            },
        },
        "parameter_recovery_domain": recovery,
        "leave_one_voltage_out": cross_validation,
        "fit_diagnostics": diagnostics,
        "egl_mutant_difference_control": {
            name: {
                "late_mean_pA": np.mean(values[:, -1000:], axis=1).tolist(),
                "contains_positive_and_negative_conditions": bool(
                    np.any(np.mean(values[:, -1000:], axis=1) > 0.0)
                    and np.any(np.mean(values[:, -1000:], axis=1) < 0.0)
                ),
            }
            for name, values in egl_controls.items()
        },
        "data_provenance": {
            K_FILE.name: _sha256(K_FILE),
            CA_FILE.name: _sha256(CA_FILE),
            "SHK-1_waves": list(shk_data.source_waves),
            "EGL-19_waves": list(egl_data.source_waves),
            "packed_metadata_verification": [
                verify_packed_metadata(K_FILE),
                verify_packed_metadata(CA_FILE),
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": 20260902,
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
        shk_n_inf_points_unclipped=shk_points["n_inf_unclipped"],
        shk_tau_n_points_ms=shk_points["tau_n_ms"],
        egl_time_ms=egl_data.step_time_ms,
        egl_voltages_mV=egl_data.voltages_mV,
        egl_observed_pA=egl_data.currents_pA,
        egl_predicted_pA=egl_prediction,
        egl_m_inf_points=egl_points["m_inf"],
        egl_tau_m_points_ms=egl_points["tau_m_ms"],
        egl_wt_minus_n582_pA=egl_controls["n582"],
        egl_wt_minus_ad1006_pA=egl_controls["ad1006"],
        **validation_arrays,
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

    shk_pred = np.asarray(
        simulate_shk_braincell(shk_data.step_time_ms, shk_data.voltages_mV, shk_params)
    )
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
    axes[0].scatter(shk_data.voltages_mV, shk_points["n_inf_unclipped"], color="#c4473d", s=34, zorder=3, label="Model-conditioned trace fits")
    axes[0].axhline(1.0, color="#777777", lw=0.8, ls=":")
    axes[0].set(xlabel="Voltage (mV)", ylabel="Steady-state activation, n_inf", ylim=(-0.03, 1.04))
    axes[0].legend(frameon=False)
    axes[1].plot(voltage_grid, shk_tau_n(voltage_grid, shk_params), color="#1769aa", lw=2)
    axes[1].scatter(shk_data.voltages_mV, shk_points["tau_n_ms"], color="#c4473d", s=34, zorder=3)
    axes[1].set(xlabel="Voltage (mV)", ylabel="Activation time constant, tau_n (ms)")
    fig.tight_layout()
    fig.savefig(RESULTS / "shk1_gating.png", dpi=220)
    plt.close(fig)

    egl_pred = np.asarray(
        simulate_egl_braincell(egl_data.step_time_ms, egl_data.voltages_mV, egl_params)
    )
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
    m_inf, tau_m = egl_gate_functions(voltage_grid, egl_params)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(voltage_grid, m_inf, color="#1769aa", lw=2, label="Fitted function")
    axes[0].scatter(egl_data.voltages_mV, egl_points["m_inf"], color="#c4473d", s=34, marker="o", label="Model-conditioned trace fits")
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
    brainstate.random.seed(20260902)
    np.random.seed(20260902)
    shk_data = load_shk_data()
    egl_data = load_egl_data()
    egl_controls = load_egl_mutant_controls()
    shk_params, shk_points, shk_diagnostics = fit_shk(shk_data)
    egl_params, egl_diagnostics = fit_egl(egl_data)
    egl_points = extract_egl_points(egl_data, egl_params)
    egl_diagnostics["gate_point_fit"] = egl_points["fit_diagnostics"]
    recovery, recovery_arrays = run_recovery_domain(shk_data, egl_data)
    cross_validation, cross_validation_arrays = run_leave_one_voltage_out(
        shk_data, egl_data
    )
    report = save_results(
        shk_data,
        shk_params,
        shk_points,
        egl_data,
        egl_params,
        egl_points,
        {"SHK-1": shk_diagnostics, "EGL-19": egl_diagnostics},
        egl_controls,
        recovery,
        cross_validation,
        {**recovery_arrays, **cross_validation_arrays},
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
