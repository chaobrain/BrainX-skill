from pathlib import Path

import numpy as np

import brainunit as u

from cellegans_hh.data import FIT_TRACE, TEST_TRACES, current_protocol, initial_voltage, load_experiment
from cellegans_hh.model import PARAMETER_SPECS, initial_parameter_vector, simulate
from cellegans_hh.inference import InferenceProblem, spike_times_ms


ROOT = Path(__file__).resolve().parents[1]


def test_data_contract_and_split():
    data = load_experiment(ROOT / "Fig4A-D.txt")
    assert data.time.shape == (5000,)
    assert FIT_TRACE == 8
    assert TEST_TRACES == (6, 7, 9)
    assert data.sha256 == "7f6ff6a74b17e79a7d1122dc85ea4e00daf3f30a7b07d6bc2b7f60e8a4cd7df7"
    assert np.isclose(np.diff(data.time.to_decimal(u.ms)).mean(), 0.1)


def test_parameter_map_is_labeled_and_bounded():
    names = [spec.name for spec in PARAMETER_SPECS]
    assert names == [
        "g_shk1",
        "g_egl19",
        "g_slo2",
        "g_kr",
        "g_na",
        "g_leak",
        "capacitance",
    ]
    for spec in PARAMETER_SPECS:
        assert spec.lower < spec.initial < spec.upper


def test_nominal_rollout_is_finite_and_deterministic():
    data = load_experiment(ROOT / "Fig4A-D.txt")
    target = data.voltage_by_trace[FIT_TRACE]
    initial_v = initial_voltage(target, data.time)
    current = current_protocol(data.time[:200], 25.0 * u.pA)
    first = simulate(initial_parameter_vector(), current, initial_v)
    second = simulate(initial_parameter_vector(), current, initial_v)
    first_mV = first.to_decimal(u.mV)
    assert first.shape == (200,)
    assert np.isfinite(first_mV).all()
    np.testing.assert_allclose(first_mV, second.to_decimal(u.mV), rtol=0, atol=1e-6)


def test_nominal_full_state_is_finite_and_in_range():
    data = load_experiment(ROOT / "Fig4A-D.txt")
    target = data.voltage_by_trace[FIT_TRACE]
    time = data.time[:1000]
    states = simulate(
        initial_parameter_vector(),
        current_protocol(time, 25.0 * u.pA),
        initial_voltage(target, data.time),
        return_states=True,
    )
    assert set(states) == {
        "voltage", "shk1_m", "shk1_h", "egl19_m", "egl19_h",
        "kr_n", "na_m", "na_h", "calcium_i",
    }
    assert np.isfinite(states["voltage"].to_decimal(u.mV)).all()
    assert np.isfinite(states["calcium_i"].to_decimal(u.mM)).all()
    assert np.all(states["calcium_i"].to_decimal(u.mM) > 0.0)
    for name in ("shk1_m", "shk1_h", "egl19_m", "egl19_h", "kr_n", "na_m", "na_h"):
        values = np.asarray(states[name])
        assert np.isfinite(values).all()
        assert values.min() >= -1e-6
        assert values.max() <= 1.0 + 1e-6


def test_batched_candidates_have_independent_state_and_scalar_parity():
    data = load_experiment(ROOT / "Fig4A-D.txt")
    target = data.voltage_by_trace[FIT_TRACE]
    initial_v = initial_voltage(target, data.time)
    current = current_protocol(data.time[:200], 25.0 * u.pA)
    nominal = np.asarray(initial_parameter_vector())
    batch = np.stack((nominal, nominal))
    batched = simulate(batch, current, initial_v).to_decimal(u.mV)
    scalar = simulate(nominal, current, initial_v).to_decimal(u.mV)
    assert batched.shape == (200, 2)
    np.testing.assert_allclose(batched[:, 0], scalar, rtol=0, atol=1e-6)
    np.testing.assert_allclose(batched[:, 1], scalar, rtol=0, atol=1e-6)


def test_vectorized_objective_returns_one_finite_loss_per_candidate():
    data = load_experiment(ROOT / "Fig4A-D.txt")
    target = data.voltage_by_trace[FIT_TRACE][:200]
    time = data.time[:200]
    problem = InferenceProblem(
        time=time,
        target=target,
        current=current_protocol(time, 25.0 * u.pA),
        initial_v=initial_voltage(data.voltage_by_trace[FIT_TRACE], data.time),
    )
    nominal = np.asarray(initial_parameter_vector())
    scipy_batch = np.stack((nominal, nominal), axis=1)
    losses = problem.vectorized_objective(scipy_batch)
    assert losses.shape == (2,)
    assert np.isfinite(losses).all()
    assert len(problem.records) == 1
    assert problem.records[0].candidates == 2


def test_spike_detector_can_check_prestimulus_window():
    time_ms = np.arange(0.0, 100.0, 0.1)
    voltage_mV = np.full(time_ms.shape, -40.0)
    voltage_mV[100] = 20.0
    assert spike_times_ms(time_ms, voltage_mV).size == 0
    prestim = spike_times_ms(time_ms, voltage_mV, start_ms=None, end_ms=50.0)
    np.testing.assert_allclose(prestim, [10.0])
