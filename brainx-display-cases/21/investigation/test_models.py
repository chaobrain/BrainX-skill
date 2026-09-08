"""Focused equation, execution, independence and numerical checks."""

import json
from pathlib import Path
import time

import brainstate
import brainunit as u
import jax
import numpy as np

from experiment import host_result, metrics
from models import Cell, Culture, GD, ID, cell_rollout, culture_runner


def main():
    brainstate.environ.set(precision=32)
    evidence = {}
    cell = Cell(2, np.array([0., 10.]) * GD, g_k=18,
                tau_na_ms=10, depth_um=.01)
    cell.init_state()
    cell.reset_state()
    assert np.allclose(cell.nak.potassium_current(-77*u.mV).to_decimal(ID), 0)
    outward = cell.nak.potassium_current(-40*u.mV).to_decimal(ID)
    assert outward[0] == 0 and outward[1] < 0
    cell.nak.compute_derivative(-65*u.mV)
    assert (cell.nak.sodium.derivative > 0*u.mM/u.ms).all()
    cell.nak.sodium.value = np.array([100., 100.])*u.mM
    cell.nak.compute_derivative(50*u.mV)
    assert (cell.nak.sodium.derivative < 0*u.mM/u.ms).all()
    assert all(np.isfinite(x).all() for x in cell.nak.rates(-40*u.mV))
    evidence['equations_and_units'] = 'pass: K reversal/sign, Na loading/removal, finite rates'

    culture = Culture([0, .5, 1], n=8, weight=0, seed=2,
                      g_kna=10, g_k=18, tau_na_ms=10, depth_um=.01)
    t, runner = culture_runner(culture, duration=100)
    timings, results = [], []
    for _ in range(2):
        start = time.perf_counter()
        with brainstate.environ.context(dt=.025*u.ms):
            result = runner(20*ID)
            jax.block_until_ready(result)
        timings.append(time.perf_counter()-start)
        results.append(host_result(result))
    assert all(np.array_equal(a, b) for a, b in zip(*results))
    kd = np.asarray(culture.kd)
    assert kd.sum(1).tolist() == [0, 4, 8]
    assert not np.diag(np.asarray(culture.adjacency)).any()
    for x in results[0][:4]:
        assert np.array_equal(x[:, 1, kd[1]], x[:, 2, kd[1]])
        assert np.array_equal(x[:, 1, ~kd[1]], x[:, 0, ~kd[1]])
    assert np.count_nonzero(results[0][4]) == 0
    evidence['reset_and_uncoupled_independence'] = 'pass: bit-identical State and output'
    evidence['compile_plus_first_run_seconds'] = timings[0]
    evidence['warm_run_seconds'] = timings[1]
    evidence['acceleration_decision'] = 'unchanged: compiled BrainState loop; no approximate optimization'

    summaries = []
    for dt in [.025, .0125]:
        t, result = cell_rollout(np.array([[20., 120.]]), np.array([[0.], [10.]]),
                                dt=dt, duration=500, g_k=18,
                                tau_na_ms=10, depth_um=.01)
        summary = metrics(np.asarray(t.to_decimal(u.ms)), host_result(result), 100, 600)
        summaries.append(summary)
    rate_difference = np.max(np.abs(summaries[0]['rate_hz']-summaries[1]['rate_hz']))
    voltage_difference = np.max(np.abs(summaries[0]['late_voltage_mV']-summaries[1]['late_voltage_mV']))
    assert rate_difference <= 2.1
    assert voltage_difference <= 1.
    assert np.array_equal(summaries[0]['block'], summaries[1]['block'])
    assert summaries[0]['rate_hz'][0, 0] > summaries[0]['rate_hz'][1, 0]
    assert summaries[0]['block'][0, 1] and not summaries[0]['block'][1, 1]
    evidence['cell_dt_refinement'] = dict(max_rate_difference_hz=float(rate_difference),
        max_late_voltage_difference_mV=float(voltage_difference), block_classification='unchanged')

    # Zero threshold crossings alone must never label a non-depolarized trace as block.
    t = np.arange(100., 600., .025)
    shape = (len(t), 1)
    quiet = (np.full(shape, -65.), np.zeros(shape), np.full(shape, .6), np.full(shape, 5.))
    assert not metrics(t, quiet, 100, 600)['block'].any()
    evidence['silence_is_not_block'] = 'pass'
    evidence['status'] = 'pass'
    path = Path('validation.json')
    path.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
