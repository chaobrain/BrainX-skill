"""Compare saved solver/refinement results without rerunning simulations."""

import json
from pathlib import Path

import numpy as np

from analyze import read_rows


def index_rows(rows, keys):
    return {tuple(row[key] for key in keys): row for row in rows}


def compare(a, b, fields):
    keys = sorted(set(a) & set(b))
    return {field: max(abs(a[key][field]-b[key][field]) for key in keys) for field in fields}


def main():
    cell_keys = ['g_kna_mS_cm2', 'current_uA_cm2']
    cell = index_rows(read_rows(Path('runs/cell_fast/metrics.csv')), cell_keys)
    exp = index_rows(read_rows(Path('runs/cell_exp_euler/metrics.csv')), cell_keys)
    exp_half = index_rows(read_rows(Path('runs/cell_exp_euler_half/metrics.csv')), cell_keys)
    fields = ['rate_hz', 'late_voltage_mV', 'block', 'calcium_voltage_proxy']
    result = dict(rk4_vs_exp_euler=compare(cell, exp, fields),
                  exp_euler_refinement=compare(exp, exp_half, fields))
    net_keys = ['seed', 'current_uA_cm2', 'fraction', 'group']
    network = index_rows(read_rows(Path('runs/block_validation/metrics.csv')), net_keys)
    half = index_rows(read_rows(Path('runs/block_dt_half/metrics.csv')), net_keys)
    result['network_refinement'] = compare(network, half, fields)
    for name in ['rk4_vs_exp_euler', 'exp_euler_refinement']:
        assert result[name]['rate_hz'] <= 2.1
        assert result[name]['late_voltage_mV'] <= 1.
        assert result[name]['block'] == 0
    assert result['network_refinement']['rate_hz'] <= 5.
    assert result['network_refinement']['block'] <= .050001
    with np.load('runs/block_validation/seed1_current2.npz') as a:
        with np.load('runs/block_dt_half/seed1_current0.npz') as b:
            assert np.array_equal(a['kd'], b['kd'])
            assert np.array_equal(a['adjacency'], b['adjacency'])
    assessment = json.loads(Path('assessment.json').read_text())
    reference = next(r for r in assessment['block_validation']
                     if r['seed']==1 and r['current_uA_cm2']==120)
    refined = assessment['block_dt_half'][0]
    for r in [reference, refined]:
        assert r['optical_within_mosaic_at_half'] > 0
        assert r['optical_full_at_half'] < 1
    result['optical_signs_at_half'] = 'preserved'
    result['network_pairing'] = 'exact topology and targeting match'
    result['status'] = 'pass'
    Path('numerical-assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
