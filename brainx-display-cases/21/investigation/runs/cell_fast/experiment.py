"""Run immutable illustrative experiments and retain raw mechanism observables."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import brainstate
import brainunit as u
import jax
import numpy as np

from models import Cell, Culture, GD, ID, cell_rollout, culture_runner


def metrics(t, result, start, stop):
    window = (t >= start) & (t < stop)
    v, spikes, h, sodium = result[:4]
    rate = spikes[window].sum(0) * 1000 / (stop - start)
    last = (t >= stop - 100) & (t < stop)
    vmean, vstd = v[last].mean(0), v[last].std(0)
    late_spikes = spikes[last].sum(0)
    block = (late_spikes == 0) & (vmean > -50) & (vstd < 3) & (h[last].mean(0) < .3)
    calcium_v = (1 / (1 + np.exp(-(v[window] + 30) / 6))
                 * np.maximum(120 - v[window], 0) / 120).mean(0)
    summary = dict(rate_hz=rate, late_voltage_mV=vmean, late_voltage_sd_mV=vstd,
                   h_available=h[last].mean(0), sodium_mM=sodium[last].mean(0),
                   block=block, calcium_voltage_proxy=calcium_v,
                   late_spikes=late_spikes)
    if len(result) > 4:
        g = result[4][window]
        summary['synaptic_g_mS_cm2'] = g.mean(0)
        summary['synaptic_inward_uA_cm2'] = (g * np.maximum(-v[window], 0)).mean(0)
    return summary


def host_result(result):
    units = (u.mV, None, None, u.mM, GD)
    raw = tuple(np.asarray(x.to_decimal(unit) if unit is not None else x)
                for x, unit in zip(result, units))
    if not all(np.isfinite(x).all() for x in raw):
        raise ValueError('Nonfinite trajectory; reject this run.')
    if (raw[2].min() < -1e-5 or raw[2].max() > 1 + 1e-5
            or raw[3].min() < 0):
        raise ValueError('Invalid gate or sodium concentration; reject this run.')
    return raw


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), Path(__file__).with_name('models.py'))}
    for filename in hashes:
        (out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    (out / 'environment.json').write_text(json.dumps(dict(
        python=sys.executable, python_version=platform.python_version(),
        backend=jax.default_backend(), devices=list(map(str, jax.devices())),
        code_sha256=hashes, command=sys.argv), indent=2) + '\n')
    started = time.perf_counter()
    rows = []
    dt, duration = config.get('dt_ms', .025), config.get('duration_ms', 1000.)
    brainstate.environ.set(precision=32)
    if config['mode'] == 'cell':
        currents = np.array(config['currents_uA_cm2'])
        conductances = np.array(config['g_kna_mS_cm2'])
        times, result = cell_rollout(currents[None, :], conductances[:, None],
                                    dt=dt, duration=duration, solver=config.get('solver', 'rk4'),
                                    **config.get('cell', {}))
        t, raw = np.asarray(times.to_decimal(u.ms)), host_result(result)
        np.savez_compressed(out / 'raw.npz', t_ms=t, voltage_mV=raw[0],
                            spikes=raw[1], h=raw[2], sodium_mM=raw[3])
        summary = metrics(t, raw, 100., 100. + duration)
        for i, g in enumerate(conductances):
            for j, current in enumerate(currents):
                rows.append(dict(g_kna_mS_cm2=g, current_uA_cm2=current,
                                 **{k: float(v[i, j]) for k, v in summary.items()}))
    else:
        fractions = config.get('fractions', np.linspace(0, 1, 11).tolist())
        for seed in config['seeds']:
            culture = Culture(fractions, seed=seed, **config['network'])
            times, runner = culture_runner(culture, dt=dt, duration=duration)
            for index, current in enumerate(config['currents_uA_cm2']):
                tic = time.perf_counter()
                with brainstate.environ.context(dt=dt * u.ms):
                    result = runner(current * ID)
                    jax.block_until_ready(result)
                t, raw = np.asarray(times.to_decimal(u.ms)), host_result(result)
                name = f'seed{seed}_current{index}'
                np.savez_compressed(out / f'{name}.npz', t_ms=t, voltage_mV=raw[0],
                                    spikes=raw[1], h=raw[2], sodium_mM=raw[3],
                                    synaptic_g_mS_cm2=raw[4], kd=np.asarray(culture.kd),
                                    adjacency=np.asarray(culture.adjacency))
                summary = metrics(t, raw, 100., 100. + duration)
                for i, fraction in enumerate(fractions):
                    for group, mask in [('all', np.ones(culture.kd.shape[1], bool)),
                                        ('KD', np.asarray(culture.kd[i])),
                                        ('control', ~np.asarray(culture.kd[i]))]:
                        if mask.any():
                            rows.append(dict(seed=seed, fraction=fraction, group=group,
                                             current_uA_cm2=current, n=int(mask.sum()),
                                             **{k: float(v[i, mask].mean()) for k, v in summary.items()}))
                message = f'{name} completed in {time.perf_counter() - tic:.2f} s'
                print(message, flush=True)
                with (out / 'run.log').open('a') as handle:
                    handle.write(message + '\n')
    write_csv(out / 'metrics.csv', rows)
    elapsed = time.perf_counter() - started
    with (out / 'run.log').open('a') as handle:
        handle.write(f'Completed all finite trajectories in {elapsed:.2f} s\n')
    (out / 'status.json').write_text(json.dumps(dict(status='done', elapsed_seconds=elapsed,
                                                    finite=all(np.isfinite(r['rate_hz']) for r in rows)), indent=2))
    print(f'DONE {out}: {elapsed:.2f} s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out)
