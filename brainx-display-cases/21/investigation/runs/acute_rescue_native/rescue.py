"""Acute recovery interventions in otherwise identical blocked cells."""

import json
from pathlib import Path

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

from experiment import host_result, metrics
from models import Cell, GD, ID


def main():
    out = Path('runs/acute_rescue_native')
    out.mkdir(exist_ok=False)
    cell = Cell(3, u.math.zeros(3)*GD, g_k=18, tau_na_ms=10, depth_um=.01)
    cell.init_state()
    with brainstate.environ.context(dt=.025*u.ms):
        reduced_drive = braintools.input.Constant([
            (0*ID, 100*u.ms), (120*ID, 300*u.ms), (20*ID, 400*u.ms)])
        sustained_drive = braintools.input.Constant([
            (0*ID, 100*u.ms), (120*ID, 700*u.ms)])
        currents = u.math.stack([reduced_drive(), sustained_drive(), sustained_drive()], axis=1)
        times = u.math.arange(0*u.ms, reduced_drive.duration, .025*u.ms)

    def step(t, injected_current):
        cell.nak.g_kna.value = u.math.where(
            (u.math.arange(3) == 1) & (t >= 400*u.ms), 10., 0.)*GD
        with brainstate.environ.context(t=t):
            spike = cell.update(injected_current)
        return cell.V.value, spike, cell.nak.h.value, cell.nak.sodium.value

    @brainstate.transform.jit
    def run():
        cell.reset_state()
        return brainstate.transform.for_loop(step, times, currents)

    with brainstate.environ.context(dt=.025*u.ms, precision=32):
        result = run()
        jax.block_until_ready(result)
    t, raw = np.asarray(times.to_decimal(u.ms)), host_result(result)
    np.savez_compressed(out/'raw.npz', t_ms=t, voltage_mV=raw[0], spikes=raw[1],
                        h=raw[2], sodium_mM=raw[3],
                        currents_uA_cm2=np.asarray(currents.to_decimal(ID)))
    summary = {}
    for name, start, stop in [('before', 200., 400.), ('after', 600., 800.)]:
        summary[name] = {k: v.tolist() for k, v in metrics(t, raw, start, stop).items()}
    summary['conditions'] = ['reduce drive to 20', 'restore gKNa to 10', 'no intervention']
    summary['protocol'] = '0-100 ms rest, 100-400 ms current 120, intervention 400-800 ms'
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'rescue.py').write_bytes(Path(__file__).read_bytes())
    (out/'models.py').write_bytes(Path(__file__).with_name('models.py').read_bytes())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
