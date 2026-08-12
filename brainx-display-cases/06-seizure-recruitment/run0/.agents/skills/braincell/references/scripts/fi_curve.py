"""
Reference script mirrored from:
https://brainx.chaobrain.com/braincell/examples/fi_curve.html

Purpose:
Canonical vectorized current-sweep script. Use when the task asks for FI curves, current sweeps, firing-rate extraction, warm-up discard, or using `size=N` as independent point neurons.

Use this when:
- The task asks for FI curves, current sweeps, spike counts, or firing-rate extraction.
- The agent needs a vectorized `SingleCompartment` pattern where `size=N` means independent point neurons.

Do not use this when:
- The task asks for morphology, network connectivity, or one detailed single-cell trace only.
"""

import brainstate
import brainunit as u
import numpy as np
import matplotlib.pyplot as plt
import braincell


class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='exp_euler'):
        super().__init__(size, V_th=20. * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))
        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))
        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,
                                       g_max=0.03 * (u.mS / u.cm ** 2))


n_levels = 11
amplitudes = np.linspace(0., 20., n_levels)        # uA/cm^2
I = amplitudes * (u.uA / u.cm ** 2)

# `HH(n_levels)` creates independent point neurons, not compartments.
net = HH(n_levels)
net.init_state()

warmup = 100. * u.ms
total = 600. * u.ms


def step(t):
    with brainstate.environ.context(t=t):
        net.update(I)
    return t, net.spike.value


with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, total, brainstate.environ.get_dt())
    ts, spikes = brainstate.transform.for_loop(step, times)

# Discard warm-up before converting spike counts to firing rate.
mask = (ts >= warmup)
counts = np.asarray(u.math.sum(spikes[mask], axis=0))
rate = counts / float((total - warmup) / u.second)   # Hz
print('spike counts:', counts.astype(int).tolist())

plt.figure(figsize=(5, 4))
plt.plot(amplitudes, rate, 'o-')
plt.xlabel('Input current density (uA/cm$^2$)')
plt.ylabel('Firing rate (Hz)')
plt.title('F\u2013I curve of an HH neuron')
plt.tight_layout()
plt.show()
