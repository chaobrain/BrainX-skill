"""
Reference script mirrored from:
https://brainx.chaobrain.com/braincell/examples/hh_neuron_basics.html

Purpose:
Canonical minimal single-compartment HH simulation. Use as the default full-script reference for building one `SingleCompartment` neuron, adding Na/K/leak currents, initializing state, injecting current through `update(I)`, running a loop, and plotting voltage/spikes.

Use this when:
- The task asks for one end-to-end HH point-neuron current-clamp simulation.
- The agent needs the default `SingleCompartment` Na/K/leak construction pattern.

Do not use this when:
- The task needs morphology, network wiring, or a channel-level diagnostic instead of a membrane trace.
"""

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import braincell


# Subclass `SingleCompartment`; `size=1` below means one point neuron.
class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='exp_euler'):
        super().__init__(size, V_th=20. * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))
        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))
        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,
                                       g_max=0.03 * (u.mS / u.cm ** 2))


neuron = HH(1)
neuron.init_state()

# Current clamp is direct: pass density-based injected current to `update(I)`.
I = 5. * u.uA / u.cm ** 2


def step(t):
    with brainstate.environ.context(t=t):
        neuron.update(I)
    return neuron.V.value, neuron.spike.value


with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, 100. * u.ms, brainstate.environ.get_dt())
    vs, spikes = brainstate.transform.for_loop(step, times)

print('number of spikes:', int(u.math.sum(spikes)))

plt.figure(figsize=(8, 3))
plt.plot(times / u.ms, u.math.squeeze(vs) / u.mV, linewidth=1.2)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('HH neuron under 5 uA/cm^2 constant current')
plt.tight_layout()
plt.show()
