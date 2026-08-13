"""
Reference script mirrored from:
https://brainx.chaobrain.com/braincell/examples/channel_ablation.html

Purpose:
Canonical ablation script. Use when the task asks to remove, suppress, or compare a channel/current contribution, especially by setting conductance such as `gK=0` while keeping the ion/channel structure intact.

Use this when:
- The task asks for intact-vs-ablated point-neuron comparisons.
- The agent needs to suppress a current by changing `g_max` while preserving the ion/channel structure.

Do not use this when:
- The task asks for custom channel authoring or morphology-based mechanism removal.
"""

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import braincell


class HH(braincell.SingleCompartment):
    def __init__(self, size, gK=36. * (u.mS / u.cm ** 2), solver='exp_euler'):
        super().__init__(size, V_th=20. * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))
        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        # Ablation is controlled by `gK`; the potassium ion/channel path remains present.
        self.k.add(IK=braincell.channel.K_HH1952(size, g_max=gK))
        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,
                                       g_max=0.03 * (u.mS / u.cm ** 2))


intact = HH(1)
ablated = HH(1, gK=0. * (u.mS / u.cm ** 2))
intact.init_state()
ablated.init_state()

I = 5. * u.uA / u.cm ** 2


def step(t):
    with brainstate.environ.context(t=t):
        intact.update(I)
        ablated.update(I)
    return intact.V.value, ablated.V.value


with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, 80. * u.ms, brainstate.environ.get_dt())
    v_intact, v_ablated = brainstate.transform.for_loop(step, times)

plt.figure(figsize=(8, 3))
plt.plot(times / u.ms, u.math.squeeze(v_intact) / u.mV, label='intact')
plt.plot(times / u.ms, u.math.squeeze(v_ablated) / u.mV, label='no $I_K$', linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Effect of ablating the potassium current')
plt.legend()
plt.tight_layout()
plt.show()
