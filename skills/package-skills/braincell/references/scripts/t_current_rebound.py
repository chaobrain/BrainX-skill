"""
Reference script mirrored from:
https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html

Purpose:
Advanced rebound-bursting script. Use when the task asks for post-inhibitory rebound, T-type calcium current, thalamic-style rebound, hyperpolarizing current steps, or release from inhibition.

Use this when:
- The task asks for rebound bursting after a hyperpolarizing current step.
- The agent needs T-type calcium and HCN currents in a single-compartment thalamic-style cell.

Do not use this when:
- The task asks for a generic HH current-clamp trace without rebound or calcium dynamics.
"""

import brainstate
import braintools
import brainunit as u
import numpy as np
import matplotlib.pyplot as plt
import braincell


class ThalamicCell(braincell.SingleCompartment):
    """Thalamic-relay-style cell with a low-threshold T-type Ca current."""
    def __init__(self, size, solver='ind_exp_euler'):
        super().__init__(size, V_initializer=braintools.init.Constant(-65. * u.mV),
                         V_th=20. * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30 * u.mV))
        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV, q10=2.0, temp=u.celsius2kelvin(16.)))
        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM,
                                                tau=10. * u.ms, d=0.5 * u.um)
        # T-type calcium current supports post-inhibitory rebound.
        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))
        self.Ih = braincell.channel.HCN_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)
        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)


cell = ThalamicCell(1)
cell.init_state()


def I_of_t(t):
    return u.math.where(t < 200. * u.ms,
                        -2. * u.uA / u.cm ** 2,
                        0. * u.uA / u.cm ** 2)


def step(t):
    with brainstate.environ.context(t=t):
        cell.update(I_of_t(t))
    return cell.V.value, cell.spike.value


with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, 500. * u.ms, brainstate.environ.get_dt())
    vs, spikes = brainstate.transform.for_loop(step, times)

spike_times = np.asarray(times[u.math.squeeze(spikes) > 0] / u.ms)
rebound = spike_times[spike_times > 200.]
print('spikes after release (ms):', np.round(rebound[:10], 1).tolist())

plt.figure(figsize=(8, 3))
plt.plot(times / u.ms, u.math.squeeze(vs) / u.mV, linewidth=0.8)
plt.axvline(200., color='k', linestyle=':', label='release')
plt.xlabel('Time (ms)'); plt.ylabel('Membrane potential (mV)')
plt.title('Post-inhibitory rebound burst (T-type Ca current)')
plt.legend(); plt.tight_layout(); plt.show()
