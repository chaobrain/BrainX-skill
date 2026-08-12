"""Controlled AHP ablation in a calibrated BrainCell teaching model.

Mechanism pattern:
https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html

Ablation pattern:
https://brainx.chaobrain.com/braincell/examples/channel_ablation.html

This is not a reproduction of one literature neuron. It combines classical HH
spiking with dynamic calcium and AHP_De1994, then sweeps AHP strength so the
causal effect is visible and sensitivity to the calibrated value is explicit.
"""

import braincell
import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np


class AdaptingCell(braincell.SingleCompartment):
    """Single-compartment teaching model with removable Ca-activated K current."""

    def __init__(self, size, g_ahp, solver="exp_euler"):
        super().__init__(
            size,
            C=1.0 * u.uF / u.cm**2,
            V_initializer=braintools.init.Constant(-75.0 * u.mV),
            V_th=20.0 * u.mV,
            solver=solver,
        )
        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        # These teaching values are calibrated, not a sourced cell phenotype.
        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5.0e-5 * u.mM,
            tau=80.0 * u.ms,
            d=0.5 * u.um,
        )
        self.ca.add(
            ICaL=braincell.channel.CaL_IS2008(
                size,
                g_max=5.0 * u.mS / u.cm**2,
            )
        )

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(
            IAHP=braincell.channel.AHP_De1994(size, g_max=g_ahp)
        )
        self.IL = braincell.channel.IL(
            size,
            E=-54.387 * u.mV,
            g_max=0.03 * u.mS / u.cm**2,
        )


dt = 0.01 * u.ms
stimulus_on = 50.0 * u.ms
stimulus_off = 550.0 * u.ms
holding_current = -0.5 * u.uA / u.cm**2
g_ahp = u.math.asarray([0.0, 0.3, 1.0]) * u.mS / u.cm**2
current = u.math.asarray([10.0, 10.0, 10.0]) * u.uA / u.cm**2

cell = AdaptingCell(size=g_ahp.shape, g_ahp=g_ahp)

with brainstate.environ.context(dt=dt):
    cell.init_state()
    times = u.math.arange(0.0 * u.ms, 600.0 * u.ms, dt)

    def step(t):
        injected = u.math.where(
            (t >= stimulus_on) & (t < stimulus_off),
            current,
            holding_current,
        )
        with brainstate.environ.context(t=t):
            spike = cell.update(injected)
        return cell.V.value, spike

    voltages, spikes = brainstate.transform.for_loop(step, times)

times_ms = np.asarray(times.to_decimal(u.ms))
pre_stimulus_spikes = np.asarray(spikes[times < stimulus_on]).sum()
assert pre_stimulus_spikes == 0

isis = []
for i, strength in enumerate(g_ahp.to_decimal(u.mS / u.cm**2)):
    spike_times = times_ms[np.asarray(spikes[:, i]) > 0]
    spike_times = spike_times[
        (spike_times >= stimulus_on.to_decimal(u.ms))
        & (spike_times < stimulus_off.to_decimal(u.ms))
    ]
    isi = np.diff(spike_times)
    isis.append(isi)
    print(
        f"g_AHP={strength:.1f} mS/cm^2: {spike_times.size} spikes, "
        f"first ISI={isi[0]:.2f} ms, last ISI={isi[-1]:.2f} ms"
    )

# The ablated lane is tonic; the calibrated present-AHP lane adapts strongly.
assert abs(isis[0][-1] - isis[0][0]) / isis[0][0] < 0.10
assert isis[-1][-1] / isis[-1][0] > 1.50

fig, axes = plt.subplots(2, 1, figsize=(8, 5))
for i, label in ((0, "AHP removed"), (-1, "AHP present (teaching value)")):
    axes[0].plot(
        times_ms,
        np.asarray(voltages[:, i].to_decimal(u.mV)),
        linewidth=0.8,
        label=label,
    )
    axes[1].plot(
        np.arange(1, isis[i].size + 1),
        isis[i],
        "o-",
        markersize=3,
        label=label,
    )
axes[0].set(
    xlabel="Time (ms)",
    ylabel="V (mV)",
    title="Matched steady-current conditions from a quiet baseline",
)
axes[1].set(xlabel="ISI index", ylabel="ISI (ms)", title="AHP-dependent slowing")
for axis in axes:
    axis.legend(frameon=False)
plt.tight_layout()
plt.show()
