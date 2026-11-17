"""Causal same-clock composition of point, cellular, and regional dynamics."""
import braincell
import brainmass
import brainstate
import brainunit as u
from native_workflows import LIF


class HH(braincell.SingleCompartment):
    def __init__(self):
        super().__init__(1, V_th=20.0 * u.mV, solver="exp_euler")
        self.na = braincell.ion.SodiumFixed(1, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(1))
        self.k = braincell.ion.PotassiumFixed(1, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(1))
        self.IL = braincell.channel.IL(1, E=-54.387 * u.mV, g_max=0.03 * u.mS / u.cm**2)


class CoupledScales(brainstate.nn.Module):
    """Every target consumes source values from the previous completed step."""
    def __init__(self, gain=1.0):
        super().__init__()
        self.point = LIF(1)
        self.cell = HH()
        self.mass = brainmass.HopfStep(in_size=1, a=-0.25, w=0.3)
        self.gain = gain

    def update(self):
        previous_spikes = self.point.get_spike()
        previous_mass = self.mass.x.value
        previous_cell = (self.cell.V.value + 65.0 * u.mV) / (100.0 * u.mV)
        rate = previous_spikes / brainstate.environ.get_dt()
        self.point(0.5 * u.nA + self.gain * (previous_mass + previous_cell) * 0.05 * u.nA)
        self.cell.update(self.gain * previous_spikes * 0.1 * u.uA / u.cm**2)
        self.mass.update(x_inp=self.gain * rate / (1000.0 * u.Hz), y_inp=0.0)
