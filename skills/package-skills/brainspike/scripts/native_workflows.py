"""Public BrainSpike Neuron composition; independent of the NEST Simulator."""
import brainspike
import brainstate
import braintrace
import braintools
import brainunit as u


class LIF(brainspike.Neuron):
    """Euler LIF with a unit-aware drive and an explicit soft reset."""
    def __init__(self, size):
        super().__init__(size, spk_fun=braintools.surrogate.InvSquareGrad())
        self.size = size
        self.tau = 10.0 * u.ms
        self.resistance = 100.0 * u.Mohm
        self.rest = -65.0 * u.mV
        self.threshold = -50.0 * u.mV

    def init_state(self, batch_size=None, **kwargs):
        shape = (self.size,) if batch_size is None else (batch_size, self.size)
        self.V = brainstate.HiddenState(u.math.full(shape, self.rest))
        self.spike = brainstate.HiddenState(u.math.zeros(shape))

    def reset_state(self, batch_size=None, **kwargs):
        shape = (self.size,) if batch_size is None else (batch_size, self.size)
        self.V.value = u.math.full(shape, self.rest)
        self.spike.value = u.math.zeros(shape)

    def get_spike(self):
        return self.spike.value

    def update(self, current):
        current = self.sum_current_inputs(current)
        voltage = self.V.value + brainstate.environ.get_dt() / self.tau * (
            self.rest - self.V.value + self.resistance * current
        )
        spike = self.spk_fun((voltage - self.threshold) / (10.0 * u.mV))
        self.V.value = voltage - spike * (self.threshold - self.rest)
        self.spike.value = spike
        return spike


class RecurrentSNN(brainstate.nn.Module):
    """ETP-aware input/recurrent weights and a trainable spike readout."""
    def __init__(self, size=3):
        super().__init__()
        self.drive = braintrace.nn.Linear(
            1 + size, size,
            w_init=braintools.init.Constant(0.4),
            b_init=braintools.init.ZeroInit(),
        )
        self.neuron = LIF(size)
        self.readout = braintrace.nn.Linear(size, 1)

    def update(self, x):
        features = u.math.concatenate((x, self.neuron.get_spike()), axis=-1)
        spikes = self.neuron(self.drive(features) * u.nA)
        return self.readout(spikes)


class ExponentialConductance(brainspike.Synapse):
    """Post-aligned linear filter: decay old conductance, then add new events."""
    def __init__(self, size, tau=5.0 * u.ms):
        super().__init__(size)
        self.size, self.tau = size, tau

    def init_state(self, batch_size=None, **kwargs):
        shape = (self.size,) if batch_size is None else (batch_size, self.size)
        self.g = brainstate.HiddenState(u.math.zeros(shape) * u.nS)

    def reset_state(self, batch_size=None, **kwargs):
        shape = (self.size,) if batch_size is None else (batch_size, self.size)
        self.g.value = u.math.zeros(shape) * u.nS

    def update(self, increment):
        self.g.value = self.g.value * u.math.exp(-brainstate.environ.get_dt() / self.tau) + increment
        return self.g.value
