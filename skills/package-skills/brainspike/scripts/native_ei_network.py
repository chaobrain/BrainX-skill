"""Small native conductance E/I network; run from this file's directory or directly."""
import json
import brainevent
import brainstate
import brainunit as u
from native_workflows import LIF, ExponentialConductance


class EINetwork(brainstate.nn.Module):
    def __init__(self, n_exc=16, n_inh=4):
        super().__init__()
        self.n_exc, self.n_inh = n_exc, n_inh
        self.exc, self.inh = LIF(n_exc), LIF(n_inh)
        n = n_exc+n_inh
        self.w_exc = brainstate.random.bernoulli(0.3, size=(n_exc,n))*0.3*u.nS
        self.w_inh = brainstate.random.bernoulli(0.3, size=(n_inh,n))*0.8*u.nS
        self.syn_exc = ExponentialConductance(n, tau=5*u.ms)
        self.syn_inh = ExponentialConductance(n, tau=10*u.ms)

    def update(self, drive):
        # Snapshot previous events and voltage before advancing either population.
        exc_spikes, inh_spikes = self.exc.get_spike(), self.inh.get_spike()
        voltage = u.math.concatenate((self.exc.V.value, self.inh.V.value))
        g_exc = self.syn_exc(brainevent.BinaryArray(exc_spikes != 0) @ self.w_exc)
        g_inh = self.syn_inh(brainevent.BinaryArray(inh_spikes != 0) @ self.w_inh)
        i_exc = g_exc*(0*u.mV-voltage)
        i_inh = g_inh*(-80*u.mV-voltage)
        current = drive+i_exc+i_inh
        exc = self.exc(current[:self.n_exc])
        inh = self.inh(current[self.n_exc:])
        return {"spikes": u.math.concatenate((exc,inh)), "g_exc": g_exc,
                "g_inh": g_inh, "i_exc": i_exc, "i_inh": i_inh,
                "voltage": u.math.concatenate((self.exc.V.value,self.inh.V.value))}


def simulate(steps=300):
    brainstate.environ.set(precision=64,dt=0.1*u.ms)
    brainstate.random.seed(7)
    net = EINetwork()
    brainstate.nn.init_all_states(net)
    outputs = brainstate.transform.for_loop(lambda _: net(0.2*u.nA), u.math.arange(steps))
    return net, outputs


if __name__ == "__main__":
    _, outputs = simulate()
    print(json.dumps({"spikes":float(outputs["spikes"].sum()), "shape":outputs["spikes"].shape}))
