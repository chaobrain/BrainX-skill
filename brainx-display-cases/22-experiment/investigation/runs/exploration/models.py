"""Minimal population models; activity and drives are dimensionless."""

import brainmass
import brainstate
import braintools
import brainunit as u


class EICircuit(brainstate.nn.Module):
    """Stock Wilson-Cowan dynamics with explicit somatic/synaptic interventions."""

    def __init__(self, conditions, method="rk4", initial=0.05):
        super().__init__()
        self.in_size = len(conditions)
        names = ("wEE", "wEI", "wIE", "wII", "drive_E", "drive_I",
                 "soma_E", "soma_I", "release_E", "release_I", "gate_loss")
        self.p = {key: u.math.asarray([c[key] for c in conditions]) for key in names}
        self.node = brainmass.WilsonCowanStep(
            in_size=self.in_size,
            tau_E=10.0 * u.ms, tau_I=20.0 * u.ms,
            a_E=1.2, theta_E=2.8, a_I=1.0, theta_I=4.0, r=1.0,
            wEE=self.p["wEE"], wEI=self.p["wEI"],
            wIE=self.p["wIE"], wII=self.p["wII"],
            rE_init=braintools.init.Constant(initial),
            rI_init=braintools.init.Constant(initial), method=method,
        )

    def update(self, activation):
        p = self.p
        e, i = self.node.rE.value, self.node.rI.value
        loss_e = activation * p["release_E"]
        loss_i = activation * p["release_I"]
        # Corrections scale all outgoing synapses using the same pre-update State.
        inp_e = (p["drive_E"] - activation * p["soma_E"]
                 - loss_e * p["wEE"] * e + loss_i * p["wEI"] * i)
        inp_i = (p["drive_I"] - activation * (p["soma_I"] + p["gate_loss"])
                 - loss_e * p["wIE"] * e + loss_i * p["wII"] * i)
        self.node.update(inp_e, inp_i)
        return self.node.rE.value, self.node.rI.value


def simulate(conditions, dt_ms=0.2, duration_ms=3000.0, onset_ms=1000.0,
             ramp_ms=200.0, offset_ms=None, method="rk4", initial=0.05,
             sample_every=5, jit=True):
    circuit = EICircuit(conditions, method=method, initial=initial)

    def protocol(_index, time):
        h = u.math.clip((time - onset_ms * u.ms) / (ramp_ms * u.ms), 0.0, 1.0)
        if offset_ms is not None:
            h *= 1.0 - u.math.clip(
                (time - offset_ms * u.ms) / (ramp_ms * u.ms), 0.0, 1.0)
        return h

    result = brainmass.Simulator(circuit, dt=dt_ms * u.ms).run(
        duration_ms * u.ms, inputs=protocol,
        monitors={"E": lambda m: m.node.rE.value,
                  "I": lambda m: m.node.rI.value},
        sample_every=sample_every, jit=jit,
    )
    return result


def condition(**overrides):
    value = dict(label="baseline", wEE=16.0, wEI=12.0, wIE=15.0, wII=3.0,
                 drive_E=1.0, drive_I=1.0, soma_E=0.0, soma_I=0.0,
                 release_E=0.0, release_I=0.0, gate_loss=0.0)
    value.update(overrides)
    return value
