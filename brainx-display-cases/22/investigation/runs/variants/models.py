"""Minimal population models; activity and drives are dimensionless."""

import brainmass
import brainstate
import braintools
import brainunit as u


class EICircuit(brainmass.WilsonCowanStep):
    """Stock Wilson-Cowan dynamics with explicit somatic/synaptic interventions."""

    def __init__(self, conditions, method="rk4", initial=0.05):
        names = ("wEE", "wEI", "wIE", "wII", "drive_E", "drive_I",
                 "soma_E", "soma_I", "release_E", "release_I", "gate_loss")
        self.p = {key: u.math.asarray([c[key] for c in conditions]) for key in names}
        super().__init__(
            in_size=len(conditions),
            tau_E=10.0 * u.ms, tau_I=20.0 * u.ms,
            a_E=1.2, theta_E=2.8, a_I=1.0, theta_I=4.0, r=1.0,
            wEE=self.p["wEE"], wEI=self.p["wEI"],
            wIE=self.p["wIE"], wII=self.p["wII"],
            rE_init=braintools.init.Constant(initial),
            rI_init=braintools.init.Constant(initial), method=method,
        )
        self.activation = brainstate.ShortTermState(u.math.zeros(len(conditions)))

    def drE(self, e, i, ext):
        p, h = self.p, self.activation.value
        correction = -h * p["release_E"] * p["wEE"] * e + h * p["release_I"] * p["wEI"] * i
        return super().drE(e, i, ext + correction)

    def drI(self, i, e, ext):
        p, h = self.p, self.activation.value
        correction = -h * p["release_E"] * p["wIE"] * e + h * p["release_I"] * p["wII"] * i
        return super().drI(i, e, ext + correction)

    def update(self, activation):
        p = self.p
        self.activation.value = u.math.ones_like(self.rE.value) * activation
        inp_e = p["drive_E"] - activation * p["soma_E"]
        inp_i = p["drive_I"] - activation * (p["soma_I"] + p["gate_loss"])
        return super().update(inp_e, inp_i)


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
        monitors={"E": "rE", "I": "rI", "activation": "activation"},
        sample_every=sample_every, jit=jit,
    )
    return result


def condition(**overrides):
    value = dict(label="baseline", wEE=16.0, wEI=12.0, wIE=15.0, wII=3.0,
                 drive_E=1.0, drive_I=1.0, soma_E=0.0, soma_I=0.0,
                 release_E=0.0, release_I=0.0, gate_loss=0.0)
    value.update(overrides)
    return value
