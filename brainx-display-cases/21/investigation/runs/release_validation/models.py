"""Illustrative KCNT2 mechanisms, not a fitted model of human iNeurons."""

import braincell
import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax

GD = u.mS / u.cm**2
ID = u.uA / u.cm**2


class SodiumKNa(braincell.Channel):
    """HH sodium current coupled to a local Na pool and instantaneous KNa."""

    root_type = braincell.HHTypedNeuron

    def __init__(self, size, g_kna, depth=0.1 * u.um, tau_na=100 * u.ms,
                 half_na=20 * u.mM, g_na=120 * GD):
        super().__init__(size)
        self.g_kna = brainstate.ParamState(g_kna)
        self.g_na = g_na
        self.depth, self.tau_na, self.half_na = depth, tau_na, half_na
        self.ena, self.ek, self.na_rest = 50 * u.mV, -77 * u.mV, 5 * u.mM

    @staticmethod
    def rates(v):
        x = v.to_decimal(u.mV)
        return (1 / u.math.exprel(-(x + 40) / 10),
                4 * u.math.exp(-(x + 65) / 18),
                0.07 * u.math.exp(-(x + 65) / 20),
                1 / (1 + u.math.exp(-(x + 35) / 10)))

    def init_state(self, v, batch_size=None):
        am, bm, ah, bh = self.rates(v)
        self.m = braincell.quad.DiffEqState(am / (am + bm))
        self.h = braincell.quad.DiffEqState(ah / (ah + bh))
        self.sodium = braincell.quad.DiffEqState(u.math.ones_like(v / u.mV) * self.na_rest)

    def reset_state(self, v, batch_size=None):
        am, bm, ah, bh = self.rates(v)
        self.m.value, self.h.value = am / (am + bm), ah / (ah + bh)
        self.sodium.value = u.math.ones_like(v / u.mV) * self.na_rest

    def sodium_current(self, v):
        return self.g_na * self.m.value**3 * self.h.value * (self.ena - v)

    def potassium_current(self, v):
        ratio = self.sodium.value / self.half_na
        activation = ratio**3 / (1 + ratio**3)
        return self.g_kna.value * activation * (self.ek - v)

    def current(self, v):
        return self.sodium_current(v) + self.potassium_current(v)

    def compute_derivative(self, v):
        am, bm, ah, bh = self.rates(v)
        self.m.derivative = (am * (1 - self.m.value) - bm * self.m.value) / u.ms
        self.h.derivative = (ah * (1 - self.h.value) - bh * self.h.value) / u.ms
        influx = u.math.maximum(self.sodium_current(v), 0 * ID)
        self.sodium.derivative = (influx / (u.faraday_constant * self.depth)
                                  - (self.sodium.value - self.na_rest) / self.tau_na)


class Cell(braincell.SingleCompartment):
    def __init__(self, size, g_kna, initial=-65 * u.mV, solver="rk4", g_k=36.,
                 tau_na_ms=100., depth_um=0.1, **kwargs):
        super().__init__(size, V_initializer=braintools.init.Constant(initial),
                         V_th=-20 * u.mV, C=1 * u.uF / u.cm**2, solver=solver)
        self.nak = SodiumKNa(size, g_kna, tau_na=tau_na_ms * u.ms,
                            depth=depth_um * u.um, **kwargs)
        self.k = braincell.ion.PotassiumFixed(size, E=-77 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size, g_max=g_k * GD, V_sh=-45 * u.mV))
        self.leak = braincell.channel.IL(size, g_max=0.3 * GD, E=-54.387 * u.mV)


class EventComm(brainstate.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = brainstate.ParamState(weights)

    def update(self, spikes):
        return jax.vmap(lambda s, w: brainevent.BinaryArray(s) @ w)(spikes, self.weights.value)


class Culture(brainstate.nn.Module):
    """Independent fraction conditions share a topology, never dynamical State."""

    def __init__(self, fractions, n=100, seed=0, g_kna=5., residual=0.,
                 weight=0.05, probability=0.3, synaptic_loss=0., **cell_kwargs):
        super().__init__()
        brainstate.random.seed(seed)
        shape = (len(fractions), n)
        ranks = brainstate.random.permutation(n)
        self.kd = ranks[None, :] < u.math.asarray(fractions)[:, None] * n
        g = g_kna * u.math.where(self.kd, residual, 1.) * GD
        self.heterogeneity = 1 + 0.10 * brainstate.random.randn(n)
        initial = (-65 + 2 * brainstate.random.randn(n)) * u.mV
        self.cells = Cell(shape, g, initial=initial, **cell_kwargs)
        adjacency = brainstate.random.bernoulli(probability, size=(n, n))
        adjacency = adjacency & ~u.math.eye(n, dtype=bool)
        self.adjacency = adjacency
        self.release = u.math.where(self.kd, 1 - synaptic_loss, 1.)
        weights = self.release[:, :, None] * adjacency[None, :, :] * weight * GD
        self.proj = brainpy.state.AlignPostProj(
            comm=EventComm(weights),
            syn=brainpy.state.Expon(shape, tau=5 * u.ms,
                                    g_initializer=braintools.init.Constant(0 * GD)),
            out=brainpy.state.COBA(E=0 * u.mV), post=self.cells)
        self.initial = initial

    def init_state(self):
        pass

    def update(self, t, injected):
        with brainstate.environ.context(t=t):
            spikes = self.cells.spike.value != 0
            self.proj(spikes)
            out = self.cells.update(injected * self.heterogeneity)
        return out


def culture_runner(culture, dt=0.025, duration=1000., baseline=100.):
    protocol, times = current_protocol(dt, duration, baseline)
    with brainstate.environ.context(dt=dt * u.ms):
        brainstate.nn.init_all_states(culture)

    @brainstate.transform.jit
    def run(current):
        culture.cells.reset_state()
        culture.proj.syn.reset_state()

        def step(t, amplitude):
            spike = culture.update(t, amplitude * current)
            return (culture.cells.V.value, spike, culture.cells.nak.h.value,
                    culture.cells.nak.sodium.value, culture.proj.syn.g.value)

        return brainstate.transform.for_loop(step, times, protocol)

    return times, run


def current_protocol(dt, duration=1000., baseline=100.):
    with brainstate.environ.context(dt=dt * u.ms):
        protocol = braintools.input.Constant([(0., baseline * u.ms),
                                              (1., duration * u.ms)])
        return protocol(), u.math.arange(0 * u.ms, protocol.duration, dt * u.ms)


def cell_rollout(currents, conductances, dt=0.025, duration=1000., solver="rk4", **kwargs):
    currents, conductances = u.math.broadcast_arrays(u.math.asarray(currents),
                                                     u.math.asarray(conductances))
    cells = Cell(currents.shape, conductances * GD, solver=solver, **kwargs)
    cells.init_state()
    protocol, times = current_protocol(dt, duration)

    def step(t, amplitude):
        with brainstate.environ.context(t=t):
            spike = cells.update(amplitude * currents * ID)
        return cells.V.value, spike, cells.nak.h.value, cells.nak.sodium.value

    @brainstate.transform.jit
    def run():
        cells.reset_state()
        return brainstate.transform.for_loop(step, times, protocol)

    with brainstate.environ.context(dt=dt * u.ms):
        result = run()
    return times, result
