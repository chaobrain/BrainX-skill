"""COBA E/I balanced network -- a minimal teaching model on BrainEvent CSR.

The network is the classic Vogels & Abbott (2005) conductance-based (COBA)
excitatory/inhibitory random network:

    * ``N`` leaky integrate-and-fire neurons with an absolute refractory period,
      split 80% excitatory / 20% inhibitory.
    * Every neuron projects to ``CONN_NUM`` randomly chosen postsynaptic targets
      (fixed out-degree), stored as one connection matrix per population. The
      storage format is selectable -- ``fcn``, ``csr`` or ``dense``, see
      ``make_connection`` -- and does not change the dynamics.
    * Each spike deposits conductance into an exponentially decaying synapse
      (``Expon``); the conductance drives the membrane through a reversal
      potential (``COBA``).

The whole simulation is a single loop over time:

    spikes -> CSR mat-vec -> synaptic conductance -> membrane -> spikes

Everything is spelled out inline: one connection builder, one network class,
one simulation function.

Run with the BrainEvent environment::

    python coba_ei_teaching.py --conn csr
"""

from __future__ import annotations

import argparse

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Parameters (mirrors config.txt)
# --------------------------------------------------------------------------

PRECISION = 32          # float32 everywhere
DT = 0.1 * u.ms         # integration step
DURATION = 100. * u.ms  # simulated time

SCALE = 1               # network size multiplier
N_BASE = 4000           # neurons per unit of scale
EXC_FRACTION = 0.8      # 80% excitatory / 20% inhibitory

# regime.
CONN_NUM = 80           # out-degree: targets per presynaptic neuron
CONN_NUM_BASE = 80      # reference out-degree the weights were tuned for

# Synaptic weights. The reference COBA model is dimensionless:
#     dV/dt = (V_rest - V + I_offset + g*(E - V)) / tau_m
# with V in "mV numbers" and g a conductance relative to the leak. BrainPy's
# LIFRef is dimensionful -- dV/dt = (V_rest - V + R*I) / tau_m with R = 1 ohm --
# so the term R*g*(E - V) reproduces the reference only when g is expressed in
# *siemens*: 1 ohm * 1 S * 1 mV = 1 mV. Writing these as mS would make the
# synapses 1000x too weak and leave the network driven purely by I_OFFSET.
EXC_WEIGHT_BASE = 0.6 * u.siemens
INH_WEIGHT_BASE = 6.7 * u.siemens

# LIF neuron
V_REST = -60. * u.mV
V_RESET = -60. * u.mV
V_TH = -50. * u.mV
TAU_M = 20. * u.ms
TAU_REF = 5. * u.ms
V_INIT_MEAN = -55. * u.mV
V_INIT_STD = 5. * u.mV

# COBA synapses
E_EXC = 0. * u.mV
E_INH = -80. * u.mV
TAU_SYN_EXC = 5. * u.ms
TAU_SYN_INH = 10. * u.ms

# Constant drive. Same convention as the weights: R * 20 mA = 1 ohm * 20 mA
# = 20 mV, matching the reference model's dimensionless i_offset = 20.
I_OFFSET = 20. * u.mA

CONN_SEED = 123
STATE_SEED = 456

# Neurons whose analogue variables (V, conductances) get recorded -- think of
# it as a 64-electrode array. Most neurons in this network are silent over a
# 100 ms window, so recording a group and plotting the most active one is more
# informative than picking a fixed index.
PROBE_NEURONS = np.arange(64)

brainstate.environ.set(dt=DT, precision=PRECISION)


# --------------------------------------------------------------------------
# Connectivity
#
# The same connection pattern can be stored three different ways. All of them
# answer `spikes @ conn` with the same postsynaptic input; they differ only in
# what is kept in memory and which kernel runs:
#
#   'fcn'    brainevent.FixedNumPerPre -- an (n_pre, out_degree) index matrix.
#            Exploits the fact that every row has the same length, so no row
#            pointer is needed. This is what the reference benchmark uses.
#   'csr'    brainevent.CSR -- flat indices plus a row pointer. The general
#            sparse format; it also allows rows of unequal length.
#   'dense'  a plain (n_pre, n_post) array. Easiest to read, but O(N^2) memory
#            and the kernel touches every entry, spike or not.
#
# All three share `sample_targets`, so they produce bit-identical spike trains.
#
# Weights are homogeneous (one shared scalar), so for 'fcn' and 'csr' the
# connectivity costs only the indices.
# --------------------------------------------------------------------------

CONN_KIND = 'fcn'       # 'fcn' | 'csr' | 'dense'


def sample_targets(n_pre: int, n_post: int, out_degree: int, seed: int) -> np.ndarray:
    """Row ``i`` lists the ``out_degree`` postsynaptic targets of neuron ``i``."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_post, size=(n_pre, out_degree), dtype=np.int32)


def make_connection(n_pre: int, n_post: int, out_degree: int, weight, seed: int, kind=CONN_KIND):
    """Build a ``(n_pre, n_post)`` connection matrix in the requested format."""
    w = u.math.asarray(weight, dtype=brainstate.environ.dftype())
    shape = (n_pre, n_post)

    if kind == 'fcn':
        targets = sample_targets(n_pre, n_post, out_degree, seed)
        return brainevent.FixedNumPerPre(w, targets, shape=shape)

    if kind == 'csr':
        # Every row has the same length, so the row pointer is just a ramp.
        indices = sample_targets(n_pre, n_post, out_degree, seed).reshape(-1)
        indptr = np.arange(n_pre + 1, dtype=np.int32) * np.int32(out_degree)
        return brainevent.CSR((w, indices, indptr), shape=shape)

    if kind == 'dense':
        # `add.at` accumulates, so a target drawn twice gets twice the weight --
        # the same thing the sparse formats do with duplicate column indices.
        targets = sample_targets(n_pre, n_post, out_degree, seed)
        mask = np.zeros(shape, dtype=brainstate.environ.dftype())
        np.add.at(mask, (np.repeat(np.arange(n_pre), out_degree), targets.reshape(-1)), 1.)
        return u.math.asarray(mask) * w

    raise ValueError(f"kind must be 'fcn', 'csr', 'dense', got {kind!r}.")


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

class EINet(brainstate.nn.Module):
    """Excitatory/inhibitory COBA network.

    Layout: neurons ``[0, n_exc)`` are excitatory, ``[n_exc, num)`` inhibitory.
    Both populations project onto *all* ``num`` neurons, so the two CSR matrices
    have shapes ``(n_exc, num)`` and ``(n_inh, num)``.
    """

    def __init__(self, scale: float = SCALE, conn_num: int = CONN_NUM, conn_kind: str = CONN_KIND):
        super().__init__()

        self.num = int(N_BASE * scale)
        self.n_exc = int(EXC_FRACTION * self.num)
        self.n_inh = self.num - self.n_exc
        self.conn_kind = conn_kind

        # Weights were tuned at CONN_NUM_BASE synapses per neuron; keep the total
        # synaptic drive constant when the out-degree changes.
        exc_weight = EXC_WEIGHT_BASE * CONN_NUM_BASE / conn_num
        inh_weight = INH_WEIGHT_BASE * CONN_NUM_BASE / conn_num

        self.exc_conn = make_connection(self.n_exc, self.num, conn_num, exc_weight,
                                        CONN_SEED, kind=conn_kind)
        self.inh_conn = make_connection(self.n_inh, self.num, conn_num, inh_weight,
                                        CONN_SEED + 1, kind=conn_kind)

        # Neurons.
        brainstate.random.seed(STATE_SEED)
        self.N = brainpy.state.LIFRef(
            self.num,
            V_rest=V_REST,
            V_th=V_TH,
            V_reset=V_RESET,
            tau=TAU_M,
            tau_ref=TAU_REF,
            V_initializer=braintools.init.Normal(V_INIT_MEAN, V_INIT_STD),
        )

        # Exponentially decaying conductances, one channel per population.
        self.exc_syn = brainpy.state.Expon(self.num, tau=TAU_SYN_EXC)
        self.inh_syn = brainpy.state.Expon(self.num, tau=TAU_SYN_INH)

        # Conductance -> current, through the reversal potentials.
        self.exc_out = brainpy.state.COBA(E=E_EXC)
        self.inh_out = brainpy.state.COBA(E=E_INH)
        self.N.add_current_input('exc_coba', self.exc_out)
        self.N.add_current_input('inh_coba', self.inh_out)

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.

            # The event-driven mat-vec. Wrapping the spike vector in a
            # `BinaryArray` is what tells brainevent to dispatch an event
            # kernel: only the rows of neurons that spiked this step are read,
            # and their weights are scattered onto the postsynaptic targets.
            # This line is identical for every CONN_KIND -- the format of
            # `exc_conn` picks the kernel.
            exc_input = brainevent.BinaryArray(spk[:self.n_exc]) @ self.exc_conn
            inh_input = brainevent.BinaryArray(spk[self.n_exc:]) @ self.inh_conn

            self.exc_out.bind_cond(self.exc_syn(exc_input))
            self.inh_out.bind_cond(self.inh_syn(inh_input))

            return self.N(inp)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(net: EINet, duration=DURATION, probe=PROBE_NEURONS):
    """Run the network for ``duration``; return spikes plus the probe traces.

    The whole time loop is compiled as one XLA program: ``for_loop`` scans over
    the time points, carrying the neuron/synapse states, and stacks whatever the
    step function returns.

    Spikes are recorded for the whole population (that is what the raster and
    the firing rate need). The analogue variables -- membrane potential and the
    two synaptic conductances -- are recorded only for the neurons in ``probe``,
    like sticking electrodes into a handful of cells: recording all of them
    would cost ``n_steps x num`` floats per variable.

    States are created *before* the JIT boundary so that ``net.N.V`` and the
    synaptic conductances still hold real values after the run.
    """
    brainstate.nn.init_all_states(net)
    times = u.math.arange(0. * u.ms, duration, DT)

    @brainstate.transform.jit
    def run():
        def step(t):
            spikes = net.update(t, I_OFFSET)
            return (spikes,
                    net.N.V.value[probe],
                    net.exc_syn.g.value[probe],
                    net.inh_syn.g.value[probe])

        return brainstate.transform.for_loop(step, times)

    spikes, V, g_exc, g_inh = run()
    return times, spikes, V, g_exc, g_inh


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_summary(net, times, spikes, V, g_exc, g_inh, path='coba_ei_summary.png'):
    """Four views of the same run."""
    t = np.asarray(times.to_decimal(u.ms))
    spikes = np.asarray(spikes) != 0.
    dt_s = DT.to_decimal(u.second)

    # Panels 3 and 4 follow one cell: the most active of the recorded neurons.
    col = int(spikes[:, PROBE_NEURONS].sum(0).argmax())
    cell = int(PROBE_NEURONS[col])
    V, g_exc, g_inh = V[:, col], g_exc[:, col], g_inh[:, col]
    cell_rate = spikes[:, cell].sum() / DURATION.to_decimal(u.second)

    # The two synaptic currents, as they enter dV/dt: R * g * (E - V).
    i_exc = (net.N.R * g_exc * (E_EXC - V)).to_decimal(u.mV)
    i_inh = (net.N.R * g_inh * (E_INH - V)).to_decimal(u.mV)
    i_ext = (net.N.R * I_OFFSET).to_decimal(u.mV)

    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    # 1. Raster: who spiked when.
    step, neuron = np.nonzero(spikes)
    ax[0].scatter(t[step], neuron, s=0.4, c='k', marker='.')
    ax[0].axhline(net.n_exc, color='tab:red', lw=0.8)
    ax[0].set_ylabel('neuron index')
    ax[0].set_title(f'raster ({net.num} neurons, red line = exc/inh boundary)')

    # 2. Population rate: the network view of the same spikes.
    bin_steps = max(1, int(round(1. * u.ms / DT)))            # 1 ms bins
    n_bins = spikes.shape[0] // bin_steps
    pop = spikes[:n_bins * bin_steps].reshape(n_bins, bin_steps, -1).sum((1, 2))
    ax[1].plot(t[:n_bins * bin_steps:bin_steps], pop / net.num / (bin_steps * dt_s), 'k-', lw=.9)
    ax[1].set_ylabel('population rate (Hz)')
    ax[1].set_title('population rate')

    # 3. Membrane potential of that one neuron.
    v_mv = V.to_decimal(u.mV)
    ax[2].plot(t, v_mv, 'k-', lw=.8)
    ax[2].plot(t[spikes[:, cell]], v_mv[spikes[:, cell]], '|', color='tab:orange',
               ms=10, mew=1.2, label='spike')
    ax[2].axhline(V_TH.to_decimal(u.mV), color='tab:red', ls='--', lw=.8, label='V_th')
    ax[2].axhline(V_REST.to_decimal(u.mV), color='tab:blue', ls=':', lw=.8, label='V_rest')
    ax[2].set_ylabel('V (mV)')
    ax[2].set_title(f'membrane potential of neuron {cell}')
    ax[2].legend(loc='upper right', fontsize=8)

    # 4. The synaptic currents driving that neuron. The point of the balanced
    #    state: excitation and inhibition are each far larger than the external
    #    drive, and nearly cancel. What is left fluctuates around threshold.
    ax[3].plot(t, i_exc, color='tab:red', lw=.7, label='R*g_exc*(E_exc - V)')
    ax[3].plot(t, i_inh, color='tab:blue', lw=.7, label='R*g_inh*(E_inh - V)')
    ax[3].plot(t, i_exc + i_inh + i_ext, color='k', lw=.9, label='total (incl. I_offset)')
    ax[3].axhline(0., color='0.6', lw=.5)
    ax[3].set_ylabel('contribution to dV/dt (mV)')
    ax[3].set_xlabel('time (ms)')
    ax[3].set_title('synaptic input to that neuron')
    ax[3].legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main(conn_kind: str = CONN_KIND):

    net = EINet(conn_kind=conn_kind)

    times, spikes, V, g_exc, g_inh = simulate(net)

    spk = np.asarray(spikes) != 0.
    rate = spk.sum() / net.num / DURATION.to_decimal(u.second)
    print(f'{net.num} neurons ({net.n_exc} exc / {net.n_inh} inh), '
          f'{CONN_NUM} synapses per neuron, conn_kind={conn_kind!r} '
          f'({type(net.exc_conn).__name__})')
    print(f'mean firing rate over {DURATION}: {rate:.2f} Hz')

    # Irregularity: the coefficient of variation of the inter-spike intervals,
    # pooled over neurons. ~1 means Poisson-like, ~0 means clock-like.
    dt_ms = DT.to_decimal(u.ms)
    isis = np.concatenate([np.diff(np.flatnonzero(spk[:, i])) * dt_ms
                           for i in range(net.num) if spk[:, i].sum() > 1])
    print(f'ISI CV = {isis.std() / isis.mean():.2f}   (1.0 = Poisson, 0 = regular)')

    print(f'summary figure saved to {plot_summary(net, times, spikes, V, g_exc, g_inh)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conn', default=CONN_KIND, choices=['fcn', 'csr', 'dense'],
                        help='connectivity storage format (default: %(default)s)')
    main(parser.parse_args().conn)
