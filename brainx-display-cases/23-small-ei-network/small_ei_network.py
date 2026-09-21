"""A compact recurrent 16-excitatory/4-inhibitory BrainX network."""

from __future__ import annotations

import brainpy
import brainstate
import braintools
import brainunit as u


N_EXC = 16
N_INH = 4
N_NEURONS = N_EXC + N_INH
DT = 0.1 * u.ms
DURATION = 100.0 * u.ms
EXTERNAL_CURRENT = 20.0 * u.mA
CONNECTION_PROBABILITY = 0.25
SEED = 1234


class EINetwork(brainstate.nn.Module):
    """Recurrent conductance-based E/I network with one shared neuron array."""

    def __init__(
        self,
        n_exc: int = N_EXC,
        n_inh: int = N_INH,
        connection_probability: float = CONNECTION_PROBABILITY,
    ):
        super().__init__()
        if n_exc <= 0 or n_inh <= 0:
            raise ValueError("both populations must contain at least one neuron")

        self.n_exc = int(n_exc)
        self.n_inh = int(n_inh)
        self.num_neurons = self.n_exc + self.n_inh

        self.neurons = brainpy.state.LIFRef(
            self.num_neurons,
            R=1.0 * u.ohm,
            tau=20.0 * u.ms,
            tau_ref=5.0 * u.ms,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        self.exc_projection = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                self.n_exc,
                self.num_neurons,
                connection_probability,
                0.6 * u.mS,
            ),
            syn=brainpy.state.Expon.desc(self.num_neurons, tau=5.0 * u.ms),
            out=brainpy.state.COBA.desc(E=0.0 * u.mV),
            post=self.neurons,
        )
        self.inh_projection = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                self.n_inh,
                self.num_neurons,
                connection_probability,
                6.7 * u.mS,
            ),
            syn=brainpy.state.Expon.desc(self.num_neurons, tau=10.0 * u.ms),
            out=brainpy.state.COBA.desc(E=-80.0 * u.mV),
            post=self.neurons,
        )

    def update(self, t, drive: u.Quantity):
        """Advance one step: communicate spikes, then integrate the neurons."""
        with brainstate.environ.context(t=t):
            previous_spikes = self.neurons.get_spike() != 0.0
            self.exc_projection(previous_spikes[: self.n_exc])
            self.inh_projection(previous_spikes[self.n_exc :])
            return self.neurons(drive)


def run(
    duration: u.Quantity = DURATION,
    drive: u.Quantity = EXTERNAL_CURRENT,
):
    """Simulate the network and return `(times, spikes)`."""
    with brainstate.environ.context(dt=DT):
        brainstate.random.seed(SEED)
        net = EINetwork()
        brainstate.nn.init_all_states(net)
        times = u.math.arange(0.0 * u.ms, duration, brainstate.environ.get_dt())
        spikes = brainstate.transform.for_loop(
            lambda t: net.update(t, drive),
            times,
        )
    return times, spikes


def plot_raster(times, spikes, output_path=None):
    """Render the recorded spikes and optionally persist the figure."""
    import matplotlib.pyplot as plt
    import braintools.visualize as btvis

    time_indices, neuron_indices = u.math.where(spikes)
    spike_times = times[time_indices].to_decimal(u.ms)
    neuron_indices = neuron_indices.astype(int)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    btvis.spike_raster(
        spike_times,
        neuron_indices,
        time_range=(0.0, float(times[-1].to_decimal(u.ms))),
        neuron_range=(0, N_NEURONS - 1),
        ax=ax,
        markersize=3.0,
        xlabel="Time (ms)",
        ylabel="Neuron index (E: 0–15, I: 16–19)",
        title="20-neuron recurrent E/I network",
    )
    ax.axhline(N_EXC - 0.5, color="0.7", linewidth=0.8, linestyle="--")
    fig.tight_layout()

    if output_path is not None:
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    times, spikes = run()
    plot_raster(times, spikes, "outputs/small_ei_network_raster.png")
