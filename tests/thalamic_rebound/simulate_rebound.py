"""Demonstrate T-type-calcium-dependent rebound bursting in a relay neuron."""

from pathlib import Path

import braincell
import brainstate
import braintools
import brainunit as u
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DT = 0.02 * u.ms
T_STOP = 500.0 * u.ms
PULSE_START = 100.0 * u.ms
PULSE_END = 300.0 * u.ms
PULSE_CURRENT = -0.50 * u.uA / u.cm**2
ZERO_CURRENT = 0.0 * u.uA / u.cm**2
REBOUND_WINDOW = 100.0 * u.ms


class ThalamicRelayNeuron(braincell.SingleCompartment):
    """Single-compartment relay cell with optional low-threshold T current."""

    def __init__(self, with_t_channel: bool):
        super().__init__(
            1,
            C=1.0 * u.uF / u.cm**2,
            V_th=0.0 * u.mV,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            solver="exp_euler",
        )

        self.na = braincell.ion.SodiumFixed(1, E=50.0 * u.mV)
        self.na.add(
            INa=braincell.channel.Na_Ba2002(
                1,
                g_max=90.0 * u.mS / u.cm**2,
            )
        )

        self.k = braincell.ion.PotassiumFixed(1, E=-90.0 * u.mV)
        self.k.add(
            IKL=braincell.channel.K_Leak(
                1,
                g_max=0.01 * u.mS / u.cm**2,
            )
        )
        self.k.add(
            IK=braincell.channel.KDR_Ba2002(
                1,
                g_max=10.0 * u.mS / u.cm**2,
            )
        )

        if with_t_channel:
            self.ca = braincell.ion.CalciumFixed(1, E=120.0 * u.mV)
            self.ca.add(
                IT=braincell.channel.CaT_HM1992(
                    1,
                    g_max=2.0 * u.mS / u.cm**2,
                )
            )

        self.IL = braincell.channel.IL(
            1,
            E=-70.0 * u.mV,
            g_max=0.04 * u.mS / u.cm**2,
        )


def simulate(with_t_channel: bool) -> dict[str, np.ndarray]:
    """Run one current-clamp simulation and return unit-labeled raw arrays."""
    neuron = ThalamicRelayNeuron(with_t_channel)

    with brainstate.environ.context(dt=DT):
        neuron.init_state()
        times = u.math.arange(0.0 * u.ms, T_STOP, brainstate.environ.get_dt())
        current = u.math.where(
            (times >= PULSE_START) & (times < PULSE_END),
            PULSE_CURRENT,
            ZERO_CURRENT,
        )

        def step(t, i_ext):
            with brainstate.environ.context(t=t):
                spike = neuron.update(i_ext)
            t_inactivation = (
                neuron.ca.IT.q.value
                if with_t_channel
                else u.math.zeros(neuron.varshape)
            )
            return neuron.V.value, spike, t_inactivation

        voltage, spikes, t_inactivation = brainstate.transform.for_loop(
            step,
            times,
            current,
        )

    return {
        "time_ms": np.asarray(times.to_decimal(u.ms)),
        "current_uA_cm2": np.asarray(
            current.to_decimal(u.uA / u.cm**2)
        ),
        "voltage_mV": np.asarray(voltage.to_decimal(u.mV)).squeeze(),
        "spikes": np.asarray(spikes).squeeze().astype(bool),
        "t_inactivation": np.asarray(t_inactivation).squeeze(),
    }


def rebound_spike_times(result: dict[str, np.ndarray]) -> np.ndarray:
    """Return spike times in the first 100 ms following pulse release."""
    release_ms = float(PULSE_END.to_decimal(u.ms))
    window_end_ms = float((PULSE_END + REBOUND_WINDOW).to_decimal(u.ms))
    mask = (
        (result["time_ms"] >= release_ms)
        & (result["time_ms"] < window_end_ms)
        & result["spikes"]
    )
    return result["time_ms"][mask]


def validate_results(
    with_t: dict[str, np.ndarray],
    without_t: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Check that the simulation expresses the requested causal comparison."""
    release_ms = float(PULSE_END.to_decimal(u.ms))
    pre_pulse = with_t["time_ms"] < float(PULSE_START.to_decimal(u.ms))
    after_release = with_t["time_ms"] >= release_ms
    with_t_rebound = rebound_spike_times(with_t)
    without_t_rebound = rebound_spike_times(without_t)

    assert not with_t["spikes"][pre_pulse].any()
    assert not without_t["spikes"][pre_pulse].any()
    assert len(with_t_rebound) >= 2, "T-intact neuron did not rebound-burst."
    assert len(without_t_rebound) == 0, "T-channel ablation still spiked."
    assert np.all(with_t["current_uA_cm2"][after_release] == 0.0)
    assert np.isfinite(with_t["voltage_mV"]).all()
    assert np.isfinite(without_t["voltage_mV"]).all()
    return with_t_rebound, without_t_rebound


def plot_results(
    with_t: dict[str, np.ndarray],
    without_t: dict[str, np.ndarray],
    with_t_rebound: np.ndarray,
    without_t_rebound: np.ndarray,
    output_path: Path,
) -> None:
    """Create an aligned comparison figure."""
    colors = {
        "current": "#0E7C86",
        "with_t": "#C53A46",
        "without_t": "#343A40",
        "gate": "#2866A6",
        "pulse": "#DCEFF1",
        "grid": "#D9DEE2",
        "muted": "#64717B",
    }
    release_ms = float(PULSE_END.to_decimal(u.ms))
    pulse_start_ms = float(PULSE_START.to_decimal(u.ms))
    rebound_end_ms = float((PULSE_END + REBOUND_WINDOW).to_decimal(u.ms))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.0, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 2.0, 2.0, 1.0], "hspace": 0.18},
    )
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.11, right=0.97)
    fig.suptitle(
        "Post-inhibitory rebound requires the T-type calcium channel",
        x=0.11,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#202529",
    )
    fig.text(
        0.11,
        0.925,
        r"Same $-0.50\ \mu$A cm$^{-2}$ pulse in both cells; current returns to zero at 300 ms",
        ha="left",
        fontsize=10.5,
        color=colors["muted"],
    )

    for ax in axes:
        ax.axvspan(
            pulse_start_ms,
            release_ms,
            color=colors["pulse"],
            alpha=0.75,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(
            release_ms,
            color=colors["muted"],
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
        ax.grid(axis="y", color=colors["grid"], linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#87919A")
        ax.spines["bottom"].set_color("#87919A")
        ax.tick_params(colors="#4C555C", length=3)

    t = with_t["time_ms"]
    axes[0].plot(
        t,
        with_t["current_uA_cm2"],
        color=colors["current"],
        linewidth=2.0,
    )
    axes[0].set_ylabel(r"$I_{\mathrm{ext}}$" + "\n" + r"($\mu$A cm$^{-2}$)")
    axes[0].set_ylim(-0.62, 0.13)
    axes[0].set_yticks([-0.5, 0.0])
    axes[0].annotate(
        "release",
        xy=(release_ms, 0.0),
        xytext=(release_ms + 24.0, -0.35),
        arrowprops={"arrowstyle": "-|>", "color": colors["muted"], "lw": 0.9},
        color=colors["muted"],
        fontsize=9,
    )

    axes[1].plot(
        t,
        with_t["voltage_mV"],
        color=colors["with_t"],
        linewidth=1.35,
    )
    axes[1].set_ylabel("Membrane\npotential (mV)")
    axes[1].set_ylim(-105, 62)
    axes[1].set_yticks([-100, -50, 0, 50])
    axes[1].set_title(
        "T-type channel intact",
        loc="left",
        fontweight="bold",
        color=colors["with_t"],
        pad=4,
    )
    axes[1].text(
        0.985,
        0.89,
        f"{len(with_t_rebound)} rebound spikes",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color=colors["with_t"],
        fontweight="bold",
    )
    for spike_time in with_t_rebound:
        axes[1].plot(
            spike_time,
            55,
            marker="v",
            markersize=4.5,
            color=colors["with_t"],
            clip_on=False,
        )

    axes[2].plot(
        t,
        without_t["voltage_mV"],
        color=colors["without_t"],
        linewidth=1.35,
    )
    axes[2].set_ylabel("Membrane\npotential (mV)")
    axes[2].set_ylim(-105, 62)
    axes[2].set_yticks([-100, -50, 0, 50])
    axes[2].set_title(
        "T-type channel removed",
        loc="left",
        fontweight="bold",
        color=colors["without_t"],
        pad=4,
    )
    axes[2].text(
        0.985,
        0.89,
        f"{len(without_t_rebound)} rebound spikes",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        color=colors["without_t"],
        fontweight="bold",
    )

    axes[3].plot(
        t,
        with_t["t_inactivation"],
        color=colors["gate"],
        linewidth=1.8,
    )
    axes[3].fill_between(
        t,
        0.0,
        with_t["t_inactivation"],
        color=colors["gate"],
        alpha=0.12,
        linewidth=0,
    )
    axes[3].set_ylabel(r"$I_T$ gate $q$")
    axes[3].set_ylim(-0.02, 0.30)
    axes[3].set_yticks([0.0, 0.1, 0.2, 0.3])
    axes[3].set_xlabel("Time (ms)")
    axes[3].annotate(
        "T channels become available",
        xy=(release_ms - 2.0, with_t["t_inactivation"][np.searchsorted(t, release_ms) - 1]),
        xytext=(175.0, 0.265),
        arrowprops={"arrowstyle": "-|>", "color": colors["gate"], "lw": 0.9},
        color=colors["gate"],
        fontsize=9,
    )

    axes[3].set_xlim(0.0, float(T_STOP.to_decimal(u.ms)))
    axes[3].set_xticks(np.arange(0.0, 501.0, 50.0))
    axes[3].axvspan(
        release_ms,
        rebound_end_ms,
        color=colors["with_t"],
        alpha=0.045,
        linewidth=0,
        zorder=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    output_path = Path(__file__).resolve().parent / "thalamic_rebound_comparison.png"
    with_t = simulate(with_t_channel=True)
    without_t = simulate(with_t_channel=False)
    with_t_rebound, without_t_rebound = validate_results(with_t, without_t)
    plot_results(
        with_t,
        without_t,
        with_t_rebound,
        without_t_rebound,
        output_path,
    )

    with_times = ", ".join(f"{time:.2f}" for time in with_t_rebound)
    print(f"T-type intact: {len(with_t_rebound)} rebound spikes at {with_times} ms")
    print(f"T-type removed: {len(without_t_rebound)} rebound spikes")
    print(f"Figure: {output_path}")


if __name__ == "__main__":
    main()
