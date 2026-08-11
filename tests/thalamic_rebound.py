"""Post-inhibitory rebound in a thalamic relay neuron.

The script compares the same single-compartment relay-cell model with its
T-type calcium conductance intact and ablated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import braincell
import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np


DT = 0.01 * u.ms
DURATION = 500.0 * u.ms
RELEASE_TIME = 200.0 * u.ms
INHIBITORY_CURRENT = -2.0 * u.uA / u.cm**2
ZERO_CURRENT = 0.0 * u.uA / u.cm**2
T_CONDUCTANCE = 2.1 * u.mS / u.cm**2


class ThalamicRelayCell(braincell.SingleCompartment):
    """Single-compartment relay cell based on Huguenard-McCormick currents."""

    def __init__(self, g_cat, size: int = 1, solver: str = "ind_exp_euler"):
        super().__init__(
            size,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            V_th=20.0 * u.mV,
            solver=solver,
        )

        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(
            INa=braincell.channel.Na_Ba2002(size, V_sh=-30.0 * u.mV)
        )

        self.k = braincell.ion.PotassiumFixed(size, E=-90.0 * u.mV)
        self.k.add(
            IKL=braincell.channel.K_Leak(
                size, g_max=0.01 * u.mS / u.cm**2
            )
        )
        self.k.add(
            IDR=braincell.channel.KDR_Ba2002(
                size,
                V_sh=-30.0 * u.mV,
                q10=2.0,
                temp=u.celsius2kelvin(16.0),
            )
        )

        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5.0e-5 * u.mM,
            tau=10.0 * u.ms,
            d=0.5 * u.um,
        )
        self.ca.add(
            ICaT=braincell.channel.CaT_HM1992(size, g_max=g_cat)
        )
        self.ca.add(
            ICaHT=braincell.channel.CaHT_HM1992(
                size, g_max=3.0 * u.mS / u.cm**2
            )
        )

        self.Ih = braincell.channel.HCN_HM1992(
            size,
            g_max=0.01 * u.mS / u.cm**2,
            E=-43.0 * u.mV,
        )
        self.IL = braincell.channel.IL(
            size,
            g_max=0.0075 * u.mS / u.cm**2,
            E=-70.0 * u.mV,
        )


def injected_current(t):
    """Hyperpolarize until RELEASE_TIME, then release the current to zero."""
    return u.math.where(
        t < RELEASE_TIME,
        INHIBITORY_CURRENT,
        ZERO_CURRENT,
    )


def run_comparison():
    intact = ThalamicRelayCell(g_cat=T_CONDUCTANCE)
    no_t = ThalamicRelayCell(g_cat=0.0 * u.mS / u.cm**2)
    intact.init_state()
    no_t.init_state()

    def step(t):
        current = injected_current(t)
        with brainstate.environ.context(t=t):
            intact.update(current)
            no_t.update(current)
        return (
            intact.V.value,
            intact.spike.value,
            no_t.V.value,
            no_t.spike.value,
            current,
        )

    with brainstate.environ.context(dt=DT):
        times = u.math.arange(
            0.0 * u.ms,
            DURATION,
            brainstate.environ.get_dt(),
        )
        v_intact, s_intact, v_no_t, s_no_t, currents = (
            brainstate.transform.for_loop(step, times)
        )

    return times, currents, v_intact, s_intact, v_no_t, s_no_t


def spike_times_ms(times, spikes):
    spike_mask = u.math.squeeze(spikes) > 0
    return np.asarray(times[spike_mask].to_decimal(u.ms))


def summarize(times, s_intact, s_no_t):
    release_ms = float(RELEASE_TIME.to_decimal(u.ms))
    intact_spikes = spike_times_ms(times, s_intact)
    no_t_spikes = spike_times_ms(times, s_no_t)
    intact_rebound = intact_spikes[intact_spikes > release_ms]
    no_t_rebound = no_t_spikes[no_t_spikes > release_ms]

    print(f"Release from inhibition: {release_ms:.1f} ms")
    print(
        "T-type intact rebound spikes:",
        np.round(intact_rebound, 2).tolist(),
    )
    print(
        "T-type removed rebound spikes:",
        np.round(no_t_rebound, 2).tolist(),
    )
    if intact_rebound.size:
        print(
            "Intact first-spike latency:",
            f"{intact_rebound[0] - release_ms:.2f} ms",
        )

    return intact_rebound, no_t_rebound


def make_figure(
    times,
    currents,
    v_intact,
    v_no_t,
    intact_rebound,
    no_t_rebound,
    output_path: Path,
    show: bool,
):
    time_ms = np.asarray(times.to_decimal(u.ms))
    current_density = np.asarray(currents.to_decimal(u.uA / u.cm**2))
    intact_mv = np.asarray(
        u.math.squeeze(v_intact).to_decimal(u.mV)
    )
    no_t_mv = np.asarray(u.math.squeeze(v_no_t).to_decimal(u.mV))
    release_ms = float(RELEASE_TIME.to_decimal(u.ms))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 2.4, 2.4]},
    )

    axes[0].plot(time_ms, current_density, color="#30343b", linewidth=1.8)
    axes[0].fill_between(
        time_ms,
        current_density,
        0.0,
        where=current_density < 0.0,
        color="#8b5e3c",
        alpha=0.22,
    )
    axes[0].set_ylabel(r"$I_{\mathrm{ext}}$" + "\n" + r"($\mu$A/cm$^2$)")
    axes[0].set_ylim(-2.5, 0.5)

    axes[1].plot(time_ms, intact_mv, color="#007f73", linewidth=1.15)
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].set_title(
        f"T-type Ca$^{{2+}}$ intact: {len(intact_rebound)} rebound spikes",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    axes[2].plot(time_ms, no_t_mv, color="#b5473e", linewidth=1.15)
    axes[2].set_ylabel("Voltage (mV)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title(
        f"T-type Ca$^{{2+}}$ removed: {len(no_t_rebound)} rebound spikes",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_ylim(-150.0, 65.0)
    axes[2].set_ylim(-150.0, 65.0)

    for ax in axes:
        ax.axvline(
            release_ms,
            color="#111111",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
        ax.grid(axis="y", alpha=0.25)

    axes[0].annotate(
        "release to 0",
        xy=(release_ms, 0.0),
        xytext=(release_ms + 13.0, -1.05),
        arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 0.9},
        fontsize=10,
    )
    fig.suptitle(
        "Post-inhibitory rebound in a thalamic relay neuron",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare thalamic rebound with and without T-type calcium."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("thalamic_rebound_comparison.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    args = parser.parse_args()

    results = run_comparison()
    times, currents, v_intact, s_intact, v_no_t, s_no_t = results
    intact_rebound, no_t_rebound = summarize(times, s_intact, s_no_t)
    make_figure(
        times,
        currents,
        v_intact,
        v_no_t,
        intact_rebound,
        no_t_rebound,
        args.output,
        args.show,
    )


if __name__ == "__main__":
    main()
