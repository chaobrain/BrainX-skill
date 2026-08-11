"""Spike-frequency adaptation and AHP-current ablation with BrainX.

The only difference between the adapted and ablated conditions is the maximal
conductance of the calcium-activated potassium (AHP) channel. Setting it to
zero removes that current without changing sodium, delayed-rectifier,
calcium, leak, stimulus, or integration parameters.
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
DURATION = 600.0 * u.ms
STIMULUS_ON = 50.0 * u.ms
STIMULUS_OFF = 550.0 * u.ms
EARLY_WINDOW = (50.0 * u.ms, 200.0 * u.ms)
LATE_WINDOW = (400.0 * u.ms, 550.0 * u.ms)


class AdaptingCell(braincell.SingleCompartment):
    """Classical HH cell plus an explicit Ca-activated K current."""

    def __init__(self, size, g_ahp, solver="exp_euler"):
        super().__init__(
            size,
            C=1.0 * u.uF / u.cm**2,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            V_th=20.0 * u.mV,
            solver=solver,
        )

        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        # Spike-driven Ca influx raises Ci, the slow signal for AHP activation.
        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5.0e-5 * u.mM,
            tau=80.0 * u.ms,
            d=0.5 * u.um,
        )
        self.ca.add(
            ICaL=braincell.channel.CaL_IS2008(
                size, g_max=5.0 * u.mS / u.cm**2
            )
        )

        # Potassium supplies E_K and calcium supplies the changing Ci.
        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(
            IAHP=braincell.channel.AHP_De1994(size, g_max=g_ahp)
        )

        self.IL = braincell.channel.IL(
            size,
            E=-54.387 * u.mV,
            g_max=0.03 * u.mS / u.cm**2,
        )


def build_parameter_grid(input_currents, adaptation_strengths):
    """Create a current x AHP-conductance grid with BrainState vmap."""

    def pair(current, g_ahp):
        return current, g_ahp

    across_strengths = brainstate.transform.vmap(
        pair,
        in_axes=(None, 0),
        out_axes=0,
    )
    across_currents = brainstate.transform.vmap(
        across_strengths,
        in_axes=(0, None),
        out_axes=0,
    )
    return across_currents(input_currents, adaptation_strengths)


def run_experiment():
    """Simulate the full parameter sweep and return unit-bearing traces."""

    input_currents = u.math.asarray([7.0, 10.0, 13.0]) * u.uA / u.cm**2
    adaptation_strengths = (
        u.math.asarray([0.0, 0.30, 1.00]) * u.mS / u.cm**2
    )
    current_grid, g_ahp_grid = build_parameter_grid(
        input_currents,
        adaptation_strengths,
    )

    cell = AdaptingCell(current_grid.shape, g_ahp=g_ahp_grid)

    with brainstate.environ.context(dt=DT):
        cell.init_state()
        times = u.math.arange(0.0 * u.ms, DURATION, brainstate.environ.get_dt())

        def step(t):
            injected_current = u.math.where(
                (t >= STIMULUS_ON) & (t < STIMULUS_OFF),
                current_grid,
                0.0 * u.uA / u.cm**2,
            )
            with brainstate.environ.context(t=t):
                spike = cell.update(injected_current)
            return cell.V.value, spike, cell.ca.Ci.value

        voltages, spikes, calcium = brainstate.transform.for_loop(step, times)

    return {
        "times": times,
        "input_currents": input_currents,
        "adaptation_strengths": adaptation_strengths,
        "voltages": voltages,
        "spikes": spikes,
        "calcium": calcium,
    }


def firing_rates(times, spikes):
    """Compute early and late rates for each sweep lane with nested vmap."""

    condition_first = u.math.moveaxis(spikes, 0, -1)

    def rates_for_one(spike_train):
        early_mask = (times >= EARLY_WINDOW[0]) & (times < EARLY_WINDOW[1])
        late_mask = (times >= LATE_WINDOW[0]) & (times < LATE_WINDOW[1])
        spike_events = spike_train > 0
        early_rate = u.math.sum(spike_events & early_mask) / (
            EARLY_WINDOW[1] - EARLY_WINDOW[0]
        )
        late_rate = u.math.sum(spike_events & late_mask) / (
            LATE_WINDOW[1] - LATE_WINDOW[0]
        )
        return early_rate, late_rate

    across_strengths = brainstate.transform.vmap(rates_for_one, in_axes=0)
    across_currents = brainstate.transform.vmap(across_strengths, in_axes=0)
    return across_currents(condition_first)


def spike_times_ms(data, current_index, adaptation_index):
    times_ms = data["times"].to_decimal(u.ms)
    spike_mask = np.asarray(
        data["spikes"][:, current_index, adaptation_index], dtype=bool
    )
    return np.asarray(times_ms)[spike_mask]


def validate_adaptation(data, early_rates, late_rates):
    """Assert the sweep contains the intended adaptation/ablation contrast."""

    selected_current = 1
    stimulus_bounds_ms = (
        STIMULUS_ON.to_decimal(u.ms),
        STIMULUS_OFF.to_decimal(u.ms),
    )

    def stimulus_isi(adaptation_index):
        spike_times = spike_times_ms(data, selected_current, adaptation_index)
        spike_times = spike_times[
            (spike_times >= stimulus_bounds_ms[0])
            & (spike_times < stimulus_bounds_ms[1])
        ]
        return np.diff(spike_times)

    removed_isi = stimulus_isi(0)
    intact_isi = stimulus_isi(-1)
    assert removed_isi.size > 10 and intact_isi.size > 10
    assert abs(removed_isi[-1] - removed_isi[0]) / removed_isi[0] < 0.10
    assert intact_isi[-1] / intact_isi[0] > 1.50

    late_hz = np.asarray(late_rates.to_decimal(u.Hz))
    assert late_hz[selected_current, -1] < 0.70 * late_hz[selected_current, 0]

    # These conversions also enforce the dimensional output contract.
    data["voltages"].in_unit(u.mV)
    data["calcium"].in_unit(u.uM)
    early_rates.in_unit(u.Hz)


def print_summary(data, early_rates, late_rates):
    currents = data["input_currents"].to_decimal(u.uA / u.cm**2)
    strengths = data["adaptation_strengths"].to_decimal(u.mS / u.cm**2)
    early_hz = np.asarray(early_rates.to_decimal(u.Hz))
    late_hz = np.asarray(late_rates.to_decimal(u.Hz))

    print("current (uA/cm^2) | g_AHP (mS/cm^2) | early Hz | late Hz")
    for current_i, current in enumerate(currents):
        for strength_i, strength in enumerate(strengths):
            print(
                f"{current:17.2f} | {strength:16.2f} | "
                f"{early_hz[current_i, strength_i]:8.1f} | "
                f"{late_hz[current_i, strength_i]:7.1f}"
            )

    selected_current = 1
    for strength_i, label in ((0, "AHP removed"), (-1, "AHP intact")):
        spike_times = spike_times_ms(data, selected_current, strength_i)
        stimulus_spikes = spike_times[
            (spike_times >= STIMULUS_ON.to_decimal(u.ms))
            & (spike_times < STIMULUS_OFF.to_decimal(u.ms))
        ]
        isi = np.diff(stimulus_spikes)
        if isi.size:
            print(
                f"{label}: {stimulus_spikes.size} spikes, "
                f"first ISI={isi[0]:.2f} ms, last ISI={isi[-1]:.2f} ms"
            )


def make_figure(data, early_rates, late_rates, output_path):
    times_ms = np.asarray(data["times"].to_decimal(u.ms))
    voltages_mv = np.asarray(data["voltages"].to_decimal(u.mV))
    calcium_um = np.asarray(data["calcium"].to_decimal(u.uM))
    currents = np.asarray(
        data["input_currents"].to_decimal(u.uA / u.cm**2)
    )
    strengths = np.asarray(
        data["adaptation_strengths"].to_decimal(u.mS / u.cm**2)
    )
    early_hz = np.asarray(early_rates.to_decimal(u.Hz))
    late_hz = np.asarray(late_rates.to_decimal(u.Hz))

    selected_current = 1
    removed = 0
    intact = len(strengths) - 1
    colors = {"removed": "#2878B5", "intact": "#C43C39"}

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))

    ax = axes[0, 0]
    ax.plot(
        times_ms,
        voltages_mv[:, selected_current, removed],
        color=colors["removed"],
        linewidth=0.8,
        label=r"AHP removed ($g_{AHP}=0$)",
    )
    ax.plot(
        times_ms,
        voltages_mv[:, selected_current, intact],
        color=colors["intact"],
        linewidth=0.8,
        label=rf"AHP intact ($g_{{AHP}}={strengths[intact]:.2f}$ mS/cm$^2$)",
    )
    ax.set_title(rf"Steady drive: {currents[selected_current]:.1f} $\mu$A/cm$^2$")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    for strength_i, label, color in (
        (removed, "AHP removed", colors["removed"]),
        (intact, "AHP intact", colors["intact"]),
    ):
        spike_times = spike_times_ms(data, selected_current, strength_i)
        stimulus_spikes = spike_times[
            (spike_times >= STIMULUS_ON.to_decimal(u.ms))
            & (spike_times < STIMULUS_OFF.to_decimal(u.ms))
        ]
        isi = np.diff(stimulus_spikes)
        ax.plot(np.arange(1, isi.size + 1), isi, "o-", ms=3, label=label, color=color)
    ax.set_title("Inter-spike intervals")
    ax.set_xlabel("Interval number")
    ax.set_ylabel("ISI (ms)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(
        times_ms,
        calcium_um[:, selected_current, removed],
        color=colors["removed"],
        linewidth=1.2,
        label="AHP removed",
    )
    ax.plot(
        times_ms,
        calcium_um[:, selected_current, intact],
        color=colors["intact"],
        linewidth=1.2,
        label="AHP intact",
    )
    ax.set_title(r"Spike-driven intracellular Ca$^{2+}$")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$[Ca^{2+}]_i$ ($\mu$M)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    x = np.arange(len(strengths))
    width = 0.12
    offsets = np.linspace(-width, width, len(currents))
    for current_i, (current, offset) in enumerate(zip(currents, offsets)):
        ax.plot(
            x + offset,
            early_hz[current_i],
            "o-",
            color="#555555",
            alpha=0.45 + 0.2 * current_i,
            linewidth=1.0,
        )
        ax.plot(
            x + offset,
            late_hz[current_i],
            "s--",
            label=rf"{current:.1f} $\mu$A/cm$^2$",
            linewidth=1.4,
        )
    ax.set_xticks(x, [f"{strength:.2f}" for strength in strengths])
    ax.set_title("Early (circles) versus late (squares) rate")
    ax.set_xlabel(r"$g_{AHP}$ (mS/cm$^2$)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15, linewidth=0.6)

    fig.suptitle("Calcium-activated potassium current causes spike-frequency adaptation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spike_frequency_adaptation.png"),
        help="Path for the generated figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure after saving it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = run_experiment()
    early_rates, late_rates = firing_rates(data["times"], data["spikes"])
    validate_adaptation(data, early_rates, late_rates)
    print_summary(data, early_rates, late_rates)
    fig = make_figure(data, early_rates, late_rates, args.output)
    print(f"saved figure: {args.output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
