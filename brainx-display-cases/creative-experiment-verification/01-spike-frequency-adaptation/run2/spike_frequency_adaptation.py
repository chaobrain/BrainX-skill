"""Demonstrate spike-frequency adaptation and controlled AHP ablation.

This teaching model combines classical Hodgkin-Huxley spiking with dynamic
calcium and the BrainCell AHP_De1994 calcium-activated potassium current. It is
not intended as a reproduction of a particular biological cell type.
"""

from pathlib import Path

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DT = 0.01 * u.ms
STIMULUS_ON = 50.0 * u.ms
STIMULUS_OFF = 550.0 * u.ms
DURATION = 600.0 * u.ms
CURRENT_LEVELS = u.math.asarray([8.0, 10.0, 12.0]) * u.uA / u.cm**2
AHP_STRENGTHS = u.math.asarray([0.0, 0.3, 1.0]) * u.mS / u.cm**2
HOLDING_CURRENT = -0.5 * u.uA / u.cm**2
FIGURE_PATH = Path(__file__).with_name("spike_frequency_adaptation.png")


class AdaptingCell(braincell.SingleCompartment):
    """Single-compartment cell with a removable Ca-activated K current."""

    def __init__(self, size, g_ahp, solver="exp_euler"):
        super().__init__(
            size,
            C=1.0 * u.uF / u.cm**2,
            V_initializer=braintools.init.Constant(-75.0 * u.mV),
            V_th=20.0 * u.mV,
            solver=solver,
        )

        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        # Calcium accumulation supplies the slow memory of recent firing.
        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5.0e-5 * u.mM,
            tau=80.0 * u.ms,
            d=0.5 * u.um,
        )
        self.ca.add(
            ICaL=braincell.channel.CaL_IS2008(
                size,
                g_max=5.0 * u.mS / u.cm**2,
            )
        )

        # Potassium supplies E_K; calcium controls AHP channel activation.
        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(
            IAHP=braincell.channel.AHP_De1994(size, g_max=g_ahp)
        )

        self.IL = braincell.channel.IL(
            size,
            E=-54.387 * u.mV,
            g_max=0.03 * u.mS / u.cm**2,
        )


def run_experiment():
    """Run every (AHP strength, input current) condition in one state graph."""
    condition_shape = (AHP_STRENGTHS.shape[0], CURRENT_LEVELS.shape[0])
    g_ahp_grid = u.math.broadcast_to(
        AHP_STRENGTHS[:, None], condition_shape
    )
    current_grid = u.math.broadcast_to(
        CURRENT_LEVELS[None, :], condition_shape
    )
    cell = AdaptingCell(size=condition_shape, g_ahp=g_ahp_grid)

    with brainstate.environ.context(dt=DT):
        cell.init_state()
        times = u.math.arange(0.0 * u.ms, DURATION, DT)

        def step(t):
            injected = u.math.where(
                (t >= STIMULUS_ON) & (t < STIMULUS_OFF),
                current_grid,
                HOLDING_CURRENT,
            )
            with brainstate.environ.context(t=t):
                spike = cell.update(injected)
            return cell.V.value, cell.ca.Ci.value, spike

        voltages, calcium, spikes = brainstate.transform.for_loop(step, times)

    return times, voltages, calcium, spikes


def summarize_spikes(spikes, times, dt_ms):
    """Use nested BrainState vmaps to summarize every condition's spike train."""
    stimulus_mask = (times >= STIMULUS_ON) & (times < STIMULUS_OFF)
    # Move time last so each mapped call receives one complete spike train.
    spike_trains = u.math.moveaxis(spikes[stimulus_mask], 0, -1)

    def one_trace(spike_trace):
        fired = spike_trace.astype(bool)
        indices = jnp.arange(fired.shape[0])
        count = jnp.sum(fired)

        first = jnp.min(jnp.where(fired, indices, fired.shape[0]))
        second = jnp.min(
            jnp.where(fired & (indices > first), indices, fired.shape[0])
        )
        last = jnp.max(jnp.where(fired, indices, -1))
        penultimate = jnp.max(jnp.where(fired & (indices < last), indices, -1))
        enough_spikes = count >= 2
        first_isi = jnp.where(enough_spikes, (second - first) * dt_ms, jnp.nan)
        last_isi = jnp.where(
            enough_spikes, (last - penultimate) * dt_ms, jnp.nan
        )
        ratio = last_isi / first_isi
        return jnp.stack((count, first_isi, last_isi, ratio))

    summarize_currents = brainstate.transform.vmap(
        one_trace, in_axes=0, out_axes=0
    )
    summarize_grid = brainstate.transform.vmap(
        summarize_currents, in_axes=0, out_axes=0
    )
    return summarize_grid(spike_trains)


def spike_times_for(spikes, times_ms, strength_index, current_index):
    fired = np.asarray(spikes[:, strength_index, current_index]).astype(bool)
    spike_times = times_ms[fired]
    return spike_times[
        (spike_times >= STIMULUS_ON.to_decimal(u.ms))
        & (spike_times < STIMULUS_OFF.to_decimal(u.ms))
    ]


def print_summary(metrics):
    print("\nSteady-input response (ISI ratio = last ISI / first ISI)")
    print("g_AHP (mS/cm^2)  I (uA/cm^2)  spikes  first ISI  last ISI  ratio")
    strengths = AHP_STRENGTHS.to_decimal(u.mS / u.cm**2)
    currents = CURRENT_LEVELS.to_decimal(u.uA / u.cm**2)
    metrics_np = np.asarray(metrics)
    for strength_index, strength in enumerate(strengths):
        for current_index, current in enumerate(currents):
            count, first_isi, last_isi, ratio = metrics_np[
                strength_index, current_index
            ]
            print(
                f"{strength:16.1f}  {current:12.1f}  {count:6.0f}  "
                f"{first_isi:9.2f}  {last_isi:8.2f}  {ratio:5.2f}"
            )


def validate_contrast(metrics):
    """Check the direct, matched-current AHP ablation contrast."""
    metrics_np = np.asarray(metrics)
    current_index = int(
        np.argmin(
            np.abs(CURRENT_LEVELS.to_decimal(u.uA / u.cm**2) - 10.0)
        )
    )
    removed_ratio = metrics_np[0, current_index, 3]
    present_ratio = metrics_np[-1, current_index, 3]
    assert removed_ratio < 1.10, (
        "The AHP-removed condition unexpectedly adapted strongly: "
        f"ISI ratio={removed_ratio:.2f}"
    )
    assert present_ratio > 1.50, (
        "The AHP-present condition did not adapt strongly enough: "
        f"ISI ratio={present_ratio:.2f}"
    )
    assert present_ratio > removed_ratio + 0.50


def make_figure(
    times, voltages, calcium, spikes, metrics, output_path=FIGURE_PATH
):
    times_ms = np.asarray(times.to_decimal(u.ms))
    current_index = int(
        np.argmin(
            np.abs(CURRENT_LEVELS.to_decimal(u.uA / u.cm**2) - 10.0)
        )
    )
    labels = ((0, "AHP removed"), (-1, "AHP present"))
    colors = ("#D14B3F", "#26706A")

    fig = plt.figure(figsize=(12.0, 7.2))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0))
    voltage_axis = fig.add_subplot(grid[0, :])
    isi_axis = fig.add_subplot(grid[1, 0])
    calcium_axis = fig.add_subplot(grid[1, 1])
    sweep_axis = fig.add_subplot(grid[1, 2])

    for (strength_index, label), color in zip(labels, colors):
        voltage_axis.plot(
            times_ms,
            np.asarray(
                voltages[:, strength_index, current_index].to_decimal(u.mV)
            ),
            color=color,
            linewidth=0.85,
            label=label,
        )
        spike_times = spike_times_for(
            spikes, times_ms, strength_index, current_index
        )
        isis = np.diff(spike_times)
        isi_axis.plot(
            np.arange(1, isis.size + 1),
            isis,
            "o-",
            color=color,
            linewidth=1.4,
            markersize=3.5,
            label=label,
        )

    voltage_axis.axvspan(
        STIMULUS_ON.to_decimal(u.ms),
        STIMULUS_OFF.to_decimal(u.ms),
        color="#D9D5CB",
        alpha=0.35,
        linewidth=0,
    )
    voltage_axis.set(
        xlim=(0.0, DURATION.to_decimal(u.ms)),
        xlabel="Time (ms)",
        ylabel="Membrane voltage (mV)",
        title="Matched 10 uA/cm^2 steady input",
    )
    isi_axis.set(
        xlabel="Interspike interval index",
        ylabel="Interspike interval (ms)",
        title="AHP current progressively lengthens intervals",
    )
    voltage_axis.legend(frameon=False, ncol=2)
    isi_axis.legend(frameon=False)

    calcium_axis.plot(
        times_ms,
        np.asarray(calcium[:, -1, current_index].to_decimal(u.uM)),
        color=colors[1],
        linewidth=1.5,
    )
    calcium_axis.axvspan(
        STIMULUS_ON.to_decimal(u.ms),
        STIMULUS_OFF.to_decimal(u.ms),
        color="#D9D5CB",
        alpha=0.35,
        linewidth=0,
    )
    calcium_axis.set(
        xlim=(0.0, DURATION.to_decimal(u.ms)),
        xlabel="Time (ms)",
        ylabel="Intracellular Ca (uM)",
        title="Slow calcium signal gates AHP",
    )

    adaptation_ratios = np.asarray(metrics)[:, :, 3]
    image = sweep_axis.imshow(
        adaptation_ratios,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=1.0,
    )
    sweep_axis.set(
        xticks=np.arange(CURRENT_LEVELS.shape[0]),
        xticklabels=[
            f"{value:g}"
            for value in CURRENT_LEVELS.to_decimal(u.uA / u.cm**2)
        ],
        yticks=np.arange(AHP_STRENGTHS.shape[0]),
        yticklabels=[
            f"{value:g}"
            for value in AHP_STRENGTHS.to_decimal(u.mS / u.cm**2)
        ],
        xlabel="Input current (uA/cm^2)",
        ylabel="g_AHP (mS/cm^2)",
        title="Adaptation ratio across conditions",
    )
    for row in range(adaptation_ratios.shape[0]):
        for column in range(adaptation_ratios.shape[1]):
            sweep_axis.text(
                column,
                row,
                f"{adaptation_ratios[row, column]:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if adaptation_ratios[row, column]
                    > np.nanmean(adaptation_ratios)
                    else "black"
                ),
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=sweep_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Last ISI / first ISI")

    fig.suptitle("Spike-frequency adaptation from a removable Ca-activated K current")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    times, voltages, calcium, spikes = run_experiment()
    metrics = summarize_spikes(spikes, times, DT.to_decimal(u.ms))
    print_summary(metrics)
    validate_contrast(metrics)
    make_figure(times, voltages, calcium, spikes, metrics)
    print(f"\nSaved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
