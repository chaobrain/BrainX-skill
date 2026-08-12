"""Generate and perturb an alpha rhythm with a Jansen-Rit cortical column."""

from pathlib import Path

import braintools
import braintools.init
import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DT = 0.1 * u.ms
DURATION = 6000.0 * u.ms
TRANSIENT = 2000.0 * u.ms
SAMPLE_EVERY = 10
RECORDED_DT = DT * SAMPLE_EVERY
EXTERNAL_DRIVE = 220.0 * u.Hz

# Jansen-Rit synaptic time constants. The model takes their inverse rates.
TAU_EXCITATORY = 10.0 * u.ms
TAU_INHIBITORY = 20.0 * u.ms
EXCITATORY_GAIN = 3.25 * u.mV
INHIBITORY_GAINS = jnp.array([22.0, 17.6]) * u.mV
INITIAL_EXCITATORY_POTENTIALS = jnp.array([0.0, 0.2, 0.4]) * u.mV
CONDITION_NAMES = ("Resting baseline", "20% weaker inhibition")

ALPHA_BAND_HZ = (8.0, 13.0)
SEARCH_BAND_HZ = (1.0, 40.0)
RMS_FLOOR_MV = 1e-3
OUTPUT_PATH = Path(__file__).with_name("alpha_rhythm_comparison.png")


def simulate_one(inhibitory_gain, initial_excitatory_potential):
    """Run one unit-aware cortical column; vmap supplies independent instances."""
    column = brainmass.JansenRitStep(
        in_size=1,
        Ae=EXCITATORY_GAIN,
        Ai=inhibitory_gain,
        be=1.0 / TAU_EXCITATORY,
        bi=1.0 / TAU_INHIBITORY,
        C=135.0,
        s_max=5.0 * u.Hz,
        E_init=braintools.init.Constant(initial_excitatory_potential),
    )
    times = u.math.arange(0.0 * u.ms, DURATION, DT)
    step_indices = jnp.arange(times.shape[0])

    with brainstate.environ.context(dt=DT):
        column.init_all_states()

        def step(t, index):
            with brainstate.environ.context(t=t, i=index):
                column.update(E_inp=EXTERNAL_DRIVE)
            return column.eeg()

        # Model State carries all six Jansen-Rit dynamical variables through time.
        return brainstate.transform.for_loop(step, times, step_indices)[:, 0]


def run_experiment():
    """Map complete simulations over inhibition and initial-condition axes."""
    inhibitory_grid, initial_grid = u.math.meshgrid(
        INHIBITORY_GAINS,
        INITIAL_EXCITATORY_POTENTIALS,
        indexing="ij",
    )
    flat_eeg = brainstate.transform.vmap(simulate_one)(
        inhibitory_grid.reshape((-1,)),
        initial_grid.reshape((-1,)),
    )

    n_steps = int(DURATION / DT)
    transient_steps = int(TRANSIENT / DT)
    eeg = flat_eeg.reshape(inhibitory_grid.shape + (n_steps,))
    eeg = eeg[:, :, transient_steps::SAMPLE_EVERY].transpose(2, 0, 1)

    # Welch analysis and plotting are explicit raw-array boundaries in millivolts.
    eeg_mv = np.asarray(eeg.to_decimal(u.mV))
    eeg_mv = eeg_mv - eeg_mv.mean(axis=0, keepdims=True)
    return eeg, eeg_mv


def analyze(eeg_mv):
    """Return spectra and per-run summaries for [time, condition, initial]."""
    n_time, n_condition, n_initial = eeg_mv.shape
    frequencies, psd = braintools.metric.power_spectral_density(
        eeg_mv.reshape((n_time, -1)),
        RECORDED_DT,
        nperseg=2000,
    )
    frequencies = np.asarray(frequencies)
    psd = np.asarray(psd).reshape((frequencies.size, n_condition, n_initial))

    alpha_mask = (
        (frequencies >= ALPHA_BAND_HZ[0])
        & (frequencies <= ALPHA_BAND_HZ[1])
    )
    search_mask = (
        (frequencies >= SEARCH_BAND_HZ[0])
        & (frequencies <= SEARCH_BAND_HZ[1])
    )
    rms_mv = eeg_mv.std(axis=0)
    alpha_power = np.trapezoid(psd[alpha_mask], frequencies[alpha_mask], axis=0)
    search_power = np.trapezoid(psd[search_mask], frequencies[search_mask], axis=0)
    relative_alpha_power = alpha_power / search_power

    dominant_hz = np.full((n_condition, n_initial), np.nan)
    search_frequencies = frequencies[search_mask]
    for condition in range(n_condition):
        for initial in range(n_initial):
            if rms_mv[condition, initial] >= RMS_FLOOR_MV:
                peak = np.argmax(psd[search_mask, condition, initial])
                dominant_hz[condition, initial] = search_frequencies[peak]

    return {
        "frequencies": frequencies,
        "psd": psd,
        "rms_mv": rms_mv,
        "alpha_power": alpha_power,
        "relative_alpha_power": relative_alpha_power,
        "dominant_hz": dominant_hz,
    }


def print_summary(metrics):
    initial_mv = np.asarray(INITIAL_EXCITATORY_POTENTIALS.to_decimal(u.mV))
    gains_mv = np.asarray(INHIBITORY_GAINS.to_decimal(u.mV))
    print("\nJansen-Rit EEG proxy after the 2 s transient")
    print("condition                  Ai (mV)  init E (mV)  RMS (mV)   alpha power (mV^2)  peak")
    for condition, name in enumerate(CONDITION_NAMES):
        for initial, initial_value in enumerate(initial_mv):
            peak = metrics["dominant_hz"][condition, initial]
            peak_text = f"{peak:4.1f} Hz" if np.isfinite(peak) else "n/a*"
            print(
                f"{name:26s} {gains_mv[condition]:7.1f}"
                f" {initial_value:12.1f}"
                f" {metrics['rms_mv'][condition, initial]:10.4g}"
                f" {metrics['alpha_power'][condition, initial]:20.4g}"
                f"  {peak_text}"
            )
    print(f"* Peak is not reported below the predeclared {RMS_FLOOR_MV:g} mV RMS floor.")


def plot_comparison(eeg_mv, metrics):
    colors = ("#176B87", "#C44E52")
    initial_mv = np.asarray(INITIAL_EXCITATORY_POTENTIALS.to_decimal(u.mV))
    time_s = np.arange(eeg_mv.shape[0]) * RECORDED_DT.to_decimal(u.second)
    display = time_s <= 1.0

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    trace_ax, spectrum_ax, alpha_ax, rms_ax = axes.flat

    representative_initial = 1
    for condition, name in enumerate(CONDITION_NAMES):
        trace_ax.plot(
            time_s[display],
            eeg_mv[display, condition, representative_initial],
            color=colors[condition],
            linewidth=1.4,
            label=name,
        )
    trace_ax.set(title="Simulated EEG", xlabel="Time after transient (s)", ylabel="EEG proxy (mV)")
    trace_ax.legend(frameon=False)

    frequencies = metrics["frequencies"]
    spectrum_mask = (frequencies >= 1.0) & (frequencies <= 40.0)
    spectrum_ax.axvspan(*ALPHA_BAND_HZ, color="#D8D8D8", alpha=0.55, linewidth=0)
    for condition, name in enumerate(CONDITION_NAMES):
        spectrum_ax.semilogy(
            frequencies[spectrum_mask],
            metrics["psd"][spectrum_mask, condition].mean(axis=1),
            color=colors[condition],
            linewidth=1.8,
            label=name,
        )
    spectrum_ax.set(title="Power spectrum", xlabel="Frequency (Hz)", ylabel="PSD (mV^2/Hz)")
    spectrum_ax.legend(frameon=False)

    x_positions = np.arange(len(CONDITION_NAMES))
    offsets = np.linspace(-0.10, 0.10, initial_mv.size)
    for initial, (initial_value, offset) in enumerate(zip(initial_mv, offsets)):
        alpha_ax.scatter(
            x_positions + offset,
            metrics["alpha_power"][:, initial],
            s=36,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            label=f"E(0) = {initial_value:.1f} mV",
            zorder=3,
        )
        rms_ax.scatter(
            x_positions + offset,
            metrics["rms_mv"][:, initial],
            s=36,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    short_names = ("Baseline", "Weaker inhibition")
    for axis, title, ylabel in (
        (alpha_ax, "Alpha-band power", "8-13 Hz power (mV^2)"),
        (rms_ax, "Signal amplitude", "RMS (mV)"),
    ):
        axis.set_yscale("log")
        axis.set_xticks(x_positions, short_names)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
    alpha_ax.legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Weakening inhibition suppresses the resting alpha rhythm", fontsize=14)
    fig.savefig(OUTPUT_PATH, dpi=180)


def main():
    eeg, eeg_mv = run_experiment()
    metrics = analyze(eeg_mv)

    baseline_peaks = metrics["dominant_hz"][0]
    assert eeg.unit == u.mV
    assert np.isfinite(eeg_mv).all()
    assert np.all(
        (baseline_peaks >= ALPHA_BAND_HZ[0])
        & (baseline_peaks <= ALPHA_BAND_HZ[1])
    ), "The baseline did not settle into the alpha band."

    print_summary(metrics)
    plot_comparison(eeg_mv, metrics)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
