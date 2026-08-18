"""Generate alpha-like EEG from a Jansen-Rit cortical column.

The intervention changes only the inhibitory postsynaptic gain ``Ai``. Three
matched initial conditions are simulated for each inhibition strength.
"""

from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


DT = 0.1 * u.ms
DURATION = 2500.0 * u.ms
TRANSIENT = 500.0 * u.ms
EXTERNAL_DRIVE = 220.0 * u.Hz
INHIBITORY_GAINS = jnp.asarray([22.0, 15.4]) * u.mV
INITIAL_OFFSETS = jnp.asarray([-0.5, 0.0, 0.5]) * u.mV
CONDITION_NAMES = ("Resting baseline", "Weakened inhibition")
COLORS = ("#176B5B", "#C43D4F")


def simulate_one(inhibitory_gain, initial_offset):
    """Run one stateful column for one gain and one initial condition."""
    column = brainmass.JansenRitStep(
        in_size=1,
        Ae=3.25 * u.mV,
        Ai=inhibitory_gain,
        be=100.0 * u.Hz,
        bi=50.0 * u.Hz,
        C=135.0,
        M_init=braintools.init.Constant(initial_offset),
        E_init=braintools.init.Constant(0.5 * initial_offset),
        I_init=braintools.init.Constant(-0.5 * initial_offset),
    )

    with brainstate.environ.context(dt=DT):
        brainstate.nn.init_all_states(column)
        times = u.math.arange(0.0 * u.ms, DURATION, DT)

        def step(t):
            with brainstate.environ.context(t=t):
                column.update(0.0 * u.mV, EXTERNAL_DRIVE, 0.0 * u.mV)
                return column.eeg()[0]

        return brainstate.transform.for_loop(step, times)


def simulate_conditions():
    """Map the complete rollout over gain/initial-condition pairs."""
    gain_grid = u.math.repeat(INHIBITORY_GAINS, INITIAL_OFFSETS.size)
    initial_grid = u.math.tile(INITIAL_OFFSETS, INHIBITORY_GAINS.size)
    mapped_rollout = brainstate.transform.vmap(
        simulate_one,
        in_axes=(0, 0),
        out_axes=0,
    )
    traces = mapped_rollout(gain_grid, initial_grid)
    return traces.reshape(
        (INHIBITORY_GAINS.size, INITIAL_OFFSETS.size, -1)
    )


def power_spectrum(signal, sample_rate_hz):
    """Return a one-sided windowed periodogram at the plotting boundary."""
    centered = signal - signal.mean()
    window = np.hanning(centered.size)
    spectrum = np.fft.rfft(centered * window)
    power = np.abs(spectrum) ** 2 / (sample_rate_hz * np.sum(window**2))
    if power.size > 2:
        power[1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    return frequencies, power


def summarize(eeg_mv, sample_rate_hz):
    """Compute phase-insensitive metrics for every matched simulation."""
    rows = []
    spectra = []
    for condition_index, name in enumerate(CONDITION_NAMES):
        condition_spectra = []
        for initial_index, trace in enumerate(eeg_mv[condition_index]):
            frequencies, power = power_spectrum(trace, sample_rate_hz)
            peak_band = (frequencies >= 4.0) & (frequencies <= 20.0)
            alpha_band = (frequencies >= 8.0) & (frequencies <= 13.0)
            peak_hz = frequencies[peak_band][np.argmax(power[peak_band])]
            alpha_power = np.trapezoid(power[alpha_band], frequencies[alpha_band])
            rows.append(
                (
                    name,
                    float(INITIAL_OFFSETS.to_decimal(u.mV)[initial_index]),
                    peak_hz,
                    float(trace.mean()),
                    float(np.sqrt(np.mean((trace - trace.mean()) ** 2))),
                    alpha_power,
                )
            )
            condition_spectra.append(power)
        spectra.append(np.asarray(condition_spectra))
    return frequencies, np.asarray(spectra), rows


def plot_results(times_s, eeg_mv, frequencies, spectra, output_path):
    """Plot matched traces and ensemble spectra."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 6.6),
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )
    display = (times_s >= 1.0) & (times_s <= 2.0)
    spectral_band = frequencies <= 40.0

    for row, (name, color) in enumerate(zip(CONDITION_NAMES, COLORS)):
        trace_ax, spectrum_ax = axes[row]
        for initial_index, initial_mv in enumerate(INITIAL_OFFSETS.to_decimal(u.mV)):
            trace_ax.plot(
                times_s[display],
                eeg_mv[row, initial_index, display],
                color=color,
                alpha=0.38 if initial_index != 1 else 1.0,
                linewidth=0.8 if initial_index != 1 else 1.25,
                label=f"initial M = {initial_mv:+.1f} mV",
            )
        trace_ax.set_title(name)
        trace_ax.set_xlabel("Time (s)")
        trace_ax.set_ylabel("EEG proxy E - I (mV)")
        trace_ax.grid(alpha=0.18)
        trace_ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")

        mean_power = spectra[row].mean(axis=0)
        low_power = spectra[row].min(axis=0)
        high_power = spectra[row].max(axis=0)
        spectrum_ax.axvspan(8.0, 13.0, color="#D8A31A", alpha=0.16, label="alpha band")
        spectrum_ax.fill_between(
            frequencies[spectral_band],
            low_power[spectral_band],
            high_power[spectral_band],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        spectrum_ax.plot(
            frequencies[spectral_band],
            mean_power[spectral_band],
            color=color,
            linewidth=1.5,
            label="mean spectrum",
        )
        spectrum_ax.set_yscale("log")
        spectrum_ax.set_title(f"{name}: power spectrum")
        spectrum_ax.set_xlabel("Frequency (Hz)")
        spectrum_ax.set_ylabel("Power (mV^2/Hz)")
        spectrum_ax.grid(alpha=0.18)
        spectrum_ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Jansen-Rit cortical EEG under reduced inhibitory gain", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    brainstate.random.seed(7)
    eeg = simulate_conditions()

    times = u.math.arange(0.0 * u.ms, DURATION, DT)
    times_s = np.asarray(times.to_decimal(u.second))
    eeg_mv = np.asarray(eeg.to_decimal(u.mV))
    settled = times_s >= TRANSIENT.to_decimal(u.second)
    settled_times_s = times_s[settled]
    settled_eeg_mv = eeg_mv[..., settled]
    sample_rate_hz = 1.0 / float(DT.to_decimal(u.second))

    frequencies, spectra, rows = summarize(settled_eeg_mv, sample_rate_hz)
    baseline_rows = rows[: INITIAL_OFFSETS.size]
    weakened_rows = rows[INITIAL_OFFSETS.size :]
    if not all(8.0 <= row[2] <= 13.0 for row in baseline_rows):
        raise RuntimeError("Baseline did not settle into the expected alpha band.")
    if not all(weak[4] < 0.1 * base[4] for base, weak in zip(baseline_rows, weakened_rows)):
        raise RuntimeError("Weakened inhibition did not suppress the alpha oscillation.")

    output_path = Path(__file__).with_name("alpha_rhythm_inhibition.png")
    plot_results(settled_times_s, settled_eeg_mv, frequencies, spectra, output_path)

    print("condition             init_M  peak_Hz  mean_mV  RMS_mV  alpha_power_mV2")
    for name, initial_mv, peak_hz, mean_mv, rms_mv, alpha_power in rows:
        print(
            f"{name:22s} {initial_mv:+6.1f}  {peak_hz:7.2f}  "
            f"{mean_mv:7.2f}  {rms_mv:6.3f}  {alpha_power:15.6f}"
        )
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
