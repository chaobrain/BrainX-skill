"""Generate an alpha-like cortical rhythm and weaken inhibitory feedback."""

from pathlib import Path

import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


DT = 0.1 * u.ms
DURATION = 4.0 * u.second
TRANSIENT = 1.0 * u.second
TAU_E = 15.0 * u.ms
TAU_I = 22.5 * u.ms
RATE_SCALE = 100.0 * u.Hz
DRIVE_E = 100.0 * u.Hz
DRIVE_I = 0.0 * u.Hz
INHIBITION = jnp.array([12.0, 8.0])  # Wilson-Cowan wEI, dimensionless
INITIAL_RATES = jnp.array(
    [
        [0.01, 0.01],
        [0.10, 0.04],
        [0.30, 0.12],
    ]
) * RATE_SCALE
OUTPUT = Path(__file__).with_name("alpha_rhythm.png")


def simulate_one(w_ei, initial_rates):
    """Run one independent E/I population circuit and return its EEG-like proxy."""
    circuit = brainmass.WilsonCowanStep(
        in_size=1,
        tau_E=TAU_E,
        tau_I=TAU_I,
        wEE=16.0,
        wEI=w_ei,
        wIE=15.0,
        wII=3.0,
    )

    with brainstate.environ.context(dt=DT):
        circuit.init_all_states()
        circuit.rE.value = jnp.asarray([initial_rates[0] / RATE_SCALE])
        circuit.rI.value = jnp.asarray([initial_rates[1] / RATE_SCALE])
        times = u.math.arange(0.0 * u.ms, DURATION, DT)
        indices = jnp.arange(times.shape[0])

        def step(t, index):
            with brainstate.environ.context(t=t, i=index):
                circuit.update(DRIVE_E / RATE_SCALE, DRIVE_I / RATE_SCALE)
            # A local field/EEG-like population proxy, not a calibrated scalp voltage.
            return (circuit.rE.value - circuit.rI.value)[0] * RATE_SCALE

        return brainstate.transform.for_loop(step, times, indices)


def power_spectrum(signal):
    centered = signal - signal.mean()
    frequencies = np.fft.rfftfreq(centered.size, d=DT.to_decimal(u.second))
    power = np.abs(np.fft.rfft(centered)) ** 2 / centered.size
    return frequencies, power


def run_experiment():
    # Map complete stateful rollouts over inhibition and initial-condition lanes.
    w_ei, initial_conditions = jnp.meshgrid(
        INHIBITION,
        jnp.arange(INITIAL_RATES.shape[0]),
        indexing="ij",
    )
    trajectories_hz = brainstate.transform.vmap(
        simulate_one,
        in_axes=(0, 0),
    )(
        w_ei.reshape(-1),
        INITIAL_RATES[initial_conditions.reshape(-1)],
    )
    trajectories = np.asarray(trajectories_hz.to_decimal(u.Hz)).reshape(
        INHIBITION.size,
        INITIAL_RATES.shape[0],
        -1,
    )

    transient_samples = int(TRANSIENT / DT)
    return trajectories[:, :, transient_samples:]


def measure(trace):
    frequencies, power = power_spectrum(trace)
    analysis_band = (frequencies >= 1.0) & (frequencies <= 40.0)
    alpha_band = (frequencies >= 8.0) & (frequencies <= 13.0)
    rms = np.sqrt(np.mean((trace - trace.mean()) ** 2))
    peak_hz = (
        frequencies[analysis_band][np.argmax(power[analysis_band])]
        if rms > 1e-4
        else np.nan
    )
    return peak_hz, rms, power[alpha_band].sum()


def main():
    settled = run_experiment()
    metrics = np.asarray(
        [[measure(trace) for trace in condition] for condition in settled]
    )

    # Validate every initial-condition lane before reporting the intervention effect.
    assert np.all((metrics[0, :, 0] >= 8.0) & (metrics[0, :, 0] <= 13.0))
    assert np.all(metrics[0, :, 1] > (5.0 * u.Hz).to_decimal(u.Hz))
    assert np.all(metrics[1, :, 1] < (1e-4 * u.Hz).to_decimal(u.Hz))

    traces = settled[:, 0]  # representative lane; metrics retain all lanes
    time_s = np.arange(traces.shape[-1]) * DT.to_decimal(u.second)
    spectra = [power_spectrum(trace) for trace in traces]

    labels = ["Baseline inhibition", "Weakened inhibition"]
    colors = ["#1f6f78", "#c14924"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for label, color, trace, (frequencies, power) in zip(
        labels, colors, traces, spectra
    ):
        axes[0].plot(time_s[:10000], trace[:10000], color=color, label=label)
        axes[1].plot(frequencies, power, color=color, label=label)

    axes[0].set(
        title="Resting cortical population signal",
        xlabel="Time (s)",
        ylabel="EEG-like E-I rate proxy (Hz)",
    )
    axes[1].set(
        title="Power spectrum",
        xlabel="Frequency (Hz)",
        ylabel="Power (Hz squared)",
        xlim=(1.0, 40.0),
    )
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT, dpi=180)

    print("condition,initial_condition,wEI,peak_hz,rms_hz,alpha_power")
    for condition_index, (label, inhibition) in enumerate(
        zip(labels, np.asarray(INHIBITION))
    ):
        for initial_index, (peak_hz, rms, alpha_power) in enumerate(
            metrics[condition_index]
        ):
            print(
                f"{label},{initial_index},{inhibition:.1f},"
                f"{peak_hz:.2f},{rms:.6f},{alpha_power:.3f}"
            )
    print(f"figure={OUTPUT}")


if __name__ == "__main__":
    main()
