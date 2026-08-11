"""Spiking head-direction ring attractor with dark turns and a wedge lesion.

The recurrent ring is a phenomenological continuous attractor. Binary spikes
are communicated by BrainEvent, BrainPy-State owns the LIF and synaptic
dynamics, BrainState owns time and condition axes, and BrainUnit keeps all
physical parameters dimensionally explicit.
"""

from __future__ import annotations

import csv
from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.util import filter as state_filter


# Ring and integration
N_NEURONS = 72
DT = 1.0 * u.ms
AXONAL_DELAY = 2.0 * u.ms
TAU_MEMBRANE = 20.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
TAU_SYNAPSE = 10.0 * u.ms
TAU_READOUT = 50.0 * u.ms

# LIF operating point
RESISTANCE = 100.0 * u.Mohm
V_REST = -65.0 * u.mV
V_RESET = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
BIAS_CURRENT = 0.105 * u.nA
CUE_CURRENT = 0.11 * u.nA

# Phenomenological attractor calibration
LOCAL_WEIGHT = 0.052 * u.nA
GLOBAL_INHIBITION = 0.007 * u.nA
RECURRENT_KAPPA = 4.0
CUE_KAPPA = 7.0
VELOCITY_SHIFT = 10.0  # degrees between base and shifted recurrent kernels
VELOCITY_MIX = 0.10
REFERENCE_TURN_SPEED = 90.0 * u.degree / u.second

# Protocol timing
CUE_DURATION = 300.0 * u.ms
DARK_SETTLE_DURATION = 250.0 * u.ms
TURN_DURATION = 1000.0 * u.ms
DARK_HOLD_DURATION = 350.0 * u.ms

LESION_PRE_DURATION = 300.0 * u.ms
LESION_TEST_DURATION = 900.0 * u.ms
LESION_CENTER_DEG = 0.0
LESION_WIDTH_DEG = 60.0
N_TEST_HEADINGS = N_NEURONS

# Outcome predicates
DEPARTURE_THRESHOLD_DEG = 12.0
RECOVERY_THRESHOLD_DEG = 10.0
RECOVERY_WINDOW = 200.0 * u.ms
MIN_FINAL_VECTOR_STRENGTH = 0.30
MIN_FINAL_RATE_RATIO = 0.45

TAU_AXES = (
    state_filter.Any(
        state_filter.OfType(brainstate.HiddenState),
        state_filter.OfType(brainstate.ShortTermState),
    )
)


def _steps(duration) -> int:
    """Convert a grid-aligned duration to a checked integer step count."""
    value = float(duration.to_decimal(u.ms) / DT.to_decimal(u.ms))
    rounded = int(round(value))
    if not np.isclose(value, rounded):
        raise ValueError(f"{duration} is not an integer multiple of dt={DT}.")
    return rounded


def wrap_angle(angle):
    """Wrap radians to [-pi, pi)."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def circular_difference(angle, reference):
    """Signed shortest difference between two radian angles."""
    return wrap_angle(angle - reference)


def ring_angles(n_neurons: int = N_NEURONS):
    return jnp.linspace(-jnp.pi, jnp.pi, n_neurons, endpoint=False)


def recurrent_kernel(preferred_angles, shift_degrees: float = 0.0):
    """Dense pre-by-post event weights for the circular attractor."""
    delta = (
        preferred_angles[None, :]
        - preferred_angles[:, None]
        - jnp.deg2rad(shift_degrees)
    )
    local = jnp.exp(RECURRENT_KAPPA * (jnp.cos(delta) - 1.0))
    return LOCAL_WEIGHT * local - GLOBAL_INHIBITION


def advance_spike_history(history, current_spikes, delay_steps):
    """Insert current events and retrieve an exact grid-aligned delay tap."""
    updated = jnp.concatenate((current_spikes[None, :], history[:-1]), axis=0)
    return updated, updated[delay_steps]


def cue_profile(preferred_angles, headings):
    """Unit-peak von Mises current profile for each initial heading."""
    delta = preferred_angles[None, :] - headings[:, None]
    return jnp.exp(CUE_KAPPA * (jnp.cos(delta) - 1.0))


def decode_bump(rates, preferred_angles):
    """Population-vector angle and normalized vector strength."""
    x = jnp.sum(rates * jnp.cos(preferred_angles), axis=-1)
    y = jnp.sum(rates * jnp.sin(preferred_angles), axis=-1)
    total = jnp.sum(rates, axis=-1)
    strength = jnp.sqrt(x * x + y * y) / jnp.maximum(total, 1e-7)
    return jnp.arctan2(y, x), strength


class SpikingCompass(brainstate.nn.Module):
    """Unit-aware LIF ring with delayed event-driven recurrent input."""

    def __init__(self):
        super().__init__()
        self.preferred_angles = ring_angles()
        self.base_weights = recurrent_kernel(self.preferred_angles)
        self.clockwise_weights = recurrent_kernel(
            self.preferred_angles, VELOCITY_SHIFT
        )
        self.counterclockwise_weights = recurrent_kernel(
            self.preferred_angles, -VELOCITY_SHIFT
        )
        self.delay_steps = _steps(AXONAL_DELAY)

        self.neurons = brainpy.state.LIFRef(
            N_NEURONS,
            R=RESISTANCE,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.recurrent_synapse = brainpy.state.Expon(
            N_NEURONS,
            tau=TAU_SYNAPSE,
            g_initializer=braintools.init.ZeroInit(unit=u.nA),
        )
        self.recurrent_output = brainpy.state.CUBA(scale=1.0)
        self.neurons.add_current_input("recurrent", self.recurrent_output)
        self.rate_readout = brainpy.state.Expon(
            N_NEURONS,
            tau=TAU_READOUT,
            g_initializer=braintools.init.ZeroInit(),
        )

    def init_state(self):
        # A short pointer-free buffer gives every vmapped trial its own delay.
        self.spike_history = brainstate.HiddenState(
            jnp.zeros((self.delay_steps + 1, N_NEURONS), dtype=bool)
        )

    def update(self, time, external_current, velocity_fraction, active_mask):
        with brainstate.environ.context(t=time):
            previous_spikes = (self.neurons.get_spike() != 0.0) & active_mask
            history, delayed_spikes = advance_spike_history(
                self.spike_history.value, previous_spikes, self.delay_steps
            )
            self.spike_history.value = history

            # The three event products share binary spikes and differ only in
            # the ring kernel used to steer the bump.
            events = brainevent.BinaryArray(delayed_spikes)
            base_current = events @ self.base_weights
            clockwise_current = events @ self.clockwise_weights
            counterclockwise_current = events @ self.counterclockwise_weights

            mix = jnp.minimum(jnp.abs(velocity_fraction) * VELOCITY_MIX, 0.5)
            steered_current = u.math.where(
                velocity_fraction >= 0.0,
                base_current + mix * (clockwise_current - base_current),
                base_current + mix * (counterclockwise_current - base_current),
            )
            filtered_current = self.recurrent_synapse(
                steered_current * active_mask
            )
            self.recurrent_output.bind_cond(filtered_current * active_mask)

            spikes = (
                self.neurons(external_current * active_mask) != 0.0
            ) & active_mask
            return self.rate_readout(spikes)


def protocol_inputs(
    headings,
    duration,
    *,
    velocity_fraction=None,
    lesion_masks=None,
    lesion_onset=None,
):
    """Create complete time-major inputs before entering the stateful loop."""
    headings = jnp.asarray(headings)
    batch_size = headings.shape[0]
    times = u.math.arange(0.0 * u.ms, duration, DT)
    n_steps = times.shape[0]

    cue_steps = _steps(CUE_DURATION)
    cue = cue_profile(ring_angles(), headings)
    cue_gate = (jnp.arange(n_steps) < cue_steps)[:, None, None]
    drive = BIAS_CURRENT + CUE_CURRENT * cue_gate * cue[None, :, :]

    if velocity_fraction is None:
        velocity_fraction = jnp.zeros((n_steps, batch_size))
    else:
        velocity_fraction = jnp.asarray(velocity_fraction)

    masks = jnp.ones((n_steps, batch_size, N_NEURONS), dtype=bool)
    if lesion_masks is not None:
        if lesion_onset is None:
            raise ValueError("lesion_onset is required with lesion_masks.")
        onset_step = _steps(lesion_onset)
        lesion_gate = (jnp.arange(n_steps) >= onset_step)[:, None, None]
        masks = jnp.where(lesion_gate, lesion_masks[None, :, :], masks)

    return times, drive, velocity_fraction, masks


def run_protocol(times, drive, velocity_fraction, active_masks):
    """Map independent conditions inside one BrainState time loop."""
    batch_size = drive.shape[1]
    with brainstate.environ.context(dt=DT):
        compass = SpikingCompass()
        brainstate.nn.vmap_init_all_states(compass, axis_size=batch_size)
        mapped_step = brainstate.transform.vmap2(
            compass.update,
            in_axes=(None, 0, 0, 0),
            out_axes=0,
            state_in_axes={0: TAU_AXES},
            state_out_axes={0: TAU_AXES},
            unexpected_out_state_mapping="raise",
        )

        @brainstate.transform.jit
        def run():
            return brainstate.transform.for_loop(
                mapped_step, times, drive, velocity_fraction, active_masks
            )

        rates = run()
    return rates


def run_dark_turn():
    """Cue north, remove the cue, and rotate 90 degrees in darkness."""
    turn_start = CUE_DURATION + DARK_SETTLE_DURATION
    turn_stop = turn_start + TURN_DURATION
    duration = turn_stop + DARK_HOLD_DURATION
    n_steps = _steps(duration)

    speed_raw = REFERENCE_TURN_SPEED.to_decimal(u.degree / u.second)
    reference_raw = REFERENCE_TURN_SPEED.to_decimal(u.degree / u.second)
    turn_gate = (
        (jnp.arange(n_steps) >= _steps(turn_start))
        & (jnp.arange(n_steps) < _steps(turn_stop))
    )
    velocity = jnp.zeros((n_steps, 2))
    velocity = velocity.at[:, 1].set(
        turn_gate * (speed_raw / reference_raw)
    )

    times, drive, velocity, masks = protocol_inputs(
        jnp.zeros(2), duration, velocity_fraction=velocity
    )
    rates = run_protocol(times, drive, velocity, masks)
    decoded, strength = decode_bump(rates, ring_angles())

    dt_seconds = DT.to_decimal(u.second)
    true_turn = jnp.cumsum(
        velocity[:, 1]
        * REFERENCE_TURN_SPEED.to_decimal(u.degree / u.second)
        * dt_seconds
    )
    true_heading = jnp.stack((jnp.zeros(n_steps), jnp.deg2rad(true_turn)), axis=1)
    return {
        "times": times,
        "rates": rates,
        "decoded": decoded,
        "strength": strength,
        "true_heading": true_heading,
        "turn_start": turn_start,
        "turn_stop": turn_stop,
    }


def wedge_mask(center_degrees=LESION_CENTER_DEG, width_degrees=LESION_WIDTH_DEG):
    center = jnp.deg2rad(center_degrees)
    half_width = jnp.deg2rad(width_degrees / 2.0)
    return jnp.abs(circular_difference(ring_angles(), center)) > half_width


def classify_lesion_outcomes(
    error_degrees,
    strength,
    rate_ratio,
    *,
    departure_threshold=DEPARTURE_THRESHOLD_DEG,
    recovery_threshold=RECOVERY_THRESHOLD_DEG,
    recovery_steps=None,
):
    """Classify spared/recovered/failed from the full post-lesion traces."""
    if recovery_steps is None:
        recovery_steps = _steps(RECOVERY_WINDOW)
    departed = jnp.max(error_degrees, axis=0) > departure_threshold
    sustained_return = jnp.all(
        error_degrees[-recovery_steps:] <= recovery_threshold, axis=0
    )
    reliable = (
        jnp.mean(strength[-recovery_steps:], axis=0)
        >= MIN_FINAL_VECTOR_STRENGTH
    ) & (
        jnp.mean(rate_ratio[-recovery_steps:], axis=0)
        >= MIN_FINAL_RATE_RATIO
    )
    return jnp.where(
        ~departed & sustained_return & reliable,
        0,
        jnp.where(departed & sustained_return & reliable, 1, 2),
    )


def run_lesion_map():
    """Run matched control and wedge-lesion trials for every initial heading."""
    headings = jnp.linspace(-jnp.pi, jnp.pi, N_TEST_HEADINGS, endpoint=False)
    all_headings = jnp.concatenate((headings, headings))
    intact = jnp.ones((N_TEST_HEADINGS, N_NEURONS), dtype=bool)
    lesioned = jnp.broadcast_to(wedge_mask(), intact.shape)
    lesion_masks = jnp.concatenate((intact, lesioned), axis=0)

    lesion_onset = CUE_DURATION + LESION_PRE_DURATION
    duration = lesion_onset + LESION_TEST_DURATION
    times, drive, velocity, masks = protocol_inputs(
        all_headings,
        duration,
        lesion_masks=lesion_masks,
        lesion_onset=lesion_onset,
    )
    rates = run_protocol(times, drive, velocity, masks)
    decoded, strength = decode_bump(rates, ring_angles())

    control_phase, lesion_phase = jnp.split(decoded, 2, axis=1)
    control_strength, lesion_strength = jnp.split(strength, 2, axis=1)
    control_rate, lesion_rate = jnp.split(jnp.mean(rates, axis=-1), 2, axis=1)
    onset_step = _steps(lesion_onset)
    error_degrees = jnp.rad2deg(
        jnp.abs(circular_difference(lesion_phase, control_phase))
    )[onset_step:]
    rate_ratio = (
        lesion_rate[onset_step:] / jnp.maximum(control_rate[onset_step:], 1e-7)
    )
    outcomes = classify_lesion_outcomes(
        error_degrees,
        lesion_strength[onset_step:],
        rate_ratio,
    )
    return {
        "times": times,
        "headings": headings,
        "rates": rates,
        "control_phase": control_phase,
        "lesion_phase": lesion_phase,
        "control_strength": control_strength,
        "lesion_strength": lesion_strength,
        "error_degrees": error_degrees,
        "rate_ratio": rate_ratio,
        "outcomes": outcomes,
        "lesion_onset": lesion_onset,
    }


def save_lesion_table(result, path):
    labels = np.array(["spared", "recovered", "failed"])
    headings = np.rad2deg(np.asarray(result["headings"]))
    errors = np.asarray(result["error_degrees"])
    strength = np.asarray(result["lesion_strength"])[_steps(result["lesion_onset"]):]
    rate_ratio = np.asarray(result["rate_ratio"])
    outcomes = np.asarray(result["outcomes"])
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "starting_heading_deg",
                "outcome",
                "peak_error_deg",
                "final_error_deg",
                "final_vector_strength",
                "final_rate_ratio",
            ]
        )
        for index, heading in enumerate(headings):
            writer.writerow(
                [
                    f"{heading:.1f}",
                    labels[outcomes[index]],
                    f"{errors[:, index].max():.2f}",
                    f"{errors[-_steps(RECOVERY_WINDOW):, index].mean():.2f}",
                    f"{strength[-_steps(RECOVERY_WINDOW):, index].mean():.3f}",
                    f"{rate_ratio[-_steps(RECOVERY_WINDOW):, index].mean():.3f}",
                ]
            )
    return path


def plot_results(dark, lesion, path):
    """Show dark integration, activity motion, and lesion outcomes."""
    dark_time = np.asarray(dark["times"].to_decimal(u.second))
    dark_decoded = np.rad2deg(np.unwrap(np.asarray(dark["decoded"]), axis=0))
    dark_true = np.rad2deg(np.asarray(dark["true_heading"]))
    dark_rates = np.asarray(dark["rates"][:, 1, :])

    lesion_time = (
        np.asarray(lesion["times"].to_decimal(u.second))
        - lesion["lesion_onset"].to_decimal(u.second)
    )
    lesion_time = lesion_time[_steps(lesion["lesion_onset"]):]
    heading_degrees = np.rad2deg(np.asarray(lesion["headings"]))
    errors = np.asarray(lesion["error_degrees"])
    outcomes = np.asarray(lesion["outcomes"])
    labels = np.array(["spared", "recovered", "failed"])
    colors = np.array(["#247a4d", "#d08a00", "#b83232"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    axes[0, 0].plot(dark_time, dark_true[:, 1], color="#222222", lw=2.0, label="animal")
    axes[0, 0].plot(dark_time, dark_decoded[:, 1], color="#147d92", lw=1.4, label="bump")
    axes[0, 0].plot(dark_time, dark_decoded[:, 0], color="#9a9a9a", lw=1.0, label="no-turn control")
    axes[0, 0].axvspan(
        dark["turn_start"].to_decimal(u.second),
        dark["turn_stop"].to_decimal(u.second),
        color="#e8bd4d",
        alpha=0.22,
    )
    axes[0, 0].set(title="Dark turn", ylabel="heading (deg)", xlabel="time (s)")
    axes[0, 0].legend(frameon=False)

    image = axes[0, 1].imshow(
        dark_rates.T,
        origin="lower",
        aspect="auto",
        extent=(dark_time[0], dark_time[-1], -180.0, 180.0),
        cmap="magma",
    )
    axes[0, 1].plot(dark_time, dark_true[:, 1], color="white", lw=1.0, label="animal")
    axes[0, 1].set(title="Bump activity", ylabel="preferred direction (deg)", xlabel="time (s)")
    fig.colorbar(image, ax=axes[0, 1], label="filtered spikes")

    error_image = axes[1, 0].imshow(
        errors.T,
        origin="lower",
        aspect="auto",
        extent=(lesion_time[0], lesion_time[-1], heading_degrees[0], heading_degrees[-1]),
        cmap="viridis",
        vmin=0.0,
        vmax=max(30.0, float(np.percentile(errors, 98))),
    )
    axes[1, 0].set(
        title="Lesion departure from matched control",
        ylabel="starting heading (deg)",
        xlabel="time since lesion (s)",
    )
    fig.colorbar(error_image, ax=axes[1, 0], label="absolute angular error (deg)")

    final_error = errors[-_steps(RECOVERY_WINDOW):].mean(axis=0)
    for code, label in enumerate(labels):
        selected = outcomes == code
        axes[1, 1].scatter(
            heading_degrees[selected],
            final_error[selected],
            color=colors[code],
            s=34,
            label=label,
        )
    axes[1, 1].axvspan(
        LESION_CENTER_DEG - LESION_WIDTH_DEG / 2.0,
        LESION_CENTER_DEG + LESION_WIDTH_DEG / 2.0,
        color="#b83232",
        alpha=0.12,
        label="silenced wedge",
    )
    axes[1, 1].axhline(RECOVERY_THRESHOLD_DEG, color="#555555", ls="--", lw=1.0)
    axes[1, 1].set(
        title="All starting directions",
        ylabel="final error (deg)",
        xlabel="starting heading (deg)",
        xlim=(-180.0, 180.0),
    )
    axes[1, 1].legend(frameon=False, ncol=2)

    path = Path(path)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def validate_results(dark, lesion):
    """Fail loudly when the intended compass behavior is not present."""
    final_window = _steps(100.0 * u.ms)
    turn_error = np.rad2deg(
        np.abs(
            np.asarray(
                circular_difference(
                    dark["decoded"][-final_window:, 1],
                    dark["true_heading"][-final_window:, 1],
                )
            )
        )
    ).mean()
    control_drift = np.rad2deg(
        np.abs(np.asarray(dark["decoded"][-final_window:, 0]))
    ).mean()
    final_strength = np.asarray(dark["strength"][-final_window:, 1]).mean()
    lesion_rates = np.asarray(lesion["rates"][:, N_TEST_HEADINGS:, :])
    damaged_neurons = ~np.asarray(wedge_mask())
    final_damaged_activity = lesion_rates[
        -final_window:, :, damaged_neurons
    ].mean()
    if turn_error > 20.0:
        raise AssertionError(f"dark-turn mean error is too large: {turn_error:.1f} deg")
    if control_drift > 12.0:
        raise AssertionError(f"stationary bump drifted: {control_drift:.1f} deg")
    if final_strength < MIN_FINAL_VECTOR_STRENGTH:
        raise AssertionError(f"dark-turn bump collapsed: strength={final_strength:.2f}")
    if np.asarray(lesion["outcomes"]).shape != (N_TEST_HEADINGS,):
        raise AssertionError("lesion sweep did not retain every starting heading")
    if final_damaged_activity > 1e-3:
        raise AssertionError(
            "silenced neurons retained activity: "
            f"mean filtered spikes={final_damaged_activity:.4f}"
        )
    return {
        "dark_turn_error_deg": float(turn_error),
        "stationary_drift_deg": float(control_drift),
        "dark_turn_vector_strength": float(final_strength),
        "final_damaged_activity": float(final_damaged_activity),
    }


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    dark = run_dark_turn()
    lesion = run_lesion_map()
    metrics = validate_results(dark, lesion)
    figure = plot_results(dark, lesion, output_dir / "internal_neural_compass.png")
    table = save_lesion_table(lesion, output_dir / "lesion_outcomes.csv")

    labels = np.array(["spared", "recovered", "failed"])
    outcomes = np.asarray(lesion["outcomes"])
    counts = {
        str(label): int(np.sum(outcomes == code))
        for code, label in enumerate(labels)
    }
    print(f"dark-turn final mean error: {metrics['dark_turn_error_deg']:.2f} deg")
    print(f"stationary-control drift: {metrics['stationary_drift_deg']:.2f} deg")
    print(f"dark-turn final vector strength: {metrics['dark_turn_vector_strength']:.3f}")
    print(f"silenced-wedge final activity: {metrics['final_damaged_activity']:.6f}")
    print("lesion outcomes:", counts)
    print(f"figure: {figure}")
    print(f"table: {table}")


if __name__ == "__main__":
    main()
