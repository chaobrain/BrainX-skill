"""A phenomenological spiking head-direction ring attractor in BrainX.

The experiment first cues north, removes the landmark, and turns the animal in
darkness. It then cues every represented heading, silences a fixed wedge, and
compares each lesion trial with its matched intact control.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neural-compass-matplotlib")

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter


# Ring and neuron parameters.
NUM_NEURONS = 48
DT = 0.5 * u.ms
TAU_MEMBRANE = 20.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
TAU_SYNAPSE = 6.0 * u.ms
TAU_READOUT = 25.0 * u.ms
SYNAPTIC_DELAY = 1.0 * u.ms

V_REST = -60.0 * u.mV
V_RESET = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV

# Phenomenological ring-attractor calibration.
TONIC_CURRENT = 9.0 * u.mA
CUE_CURRENT = 8.0 * u.mA
SILENCE_CURRENT = -80.0 * u.mA
LOCAL_WEIGHT = 4.0 * u.mA
GLOBAL_WEIGHT = 0.25 * u.mA
VELOCITY_WEIGHT = 0.06 * u.mA
RECURRENT_WIDTH = 0.48
CUE_WIDTH = 0.34
ANGULAR_VELOCITY_SCALE = 90.0 * u.degree / u.second

# Protocol timings.
CUE_DURATION = 200.0 * u.ms
DARK_HOLD = 200.0 * u.ms
TURN_DURATION = 1.0 * u.second
POST_TURN_HOLD = 250.0 * u.ms
LESION_SILENCE_DURATION = 50.0 * u.ms
POST_LESION_DURATION = 550.0 * u.ms
LESION_WIDTH = 60.0 * u.degree
LESION_CENTER = 180.0 * u.degree

# Outcome predicates. These are declared because the model is calibrated, not
# fitted to a biological dataset.
FINAL_ERROR_LIMIT_DEG = 22.5
DEPARTURE_ERROR_DEG = 30.0
MIN_COHERENCE = 0.16
MIN_MATCHED_MASS_RATIO = 0.35
FINAL_WINDOW = 100.0 * u.ms

OUTPUT_FIGURE = Path("neural_compass_results.png")
OUTPUT_DATA = Path("neural_compass_results.npz")


def _steps(duration) -> int:
    return int(round(duration.to_decimal(u.ms) / DT.to_decimal(u.ms)))


def _circular_difference(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _ring_weights(preferred_angles):
    delta = preferred_angles[None, :] - preferred_angles[:, None]
    local = jnp.exp((jnp.cos(delta) - 1.0) / RECURRENT_WIDTH**2)
    recurrent = LOCAL_WEIGHT * local - GLOBAL_WEIGHT
    velocity = VELOCITY_WEIGHT * local * jnp.sin(delta)
    return recurrent, velocity


class HeadDirectionRing(brainstate.nn.Module):
    """LIF ring with delayed event communication and a velocity asymmetry."""

    def __init__(self):
        super().__init__()
        self.num = NUM_NEURONS
        self.angles = jnp.linspace(0.0, 2.0 * jnp.pi, self.num, endpoint=False)
        self.recurrent_weights, self.velocity_weights = _ring_weights(self.angles)
        self.delay_steps = _steps(SYNAPTIC_DELAY)

        self.neurons = brainpy.state.LIFRef(
            self.num,
            R=1.0 * u.ohm,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.synapse = brainpy.state.Expon(
            self.num,
            tau=TAU_SYNAPSE,
            g_initializer=braintools.init.Constant(0.0 * u.mA),
        )
        self.readout = brainpy.state.Expon(
            self.num,
            tau=TAU_READOUT,
            g_initializer=braintools.init.Constant(0.0),
        )

    def init_state(self):
        # A fixed integer delay uses a pointer-free shift register. Index zero
        # is the current sample; index d is exactly d completed steps earlier.
        self.spike_history = brainstate.HiddenState(
            jnp.zeros((self.delay_steps + 1, self.num), dtype=bool)
        )

    def update(self, t, external_current, turn_gain, lesion_mask):
        with brainstate.environ.context(t=t):
            current_spikes = self.neurons.get_spike() != 0.0
            history = jnp.concatenate(
                (current_spikes[None, :], self.spike_history.value[:-1]), axis=0
            )
            self.spike_history.value = history
            delayed_spikes = history[self.delay_steps] & ~lesion_mask

            weights = self.recurrent_weights + turn_gain * self.velocity_weights
            recurrent_event = brainevent.BinaryArray(delayed_spikes) @ weights
            recurrent_current = self.synapse(recurrent_event)
            lesion_current = u.math.where(
                lesion_mask, SILENCE_CURRENT, 0.0 * u.mA
            )
            spikes = self.neurons(external_current + recurrent_current + lesion_current)
            active_spikes = (spikes != 0.0) & ~lesion_mask
            activity = self.readout(active_spikes.astype(jnp.float32))
            return active_spikes, activity


def verify_delay_convention():
    """Prove the shift-register latency with one nonzero binary event."""
    delay_steps = _steps(SYNAPTIC_DELAY)
    history = brainstate.HiddenState(jnp.zeros(delay_steps + 1, dtype=bool))

    def step(event):
        history.value = jnp.concatenate((event[None], history.value[:-1]))
        return history.value[delay_steps]

    impulse = jnp.arange(delay_steps + 3) == 0
    observed = brainstate.transform.for_loop(step, impulse)
    expected = jnp.arange(delay_steps + 3) == delay_steps
    if not bool(jnp.array_equal(observed, expected)):
        raise AssertionError("The synaptic delay convention failed its impulse check.")


def _cue_protocol(start_angles, num_steps, cue_steps):
    neuron_angles = np.linspace(0.0, 2.0 * np.pi, NUM_NEURONS, endpoint=False)
    delta = neuron_angles[None, :] - start_angles[:, None]
    cue_shape = np.exp((np.cos(delta) - 1.0) / CUE_WIDTH**2)
    currents = np.full(
        (num_steps, start_angles.size, NUM_NEURONS),
        TONIC_CURRENT.to_decimal(u.mA),
        dtype=np.float32,
    )
    currents[:cue_steps] += CUE_CURRENT.to_decimal(u.mA) * cue_shape[None, :, :]
    return jnp.asarray(currents) * u.mA


def _simulate(external_current, turn_gain, lesion_mask):
    num_steps, num_conditions = turn_gain.shape
    times = u.math.arange(0.0 * u.ms, num_steps * DT, DT)

    with brainstate.environ.context(dt=DT):
        model = HeadDirectionRing()
        brainstate.nn.vmap_init_all_states(model, axis_size=num_conditions)
        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = vmap2(
            model.update,
            in_axes=(None, 0, 0, 0),
            out_axes=0,
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )

        @brainstate.transform.jit
        def run():
            return brainstate.transform.for_loop(
                mapped_step, times, external_current, turn_gain, lesion_mask
            )

        spikes, activity = run()

    return times, np.asarray(spikes), np.asarray(activity)


def run_dark_turn():
    cue_steps = _steps(CUE_DURATION)
    hold_steps = _steps(DARK_HOLD)
    turn_steps = _steps(TURN_DURATION)
    post_steps = _steps(POST_TURN_HOLD)
    total_steps = cue_steps + hold_steps + turn_steps + post_steps

    start_angles = np.array([0.0], dtype=np.float32)
    current = _cue_protocol(start_angles, total_steps, cue_steps)
    turn_rate = np.zeros((total_steps, 1), dtype=np.float32)
    turn_rate[cue_steps + hold_steps : cue_steps + hold_steps + turn_steps] = (
        ANGULAR_VELOCITY_SCALE.to_decimal(u.degree / u.second)
    )
    turn_gain = turn_rate / ANGULAR_VELOCITY_SCALE.to_decimal(u.degree / u.second)
    lesion = jnp.zeros((total_steps, 1, NUM_NEURONS), dtype=bool)
    times, spikes, activity = _simulate(current, jnp.asarray(turn_gain), lesion)

    dt_seconds = DT.to_decimal(u.second)
    expected = np.deg2rad(np.cumsum(turn_rate[:, 0]) * dt_seconds)
    return {
        "times_ms": np.asarray(times.to_decimal(u.ms)),
        "spikes": spikes[:, 0],
        "activity": activity[:, 0],
        "expected": expected,
        "turn_start": cue_steps + hold_steps,
        "turn_stop": cue_steps + hold_steps + turn_steps,
    }


def _lesion_neuron_mask():
    angles_deg = np.linspace(0.0, 360.0, NUM_NEURONS, endpoint=False)
    center_deg = LESION_CENTER.to_decimal(u.degree)
    distance = (angles_deg - center_deg + 180.0) % 360.0 - 180.0
    return np.abs(distance) <= LESION_WIDTH.to_decimal(u.degree) / 2.0


def run_lesion_sweep():
    cue_steps = _steps(CUE_DURATION)
    hold_steps = _steps(DARK_HOLD)
    silence_steps = _steps(LESION_SILENCE_DURATION)
    post_steps = _steps(POST_LESION_DURATION)
    total_steps = cue_steps + hold_steps + silence_steps + post_steps
    headings = np.linspace(0.0, 2.0 * np.pi, NUM_NEURONS, endpoint=False)

    # Controls and lesions use the same mapped simulation path and are paired
    # by heading: [all controls, all lesions].
    start_angles = np.concatenate((headings, headings)).astype(np.float32)
    current = _cue_protocol(start_angles, total_steps, cue_steps)
    turn_gain = jnp.zeros((total_steps, start_angles.size), dtype=jnp.float32)
    lesion = np.zeros((total_steps, start_angles.size, NUM_NEURONS), dtype=bool)
    lesion_start = cue_steps + hold_steps
    lesion_stop = lesion_start + silence_steps
    lesion[lesion_start:lesion_stop, NUM_NEURONS:, :] = _lesion_neuron_mask()
    times, spikes, activity = _simulate(current, turn_gain, jnp.asarray(lesion))
    return {
        "times_ms": np.asarray(times.to_decimal(u.ms)),
        "spikes": spikes,
        "activity": activity,
        "headings": headings,
        "lesion_start": lesion_start,
        "lesion_stop": lesion_stop,
        "lesion_mask": _lesion_neuron_mask(),
    }


def decode_activity(activity):
    angles = np.linspace(0.0, 2.0 * np.pi, NUM_NEURONS, endpoint=False)
    vector = np.sum(activity * np.exp(1j * angles), axis=-1)
    mass = np.sum(activity, axis=-1)
    heading = np.mod(np.angle(vector), 2.0 * np.pi)
    coherence = np.abs(vector) / np.maximum(mass, 1e-8)
    return heading, coherence, mass


def classify_lesions(sweep):
    n = NUM_NEURONS
    activity = sweep["activity"]
    headings = sweep["headings"]
    lesion_start = sweep["lesion_start"]
    final_steps = _steps(FINAL_WINDOW)

    decoded, coherence, mass = decode_activity(activity)
    lesion_decoded = decoded[:, n:]
    control_mass, lesion_mass = mass[:, :n], mass[:, n:]

    final_slice = slice(-final_steps, None)
    control_final_activity = activity[final_slice, :n].mean(axis=0)
    lesion_final_activity = activity[final_slice, n:].mean(axis=0)
    control_final_heading, control_final_coherence, control_final_mass = decode_activity(
        control_final_activity
    )
    lesion_final_heading, lesion_final_coherence, lesion_final_mass = decode_activity(
        lesion_final_activity
    )

    control_final_error = np.abs(_circular_difference(control_final_heading, headings))
    lesion_final_error = np.abs(_circular_difference(lesion_final_heading, headings))
    matched_mass_ratio = lesion_final_mass / np.maximum(control_final_mass, 1e-8)
    matched_coherence_floor = np.maximum(MIN_COHERENCE, 0.55 * control_final_coherence)

    final_good = (
        (np.rad2deg(lesion_final_error) <= FINAL_ERROR_LIMIT_DEG)
        & (lesion_final_coherence >= matched_coherence_floor)
        & (matched_mass_ratio >= MIN_MATCHED_MASS_RATIO)
    )
    post_error = np.abs(
        _circular_difference(lesion_decoded[lesion_start:], headings[None, :])
    )
    post_mass_ratio = lesion_mass[lesion_start:] / np.maximum(
        control_mass[lesion_start:], 1e-8
    )
    max_post_error_deg = np.rad2deg(post_error).max(axis=0)
    min_post_mass_ratio = post_mass_ratio.min(axis=0)
    departed = (max_post_error_deg > DEPARTURE_ERROR_DEG) | (
        min_post_mass_ratio < 0.55
    )

    final_error_trace = np.rad2deg(
        np.abs(
            _circular_difference(
                lesion_decoded[final_slice], headings[None, :]
            )
        )
    )
    sustained_fraction = (final_error_trace <= FINAL_ERROR_LIMIT_DEG).mean(axis=0)
    sustained = sustained_fraction >= 0.9
    labels = np.full(n, "failed", dtype="U9")
    labels[final_good & sustained & departed] = "recovered"
    labels[final_good & sustained & ~departed] = "spared"

    return {
        "decoded": decoded,
        "coherence": coherence,
        "mass": mass,
        "control_final_error_deg": np.rad2deg(control_final_error),
        "lesion_final_error_deg": np.rad2deg(lesion_final_error),
        "control_final_coherence": control_final_coherence,
        "lesion_final_coherence": lesion_final_coherence,
        "matched_mass_ratio": matched_mass_ratio,
        "max_post_error_deg": max_post_error_deg,
        "min_post_mass_ratio": min_post_mass_ratio,
        "sustained_fraction": sustained_fraction,
        "final_good": final_good,
        "departed": departed,
        "labels": labels,
    }


def _contiguous_ranges(values_deg):
    if values_deg.size == 0:
        return "none"
    step = 360.0 / NUM_NEURONS
    groups = np.split(values_deg, np.where(np.diff(values_deg) > 1.5 * step)[0] + 1)
    formatted = []
    for group in groups:
        formatted.append(
            f"{group[0]:.0f} deg"
            if group.size == 1
            else f"{group[0]:.0f}-{group[-1]:.0f} deg"
        )
    return ", ".join(formatted)


def validate_results(dark_turn, sweep, outcomes):
    dark_heading, dark_coherence, _ = decode_activity(dark_turn["activity"])
    turn_slice = slice(dark_turn["turn_start"], dark_turn["turn_stop"])
    turn_heading = np.unwrap(dark_heading[turn_slice])
    turn_displacement = np.rad2deg(turn_heading[-1] - turn_heading[0])
    commanded_turn = np.rad2deg(
        dark_turn["expected"][dark_turn["turn_stop"] - 1]
        - dark_turn["expected"][dark_turn["turn_start"]]
    )
    if abs(turn_displacement - commanded_turn) > 20.0:
        raise AssertionError("The intact bump did not follow the commanded dark turn.")
    if dark_coherence[-1] < 0.4:
        raise AssertionError("The intact bump lost coherence after the dark turn.")
    if np.max(outcomes["control_final_error_deg"]) > 5.0:
        raise AssertionError("At least one matched intact control lost its heading.")

    silenced_spikes = sweep["spikes"][
        sweep["lesion_start"] : sweep["lesion_stop"],
        NUM_NEURONS:,
        :,
    ][:, :, sweep["lesion_mask"]]
    if np.any(silenced_spikes):
        raise AssertionError("A neuron in the silenced wedge emitted an active spike.")

    sustained = outcomes["sustained_fraction"] >= 0.9
    expected_recovered = outcomes["final_good"] & sustained & outcomes["departed"]
    expected_spared = outcomes["final_good"] & sustained & ~outcomes["departed"]
    if not np.array_equal(outcomes["labels"] == "recovered", expected_recovered):
        raise AssertionError("A recovered label did not satisfy the full predicate.")
    if not np.array_equal(outcomes["labels"] == "spared", expected_spared):
        raise AssertionError("A spared label did not satisfy the full predicate.")
    if not all(np.any(outcomes["labels"] == label) for label in ("spared", "recovered", "failed")):
        raise AssertionError("The calibrated lesion sweep lost an outcome regime.")
    return turn_displacement


def plot_results(dark_turn, sweep, outcomes, path=OUTPUT_FIGURE):
    ring_deg = np.linspace(0.0, 360.0, NUM_NEURONS, endpoint=False)
    dark_heading, dark_coherence, _ = decode_activity(dark_turn["activity"])
    time = dark_turn["times_ms"]

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(
        dark_turn["activity"].T,
        origin="lower",
        aspect="auto",
        extent=(time[0], time[-1], 0.0, 360.0),
        cmap="magma",
    )
    ax.plot(time, np.rad2deg(dark_turn["expected"]) % 360.0, "c--", lw=1.5, label="integrated turn")
    ax.plot(time, np.rad2deg(dark_heading), "w", lw=1.2, label="decoded bump")
    ax.axvspan(
        time[dark_turn["turn_start"]],
        time[dark_turn["turn_stop"] - 1],
        color="white",
        alpha=0.08,
    )
    ax.set(xlabel="time (ms)", ylabel="preferred direction (deg)", title="Dark rotation from north")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(image, ax=ax, label="filtered spike activity")

    ax = fig.add_subplot(grid[0, 1])
    tracking_error = np.rad2deg(
        np.abs(_circular_difference(dark_heading, dark_turn["expected"]))
    )
    ax.plot(time, tracking_error, color="black", label="angular error")
    ax.plot(time, 90.0 * dark_coherence, color="tab:green", label="90 x bump coherence")
    ax.axvline(time[dark_turn["turn_start"]], color="0.6", ls="--", lw=0.8)
    ax.axvline(time[dark_turn["turn_stop"] - 1], color="0.6", ls="--", lw=0.8)
    ax.set(xlabel="time (ms)", ylabel="degrees / scaled coherence", title="Path-integration observables")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 0])
    ax.axvspan(
        LESION_CENTER.to_decimal(u.degree) - LESION_WIDTH.to_decimal(u.degree) / 2.0,
        LESION_CENTER.to_decimal(u.degree) + LESION_WIDTH.to_decimal(u.degree) / 2.0,
        color="0.85",
        label="50 ms wedge pulse",
    )
    ax.plot(ring_deg, outcomes["control_final_error_deg"], color="0.55", lw=1.0, label="matched control")
    ax.plot(ring_deg, outcomes["lesion_final_error_deg"], color="tab:red", marker="o", ms=3, lw=1.0, label="wedge silenced")
    ax.axhline(FINAL_ERROR_LIMIT_DEG, color="black", ls="--", lw=0.8, label="success limit")
    ax.set(xlim=(0, 360), xticks=np.arange(0, 361, 60), xlabel="starting direction (deg)", ylabel="final angular error (deg)", title="Every initial heading, aligned controls")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 1], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    colors = {"spared": "#2a9d8f", "recovered": "#e9c46a", "failed": "#d1495b"}
    for label in ("spared", "recovered", "failed"):
        selected = outcomes["labels"] == label
        ax.scatter(sweep["headings"][selected], np.ones(selected.sum()), s=42, color=colors[label], label=label)
    wedge_center = np.deg2rad(LESION_CENTER.to_decimal(u.degree))
    wedge_width = np.deg2rad(LESION_WIDTH.to_decimal(u.degree))
    ax.bar(wedge_center, 1.18, width=wedge_width, bottom=0.0, color="0.75", alpha=0.35, edgecolor="none")
    ax.set_ylim(0.0, 1.22)
    ax.set_yticks([])
    ax.set_title("Outcome map around the ring", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)

    fig.suptitle("Internal neural compass: spiking ring attractor", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def save_data(dark_turn, sweep, outcomes, path=OUTPUT_DATA):
    dark_heading, dark_coherence, _ = decode_activity(dark_turn["activity"])
    np.savez_compressed(
        path,
        dark_time_ms=dark_turn["times_ms"],
        dark_activity=dark_turn["activity"],
        dark_expected_rad=dark_turn["expected"],
        dark_decoded_heading_rad=dark_heading,
        dark_coherence=dark_coherence,
        lesion_time_ms=sweep["times_ms"],
        lesion_start_ms=sweep["times_ms"][sweep["lesion_start"]],
        lesion_stop_ms=sweep["times_ms"][sweep["lesion_stop"]],
        start_heading_rad=sweep["headings"],
        lesion_mask=sweep["lesion_mask"],
        lesion_final_error_deg=outcomes["lesion_final_error_deg"],
        control_final_error_deg=outcomes["control_final_error_deg"],
        lesion_final_coherence=outcomes["lesion_final_coherence"],
        control_final_coherence=outcomes["control_final_coherence"],
        matched_mass_ratio=outcomes["matched_mass_ratio"],
        max_post_error_deg=outcomes["max_post_error_deg"],
        min_post_mass_ratio=outcomes["min_post_mass_ratio"],
        sustained_fraction=outcomes["sustained_fraction"],
        final_good=outcomes["final_good"],
        departed=outcomes["departed"],
        outcome=outcomes["labels"],
    )
    return path


def main():
    brainstate.random.seed(7)
    verify_delay_convention()
    dark_turn = run_dark_turn()
    sweep = run_lesion_sweep()
    outcomes = classify_lesions(sweep)
    turn_displacement = validate_results(dark_turn, sweep, outcomes)

    dark_heading, dark_coherence, _ = decode_activity(dark_turn["activity"])
    final_dark_error = np.rad2deg(
        abs(_circular_difference(dark_heading[-1], dark_turn["expected"][-1]))
    )
    final_turn_deg = np.rad2deg(dark_heading[-1]) % 360.0
    counts = {label: int(np.sum(outcomes["labels"] == label)) for label in ("spared", "recovered", "failed")}
    recovered_deg = np.rad2deg(
        sweep["headings"][outcomes["labels"] == "recovered"]
    )
    failed_deg = np.rad2deg(sweep["headings"][outcomes["labels"] == "failed"])

    figure = plot_results(dark_turn, sweep, outcomes)
    data = save_data(dark_turn, sweep, outcomes)
    print(
        f"Dark turn: commanded 90 deg, bump moved {turn_displacement:.1f} deg "
        f"and ended at {final_turn_deg:.1f} deg "
        f"(error {final_dark_error:.1f} deg, coherence {dark_coherence[-1]:.2f})."
    )
    print(
        "Lesion sweep: "
        f"{counts['spared']} spared, {counts['recovered']} recovered, "
        f"{counts['failed']} failed out of {NUM_NEURONS} headings."
    )
    print(f"Recovered starting directions: {_contiguous_ranges(recovered_deg)}")
    print(f"Failed starting directions: {_contiguous_ranges(failed_deg)}")
    print(f"Saved {figure} and {data}")


if __name__ == "__main__":
    main()
