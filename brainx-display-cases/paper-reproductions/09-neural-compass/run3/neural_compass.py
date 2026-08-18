"""Spiking head-direction ring attractor with a permanent wedge lesion.

The experiment has two parts:

1. Cue a bump at north, remove visual input, and apply angular velocity in
   darkness. The decoded bump is compared with the integrated true heading.
2. Cue every represented heading, then silence a fixed wedge. Intact and
   lesioned trials run together as independent BrainState-vmapped lanes.

Run this file directly. It writes a figure, a per-heading CSV file, and a JSON
summary to ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


# Simulation and neuron parameters. The recurrent delay is one completed
# integration step because update() communicates the previous step's spikes.
N_NEURONS = 48
DT = 2.0 * u.ms
RECURRENT_DELAY = DT
TAU_MEMBRANE = 20.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
TAU_SYNAPSE = 12.0 * u.ms
TAU_READOUT = 40.0 * u.ms

V_REST = -60.0 * u.mV
V_RESET = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
MEMBRANE_RESISTANCE = 1.0 * u.ohm

TONIC_CURRENT = 9.0 * u.mA
CUE_CURRENT = 8.0 * u.mA
SILENCED_CURRENT = -20.0 * u.mA
LOCAL_WEIGHT = 1.9 * u.mA
GLOBAL_WEIGHT = 0.28 * u.mA
VELOCITY_WEIGHT = 0.033 * u.mA

RING_WIDTH_RAD = np.deg2rad(28.0)
CUE_WIDTH_RAD = np.deg2rad(20.0)
ANGULAR_SPEED_SCALE = 0.5 * np.pi * u.radian / u.second
TURN_RATE = 0.5 * np.pi * u.radian / u.second

CUE_DURATION = 250.0 * u.ms
SETTLE_DURATION = 250.0 * u.ms
TURN_DURATION = 1.0 * u.second
POST_TURN_DURATION = 250.0 * u.ms
LESION_DURATION = 750.0 * u.ms

LESION_CENTER_RAD = np.deg2rad(135.0)
LESION_WIDTH_RAD = np.deg2rad(75.0)
ALIGNMENT_THRESHOLD_RAD = np.deg2rad(20.0)
MIN_BUMP_STRENGTH = 0.35
MIN_TOTAL_ACTIVITY = 1.0
MIN_RELATIVE_ACTIVITY = 0.25
SUSTAINED_WINDOW = 150.0 * u.ms


def wrap_angle(angle):
    """Wrap radians to the half-open interval [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def circular_error(angle, reference):
    """Return the absolute shortest angular distance in radians."""
    return np.abs(wrap_angle(angle - reference))


class HeadDirectionRing(brainstate.nn.Module):
    """LIF head-direction cells with symmetric and velocity-skew ring input."""

    def __init__(self, n_neurons: int = N_NEURONS):
        super().__init__()
        self.n_neurons = n_neurons
        self.preferred = jnp.linspace(-np.pi, np.pi, n_neurons, endpoint=False)

        # Connectivity is [presynaptic heading, postsynaptic heading]. The
        # symmetric Mexican-hat term holds a localized bump. The odd derivative
        # term pushes active spikes toward the direction selected by velocity.
        delta = wrap_angle(self.preferred[None, :] - self.preferred[:, None])
        local = jnp.exp(-0.5 * (delta / RING_WIDTH_RAD) ** 2)
        velocity = (delta / RING_WIDTH_RAD) * local
        self.symmetric_connectivity = (
            LOCAL_WEIGHT * local - GLOBAL_WEIGHT
        )
        self.velocity_connectivity = VELOCITY_WEIGHT * velocity

        self.neurons = brainpy.state.LIFRef(
            n_neurons,
            R=MEMBRANE_RESISTANCE,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.recurrent_synapse = brainpy.state.Expon(
            n_neurons,
            tau=TAU_SYNAPSE,
            g_initializer=braintools.init.Constant(0.0 * u.mA),
        )
        self.velocity_synapse = brainpy.state.Expon(
            n_neurons,
            tau=TAU_SYNAPSE,
            g_initializer=braintools.init.Constant(0.0 * u.mA),
        )
        self.activity_readout = brainpy.state.Expon(
            n_neurons,
            tau=TAU_READOUT,
            g_initializer=braintools.init.Constant(0.0),
        )
        self.recurrent_output = brainpy.state.CUBA(scale=1.0)
        self.velocity_output = brainpy.state.CUBA(scale=1.0)
        self.neurons.add_current_input("ring_recurrent", self.recurrent_output)
        self.neurons.add_current_input("ring_velocity", self.velocity_output)

    def update(self, t, cue_current, angular_velocity, active_mask):
        """Advance one lane by one dt and return its filtered spike activity."""
        with brainstate.environ.context(t=t):
            previous_spikes = (self.neurons.get_spike() != 0.0) & active_mask
            events = brainevent.BinaryArray(previous_spikes)
            symmetric_input = events @ self.symmetric_connectivity
            velocity_input = events @ self.velocity_connectivity

            recurrent = self.recurrent_synapse(symmetric_input)
            velocity_gain = angular_velocity / ANGULAR_SPEED_SCALE
            velocity_drive = self.velocity_synapse(velocity_input) * velocity_gain
            self.recurrent_output.bind_cond(recurrent)
            self.velocity_output.bind_cond(velocity_drive)
            total_current = TONIC_CURRENT + cue_current
            total_current = u.math.where(
                active_mask, total_current, SILENCED_CURRENT
            )

            spikes = (self.neurons(total_current) != 0.0) & active_mask
            return self.activity_readout(spikes.astype(jnp.float32))


def cue_profiles(preferred, headings):
    """Return one circular Gaussian cue profile per heading."""
    difference = wrap_angle(preferred[None, :] - headings[:, None])
    return jnp.exp(-0.5 * (difference / CUE_WIDTH_RAD) ** 2)


def wedge_mask(preferred):
    """Boolean mask that is false inside the silenced wedge."""
    return jnp.abs(wrap_angle(preferred - LESION_CENTER_RAD)) > 0.5 * LESION_WIDTH_RAD


def make_protocol(
    preferred,
    headings,
    duration,
    angular_velocity,
    lesion_flags,
):
    """Build complete time-major cue, velocity, and lesion protocols."""
    times = u.math.arange(0.0 * u.ms, duration, DT)
    time_ms = times.to_decimal(u.ms)
    cue_on = time_ms < CUE_DURATION.to_decimal(u.ms)
    cue = (
        cue_on[:, None, None]
        * cue_profiles(preferred, headings)[None, :, :]
        * CUE_CURRENT
    )

    velocity = u.math.broadcast_to(
        angular_velocity, (times.shape[0], headings.shape[0])
    )

    intact = jnp.ones((headings.shape[0], preferred.shape[0]), dtype=bool)
    lesioned = jnp.broadcast_to(wedge_mask(preferred), intact.shape)
    post_settle = time_ms >= (CUE_DURATION + SETTLE_DURATION).to_decimal(u.ms)
    condition_mask = jnp.where(lesion_flags[:, None], lesioned, intact)
    active = jnp.where(post_settle[:, None, None], condition_mask[None], intact[None])
    return times, cue, velocity, active


def simulate_protocol(headings, duration, angular_velocity, lesion_flags):
    """Run independent conditions through one vmap-over-condition/loop-over-time path."""
    ring = HeadDirectionRing()
    times, cue, velocity, active = make_protocol(
        ring.preferred,
        headings,
        duration,
        angular_velocity,
        lesion_flags,
    )

    brainstate.nn.vmap_init_all_states(ring, axis_size=headings.shape[0])
    dynamical_state = state_filter.Any(
        state_filter.OfType(brainstate.HiddenState),
        state_filter.OfType(brainstate.ShortTermState),
    )
    mapped_step = vmap2(
        ring.update,
        in_axes=(None, 0, 0, 0),
        out_axes=0,
        state_in_axes={0: dynamical_state},
        state_out_axes={0: dynamical_state},
        unexpected_out_state_mapping="raise",
    )

    @brainstate.transform.jit
    def run():
        return brainstate.transform.for_loop(
            mapped_step, times, cue, velocity, active
        )

    activity = run()
    return (
        np.asarray(times.to_decimal(u.second)),
        np.asarray(ring.preferred),
        np.asarray(activity),
    )


def decode_activity(activity, preferred):
    """Decode population-vector heading and concentration from activity."""
    vector = np.sum(activity * np.exp(1j * preferred), axis=-1)
    total = np.sum(activity, axis=-1)
    heading = np.angle(vector)
    strength = np.abs(vector) / np.maximum(total, 1e-9)
    return heading, strength


def classify_lesion_outcomes(
    control_heading,
    lesion_heading,
    lesion_strength,
    lesion_on_index,
    sustained_steps,
    relative_activity=None,
):
    """Classify matched trials from time-resolved departure and return."""
    error = circular_error(lesion_heading, control_heading)
    post_error = error[lesion_on_index:]
    post_strength = lesion_strength[lesion_on_index:]
    if relative_activity is None:
        relative_activity = np.ones_like(lesion_strength)
    post_relative_activity = relative_activity[lesion_on_index:]
    departure = np.any(
        (post_error > ALIGNMENT_THRESHOLD_RAD)
        | (post_strength < MIN_BUMP_STRENGTH)
        | (post_relative_activity < MIN_RELATIVE_ACTIVITY),
        axis=0,
    )
    aligned_tail = (
        error[-sustained_steps:] <= ALIGNMENT_THRESHOLD_RAD
    ) & (
        lesion_strength[-sustained_steps:] >= MIN_BUMP_STRENGTH
    ) & (
        relative_activity[-sustained_steps:] >= MIN_RELATIVE_ACTIVITY
    )
    sustained_return = np.all(aligned_tail, axis=0)

    labels = np.full(error.shape[1], "failed", dtype="U9")
    labels[sustained_return & ~departure] = "spared"
    labels[sustained_return & departure] = "recovered"
    return labels, departure, sustained_return, error


def run_rotation_experiment():
    """Cue north, turn 90 degrees in darkness, and score bump tracking."""
    duration = CUE_DURATION + SETTLE_DURATION + TURN_DURATION + POST_TURN_DURATION
    headings = jnp.asarray([0.0])
    turn_start_s = (CUE_DURATION + SETTLE_DURATION).to_decimal(u.second)
    turn_end_s = turn_start_s + TURN_DURATION.to_decimal(u.second)
    times = u.math.arange(0.0 * u.ms, duration, DT).to_decimal(u.second)
    omega = jnp.where(
        (times >= turn_start_s) & (times < turn_end_s),
        TURN_RATE.to_decimal(u.radian / u.second),
        0.0,
    ) * u.radian / u.second

    time_s, preferred, activity = simulate_protocol(
        headings,
        duration,
        omega[:, None],
        jnp.asarray([False]),
    )
    decoded, strength = decode_activity(activity[:, 0], preferred)
    true_heading = np.clip(time_s - turn_start_s, 0.0, TURN_DURATION.to_decimal(u.second))
    true_heading = true_heading * TURN_RATE.to_decimal(u.radian / u.second)

    turn_start_index = int(np.searchsorted(time_s, turn_start_s))
    final_window = time_s >= turn_end_s
    unwrapped = np.unwrap(decoded)
    baseline = unwrapped[turn_start_index]
    decoded_displacement = unwrapped - baseline
    final_decoded = float(np.mean(decoded_displacement[final_window]))
    final_true = float(true_heading[-1])
    final_error = abs(final_decoded - final_true)
    turn_gain = final_decoded / final_true
    return {
        "time_s": time_s,
        "preferred": preferred,
        "activity": activity[:, 0],
        "decoded": decoded_displacement,
        "strength": strength,
        "true_heading": true_heading,
        "final_error_rad": final_error,
        "turn_gain": turn_gain,
    }


def run_lesion_sweep():
    """Run intact and lesioned versions of every represented initial heading."""
    preferred = jnp.linspace(-np.pi, np.pi, N_NEURONS, endpoint=False)
    headings = jnp.concatenate([preferred, preferred])
    lesion_flags = jnp.concatenate([
        jnp.zeros(N_NEURONS, dtype=bool),
        jnp.ones(N_NEURONS, dtype=bool),
    ])
    duration = CUE_DURATION + SETTLE_DURATION + LESION_DURATION
    n_steps = int(round(duration.to_decimal(u.ms) / DT.to_decimal(u.ms)))
    zero_velocity = jnp.zeros((n_steps, headings.shape[0])) * u.radian / u.second
    time_s, preferred_np, activity = simulate_protocol(
        headings,
        duration,
        zero_velocity,
        lesion_flags,
    )
    decoded, strength = decode_activity(activity, preferred_np)
    control_heading = decoded[:, :N_NEURONS]
    lesion_heading = decoded[:, N_NEURONS:]
    lesion_strength = strength[:, N_NEURONS:]
    control_total = activity[:, :N_NEURONS].sum(axis=-1)
    lesion_total = activity[:, N_NEURONS:].sum(axis=-1)
    relative_activity = lesion_total / np.maximum(control_total, 1e-9)
    lesion_on_s = (CUE_DURATION + SETTLE_DURATION).to_decimal(u.second)
    lesion_on_index = int(np.searchsorted(time_s, lesion_on_s))
    sustained_steps = int(round(SUSTAINED_WINDOW / DT))
    labels, departure, sustained, matched_error = classify_lesion_outcomes(
        control_heading,
        lesion_heading,
        lesion_strength,
        lesion_on_index,
        sustained_steps,
        relative_activity,
    )

    control_tail = control_heading[-sustained_steps:]
    control_error = circular_error(control_tail, preferred_np[None, :])
    control_strength = strength[-sustained_steps:, :N_NEURONS]
    control_tail_activity = control_total[-sustained_steps:]
    control_valid = np.all(
        (control_error <= ALIGNMENT_THRESHOLD_RAD)
        & (control_strength >= MIN_BUMP_STRENGTH),
        axis=0,
    ) & np.all(
        control_tail_activity >= MIN_TOTAL_ACTIVITY,
        axis=0,
    )
    return {
        "time_s": time_s,
        "preferred": preferred_np,
        "control_heading": control_heading,
        "lesion_heading": lesion_heading,
        "lesion_strength": lesion_strength,
        "relative_activity": relative_activity,
        "control_total_activity": control_total,
        "sustained_steps": sustained_steps,
        "labels": labels,
        "departure": departure,
        "sustained_return": sustained,
        "matched_error": matched_error,
        "control_valid": control_valid,
    }


def write_lesion_csv(result, path):
    """Write the exact per-heading observables used by the classifier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "initial_heading_deg",
            "inside_lesion_wedge",
            "intact_control_valid",
            "departure_observed",
            "sustained_return",
            "final_matched_error_deg",
            "maximum_matched_error_deg",
            "final_bump_strength",
            "final_relative_activity",
            "maximum_tail_error_deg",
            "minimum_tail_bump_strength",
            "minimum_tail_relative_activity",
            "minimum_tail_control_activity",
            "outcome",
        ])
        inside = ~np.asarray(wedge_mask(jnp.asarray(result["preferred"])))
        tail = slice(-result["sustained_steps"], None)
        for index, heading in enumerate(result["preferred"]):
            writer.writerow([
                f"{np.rad2deg(heading):.1f}",
                bool(inside[index]),
                bool(result["control_valid"][index]),
                bool(result["departure"][index]),
                bool(result["sustained_return"][index]),
                f"{np.rad2deg(result['matched_error'][-1, index]):.3f}",
                f"{np.rad2deg(result['matched_error'][:, index].max()):.3f}",
                f"{result['lesion_strength'][-1, index]:.4f}",
                f"{result['relative_activity'][-1, index]:.4f}",
                f"{np.rad2deg(result['matched_error'][tail, index].max()):.3f}",
                f"{result['lesion_strength'][tail, index].min():.4f}",
                f"{result['relative_activity'][tail, index].min():.4f}",
                f"{result['control_total_activity'][tail, index].min():.4f}",
                result["labels"][index],
            ])


def plot_results(rotation, lesion, path):
    """Plot bump tracking and the full lesion outcome map."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    activity = rotation["activity"]
    normalized = activity / np.maximum(activity.max(axis=1, keepdims=True), 1e-9)
    extent = [
        rotation["time_s"][0],
        rotation["time_s"][-1],
        np.rad2deg(rotation["preferred"])[0],
        np.rad2deg(rotation["preferred"])[-1],
    ]
    image = axes[0].imshow(
        normalized.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].plot(
        rotation["time_s"],
        np.rad2deg(rotation["true_heading"]),
        color="white",
        linewidth=2.0,
        label="integrated velocity",
    )
    axes[0].plot(
        rotation["time_s"],
        np.rad2deg(rotation["decoded"]),
        color="#38bdf8",
        linewidth=1.2,
        label="decoded bump",
    )
    axes[0].set(ylabel="preferred heading (deg)", title="North cue, then a 90 deg turn in darkness")
    axes[0].legend(loc="upper left", frameon=False, ncol=2, labelcolor="white")
    fig.colorbar(image, ax=axes[0], label="normalized spike trace")

    initial_deg = np.rad2deg(lesion["preferred"])
    final_control = np.rad2deg(lesion["control_heading"][-1])
    final_lesion = np.rad2deg(lesion["lesion_heading"][-1])
    wedge_low = np.rad2deg(LESION_CENTER_RAD - 0.5 * LESION_WIDTH_RAD)
    wedge_high = np.rad2deg(LESION_CENTER_RAD + 0.5 * LESION_WIDTH_RAD)
    axes[1].axvspan(wedge_low, wedge_high, color="#ef4444", alpha=0.12, label="silenced wedge")
    axes[1].plot(initial_deg, initial_deg, color="0.65", linestyle="--", linewidth=1.0)
    axes[1].plot(initial_deg, final_control, color="#2563eb", linewidth=1.3, label="intact")
    axes[1].plot(initial_deg, final_lesion, color="#111827", linewidth=1.3, label="lesioned")
    axes[1].set(
        xlim=(-180, 180),
        ylim=(-180, 180),
        xlabel="initial heading (deg)",
        ylabel="final decoded heading (deg)",
        title="Matched final headings for every ring direction",
    )
    axes[1].legend(loc="upper left", frameon=False, ncol=3)

    colors = {"spared": "#16a34a", "recovered": "#f59e0b", "failed": "#dc2626"}
    max_error = np.rad2deg(lesion["matched_error"].max(axis=0))
    for label in ("spared", "recovered", "failed"):
        selected = lesion["labels"] == label
        axes[2].scatter(
            initial_deg[selected],
            max_error[selected],
            s=35,
            color=colors[label],
            label=f"{label} ({selected.sum()})",
        )
    axes[2].axvspan(wedge_low, wedge_high, color="#ef4444", alpha=0.12)
    axes[2].axhline(
        np.rad2deg(ALIGNMENT_THRESHOLD_RAD),
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label="alignment threshold",
    )
    axes[2].set(
        xlim=(-180, 180),
        xlabel="initial heading (deg)",
        ylabel="maximum lesion-control error (deg)",
        title="Outcome map after wedge silencing",
    )
    axes[2].legend(loc="upper left", frameon=False, ncol=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(output_dir: Path):
    brainstate.environ.set(dt=DT, precision=32)
    brainstate.random.seed(7)

    rotation = run_rotation_experiment()
    lesion = run_lesion_sweep()

    if (
        rotation["final_error_rad"] > ALIGNMENT_THRESHOLD_RAD
        or rotation["strength"][-1] < MIN_BUMP_STRENGTH
        or rotation["activity"][-1].sum() < MIN_TOTAL_ACTIVITY
    ):
        raise RuntimeError(
            "Dark-turn control failed: "
            f"error={np.rad2deg(rotation['final_error_rad']):.2f} deg, "
            f"strength={rotation['strength'][-1]:.3f}, "
            f"activity={rotation['activity'][-1].sum():.3f}"
        )
    if not np.all(lesion["control_valid"]):
        invalid = np.rad2deg(lesion["preferred"][~lesion["control_valid"]])
        raise RuntimeError(f"Intact bump controls failed at headings {invalid.tolist()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "neural_compass_results.png"
    csv_path = output_dir / "lesion_sweep.csv"
    summary_path = output_dir / "summary.json"
    write_lesion_csv(lesion, csv_path)
    plot_results(rotation, lesion, figure_path)

    counts = {
        label: int(np.sum(lesion["labels"] == label))
        for label in ("spared", "recovered", "failed")
    }
    summary = {
        "n_headings_tested": N_NEURONS,
        "dt_ms": DT.to_decimal(u.ms),
        "recurrent_delay_ms": RECURRENT_DELAY.to_decimal(u.ms),
        "turn_rate_rad_per_s": TURN_RATE.to_decimal(u.radian / u.second),
        "commanded_turn_deg": 90.0,
        "decoded_turn_gain": rotation["turn_gain"],
        "final_rotation_error_deg": np.rad2deg(rotation["final_error_rad"]),
        "final_rotation_bump_strength": float(rotation["strength"][-1]),
        "lesion_center_deg": np.rad2deg(LESION_CENTER_RAD),
        "lesion_width_deg": np.rad2deg(LESION_WIDTH_RAD),
        "alignment_threshold_deg": np.rad2deg(ALIGNMENT_THRESHOLD_RAD),
        "minimum_bump_strength": MIN_BUMP_STRENGTH,
        "minimum_total_activity": MIN_TOTAL_ACTIVITY,
        "minimum_relative_activity": MIN_RELATIVE_ACTIVITY,
        "sustained_window_ms": SUSTAINED_WINDOW.to_decimal(u.ms),
        "outcome_counts": counts,
        "headings_deg_by_outcome": {
            label: [
                round(float(np.rad2deg(heading)), 1)
                for heading in lesion["preferred"][lesion["labels"] == label]
            ]
            for label in ("spared", "recovered", "failed")
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")

    print(
        f"Dark turn: gain={rotation['turn_gain']:.3f}, "
        f"final error={np.rad2deg(rotation['final_error_rad']):.2f} deg"
    )
    print(f"Lesion outcomes over {N_NEURONS} headings: {counts}")
    print(f"Figure: {figure_path}")
    print(f"Per-heading results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="directory for the figure, CSV, and JSON summary",
    )
    main(parser.parse_args().output_dir)
