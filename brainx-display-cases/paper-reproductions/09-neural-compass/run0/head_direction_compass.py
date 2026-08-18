"""Spiking head-direction ring attractor with a mapped lesion sweep.

The model is intentionally compact: one recurrent LIF population, one
current-based exponential synapse, and three dense BrainEvent pathways.  The
symmetric pathway holds a bump; two one-cell-shifted pathways are gated by
positive and negative angular velocity to move it around the ring.

Run this file to create a figure, per-heading CSV, and JSON summary in
``results/``.
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
from braintools.input import Constant
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter


# Unit-bearing simulation and neuron parameters.
DT = 1.0 * u.ms
RECURRENT_DELAY = 1.0 * u.ms  # get_spike() supplies the previous completed step
TAU_MEMBRANE = 20.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
TAU_SYNAPSE = 12.0 * u.ms
MEMBRANE_RESISTANCE = 100.0 * u.Mohm
V_REST = -65.0 * u.mV
V_RESET = -65.0 * u.mV
V_THRESHOLD = -50.0 * u.mV

BASE_CURRENT = 0.130 * u.nA
CUE_CURRENT = 0.24 * u.nA
SILENCE_CURRENT = 0.8 * u.nA
LOCAL_WEIGHT = 0.050 * u.mS
GLOBAL_INHIBITION = 0.006 * u.mS
TURN_WEIGHT = 0.0055 * u.mS
CUBA_SCALE = 1.0 * u.nA / u.mS

TURN_RATE = 180.0 * u.degree / u.second
CUE_DURATION = 250.0 * u.ms
PRE_TURN_HOLD = 350.0 * u.ms
TURN_DURATION = 500.0 * u.ms
RECOVERY_DURATION = 900.0 * u.ms
LESION_ONSET = 500.0 * u.ms
TOTAL_DURATION = CUE_DURATION + PRE_TURN_HOLD + TURN_DURATION + RECOVERY_DURATION

LESION_CENTER_DEG = 0.0
LESION_WIDTH_DEG = 70.0
FINAL_WINDOW = 250.0 * u.ms
RATE_WINDOW = 60.0 * u.ms

# These thresholds define the reported phenomenological recovery regime.
MAX_CONTROL_ERROR_DEG = 25.0
MAX_CONTROL_TURN_RMSE_DEG = 12.0
MAX_LESION_ERROR_DEG = 30.0
MIN_CONTROL_RESULTANT = 0.25
MIN_LESION_RESULTANT = 0.22
MIN_ACTIVITY_RATIO = 0.35


def wrap_radians(angle):
    """Wrap radians to [-pi, pi)."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def circular_error_deg(angle_deg, reference_deg):
    """Smallest absolute separation between headings in degrees."""
    return np.abs((np.asarray(angle_deg) - np.asarray(reference_deg) + 180.0) % 360.0 - 180.0)


def preferred_angles(num_neurons: int):
    """Neuron preferences: 0 is north and positive angles turn clockwise."""
    return 2.0 * jnp.pi * jnp.arange(num_neurons) / num_neurons


def ring_weights(num_neurons: int):
    """Return symmetric and velocity-shifted dense event matrices.

    Dense storage is appropriate here because the cosine recurrent kernel is
    genuinely dense and the demonstration ring is small.
    """
    angles = preferred_angles(num_neurons)
    delta = wrap_radians(angles[None, :] - angles[:, None])
    local = jnp.exp(3.2 * (jnp.cos(delta) - 1.0))
    hold = LOCAL_WEIGHT * local - GLOBAL_INHIBITION

    derivative = TURN_WEIGHT * jnp.sin(delta) * local
    positive = derivative
    negative = -derivative
    return hold, positive, negative


class HeadDirectionRing(brainstate.nn.Module):
    """One spiking head-direction population with event-driven recurrence."""

    def __init__(self, num_neurons: int = 36):
        super().__init__()
        self.num_neurons = num_neurons
        self.angles = preferred_angles(num_neurons)
        self.hold_weights, self.positive_weights, self.negative_weights = ring_weights(num_neurons)

        self.neurons = brainpy.state.LIFRef(
            num_neurons,
            R=MEMBRANE_RESISTANCE,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_rest=V_REST,
            V_reset=V_RESET,
            V_th=V_THRESHOLD,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.recurrent_synapse = brainpy.state.Expon(num_neurons, tau=TAU_SYNAPSE)
        self.recurrent_output = brainpy.state.CUBA(scale=CUBA_SCALE)
        self.neurons.add_current_input("recurrent", self.recurrent_output)

    def update(self, t, initial_heading, lesion_condition, cue_current, angular_velocity, lesion_gate):
        with brainstate.environ.context(t=t):
            wedge = jnp.abs(wrap_radians(self.angles - LESION_CENTER_DEG * jnp.pi / 180.0)) <= (
                LESION_WIDTH_DEG * jnp.pi / 360.0
            )
            lesion_active = lesion_condition & (lesion_gate > 0.5)
            silenced = wedge & lesion_active
            live = ~silenced

            previous_spikes = (self.neurons.get_spike() != 0.0) & live
            events = brainevent.BinaryArray(previous_spikes)
            recurrent = events @ self.hold_weights

            relative_speed = angular_velocity / TURN_RATE
            positive_gain = u.math.maximum(relative_speed, 0.0)
            negative_gain = u.math.maximum(-relative_speed, 0.0)
            recurrent = recurrent + positive_gain * (events @ self.positive_weights)
            recurrent = recurrent + negative_gain * (events @ self.negative_weights)
            recurrent = recurrent * live
            self.recurrent_output.bind_cond(self.recurrent_synapse(recurrent))

            cue_profile = jnp.exp(4.5 * (jnp.cos(self.angles - initial_heading) - 1.0))
            external = BASE_CURRENT + cue_profile * cue_current - silenced * SILENCE_CURRENT
            spikes = self.neurons(external)
            return (spikes != 0.0) & live


def build_protocols():
    """Create time-major sensory, velocity, and lesion protocols."""
    dark_duration = TOTAL_DURATION - CUE_DURATION
    turn_start = CUE_DURATION + PRE_TURN_HOLD
    after_turn = TOTAL_DURATION - turn_start - TURN_DURATION

    cue = Constant([(CUE_CURRENT, CUE_DURATION), (0.0 * u.nA, dark_duration)])()
    velocity = Constant(
        [
            (0.0 * u.degree / u.second, turn_start),
            (TURN_RATE, TURN_DURATION),
            (0.0 * u.degree / u.second, after_turn),
        ]
    )()
    lesion = Constant([(0.0, LESION_ONSET), (1.0, TOTAL_DURATION - LESION_ONSET)])()
    times = u.math.arange(0.0 * u.ms, TOTAL_DURATION, brainstate.environ.get_dt())
    return times, cue, velocity, lesion


def run_sweep(num_neurons: int = 36):
    """Run control and wedge-lesion trials for every represented heading."""
    if not np.isclose(RECURRENT_DELAY.to_decimal(u.ms), DT.to_decimal(u.ms)):
        raise ValueError("The previous-spike recurrent delay must equal DT.")

    with brainstate.environ.context(dt=DT):
        model = HeadDirectionRing(num_neurons)
        times, cue, velocity, lesion_gate = build_protocols()

        headings = preferred_angles(num_neurons)
        trial_headings = jnp.tile(headings, 2)
        lesion_conditions = jnp.repeat(jnp.asarray([False, True]), num_neurons)
        num_trials = trial_headings.shape[0]
        brainstate.nn.vmap_init_all_states(model, axis_size=num_trials)

        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = vmap2(
            model.update,
            in_axes=(None, 0, 0, None, None, None),
            out_axes=0,
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )

        @brainstate.transform.jit
        def run():
            def step(t, cue_at_t, velocity_at_t, lesion_at_t):
                return mapped_step(
                    t,
                    trial_headings,
                    lesion_conditions,
                    cue_at_t,
                    velocity_at_t,
                    lesion_at_t,
                )

            return brainstate.transform.for_loop(step, times, cue, velocity, lesion_gate)

        spikes = run()

    return {
        "times": times,
        "velocity": velocity,
        "lesion_gate": lesion_gate,
        "spikes": spikes,
        "preferred_angles": headings,
    }


def moving_rates(spikes: np.ndarray, window_steps: int, dt_seconds: float):
    """Trailing-window firing rates with time on axis zero."""
    values = spikes.astype(np.float32)
    cumulative = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, axis=0)], axis=0)
    starts = np.maximum(np.arange(values.shape[0]) - window_steps + 1, 0)
    counts = cumulative[np.arange(1, values.shape[0] + 1)] - cumulative[starts]
    widths = (np.arange(values.shape[0]) - starts + 1) * dt_seconds
    return counts / widths.reshape((-1,) + (1,) * (values.ndim - 1))


def decode_population(activity: np.ndarray, angles_rad: np.ndarray):
    """Population-vector heading, resultant length, and total activity."""
    x = np.sum(activity * np.cos(angles_rad), axis=-1)
    y = np.sum(activity * np.sin(angles_rad), axis=-1)
    total = np.sum(activity, axis=-1)
    heading = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    resultant = np.hypot(x, y) / np.maximum(total, 1e-12)
    return heading, resultant, total


def analyze(result):
    """Compare each lesion trial with its matched intact control."""
    spikes = np.asarray(result["spikes"], dtype=bool)
    angles = np.asarray(result["preferred_angles"])
    num_neurons = angles.size
    dt_s = DT.to_decimal(u.second)
    window_steps = int(round(RATE_WINDOW / DT))
    rates = moving_rates(spikes, window_steps, dt_s)
    control_heading_trace, control_resultant_trace, _ = decode_population(
        rates[:, :num_neurons], angles
    )
    lesion_heading_trace, lesion_resultant_trace, _ = decode_population(
        rates[:, num_neurons:], angles
    )

    turn_start = int(round((CUE_DURATION + PRE_TURN_HOLD) / DT))
    turn_end = turn_start + int(round(TURN_DURATION / DT))
    control_phase = np.degrees(np.unwrap(np.radians(control_heading_trace), axis=0))
    observed_turn = control_phase[turn_start:turn_end] - control_phase[turn_start - 1]
    velocity_deg_s = np.asarray(result["velocity"].to_decimal(u.degree / u.second))
    expected_turn = np.cumsum(velocity_deg_s[turn_start:turn_end] * dt_s)
    control_turn_rmse = np.sqrt(np.mean((observed_turn - expected_turn[:, None]) ** 2, axis=0))
    control_turn_displacement = control_phase[turn_end - 1] - control_phase[turn_start - 1]

    final_steps = int(round(FINAL_WINDOW / DT))
    final_counts = spikes[-final_steps:].sum(axis=0)
    final_heading, final_resultant, final_total = decode_population(final_counts, angles)

    control_heading = final_heading[:num_neurons]
    lesion_heading = final_heading[num_neurons:]
    control_resultant = final_resultant[:num_neurons]
    lesion_resultant = final_resultant[num_neurons:]
    control_total = final_total[:num_neurons]
    lesion_total = final_total[num_neurons:]

    velocity_rad_s = np.asarray(result["velocity"].to_decimal(u.radian / u.second))
    integrated_turn_deg = np.degrees(velocity_rad_s.sum() * dt_s)
    start_deg = np.degrees(angles)
    expected_deg = np.mod(start_deg + integrated_turn_deg, 360.0)
    control_error = circular_error_deg(control_heading, expected_deg)
    lesion_error = circular_error_deg(lesion_heading, control_heading)
    activity_ratio = lesion_total / np.maximum(control_total, 1.0)
    coherence_ratio = lesion_resultant / np.maximum(control_resultant, 1e-12)

    control_valid = (
        (control_error <= MAX_CONTROL_ERROR_DEG)
        & (control_turn_rmse <= MAX_CONTROL_TURN_RMSE_DEG)
        & (control_resultant >= MIN_CONTROL_RESULTANT)
        & (control_total > 0.0)
    )
    recovered = (
        control_valid
        & (lesion_error <= MAX_LESION_ERROR_DEG)
        & (lesion_resultant >= MIN_LESION_RESULTANT)
        & (activity_ratio >= MIN_ACTIVITY_RATIO)
    )

    rows = []
    for i in range(num_neurons):
        rows.append(
            {
                "start_heading_deg": float(start_deg[i]),
                "expected_final_deg": float(expected_deg[i]),
                "control_final_deg": float(control_heading[i]),
                "control_error_deg": float(control_error[i]),
                "control_turn_displacement_deg": float(control_turn_displacement[i]),
                "control_turn_rmse_deg": float(control_turn_rmse[i]),
                "control_resultant": float(control_resultant[i]),
                "control_spikes_final_window": int(control_total[i]),
                "lesion_final_deg": float(lesion_heading[i]),
                "lesion_error_vs_control_deg": float(lesion_error[i]),
                "lesion_resultant": float(lesion_resultant[i]),
                "lesion_spikes_final_window": int(lesion_total[i]),
                "activity_ratio": float(activity_ratio[i]),
                "coherence_ratio": float(coherence_ratio[i]),
                "control_valid": bool(control_valid[i]),
                "recovered": bool(recovered[i]),
            }
        )

    return {
        "rows": rows,
        "integrated_turn_deg": float(integrated_turn_deg),
        "control_valid": control_valid,
        "recovered": recovered,
        "rates": rates,
        "control_heading_trace": control_heading_trace,
        "control_resultant_trace": control_resultant_trace,
        "lesion_heading_trace": lesion_heading_trace,
        "lesion_resultant_trace": lesion_resultant_trace,
    }


def expected_heading_trace(start_deg: float, velocity):
    velocity_rad_s = np.asarray(velocity.to_decimal(u.radian / u.second))
    increments = np.degrees(velocity_rad_s * DT.to_decimal(u.second))
    return np.mod(start_deg + np.cumsum(increments), 360.0)


def plot_results(result, analysis, output_path: Path):
    """Plot the north trial and the continuous lesion-sweep observables."""
    times_ms = np.asarray(result["times"].to_decimal(u.ms))
    angles = np.asarray(result["preferred_angles"])
    num_neurons = angles.size
    rates = analysis["rates"]
    control_rates = rates[:, :num_neurons]
    control_heading = analysis["control_heading_trace"]
    control_resultant = analysis["control_resultant_trace"]
    lesion_heading = analysis["lesion_heading_trace"]
    lesion_resultant = analysis["lesion_resultant_trace"]

    north = 0
    lesion_errors = np.array([row["lesion_error_vs_control_deg"] for row in analysis["rows"]])
    failed = ~analysis["recovered"]
    example = int(np.argmax(np.where(failed, lesion_errors, -1.0))) if failed.any() else north
    start_deg = np.degrees(angles)
    expected_north = expected_heading_trace(0.0, result["velocity"])
    expected_example = expected_heading_trace(start_deg[example], result["velocity"])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    image = ax.imshow(
        control_rates[:, north].T,
        origin="lower",
        aspect="auto",
        extent=[times_ms[0], times_ms[-1], 0.0, 360.0],
        cmap="magma",
        interpolation="nearest",
    )
    ax.plot(times_ms, expected_north, color="white", lw=1.4, label="integrated turn")
    ax.plot(
        times_ms,
        np.where(control_resultant[:, north] >= 0.12, control_heading[:, north], np.nan),
        color="#35d0ba",
        lw=1.0,
        label="decoded bump",
    )
    ax.set(title="A  North cue, then a 90 degree dark turn", ylabel="preferred heading (deg)")
    ax.set_yticks([0, 90, 180, 270, 360], ["N", "E", "S", "W", "N"])
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.colorbar(image, ax=ax, label="rate (Hz)", pad=0.01)

    ax = axes[0, 1]
    ax.plot(times_ms, expected_example, color="black", ls="--", lw=1.2, label="integrated turn")
    ax.plot(
        times_ms,
        np.where(control_resultant[:, example] >= 0.12, control_heading[:, example], np.nan),
        color="#087e8b",
        lw=1.2,
        label="control",
    )
    valid_lesion = lesion_resultant[:, example] >= 0.12
    ax.plot(
        times_ms,
        np.where(valid_lesion, lesion_heading[:, example], np.nan),
        color="#e4572e",
        lw=1.2,
        label="wedge silenced",
    )
    lesion_start_ms = LESION_ONSET.to_decimal(u.ms)
    ax.axvspan(lesion_start_ms, times_ms[-1], color="#d9d9d9", alpha=0.35, lw=0)
    ax.set(
        title=f"B  Example lesion trial, start {start_deg[example]:.0f} deg",
        ylabel="decoded heading (deg)",
        ylim=(0, 360),
    )
    ax.set_yticks([0, 90, 180, 270, 360], ["N", "E", "S", "W", "N"])
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    ax = axes[1, 0]
    recovered = analysis["recovered"]
    colors = np.where(recovered, "#2a9d8f", "#d1495b")
    ax.scatter(start_deg, lesion_errors, c=colors, s=34, zorder=3)
    ax.axhline(MAX_LESION_ERROR_DEG, color="black", ls="--", lw=1.0, label="error threshold")
    ax.set(
        title="C  Final lesion error against matched control",
        xlabel="starting heading (deg)",
        ylabel="circular error (deg)",
        xlim=(-5, 365),
        ylim=(-3, 183),
    )
    ax.set_xticks([0, 90, 180, 270, 360], ["N", "E", "S", "W", "N"])
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    activity_ratio = np.array([row["activity_ratio"] for row in analysis["rows"]])
    lesion_coherence = np.array([row["lesion_resultant"] for row in analysis["rows"]])
    ax.plot(start_deg, activity_ratio, color="#f4a261", marker="o", ms=3.5, lw=1.0, label="activity ratio")
    ax.plot(start_deg, lesion_coherence, color="#5e548e", marker="s", ms=3.2, lw=1.0, label="lesion resultant")
    ax.axhline(MIN_ACTIVITY_RATIO, color="#f4a261", ls="--", lw=0.9)
    ax.axhline(MIN_LESION_RESULTANT, color="#5e548e", ls="--", lw=0.9)
    ax.scatter(start_deg, np.full_like(start_deg, -0.07), c=colors, marker="|", s=85, clip_on=False)
    ax.set(
        title="D  Continuous recovery observables",
        xlabel="starting heading (deg)",
        ylabel="ratio or resultant",
        xlim=(-5, 365),
        ylim=(0, max(1.2, float(np.nanmax(activity_ratio)) * 1.08)),
    )
    ax.set_xticks([0, 90, 180, 270, 360], ["N", "E", "S", "W", "N"])
    ax.legend(frameon=False, fontsize=8, ncol=2)

    for ax in axes.flat:
        ax.set_xlabel("time (ms)" if ax in axes[0] else ax.get_xlabel())
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Spiking head-direction compass: intact tracking and wedge-lesion recovery",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(result, analysis, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lesion_sweep.csv"
    with csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(analysis["rows"][0]))
        writer.writeheader()
        writer.writerows(analysis["rows"])

    valid = int(np.sum(analysis["control_valid"]))
    recovered = int(np.sum(analysis["recovered"]))
    failed_headings = [
        round(row["start_heading_deg"], 3) for row in analysis["rows"] if not row["recovered"]
    ]
    recovered_headings = [
        round(row["start_heading_deg"], 3) for row in analysis["rows"] if row["recovered"]
    ]
    summary = {
        "num_headings": len(analysis["rows"]),
        "integrated_turn_deg": analysis["integrated_turn_deg"],
        "valid_control_headings": valid,
        "recovered_lesion_headings": recovered,
        "recovered_lesion_headings_deg": recovered_headings,
        "failed_lesion_headings_deg": failed_headings,
        "lesion_center_deg": LESION_CENTER_DEG,
        "lesion_width_deg": LESION_WIDTH_DEG,
        "criteria": {
            "max_control_error_deg": MAX_CONTROL_ERROR_DEG,
            "max_control_turn_rmse_deg": MAX_CONTROL_TURN_RMSE_DEG,
            "max_lesion_error_vs_control_deg": MAX_LESION_ERROR_DEG,
            "min_control_resultant": MIN_CONTROL_RESULTANT,
            "min_lesion_resultant": MIN_LESION_RESULTANT,
            "min_activity_ratio": MIN_ACTIVITY_RATIO,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")

    figure_path = output_dir / "head_direction_compass.png"
    plot_results(result, analysis, figure_path)
    return figure_path, csv_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=36, help="ring neurons and tested headings")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = run_sweep(args.neurons)
    analysis = analyze(result)
    figure_path, csv_path, summary_path, summary = write_outputs(result, analysis, args.output_dir)
    print(
        f"Control tracked {summary['valid_control_headings']}/{summary['num_headings']} headings; "
        f"lesioned ring recovered {summary['recovered_lesion_headings']}/{summary['num_headings']}."
    )
    print(f"Figure: {figure_path}\nSweep: {csv_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
