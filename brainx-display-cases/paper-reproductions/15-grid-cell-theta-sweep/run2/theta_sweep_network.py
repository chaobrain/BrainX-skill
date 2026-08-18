"""Alternating direction-grid theta sweeps in aggregate firing-rate populations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-brainx-theta-sweeps-run2")

import brainmass
import brainstate
import brainstate.nn as nn
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "results"
DT = 2.0 * u.ms
THETA_FREQUENCY = 10.0 * u.Hz
THETA_PERIOD = 1.0 / THETA_FREQUENCY
REFERENCE_SPEED = 30.0 * u.cm / u.second
GRID_SCALES = u.Quantity(np.asarray([38.0, 55.0, 78.0]), unit=u.cm)
MODEL_SEED = 1405
SHUFFLE_SEED = 8128
N_SHUFFLE = 4000


def wrap_angle(value):
    """Wrap a raw radian value to [-pi, pi)."""
    return (value + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def circular_difference(first, second):
    """Return the signed raw-radian difference between two host arrays."""
    return np.angle(np.exp(1j * (np.asarray(first) - np.asarray(second))))


class DirectionGridStep(nn.Module):
    """Aggregate head-direction ring and toroidal grid attractors."""

    def __init__(
        self,
        *,
        adaptation=1.0,
        theta=1.0,
        coupling=1.0,
        initial_direction=None,
        initial_grid=None,
    ):
        super().__init__()
        self.adaptation_strength = adaptation
        self.theta_strength = theta
        self.coupling_strength = coupling
        self.n_direction = 72
        self.n_grid = 18
        self.n_grid_cell = self.n_grid**2
        self.scales_cm = jnp.asarray(GRID_SCALES.to_decimal(u.cm))
        self.n_module = self.scales_cm.size

        self.direction_angles = u.math.linspace(
            -jnp.pi,
            jnp.pi,
            self.n_direction,
            endpoint=False,
        )
        direction_distance = wrap_angle(
            self.direction_angles - self.direction_angles[0]
        )
        direction_kernel = u.math.exp(
            -0.5 * (direction_distance / 0.40) ** 2
        )
        self.direction_kernel_fft = jnp.fft.fft(
            direction_kernel / u.math.sum(direction_kernel)
        )

        phase = u.math.linspace(-jnp.pi, jnp.pi, self.n_grid, endpoint=False)
        phase_x, phase_y = u.math.meshgrid(phase, phase, indexing="ij")
        self.phase_x = u.math.reshape(phase_x, (-1,))
        self.phase_y = u.math.reshape(phase_y, (-1,))
        torus_distance2 = wrap_angle(phase_x - phase_x[0, 0]) ** 2 + wrap_angle(
            phase_y - phase_y[0, 0]
        ) ** 2
        grid_kernel = u.math.exp(-0.5 * torus_distance2 / 0.52**2)
        self.grid_kernel_fft = jnp.fft.fft2(
            grid_kernel / u.math.sum(grid_kernel)
        )

        # This oblique basis maps physical displacement to hexagonal grid phase.
        self.phase_basis = u.math.asarray(
            [[1.0, 0.0], [0.5, jnp.sqrt(3.0) / 2.0]]
        )
        if initial_direction is None:
            initial_direction = 2e-3 * brainstate.random.randn(self.n_direction)
        if initial_grid is None:
            initial_grid = 2e-3 * brainstate.random.randn(
                self.n_module, self.n_grid_cell
            )
        self.direction_u = brainstate.HiddenState(initial_direction)
        self.direction_rate = brainstate.HiddenState(
            u.math.zeros(self.n_direction)
        )
        self.direction_adaptation = brainstate.HiddenState(
            u.math.zeros(self.n_direction)
        )
        self.grid_u = brainstate.HiddenState(initial_grid)
        self.grid_rate = brainstate.HiddenState(
            u.math.zeros((self.n_module, self.n_grid_cell))
        )
        self.grid_adaptation = brainstate.HiddenState(
            u.math.zeros((self.n_module, self.n_grid_cell))
        )

    @staticmethod
    def _direction_transfer(activation):
        return u.math.sigmoid(5.0 * (activation - 0.55))

    @staticmethod
    def _grid_transfer(activation):
        rectified = u.math.relu(activation) ** 2
        return rectified / (
            1.0 + 0.10 * u.math.sum(rectified, axis=1, keepdims=True)
        )

    def _animal_phase(self, position_cm):
        projected = self.phase_basis @ position_cm
        return wrap_angle(
            2.0 * jnp.pi * projected[None, :] / self.scales_cm[:, None]
        )

    def _toroidal_fields(self, centers):
        dx = wrap_angle(
            self.phase_x[None, None, :] - centers[:, :, 0, None]
        )
        dy = wrap_angle(
            self.phase_y[None, None, :] - centers[:, :, 1, None]
        )
        return u.math.exp(-0.5 * (dx**2 + dy**2) / 0.46**2)

    def update(self, index, position_cm, heading_rad, speed_cm_s):
        dt = brainstate.environ.get_dt()
        dt_ms = dt.to_decimal(u.ms)
        theta_phase = wrap_angle(
            2.0
            * jnp.pi
            * index
            * dt.to_decimal(u.second)
            * THETA_FREQUENCY.to_decimal(u.Hz)
        )
        sweep_gate = self.theta_strength * 0.5 * (
            1.0 - u.math.cos(theta_phase)
        )
        speed_gain = speed_cm_s / REFERENCE_SPEED.to_decimal(
            u.cm / u.second
        )

        old_direction_rate = self._direction_transfer(self.direction_u.value)
        local_excitation = jnp.real(
            jnp.fft.ifft(
                jnp.fft.fft(old_direction_rate)
                * self.direction_kernel_fft
            )
        )
        sensory_anchor = u.math.exp(
            -0.5
            * (wrap_angle(self.direction_angles - heading_rad) / 0.34) ** 2
        )
        recurrent_gain = 2.20 + 1.10 * speed_gain * sweep_gate
        anchor_gain = 3.40 * (1.0 - 0.45 * speed_gain * sweep_gate)
        direction_target = (
            recurrent_gain * local_excitation
            - 1.10 * u.math.mean(old_direction_rate)
            + anchor_gain * sensory_anchor
            - self.adaptation_strength
            * 2.00
            * self.direction_adaptation.value
        )
        direction_decay = u.math.exp(-dt_ms / 12.0)
        direction_adaptation_decay = u.math.exp(-dt_ms / 60.0)
        new_direction_u = (
            direction_decay * self.direction_u.value
            + (1.0 - direction_decay) * direction_target
        )
        new_direction_rate = self._direction_transfer(new_direction_u)
        new_direction_adaptation = (
            direction_adaptation_decay * self.direction_adaptation.value
            + (1.0 - direction_adaptation_decay) * new_direction_rate
        )

        animal_phase = self._animal_phase(position_cm)
        direction_vectors = u.math.stack(
            [
                u.math.cos(self.direction_angles),
                u.math.sin(self.direction_angles),
            ],
            axis=1,
        )
        projected_direction = direction_vectors @ self.phase_basis.T
        shift_fraction = 0.19 * speed_gain * sweep_gate
        shifted_centers = wrap_angle(
            animal_phase[:, None, :]
            + 2.0
            * jnp.pi
            * shift_fraction
            * projected_direction[None, :, :]
        )
        conjunctive_fields = self._toroidal_fields(shifted_centers)
        direction_weights = new_direction_rate**5
        direction_weights = direction_weights / (
            u.math.sum(direction_weights) + 1e-8
        )
        shifted_input = u.math.sum(
            conjunctive_fields * direction_weights[None, :, None], axis=1
        )
        position_input = self._toroidal_fields(
            animal_phase[:, None, :]
        )[:, 0, :]

        old_grid_rate = self._grid_transfer(self.grid_u.value)
        grid_rate_2d = u.math.reshape(
            old_grid_rate,
            (self.n_module, self.n_grid, self.n_grid),
        )
        grid_recurrent = jnp.real(
            jnp.fft.ifft2(
                jnp.fft.fft2(grid_rate_2d) * self.grid_kernel_fft
            )
        ).reshape(self.n_module, self.n_grid_cell)
        grid_target = (
            2.25 * grid_recurrent
            - 0.85 * u.math.mean(old_grid_rate, axis=1, keepdims=True)
            + 1.90 * (1.0 - 0.72 * sweep_gate) * position_input
            + self.coupling_strength
            * (1.25 + 2.20 * sweep_gate)
            * shifted_input
            - 0.18 * self.grid_adaptation.value
        )
        grid_decay = u.math.exp(-dt_ms / 10.0)
        grid_adaptation_decay = u.math.exp(-dt_ms / 240.0)
        new_grid_u = (
            grid_decay * self.grid_u.value
            + (1.0 - grid_decay) * grid_target
        )
        new_grid_rate = self._grid_transfer(new_grid_u)
        new_grid_adaptation = (
            grid_adaptation_decay * self.grid_adaptation.value
            + (1.0 - grid_adaptation_decay) * new_grid_rate
        )

        self.direction_u.value = new_direction_u
        self.direction_rate.value = new_direction_rate
        self.direction_adaptation.value = new_direction_adaptation
        self.grid_u.value = new_grid_u
        self.grid_rate.value = new_grid_rate
        self.grid_adaptation.value = new_grid_adaptation

        decoded_direction = jnp.angle(
            u.math.sum(
                new_direction_rate * jnp.exp(1j * self.direction_angles)
            )
        )
        decoded_grid_x = jnp.angle(
            u.math.sum(
                new_grid_rate * jnp.exp(1j * self.phase_x)[None, :],
                axis=1,
            )
        )
        decoded_grid_y = jnp.angle(
            u.math.sum(
                new_grid_rate * jnp.exp(1j * self.phase_y)[None, :],
                axis=1,
            )
        )
        return (
            decoded_direction,
            u.math.stack([decoded_grid_x, decoded_grid_y], axis=1),
            new_direction_rate,
            new_grid_rate,
            theta_phase,
        )


def make_protocol(kind):
    """Build one unit-aware open-field navigation protocol."""
    dt_s = float(DT.to_decimal(u.second))
    if kind == "straight":
        duration = 6.0 * u.second
        n_step = round(duration.to_decimal(u.second) / dt_s)
        speed_cm_s = np.full(n_step, 30.0)
        heading_rad = np.zeros(n_step)
    elif kind == "speed":
        duration = 9.0 * u.second
        n_step = round(duration.to_decimal(u.second) / dt_s)
        speed_cm_s = np.repeat(
            [15.0, 30.0, 45.0],
            round(3.0 / dt_s),
        )[:n_step]
        heading_rad = np.zeros(n_step)
    elif kind == "turn":
        duration = 10.0 * u.second
        n_step = round(duration.to_decimal(u.second) / dt_s)
        speed_cm_s = np.full(n_step, 24.0)
        turn_rate = np.zeros(n_step)
        turn_rate[round(2.0 / dt_s) : round(8.0 / dt_s)] = np.deg2rad(
            42.0
        )
        heading_rad = np.cumsum(turn_rate) * dt_s
        heading_rad = np.angle(np.exp(1j * heading_rad))
    else:
        raise ValueError(f"Unknown protocol: {kind}")

    velocity_cm_s = speed_cm_s[:, None] * np.stack(
        [np.cos(heading_rad), np.sin(heading_rad)], axis=1
    )
    position_cm = np.cumsum(velocity_cm_s * dt_s, axis=0)
    return {
        "index": np.arange(n_step, dtype=np.int32),
        "time": u.Quantity(
            (np.arange(n_step) + 1) * dt_s,
            unit=u.second,
        ),
        "position": u.Quantity(position_cm, unit=u.cm),
        "heading": u.Quantity(heading_rad, unit=u.radian),
        "speed": u.Quantity(speed_cm_s, unit=u.cm / u.second),
    }


def simulate(kind, *, adaptation=1.0, theta=1.0, coupling=1.0):
    """Run one matched protocol with a complete transformed time loop."""
    protocol = make_protocol(kind)
    brainstate.random.seed(MODEL_SEED)
    model = DirectionGridStep(
        adaptation=adaptation,
        theta=theta,
        coupling=coupling,
    )

    # The attractor equations use normalized cm, cm/s, and radian magnitudes.
    position_cm = jnp.asarray(protocol["position"].to_decimal(u.cm))
    heading_rad = jnp.asarray(protocol["heading"].to_decimal(u.radian))
    speed_cm_s = jnp.asarray(
        protocol["speed"].to_decimal(u.cm / u.second)
    )
    indices = jnp.asarray(protocol["index"])

    def rollout():
        def step(index, position, heading, speed):
            with brainstate.environ.context(i=index, t=index * DT):
                return model.update(index, position, heading, speed)

        return brainstate.transform.for_loop(
            step,
            indices,
            position_cm,
            heading_rad,
            speed_cm_s,
        )

    with brainstate.environ.context(dt=DT, fit=False):
        run = brainstate.transform.jit(rollout)()
    jax.block_until_ready(run[1])
    direction, grid_phase, direction_rate, grid_rate, theta_phase = run
    return {
        "index": protocol["index"],
        "time_s": np.asarray(protocol["time"].to_decimal(u.second)),
        "position_cm": np.asarray(protocol["position"].to_decimal(u.cm)),
        "heading_rad": np.asarray(protocol["heading"].to_decimal(u.radian)),
        "speed_cm_s": np.asarray(
            protocol["speed"].to_decimal(u.cm / u.second)
        ),
        "direction_rad": np.asarray(direction),
        "grid_phase_rad": np.asarray(grid_phase),
        "direction_rate": np.asarray(direction_rate),
        "grid_rate": np.asarray(grid_rate),
        "theta_phase_rad": np.asarray(theta_phase),
    }


def simulate_adaptation_sweep(levels):
    """Map complete straight-run simulations over adaptation strengths."""
    protocol = make_protocol("straight")
    indices = jnp.asarray(protocol["index"])
    position_cm = jnp.asarray(protocol["position"].to_decimal(u.cm))
    heading_rad = jnp.asarray(protocol["heading"].to_decimal(u.radian))
    speed_cm_s = jnp.asarray(
        protocol["speed"].to_decimal(u.cm / u.second)
    )

    brainstate.random.seed(MODEL_SEED)
    initial_direction = 2e-3 * brainstate.random.randn(72)
    initial_grid = 2e-3 * brainstate.random.randn(3, 18 * 18)

    def run_one(adaptation):
        model = DirectionGridStep(
            adaptation=adaptation,
            initial_direction=initial_direction,
            initial_grid=initial_grid,
        )

        def step(index, position, heading, speed):
            decoded = model.update(index, position, heading, speed)
            return decoded[0], decoded[1]

        return brainstate.transform.for_loop(
            step,
            indices,
            position_cm,
            heading_rad,
            speed_cm_s,
        )

    with brainstate.environ.context(dt=DT, fit=False):
        mapped_sweep = brainstate.transform.jit(
            brainstate.transform.vmap(run_one)
        )
        direction, grid_phase = mapped_sweep(jnp.asarray(levels))
    jax.block_until_ready(grid_phase)

    common = {
        "index": protocol["index"],
        "time_s": np.asarray(protocol["time"].to_decimal(u.second)),
        "position_cm": np.asarray(protocol["position"].to_decimal(u.cm)),
        "heading_rad": np.asarray(protocol["heading"].to_decimal(u.radian)),
        "speed_cm_s": np.asarray(
            protocol["speed"].to_decimal(u.cm / u.second)
        ),
    }
    direction = np.asarray(direction)
    grid_phase = np.asarray(grid_phase)
    return [
        {
            **common,
            "direction_rad": direction[lane],
            "grid_phase_rad": grid_phase[lane],
        }
        for lane in range(direction.shape[0])
    ]


def decode_grid_displacement(run):
    """Decode each grid module relative to the animal's true position."""
    basis = np.asarray([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    scales_cm = np.asarray(GRID_SCALES.to_decimal(u.cm))
    animal_phase = (
        2.0
        * np.pi
        * (run["position_cm"] @ basis.T)[:, None, :]
        / scales_cm[None, :, None]
    )
    phase_error = circular_difference(run["grid_phase_rad"], animal_phase)
    displacement = np.einsum(
        "ij,tmj->tmi",
        np.linalg.inv(basis),
        phase_error,
    )
    return displacement * scales_cm[None, :, None] / (2.0 * np.pi)


def cycle_analysis(run, warmup=1.0 * u.second):
    """Sample one direction and grid endpoint at theta phase pi."""
    steps_per_cycle = round(
        THETA_PERIOD.to_decimal(u.ms) / DT.to_decimal(u.ms)
    )
    evaluation_step = steps_per_cycle // 2
    first_cycle = int(
        np.ceil(
            warmup.to_decimal(u.second)
            / THETA_PERIOD.to_decimal(u.second)
        )
    )
    n_cycle = run["time_s"].size // steps_per_cycle
    cycle = np.arange(first_cycle, n_cycle)
    indices = cycle * steps_per_cycle + evaluation_step
    heading = run["heading_rad"][indices]
    direction = run["direction_rad"][indices]
    direction_offset = circular_difference(direction, heading)
    endpoints = decode_grid_displacement(run)[indices]
    grid_direction = np.arctan2(endpoints[:, :, 1], endpoints[:, :, 0])
    grid_offset = circular_difference(grid_direction, heading[:, None])
    grid_length = np.linalg.norm(endpoints, axis=2)
    alignment = np.cos(
        circular_difference(grid_direction, direction[:, None])
    )
    return {
        "cycle": cycle,
        "index": indices,
        "time_s": run["time_s"][indices],
        "speed_cm_s": run["speed_cm_s"][indices],
        "heading_rad": heading,
        "direction_rad": direction,
        "direction_offset_rad": direction_offset,
        "grid_endpoint_cm": endpoints,
        "grid_direction_rad": grid_direction,
        "grid_offset_rad": grid_offset,
        "grid_length_cm": grid_length,
        "alignment_cosine": alignment,
    }


def alternation_score(angle, minimum_angle=np.deg2rad(5.0)):
    """Fraction of all adjacent pairs that are strong and opposite-signed."""
    values = np.asarray(angle)
    if values.size < 2:
        return float("nan")
    strong_pair = (np.abs(values[:-1]) >= minimum_angle) & (
        np.abs(values[1:]) >= minimum_angle
    )
    opposite = np.sign(values[:-1]) != np.sign(values[1:])
    return float(np.mean(strong_pair & opposite))


def shuffled_alternation(angle, n_shuffle=N_SHUFFLE):
    """Null distribution formed by permuting observed cycle order."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    values = np.asarray(angle)
    return np.asarray(
        [alternation_score(rng.permutation(values)) for _ in range(n_shuffle)]
    )


def phase_resolved_alignment(run, minimum_length_cm=1.0):
    """Mean ring-grid cosine during the active half of theta after warmup."""
    start = round(1.0 / DT.to_decimal(u.second))
    displacement = decode_grid_displacement(run)[start:]
    direction = run["direction_rad"][start:, None]
    grid_direction = np.arctan2(displacement[:, :, 1], displacement[:, :, 0])
    grid_length = np.linalg.norm(displacement, axis=2)
    active = np.cos(run["theta_phase_rad"][start:]) < 0.0
    valid = active[:, None] & (grid_length >= minimum_length_cm)
    alignment = np.cos(circular_difference(grid_direction, direction))
    result = []
    for module in range(grid_length.shape[1]):
        selected = valid[:, module]
        result.append(
            float(np.mean(alignment[selected, module]))
            if selected.any()
            else None
        )
    return result


def summarize_cycles(cycles):
    """Compute all cycle-level claims for one condition."""
    shuffle = shuffled_alternation(cycles["direction_offset_rad"])
    observed = alternation_score(cycles["direction_offset_rad"])
    return {
        "n_cycles": int(cycles["cycle"].size),
        "alternation_score": observed,
        "sweep_cycle_fraction": float(
            np.mean(
                np.abs(cycles["direction_offset_rad"]) >= np.deg2rad(5.0)
            )
        ),
        "shuffle_mean": float(np.mean(shuffle)),
        "shuffle_95_interval": np.percentile(
            shuffle, [2.5, 97.5]
        ).tolist(),
        "shuffle_p_upper": float(
            (1 + np.sum(shuffle >= observed)) / (1 + shuffle.size)
        ),
        "mean_abs_direction_angle_deg": float(
            np.degrees(np.mean(np.abs(cycles["direction_offset_rad"])))
        ),
        "mean_grid_length_cm": np.mean(
            cycles["grid_length_cm"], axis=0
        ).tolist(),
        "grid_alternation_score": [
            alternation_score(cycles["grid_offset_rad"][:, module])
            for module in range(cycles["grid_offset_rad"].shape[1])
        ],
        "mean_alignment_cosine": np.mean(
            cycles["alignment_cosine"], axis=0
        ).tolist(),
    }


def summarize_speed(cycles):
    """Summarize settled cycles at each commanded running speed."""
    levels = np.asarray([15.0, 30.0, 45.0])
    segment_starts = np.asarray([0.0, 3.0, 6.0])
    result = {
        "speed_cm_s": levels.tolist(),
        "mean_grid_length_cm": [],
        "mean_abs_direction_angle_deg": [],
        "alternation_score": [],
    }
    for level, start in zip(levels, segment_starts):
        selected = np.isclose(cycles["speed_cm_s"], level) & (
            cycles["time_s"] >= start + 0.5
        )
        result["mean_grid_length_cm"].append(
            np.mean(cycles["grid_length_cm"][selected], axis=0).tolist()
        )
        result["mean_abs_direction_angle_deg"].append(
            float(
                np.degrees(
                    np.mean(
                        np.abs(cycles["direction_offset_rad"][selected])
                    )
                )
            )
        )
        result["alternation_score"].append(
            alternation_score(cycles["direction_offset_rad"][selected])
        )
    return result


def save_cycle_csv(cycles):
    scales = np.asarray(GRID_SCALES.to_decimal(u.cm))
    with (OUT / "straight_cycle_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cycle",
                "time_s",
                "direction_offset_deg",
                *[f"grid_{scale:g}cm_length_cm" for scale in scales],
                *[f"grid_{scale:g}cm_offset_deg" for scale in scales],
                *[
                    f"grid_{scale:g}cm_alignment_cosine"
                    for scale in scales
                ],
            ]
        )
        for row in range(cycles["cycle"].size):
            writer.writerow(
                [
                    int(cycles["cycle"][row]),
                    cycles["time_s"][row],
                    np.degrees(cycles["direction_offset_rad"][row]),
                    *cycles["grid_length_cm"][row],
                    *np.degrees(cycles["grid_offset_rad"][row]),
                    *cycles["alignment_cosine"][row],
                ]
            )


def select_vector_rows(turn_cycles):
    """Select exactly 10 well-spaced theta cycles along the turning path."""
    turning = np.flatnonzero(
        (turn_cycles["time_s"] >= 2.1) & (turn_cycles["time_s"] <= 7.9)
    )
    selected = np.linspace(0, turning.size - 1, 10).astype(int)
    return turning[selected]


def draw_trajectory(ax, turn, turn_cycles, *, numbered):
    rows = select_vector_rows(turn_cycles)
    indices = turn_cycles["index"][rows]
    vectors = turn_cycles["grid_endpoint_cm"][rows, 0]
    left = turn_cycles["direction_offset_rad"][rows] > 0.0
    colors = np.where(left, "#c62828", "#1565c0")
    ax.plot(
        turn["position_cm"][:, 0],
        turn["position_cm"][:, 1],
        color="#4a4a4a",
        linewidth=1.8,
        label="Rat trajectory",
    )
    for number, (point, vector, color) in enumerate(
        zip(turn["position_cm"][indices], vectors, colors), start=1
    ):
        ax.arrow(
            point[0],
            point[1],
            vector[0],
            vector[1],
            width=0.28,
            head_width=2.0,
            head_length=2.6,
            length_includes_head=True,
            color=color,
        )
        if numbered:
            ax.text(point[0] + 1.5, point[1] + 1.5, str(number), fontsize=8)
    ax.set_aspect("equal")
    ax.set(xlabel="Arena x (cm)", ylabel="Arena y (cm)")


def save_figures(
    baseline,
    baseline_cycles,
    turn,
    turn_cycles,
    speed,
    adaptation,
    controls,
):
    steps_per_cycle = round(
        THETA_PERIOD.to_decimal(u.ms) / DT.to_decimal(u.ms)
    )
    start_cycle = 22
    window = slice(
        start_cycle * steps_per_cycle,
        (start_cycle + 8) * steps_per_cycle,
    )
    time_cycles = np.arange(window.stop - window.start) / steps_per_cycle

    fig, axes = plt.subplots(
        2, 3, figsize=(14.0, 8.0), constrained_layout=True
    )
    image = axes[0, 0].imshow(
        baseline["direction_rate"][window].T,
        aspect="auto",
        origin="lower",
        extent=[0, 8, -180, 180],
        cmap="magma",
    )
    axes[0, 0].set(
        xlabel="Theta cycles",
        ylabel="Preferred direction (deg)",
        title="Head-direction ring activity",
    )
    fig.colorbar(image, ax=axes[0, 0], label="Rate (a.u.)")

    direction_offset = np.degrees(
        circular_difference(
            baseline["direction_rad"][window],
            baseline["heading_rad"][window],
        )
    )
    displacement = decode_grid_displacement(baseline)[window]
    grid_offset = np.degrees(
        circular_difference(
            np.arctan2(displacement[:, 0, 1], displacement[:, 0, 0]),
            baseline["heading_rad"][window],
        )
    )
    grid_offset[np.linalg.norm(displacement[:, 0], axis=1) < 1.0] = np.nan
    brainmass.viz.plot_timeseries(
        direction_offset,
        ts=time_cycles,
        ax=axes[0, 1],
        color="#c62828",
        linewidth=1.5,
    )
    axes[0, 1].lines[-1].set_label("Direction ring")
    axes[0, 1].plot(
        time_cycles,
        grid_offset,
        color="#1565c0",
        linewidth=1.1,
        label="Grid, 38 cm",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 1].set(
        xlabel="Theta cycles",
        ylabel="Offset from heading (deg)",
        title="Phase-matched decoded sweeps",
    )
    axes[0, 1].legend(frameon=False)

    snapshot_cycles = np.arange(start_cycle, start_cycle + 4)
    snapshot_indices = (
        snapshot_cycles * steps_per_cycle + steps_per_cycle // 2
    )
    snapshots = np.concatenate(
        [
            baseline["grid_rate"][index, 0].reshape(18, 18)
            for index in snapshot_indices
        ],
        axis=1,
    )
    image = axes[0, 2].imshow(snapshots, origin="lower", cmap="viridis")
    for boundary in (18, 36, 54):
        axes[0, 2].axvline(boundary - 0.5, color="white", linewidth=0.7)
    axes[0, 2].set(
        xticks=[9, 27, 45, 63],
        xticklabels=[str(value) for value in snapshot_cycles],
        yticks=[],
        xlabel="Theta cycle",
        title="Grid bump at sweep phase",
    )
    fig.colorbar(image, ax=axes[0, 2], label="Rate (a.u.)")

    cycle_offset_deg = np.degrees(baseline_cycles["direction_offset_rad"])
    cycle_color = np.where(cycle_offset_deg > 0.0, "#c62828", "#1565c0")
    axes[1, 0].bar(
        np.arange(20),
        cycle_offset_deg[:20],
        color=cycle_color[:20],
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 0].set(
        xlabel="Successive theta cycle",
        ylabel="Direction offset (deg)",
        title="Left-right cycle alternation",
    )

    shuffle = shuffled_alternation(baseline_cycles["direction_offset_rad"])
    axes[1, 1].hist(
        shuffle,
        bins=24,
        color="#bdbdbd",
        edgecolor="white",
    )
    axes[1, 1].axvline(
        alternation_score(baseline_cycles["direction_offset_rad"]),
        color="#c62828",
        linewidth=2.2,
        label="Observed",
    )
    axes[1, 1].set(
        xlabel="Adjacent-cycle alternation score",
        ylabel="Shuffled cycle orders",
        title="Cycle-order shuffle control",
    )
    axes[1, 1].legend(frameon=False)

    axes[1, 2].scatter(
        np.degrees(baseline_cycles["direction_offset_rad"]),
        np.degrees(baseline_cycles["grid_offset_rad"][:, 0]),
        s=22,
        alpha=0.75,
        color="#00897b",
    )
    axes[1, 2].plot([-35, 35], [-35, 35], color="black", linewidth=0.8)
    axes[1, 2].set(
        xlim=(-35, 35),
        ylim=(-35, 35),
        xlabel="Ring sweep angle (deg)",
        ylabel="Grid sweep angle (deg)",
        title="Direction-position alignment",
    )
    fig.savefig(OUT / "population_dynamics.png", dpi=210)
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 9.0), constrained_layout=True
    )
    draw_trajectory(axes[0, 0], turn, turn_cycles, numbered=False)
    axes[0, 0].set_title("Turning trajectory and 10 sweep vectors")

    speed_levels = np.asarray(speed["speed_cm_s"])
    speed_length = np.asarray(speed["mean_grid_length_cm"])
    for module, scale in enumerate(GRID_SCALES.to_decimal(u.cm)):
        axes[0, 1].plot(
            speed_levels,
            speed_length[:, module],
            marker="o",
            label=f"{scale:g} cm",
        )
    axes[0, 1].set(
        xlabel="Running speed (cm/s)",
        ylabel="Mean sweep length (cm)",
        title="Speed-dependent sweep extent",
    )
    axes[0, 1].legend(frameon=False, title="Grid scale")

    adaptation_levels = np.asarray(adaptation["strength"])
    axes[1, 0].plot(
        adaptation_levels,
        adaptation["mean_abs_direction_angle_deg"],
        marker="o",
        color="#ef6c00",
        label="Mean angle (deg)",
    )
    axes[1, 0].plot(
        adaptation_levels,
        30.0 * np.asarray(adaptation["alternation_score"]),
        marker="s",
        color="#00897b",
        label="30 x alternation",
    )
    axes[1, 0].set(
        xlabel="Adaptation strength (relative)",
        ylabel="Angle (deg); scaled score",
        title="Adaptation regime",
    )
    axes[1, 0].legend(frameon=False)

    names = ["Baseline", "No adaptation", "No theta", "No coupling"]
    keys = ["baseline", "no_adaptation", "no_theta", "no_coupling"]
    x = np.arange(len(keys))
    axes[1, 1].bar(
        x - 0.18,
        [controls[key]["alternation_score"] for key in keys],
        width=0.36,
        color="#1565c0",
        label="Ring alternation",
    )
    axes[1, 1].bar(
        x + 0.18,
        [controls[key]["grid_alternation_score"][0] for key in keys],
        width=0.36,
        color="#ef6c00",
        label="Grid alternation",
    )
    axes[1, 1].set_xticks(x, names, rotation=16)
    axes[1, 1].set(
        ylim=(0.0, 1.05),
        ylabel="Alternation score",
        title="Matched mechanism controls",
    )
    axes[1, 1].legend(frameon=False)
    fig.savefig(OUT / "navigation_and_controls.png", dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    draw_trajectory(ax, turn, turn_cycles, numbered=True)
    ax.set_title("Open-field trajectory with decoded theta-sweep vectors")
    ax.legend(frameon=False)
    fig.savefig(OUT / "trajectory_with_10_sweep_vectors.png", dpi=220)
    plt.close(fig)


def run_full_analysis():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = simulate("straight")
    runs = {
        "baseline": baseline,
        "no_adaptation": simulate("straight", adaptation=0.0),
        "no_theta": simulate("straight", theta=0.0),
        "no_coupling": simulate("straight", coupling=0.0),
    }
    cycles = {name: cycle_analysis(run) for name, run in runs.items()}
    controls = {name: summarize_cycles(value) for name, value in cycles.items()}
    for name, run in runs.items():
        controls[name]["phase_resolved_alignment_cosine"] = (
            phase_resolved_alignment(run)
        )

    speed_run = simulate("speed")
    speed = summarize_speed(cycle_analysis(speed_run))
    turn_run = simulate("turn")
    turn_cycles = cycle_analysis(turn_run)
    turn = summarize_cycles(turn_cycles)
    turn["phase_resolved_alignment_cosine"] = phase_resolved_alignment(
        turn_run
    )
    vector_rows = select_vector_rows(turn_cycles)
    vector_values = turn_cycles["grid_endpoint_cm"][vector_rows, 0]
    vector_angles_deg = np.degrees(
        np.arctan2(vector_values[:, 1], vector_values[:, 0])
    )

    adaptation_levels = np.asarray([0.0, 0.5, 1.0, 1.5])
    adaptation_cycle_sets = [
        cycle_analysis(run)
        for run in simulate_adaptation_sweep(adaptation_levels)
    ]
    adaptation = {
        "strength": adaptation_levels.tolist(),
        "mean_abs_direction_angle_deg": [
            float(np.degrees(np.mean(np.abs(item["direction_offset_rad"]))))
            for item in adaptation_cycle_sets
        ],
        "alternation_score": [
            alternation_score(item["direction_offset_rad"])
            for item in adaptation_cycle_sets
        ],
        "mean_grid_length_cm": [
            np.mean(item["grid_length_cm"], axis=0).tolist()
            for item in adaptation_cycle_sets
        ],
    }

    summary = {
        "artifact_version": "3.0",
        "model_seed": MODEL_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "n_shuffle": N_SHUFFLE,
        "integration_dt_ms": float(DT.to_decimal(u.ms)),
        "theta_frequency_hz": float(THETA_FREQUENCY.to_decimal(u.Hz)),
        "grid_scales_cm": np.asarray(GRID_SCALES.to_decimal(u.cm)).tolist(),
        "evaluation_theta_phase_rad": float(np.pi),
        "minimum_sweep_angle_deg": 5.0,
        "adaptation_sweep_transform": "brainstate.transform.vmap",
        "conditions": controls,
        "speed": speed,
        "turn": turn,
        "trajectory_vectors": {
            "count": int(vector_rows.size),
            "time_s": turn_cycles["time_s"][vector_rows].tolist(),
            "direction_deg": vector_angles_deg.tolist(),
        },
        "adaptation_sweep": adaptation,
    }

    save_cycle_csv(cycles["baseline"])
    save_figures(
        baseline,
        cycles["baseline"],
        turn_run,
        turn_cycles,
        speed,
        adaptation,
        controls,
    )
    np.savez_compressed(
        OUT / "theta_sweep_evidence.npz",
        straight_time_s=baseline["time_s"],
        straight_position_cm=baseline["position_cm"],
        straight_heading_rad=baseline["heading_rad"],
        straight_direction_rad=baseline["direction_rad"],
        straight_grid_phase_rad=baseline["grid_phase_rad"],
        straight_direction_rate=baseline["direction_rate"],
        straight_grid_rate=baseline["grid_rate"],
        straight_cycle_index=cycles["baseline"]["cycle"],
        straight_direction_offset_rad=cycles["baseline"][
            "direction_offset_rad"
        ],
        straight_grid_endpoint_cm=cycles["baseline"]["grid_endpoint_cm"],
        straight_alignment_cosine=cycles["baseline"]["alignment_cosine"],
        turn_time_s=turn_run["time_s"],
        turn_position_cm=turn_run["position_cm"],
        turn_heading_rad=turn_run["heading_rad"],
        turn_direction_rad=turn_run["direction_rad"],
        turn_grid_endpoint_cm=turn_cycles["grid_endpoint_cm"],
        turn_vector_time_s=turn_cycles["time_s"][vector_rows],
        turn_vector_position_cm=turn_run["position_cm"][
            turn_cycles["index"][vector_rows]
        ],
        turn_vector_cm=vector_values,
        turn_vector_direction_deg=vector_angles_deg,
        speed_levels_cm_s=np.asarray(speed["speed_cm_s"]),
        speed_grid_length_cm=np.asarray(speed["mean_grid_length_cm"]),
        adaptation_strength=adaptation_levels,
        adaptation_angle_deg=np.asarray(
            adaptation["mean_abs_direction_angle_deg"]
        ),
        adaptation_alternation=np.asarray(adaptation["alternation_score"]),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def run_quick_check():
    baseline = simulate("straight")
    summary = summarize_cycles(cycle_analysis(baseline))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the straight baseline and print cycle metrics.",
    )
    arguments = parser.parse_args()
    run_quick_check() if arguments.quick else run_full_analysis()
