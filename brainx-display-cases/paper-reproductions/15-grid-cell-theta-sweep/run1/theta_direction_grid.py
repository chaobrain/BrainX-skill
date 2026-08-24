"""Theta-organized direction and grid sweeps in a firing-rate network."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-theta-direction-grid")

import brainmass
import brainstate
import brainstate.nn as nn
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parent / "results"
DT = 2.0 * u.ms
THETA_FREQUENCY = 10.0 * u.Hz
THETA_PERIOD = 1.0 / THETA_FREQUENCY
GRID_SCALES = np.asarray([38.0, 55.0, 78.0]) * u.cm
REFERENCE_SPEED = 30.0 * u.cm / u.second
MODEL_SEED = 1405
SHUFFLE_SEED = 8128


def wrap(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def circular_difference(a, b):
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


class DirectionGridStep(nn.Module):
    """Aggregate ring and toroidal grid populations with firing-rate adaptation."""

    def __init__(self, adaptation=1.0, theta=1.0, coupling=1.0):
        super().__init__()
        self.adaptation = adaptation
        self.theta = theta
        self.coupling = coupling
        self.n_direction = 72
        self.n_grid = 18
        self.n_grid_cell = self.n_grid**2
        self.scales_cm = jnp.asarray(GRID_SCALES.to_decimal(u.cm))
        self.n_module = self.scales_cm.size

        self.direction_angles = jnp.linspace(
            -jnp.pi, jnp.pi, self.n_direction, endpoint=False
        )
        distance = wrap(self.direction_angles - self.direction_angles[0])
        local_kernel = jnp.exp(-0.5 * (distance / 0.40) ** 2)
        self.direction_kernel_fft = jnp.fft.fft(local_kernel / local_kernel.sum())

        phase = jnp.linspace(-jnp.pi, jnp.pi, self.n_grid, endpoint=False)
        phase_x, phase_y = jnp.meshgrid(phase, phase, indexing="ij")
        self.phase_x = phase_x.reshape(-1)
        self.phase_y = phase_y.reshape(-1)
        torus_distance2 = wrap(phase_x - phase_x[0, 0]) ** 2 + wrap(
            phase_y - phase_y[0, 0]
        ) ** 2
        grid_kernel = jnp.exp(-0.5 * torus_distance2 / 0.52**2)
        self.grid_kernel_fft = jnp.fft.fft2(grid_kernel / grid_kernel.sum())

        # Physical displacement x maps to oblique toroidal phase B x.
        self.phase_basis = jnp.asarray(
            [[1.0, 0.0], [0.5, jnp.sqrt(3.0) / 2.0]]
        )
        perturbation = 2e-3 * brainstate.random.randn(self.n_direction)
        self.direction_u = brainstate.HiddenState(perturbation)
        self.direction_rate = brainstate.HiddenState(jnp.zeros(self.n_direction))
        self.direction_adaptation = brainstate.HiddenState(
            jnp.zeros(self.n_direction)
        )
        self.grid_u = brainstate.HiddenState(
            2e-3 * brainstate.random.randn(self.n_module, self.n_grid_cell)
        )
        self.grid_rate = brainstate.HiddenState(
            jnp.zeros((self.n_module, self.n_grid_cell))
        )
        self.grid_adaptation = brainstate.HiddenState(
            jnp.zeros((self.n_module, self.n_grid_cell))
        )

    @staticmethod
    def _direction_transfer(u_value):
        return jax.nn.sigmoid(5.0 * (u_value - 0.55))

    def _grid_transfer(self, u_value):
        rectified = jnp.maximum(u_value, 0.0) ** 2
        return rectified / (
            1.0 + 0.10 * rectified.sum(axis=1, keepdims=True)
        )

    def _animal_phase(self, position_cm):
        projected = self.phase_basis @ position_cm
        return wrap(2.0 * jnp.pi * projected[None, :] / self.scales_cm[:, None])

    def _toroidal_field(self, centers):
        dx = wrap(self.phase_x[None, None, :] - centers[:, :, 0, None])
        dy = wrap(self.phase_y[None, None, :] - centers[:, :, 1, None])
        return jnp.exp(-0.5 * (dx**2 + dy**2) / 0.46**2)

    def update(self, index, position_cm, heading, speed_cm_s):
        dt_ms = brainstate.environ.get_dt().to_decimal(u.ms)
        theta_phase = wrap(
            2.0
            * jnp.pi
            * index
            * brainstate.environ.get_dt().to_decimal(u.second)
            * THETA_FREQUENCY.to_decimal(u.Hz)
        )
        sweep_gate = self.theta * 0.5 * (1.0 - jnp.cos(theta_phase))
        speed_gain = speed_cm_s / REFERENCE_SPEED.to_decimal(u.cm / u.second)

        old_direction_rate = self._direction_transfer(self.direction_u.value)
        local_excitation = jnp.real(
            jnp.fft.ifft(
                jnp.fft.fft(old_direction_rate) * self.direction_kernel_fft
            )
        )
        anchor = jnp.exp(
            -0.5 * (wrap(self.direction_angles - heading) / 0.34) ** 2
        )
        recurrent_gain = 2.20 + 1.10 * speed_gain * sweep_gate
        anchor_gain = 3.40 * (1.0 - 0.45 * speed_gain * sweep_gate)
        direction_target = (
            recurrent_gain * local_excitation
            - 1.10 * old_direction_rate.mean()
            + anchor_gain * anchor
            - self.adaptation * 2.00 * self.direction_adaptation.value
        )
        direction_decay = jnp.exp(-dt_ms / 12.0)
        adaptation_decay = jnp.exp(-dt_ms / 60.0)
        new_direction_u = (
            direction_decay * self.direction_u.value
            + (1.0 - direction_decay) * direction_target
        )
        new_direction_rate = self._direction_transfer(new_direction_u)
        new_direction_adaptation = (
            adaptation_decay * self.direction_adaptation.value
            + (1.0 - adaptation_decay) * new_direction_rate
        )

        animal_phase = self._animal_phase(position_cm)
        direction_vectors = jnp.stack(
            [jnp.cos(self.direction_angles), jnp.sin(self.direction_angles)], axis=1
        )
        projected_direction = direction_vectors @ self.phase_basis.T
        shift_fraction = 0.19 * speed_gain * sweep_gate
        phase_shift = (
            2.0
            * jnp.pi
            * shift_fraction
            * projected_direction[None, :, :]
        )
        shifted_centers = wrap(animal_phase[:, None, :] + phase_shift)
        conjunctive_fields = self._toroidal_field(shifted_centers)
        direction_weights = new_direction_rate**5
        direction_weights /= direction_weights.sum() + 1e-8
        shifted_input = jnp.sum(
            conjunctive_fields * direction_weights[None, :, None], axis=1
        )
        anchor_input = self._toroidal_field(animal_phase[:, None, :])[:, 0, :]

        old_grid_rate = self._grid_transfer(self.grid_u.value)
        grid_rate_2d = old_grid_rate.reshape(
            self.n_module, self.n_grid, self.n_grid
        )
        grid_recurrent = jnp.real(
            jnp.fft.ifft2(jnp.fft.fft2(grid_rate_2d) * self.grid_kernel_fft)
        ).reshape(self.n_module, self.n_grid_cell)
        grid_target = (
            2.25 * grid_recurrent
            - 0.85 * old_grid_rate.mean(axis=1, keepdims=True)
            + 1.90 * (1.0 - 0.72 * sweep_gate) * anchor_input
            + self.coupling * (1.25 + 2.20 * sweep_gate) * shifted_input
            - 0.18 * self.grid_adaptation.value
        )
        grid_decay = jnp.exp(-dt_ms / 10.0)
        grid_adaptation_decay = jnp.exp(-dt_ms / 240.0)
        new_grid_u = (
            grid_decay * self.grid_u.value + (1.0 - grid_decay) * grid_target
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

        direction_decode = jnp.angle(
            jnp.sum(new_direction_rate * jnp.exp(1j * self.direction_angles))
        )
        grid_phase_x = jnp.angle(
            jnp.sum(
                new_grid_rate * jnp.exp(1j * self.phase_x)[None, :], axis=1
            )
        )
        grid_phase_y = jnp.angle(
            jnp.sum(
                new_grid_rate * jnp.exp(1j * self.phase_y)[None, :], axis=1
            )
        )
        return (
            direction_decode,
            jnp.stack([grid_phase_x, grid_phase_y], axis=1),
            new_direction_rate,
            new_grid_rate,
            theta_phase,
        )


def make_protocol(kind):
    dt_s = DT.to_decimal(u.second)
    if kind == "straight":
        duration_s = 6.0
        speed = np.full(round(duration_s / dt_s), 30.0)
        heading = np.zeros(speed.size)
    elif kind == "speed":
        duration_s = 9.0
        speed = np.repeat([15.0, 30.0, 45.0], round(3.0 / dt_s))
        heading = np.zeros(speed.size)
    elif kind == "turn":
        duration_s = 10.0
        speed = np.full(round(duration_s / dt_s), 24.0)
        turn_rate = np.zeros(speed.size)
        turn_rate[round(2.0 / dt_s) : round(8.0 / dt_s)] = np.deg2rad(42.0)
        heading = np.cumsum(turn_rate) * dt_s
        heading = np.angle(np.exp(1j * heading))
    else:
        raise ValueError(f"Unknown protocol: {kind}")
    velocity = speed[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    position = np.cumsum(velocity * dt_s, axis=0)
    return {
        "index": np.arange(speed.size, dtype=np.int32),
        "time_s": (np.arange(speed.size) + 1) * dt_s,
        "position_cm": position,
        "heading_rad": heading,
        "speed_cm_s": speed,
    }


def simulate(kind, adaptation=1.0, theta=1.0, coupling=1.0):
    protocol = make_protocol(kind)
    brainstate.random.seed(MODEL_SEED)
    model = DirectionGridStep(adaptation=adaptation, theta=theta, coupling=coupling)

    def rollout():
        def step(index, position, heading, speed):
            with brainstate.environ.context(i=index, t=index * DT):
                return model.update(index, position, heading, speed)

        return brainstate.transform.for_loop(
            step,
            jnp.asarray(protocol["index"]),
            jnp.asarray(protocol["position_cm"]),
            jnp.asarray(protocol["heading_rad"]),
            jnp.asarray(protocol["speed_cm_s"]),
        )

    with brainstate.environ.context(dt=DT):
        compiled_rollout = brainstate.transform.jit(rollout)
        direction, grid_phase, direction_rate, grid_rate, theta_phase = (
            compiled_rollout()
        )
    jax.block_until_ready(grid_phase)
    protocol.update(
        direction_rad=np.asarray(direction),
        grid_phase_rad=np.asarray(grid_phase),
        direction_rate=np.asarray(direction_rate),
        grid_rate=np.asarray(grid_rate),
        theta_phase_rad=np.asarray(theta_phase),
    )
    return protocol


def decode_grid_displacement(run):
    basis = np.asarray([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    animal_phase = (
        2.0
        * np.pi
        * (run["position_cm"] @ basis.T)[:, None, :]
        / GRID_SCALES.to_decimal(u.cm)[None, :, None]
    )
    phase_error = circular_difference(run["grid_phase_rad"], animal_phase)
    displacement = np.einsum("ij,tmj->tmi", np.linalg.inv(basis), phase_error)
    return (
        displacement
        * GRID_SCALES.to_decimal(u.cm)[None, :, None]
        / (2.0 * np.pi)
    )


def cycle_analysis(run, warmup_s=1.0):
    steps_per_cycle = round(THETA_PERIOD.to_decimal(u.ms) / DT.to_decimal(u.ms))
    evaluation_step = steps_per_cycle // 2
    displacement = decode_grid_displacement(run)
    first_cycle = int(np.ceil(warmup_s / THETA_PERIOD.to_decimal(u.second)))
    n_cycle = run["time_s"].size // steps_per_cycle
    cycle = np.arange(first_cycle, n_cycle)
    indices = cycle * steps_per_cycle + evaluation_step
    heading = run["heading_rad"][indices]
    direction = run["direction_rad"][indices]
    direction_offset = circular_difference(direction, heading)
    endpoints = displacement[indices]
    grid_direction = np.arctan2(endpoints[:, :, 1], endpoints[:, :, 0])
    grid_offset = circular_difference(grid_direction, heading[:, None])
    grid_length = np.linalg.norm(endpoints, axis=2)
    alignment = np.cos(circular_difference(grid_direction, direction[:, None]))
    return {
        "cycle": cycle,
        "index": indices,
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
    values = np.asarray(angle)
    valid_pair = (np.abs(values[1:]) >= minimum_angle) & (
        np.abs(values[:-1]) >= minimum_angle
    )
    alternates = np.sign(values[1:]) != np.sign(values[:-1])
    return float(np.mean(valid_pair & alternates)) if values.size > 1 else np.nan


def shuffled_alternation(angle, n_shuffle=4000):
    rng = np.random.default_rng(SHUFFLE_SEED)
    values = np.asarray(angle)
    return np.asarray(
        [alternation_score(rng.permutation(values)) for _ in range(n_shuffle)]
    )


def condition_summary(cycles):
    shuffle = shuffled_alternation(cycles["direction_offset_rad"])
    observed = alternation_score(cycles["direction_offset_rad"])
    grid_observed = [
        alternation_score(cycles["grid_offset_rad"][:, module])
        for module in range(cycles["grid_offset_rad"].shape[1])
    ]
    return {
        "n_cycles": int(cycles["cycle"].size),
        "alternation_score": observed,
        "sweep_cycle_fraction": float(
            np.mean(np.abs(cycles["direction_offset_rad"]) >= np.deg2rad(5.0))
        ),
        "shuffle_mean": float(shuffle.mean()),
        "shuffle_95_interval": np.percentile(shuffle, [2.5, 97.5]).tolist(),
        "shuffle_p_upper": float((1 + np.sum(shuffle >= observed)) / (1 + shuffle.size)),
        "mean_abs_direction_angle_deg": float(
            np.degrees(np.mean(np.abs(cycles["direction_offset_rad"])))
        ),
        "mean_grid_length_cm": cycles["grid_length_cm"].mean(axis=0).tolist(),
        "grid_alternation_score": grid_observed,
        "mean_alignment_cosine": cycles["alignment_cosine"].mean(axis=0).tolist(),
    }


def phase_resolved_alignment(run, warmup_s=1.0, minimum_length_cm=1.0):
    displacement = decode_grid_displacement(run)
    start = round(warmup_s / DT.to_decimal(u.second))
    direction = run["direction_rad"][start:, None]
    grid_direction = np.arctan2(
        displacement[start:, :, 1], displacement[start:, :, 0]
    )
    length = np.linalg.norm(displacement[start:], axis=2)
    active_phase = np.cos(run["theta_phase_rad"][start:]) < 0.0
    valid = (length >= minimum_length_cm) & active_phase[:, None]
    cosine = np.cos(circular_difference(grid_direction, direction))
    result = []
    for module in range(length.shape[1]):
        selected = valid[:, module]
        result.append(float(np.mean(cosine[selected, module])) if selected.any() else None)
    return result


def summarize_speed(cycles):
    levels = np.asarray([15.0, 30.0, 45.0])
    summary = {
        "speed_cm_s": levels.tolist(),
        "mean_grid_length_cm": [],
        "mean_abs_direction_angle_deg": [],
        "alternation_score": [],
    }
    for level in levels:
        selected = np.isclose(cycles["speed_cm_s"], level)
        summary["mean_grid_length_cm"].append(
            cycles["grid_length_cm"][selected].mean(axis=0).tolist()
        )
        summary["mean_abs_direction_angle_deg"].append(
            float(
                np.degrees(
                    np.mean(np.abs(cycles["direction_offset_rad"][selected]))
                )
            )
        )
        summary["alternation_score"].append(
            alternation_score(cycles["direction_offset_rad"][selected])
        )
    return summary


def single_cell_analysis(turn_run):
    steps_per_cycle = round(THETA_PERIOD.to_decimal(u.ms) / DT.to_decimal(u.ms))
    mask = (turn_run["time_s"] >= 2.0) & (turn_run["time_s"] <= 8.0)
    rates = turn_run["direction_rate"][mask]
    headings = turn_run["heading_rad"][mask]
    theta_phase = turn_run["theta_phase_rad"][mask]
    preferred = np.linspace(-np.pi, np.pi, rates.shape[1], endpoint=False)

    n_cycle = rates.shape[0] // steps_per_cycle
    cycle_rate = rates[: n_cycle * steps_per_cycle].reshape(
        n_cycle, steps_per_cycle, rates.shape[1]
    ).sum(axis=1)
    centered = cycle_rate - cycle_rate.mean(axis=0, keepdims=True)
    lag1 = np.sum(centered[1:] * centered[:-1], axis=0)
    lag2 = np.sum(centered[2:] * centered[:-2], axis=0)
    skipping = (lag2 - lag1) / (np.abs(lag2) + np.abs(lag1) + 1e-9)

    resultant = np.abs(
        np.sum(rates * np.exp(1j * headings[:, None]), axis=0)
    ) / (np.sum(rates, axis=0) + 1e-9)
    tuning_width = np.sqrt(
        np.maximum(-2.0 * np.log(np.clip(resultant, 1e-9, 1.0)), 0.0)
    )

    relative_heading = circular_difference(headings[:, None], preferred[None, :])
    phase_slope = np.full(rates.shape[1], np.nan)
    phase_correlation = np.full(rates.shape[1], np.nan)
    bin_edges = np.linspace(-np.deg2rad(70.0), np.deg2rad(70.0), 8)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for cell in range(rates.shape[1]):
        phase_means = []
        valid_centers = []
        for low, high, center in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
            selected = (relative_heading[:, cell] >= low) & (
                relative_heading[:, cell] < high
            )
            if selected.sum() < 10 or rates[selected, cell].sum() < 1e-6:
                continue
            vector = np.sum(
                rates[selected, cell] * np.exp(1j * theta_phase[selected])
            )
            phase_means.append(np.angle(vector))
            valid_centers.append(center)
        if len(phase_means) >= 4:
            x = np.asarray(valid_centers)
            y = np.unwrap(np.asarray(phase_means))
            phase_slope[cell] = np.polyfit(x, y, 1)[0]
            phase_correlation[cell] = np.corrcoef(x, y)[0, 1]
    return {
        "preferred_direction_rad": preferred,
        "theta_skipping_index": skipping,
        "directional_tuning_width_rad": tuning_width,
        "turn_phase_slope": phase_slope,
        "turn_phase_correlation": phase_correlation,
    }


def save_cycle_csv(cycles):
    path = OUT / "baseline_cycle_metrics.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cycle",
                "direction_offset_deg",
                *[f"grid_{scale:g}cm_length_cm" for scale in GRID_SCALES.to_decimal(u.cm)],
                *[f"grid_{scale:g}cm_offset_deg" for scale in GRID_SCALES.to_decimal(u.cm)],
                *[f"grid_{scale:g}cm_alignment_cosine" for scale in GRID_SCALES.to_decimal(u.cm)],
            ]
        )
        for row in range(cycles["cycle"].size):
            writer.writerow(
                [
                    int(cycles["cycle"][row]),
                    np.degrees(cycles["direction_offset_rad"][row]),
                    *cycles["grid_length_cm"][row],
                    *np.degrees(cycles["grid_offset_rad"][row]),
                    *cycles["alignment_cosine"][row],
                ]
            )


def save_figures(baseline, baseline_cycles, turn, turn_cycles, controls, speed, adaptation, cells):
    steps_per_cycle = round(THETA_PERIOD.to_decimal(u.ms) / DT.to_decimal(u.ms))
    start_cycle = 22
    window = slice(start_cycle * steps_per_cycle, (start_cycle + 8) * steps_per_cycle)
    time_cycles = np.arange(window.stop - window.start) / steps_per_cycle

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
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
            baseline["direction_rad"][window], baseline["heading_rad"][window]
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
    axes[0, 1].lines[-1].set_label("Ring")
    axes[0, 1].plot(
        time_cycles,
        grid_offset,
        color="#1565c0",
        linewidth=1.1,
        alpha=0.9,
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
    snapshot_indices = snapshot_cycles * steps_per_cycle + steps_per_cycle // 2
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
        xticklabels=[str(x) for x in snapshot_cycles],
        yticks=[],
        xlabel="Theta cycle",
        title="Grid-sheet bump at sweep phase",
    )
    fig.colorbar(image, ax=axes[0, 2], label="Rate (a.u.)")

    shuffle = shuffled_alternation(baseline_cycles["direction_offset_rad"])
    axes[1, 0].hist(shuffle, bins=24, color="#bdbdbd", edgecolor="white")
    axes[1, 0].axvline(
        alternation_score(baseline_cycles["direction_offset_rad"]),
        color="#c62828",
        linewidth=2.2,
        label="Observed",
    )
    axes[1, 0].set(
        xlabel="Adjacent-cycle alternation score",
        ylabel="Shuffled cycle orders",
        title="Left-right alternation control",
    )
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        np.degrees(baseline_cycles["direction_offset_rad"]),
        np.degrees(baseline_cycles["grid_offset_rad"][:, 0]),
        s=20,
        alpha=0.72,
        color="#00897b",
    )
    axes[1, 1].plot([-35, 35], [-35, 35], color="black", linewidth=0.8)
    axes[1, 1].set(
        xlim=(-35, 35),
        ylim=(-35, 35),
        xlabel="Ring sweep angle (deg)",
        ylabel="Grid sweep angle (deg)",
        title="Direction-position alignment",
    )

    mean_length = baseline_cycles["grid_length_cm"].mean(axis=0)
    sem_length = baseline_cycles["grid_length_cm"].std(axis=0, ddof=1) / np.sqrt(
        baseline_cycles["grid_length_cm"].shape[0]
    )
    axes[1, 2].errorbar(
        GRID_SCALES.to_decimal(u.cm),
        mean_length,
        yerr=sem_length,
        marker="o",
        capsize=3,
        color="#6a1b9a",
    )
    axes[1, 2].set(
        xlabel="Grid scale (cm)",
        ylabel="Sweep length (cm, mean +/- SEM)",
        title="Scale-dependent spatial extent",
    )
    fig.savefig(OUT / "population_and_alignment.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
    selected_cycles = np.linspace(25, turn_cycles["cycle"].size - 25, 10).astype(int)
    selected_indices = turn_cycles["index"][selected_cycles]
    axes[0, 0].plot(
        turn["position_cm"][:, 0],
        turn["position_cm"][:, 1],
        color="#505050",
        linewidth=1.5,
    )
    endpoints = turn_cycles["grid_endpoint_cm"][selected_cycles, 0]
    colors = np.where(turn_cycles["direction_offset_rad"][selected_cycles] > 0, "#c62828", "#1565c0")
    for point, vector, color in zip(turn["position_cm"][selected_indices], endpoints, colors):
        axes[0, 0].arrow(
            point[0],
            point[1],
            vector[0],
            vector[1],
            width=0.35,
            head_width=2.2,
            head_length=2.8,
            length_includes_head=True,
            color=color,
        )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set(
        xlabel="Arena x (cm)",
        ylabel="Arena y (cm)",
        title="Trajectory and 10 decoded sweep vectors",
    )

    speed_levels = np.asarray(speed["speed_cm_s"])
    speed_lengths = np.asarray(speed["mean_grid_length_cm"])
    for module, scale in enumerate(GRID_SCALES.to_decimal(u.cm)):
        axes[0, 1].plot(
            speed_levels,
            speed_lengths[:, module],
            marker="o",
            label=f"{scale:g} cm",
        )
    axes[0, 1].set(
        xlabel="Running speed (cm/s)",
        ylabel="Sweep length (cm)",
        title="Speed-dependent sweep length",
    )
    axes[0, 1].legend(frameon=False, title="Grid scale")

    axes[0, 2].plot(
        speed_levels,
        speed["mean_abs_direction_angle_deg"],
        marker="o",
        color="#ef6c00",
        label="Angle",
    )
    axes[0, 2].plot(
        speed_levels,
        30.0 * np.asarray(speed["alternation_score"]),
        marker="s",
        color="#00897b",
        label="30 x reliability",
    )
    axes[0, 2].set(
        xlabel="Running speed (cm/s)",
        ylabel="Angle (deg); scaled reliability",
        title="Speed-dependent angle and reliability",
    )
    axes[0, 2].legend(frameon=False)

    adaptation_levels = np.asarray(adaptation["scale"])
    axes[1, 0].plot(
        adaptation_levels,
        adaptation["mean_abs_direction_angle_deg"],
        marker="o",
        color="#ef6c00",
        label="Angle (deg)",
    )
    axes[1, 0].plot(
        adaptation_levels,
        30.0 * np.asarray(adaptation["alternation_score"]),
        marker="s",
        color="#00897b",
        label="30 x reliability",
    )
    axes[1, 0].set(
        xlabel="Adaptation strength (relative)",
        ylabel="Angle (deg); scaled reliability",
        title="Adaptation dependence",
    )
    axes[1, 0].legend(frameon=False)

    control_names = ["Baseline", "No adaptation", "No theta", "No coupling"]
    control_keys = ["baseline", "no_adaptation", "no_theta", "no_coupling"]
    axes[1, 1].bar(
        np.arange(4) - 0.18,
        [controls[key]["alternation_score"] for key in control_keys],
        width=0.36,
        color="#1565c0",
        label="Ring alternation",
    )
    axes[1, 1].bar(
        np.arange(4) + 0.18,
        [controls[key]["mean_alignment_cosine"][0] for key in control_keys],
        width=0.36,
        color="#ef6c00",
        label="Ring-grid alignment",
    )
    axes[1, 1].set_xticks(np.arange(4), control_names, rotation=18)
    axes[1, 1].set(
        ylim=(-1.05, 1.08),
        ylabel="Score",
        title="Matched mechanism controls",
    )
    axes[1, 1].legend(frameon=False)

    finite = np.isfinite(cells["turn_phase_slope"])
    scatter = axes[1, 2].scatter(
        np.degrees(cells["directional_tuning_width_rad"][finite]),
        cells["theta_skipping_index"][finite],
        c=cells["turn_phase_slope"][finite],
        s=28,
        cmap="coolwarm",
        alpha=0.82,
    )
    axes[1, 2].set(
        xlabel="Directional tuning width (deg)",
        ylabel="Theta-skipping index",
        title="Single-cell expression during turns",
    )
    fig.colorbar(scatter, ax=axes[1, 2], label="Theta-phase slope (rad/rad)")
    fig.savefig(OUT / "protocols_and_mechanisms.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    ax.plot(
        turn["position_cm"][:, 0],
        turn["position_cm"][:, 1],
        color="#4a4a4a",
        linewidth=2.0,
        label="Rat trajectory",
    )
    for number, (point, vector, color) in enumerate(
        zip(turn["position_cm"][selected_indices], endpoints, colors), start=1
    ):
        ax.arrow(
            point[0], point[1], vector[0], vector[1], width=0.35,
            head_width=2.2, head_length=2.8, length_includes_head=True, color=color
        )
        ax.text(point[0] + 1.5, point[1] + 1.5, str(number), fontsize=8)
    ax.set_aspect("equal")
    ax.set(
        xlabel="Arena x (cm)",
        ylabel="Arena y (cm)",
        title="Open-field trajectory with decoded theta-sweep vectors",
    )
    ax.legend(frameon=False)
    fig.savefig(OUT / "trajectory_with_10_sweep_vectors.png", dpi=220)
    plt.close(fig)


def full_main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = simulate("straight")
    control_runs = {
        "baseline": baseline,
        "no_adaptation": simulate("straight", adaptation=0.0),
        "no_theta": simulate("straight", theta=0.0),
        "no_coupling": simulate("straight", coupling=0.0),
    }
    control_cycles = {key: cycle_analysis(run) for key, run in control_runs.items()}
    controls = {key: condition_summary(value) for key, value in control_cycles.items()}
    for key, run in control_runs.items():
        controls[key]["phase_resolved_alignment_cosine"] = phase_resolved_alignment(run)

    speed_run = simulate("speed")
    speed_cycles = cycle_analysis(speed_run)
    speed = summarize_speed(speed_cycles)

    adaptation_levels = np.asarray([0.0, 0.5, 1.0, 1.5])
    adaptation_cycles = [
        cycle_analysis(simulate("straight", adaptation=float(level)))
        for level in adaptation_levels
    ]
    adaptation = {
        "scale": adaptation_levels.tolist(),
        "mean_abs_direction_angle_deg": [
            float(np.degrees(np.mean(np.abs(cycle["direction_offset_rad"]))))
            for cycle in adaptation_cycles
        ],
        "alternation_score": [
            alternation_score(cycle["direction_offset_rad"])
            for cycle in adaptation_cycles
        ],
        "mean_grid_length_cm": [
            cycle["grid_length_cm"].mean(axis=0).tolist()
            for cycle in adaptation_cycles
        ],
    }

    turn = simulate("turn")
    turn_cycles = cycle_analysis(turn)
    turn_summary = condition_summary(turn_cycles)
    turn_summary["phase_resolved_alignment_cosine"] = phase_resolved_alignment(turn)
    cells = single_cell_analysis(turn)
    finite_phase = np.isfinite(cells["turn_phase_slope"])
    width_skip = spearmanr(
        cells["directional_tuning_width_rad"], cells["theta_skipping_index"]
    )
    scale_length = spearmanr(
        GRID_SCALES.to_decimal(u.cm),
        control_cycles["baseline"]["grid_length_cm"].mean(axis=0),
    )
    speed_length = spearmanr(
        speed["speed_cm_s"], np.asarray(speed["mean_grid_length_cm"])[:, 0]
    )
    metrics = {
        "artifact_version": "2.0",
        "brainx_release": "2026.7.9",
        "model_seed": MODEL_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "integration_dt_ms": DT.to_decimal(u.ms),
        "theta_frequency_hz": THETA_FREQUENCY.to_decimal(u.Hz),
        "grid_scales_cm": GRID_SCALES.to_decimal(u.cm).tolist(),
        "evaluation_phase_rad": float(np.pi),
        "minimum_sweep_angle_deg": 5.0,
        "conditions": controls,
        "turn": turn_summary,
        "speed": speed,
        "adaptation": adaptation,
        "grid_scale_length_spearman": {
            "rho": float(scale_length.statistic),
            "p": float(scale_length.pvalue),
        },
        "speed_length_spearman_smallest_module": {
            "rho": float(speed_length.statistic),
            "p": float(speed_length.pvalue),
        },
        "single_cell": {
            "n_direction_cells": int(cells["theta_skipping_index"].size),
            "median_theta_skipping_index": float(
                np.median(cells["theta_skipping_index"])
            ),
            "tuning_width_skipping_spearman": {
                "rho": float(width_skip.statistic),
                "p": float(width_skip.pvalue),
            },
            "n_cells_with_turn_phase_code": int(finite_phase.sum()),
            "median_abs_turn_phase_correlation": float(
                np.nanmedian(np.abs(cells["turn_phase_correlation"]))
            ),
            "median_turn_phase_slope_rad_per_rad": float(
                np.nanmedian(cells["turn_phase_slope"])
            ),
        },
    }

    np.savez_compressed(
        OUT / "theta_direction_grid_evidence.npz",
        baseline_time_s=baseline["time_s"],
        baseline_position_cm=baseline["position_cm"],
        baseline_heading_rad=baseline["heading_rad"],
        baseline_direction_rad=baseline["direction_rad"],
        baseline_grid_phase_rad=baseline["grid_phase_rad"],
        baseline_direction_rate=baseline["direction_rate"],
        baseline_grid_rate=baseline["grid_rate"],
        baseline_cycle_index=control_cycles["baseline"]["cycle"],
        baseline_direction_offset_rad=control_cycles["baseline"]["direction_offset_rad"],
        baseline_grid_endpoint_cm=control_cycles["baseline"]["grid_endpoint_cm"],
        baseline_alignment_cosine=control_cycles["baseline"]["alignment_cosine"],
        turn_time_s=turn["time_s"],
        turn_position_cm=turn["position_cm"],
        turn_heading_rad=turn["heading_rad"],
        turn_direction_rad=turn["direction_rad"],
        turn_grid_endpoint_cm=turn_cycles["grid_endpoint_cm"],
        speed_levels_cm_s=np.asarray(speed["speed_cm_s"]),
        speed_grid_length_cm=np.asarray(speed["mean_grid_length_cm"]),
        adaptation_scale=adaptation_levels,
        adaptation_angle_deg=np.asarray(adaptation["mean_abs_direction_angle_deg"]),
        adaptation_alternation=np.asarray(adaptation["alternation_score"]),
        cell_theta_skipping=cells["theta_skipping_index"],
        cell_tuning_width_rad=cells["directional_tuning_width_rad"],
        cell_turn_phase_slope=cells["turn_phase_slope"],
    )
    save_cycle_csv(control_cycles["baseline"])
    save_figures(
        baseline,
        control_cycles["baseline"],
        turn,
        turn_cycles,
        controls,
        speed,
        adaptation,
        cells,
    )
    (OUT / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def quick_main():
    baseline = simulate("straight")
    cycles = cycle_analysis(baseline)
    print(json.dumps(condition_summary(cycles), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    quick_main() if args.quick else full_main()
