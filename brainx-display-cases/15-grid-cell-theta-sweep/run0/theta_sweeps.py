"""Direction-grid continuous attractor model for alternating theta sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import brainmass
import brainstate
import brainstate.nn as nn
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr


OUT = Path(__file__).resolve().parent
DT = 1.0 * u.ms
THETA_FREQUENCY = 10.0 * u.Hz
THETA_PERIOD_MS = (1.0 / THETA_FREQUENCY).to_decimal(u.ms)
REFERENCE_SPEED = 30.0 * u.cm / u.second
GRID_SCALES = np.asarray([45.0, 72.0]) * u.cm
SEED = 17
SHUFFLE_SEED = 2025
ARTIFACT_VERSION = "1.0"


def wrap_angle(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def circular_difference(a, b):
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


def circular_mean(values, axis=0, weights=None):
    values = np.asarray(values)
    if weights is None:
        z = np.mean(np.exp(1j * values), axis=axis)
    else:
        z = np.sum(np.asarray(weights) * np.exp(1j * values), axis=axis)
    return np.angle(z)


class DirectionGridNetwork(nn.Module):
    """Coupled firing-rate ring and toroidal grid attractors."""

    def __init__(
        self,
        *,
        n_hd=60,
        n_grid=20,
        grid_scales_cm=(45.0, 72.0),
        adaptation_scale=1.0,
        theta_scale=1.0,
        coupling_scale=1.0,
    ):
        super().__init__()
        self.n_hd = n_hd
        self.n_grid = n_grid
        self.n_cell = n_grid * n_grid
        self.grid_scales_cm = jnp.asarray(grid_scales_cm)
        self.adaptation_scale = adaptation_scale
        self.theta_scale = theta_scale
        self.coupling_scale = coupling_scale

        self.tau_hd_ms = 10.0
        self.tau_hd_adapt_ms = 100.0
        self.hd_adapt_gain = 1.15 * adaptation_scale
        self.hd_inhibition = 0.32
        self.hd_sensory_gain = 2.8
        self.hd_theta_gain = 0.58 * theta_scale
        self.hd_tuning_width = 0.40

        self.tau_grid_ms = 10.0
        self.tau_grid_adapt_ms = 100.0
        self.grid_adapt_gain = 0.72 * adaptation_scale
        self.grid_inhibition = 0.045
        self.grid_rate_gain = 35.0
        self.grid_sensory_gain = 1.35
        self.grid_coupling_gain = 2.7 * coupling_scale
        self.grid_theta_gain = 0.72 * theta_scale
        self.grid_tuning_width = 0.52
        self.shift_fraction = 0.17

        self.hd_angles = jnp.linspace(-jnp.pi, jnp.pi, n_hd, endpoint=False)
        hd_distance = wrap_angle(self.hd_angles - self.hd_angles[0])
        hd_kernel = jnp.exp(-0.5 * (hd_distance / 0.42) ** 2)
        hd_kernel = 3.7 * hd_kernel / jnp.sum(hd_kernel)
        self.hd_kernel_fft = jnp.fft.fft(hd_kernel)

        phase = jnp.linspace(-jnp.pi, jnp.pi, n_grid, endpoint=False)
        phase_x, phase_y = jnp.meshgrid(phase, phase, indexing="ij")
        self.phase_x = phase_x.reshape(-1)
        self.phase_y = phase_y.reshape(-1)
        grid_distance = jnp.sqrt(
            wrap_angle(phase_x - phase_x[0, 0]) ** 2
            + wrap_angle(phase_y - phase_y[0, 0]) ** 2
        )
        grid_kernel = jnp.exp(-0.5 * (grid_distance / 0.56) ** 2)
        grid_kernel = 2.6 * grid_kernel / jnp.sum(grid_kernel)
        self.grid_kernel_fft = jnp.fft.fft2(grid_kernel)

        # Oblique axes map physical space to a hexagonal grid phase torus.
        self.phase_basis = jnp.asarray(
            [[1.0, 0.0], [0.5, jnp.sqrt(3.0) / 2.0]]
        )
        self.inverse_phase_basis = jnp.linalg.inv(self.phase_basis)

        perturb_hd = 1e-3 * brainstate.random.randn(n_hd)
        perturb_grid = 1e-3 * brainstate.random.randn(len(grid_scales_cm), self.n_cell)
        self.hd_u = brainstate.HiddenState(perturb_hd)
        self.hd_adaptation = brainstate.HiddenState(jnp.zeros(n_hd))
        self.hd_rate = brainstate.HiddenState(jnp.zeros(n_hd))
        self.grid_u = brainstate.HiddenState(perturb_grid)
        self.grid_adaptation = brainstate.HiddenState(jnp.zeros_like(perturb_grid))
        self.grid_rate = brainstate.HiddenState(jnp.zeros_like(perturb_grid))

    def _hd_rates(self, voltage):
        rectified = jnp.maximum(voltage, 0.0) ** 2
        return rectified / (1.0 + self.hd_inhibition * jnp.sum(rectified))

    def _grid_rates(self, voltage):
        rectified = jnp.maximum(voltage, 0.0) ** 2
        denominator = 1.0 + self.grid_inhibition * jnp.sum(rectified, axis=1, keepdims=True)
        return self.grid_rate_gain * rectified / denominator

    def _physical_to_phase(self, position_cm):
        projected = self.phase_basis @ position_cm
        return wrap_angle(
            (2.0 * jnp.pi / self.grid_scales_cm[:, None]) * projected[None, :]
        )

    def _toroidal_gaussian(self, centers):
        dx = wrap_angle(self.phase_x[None, None, :] - centers[:, :, 0, None])
        dy = wrap_angle(self.phase_y[None, None, :] - centers[:, :, 1, None])
        distance2 = dx**2 + dy**2
        return jnp.exp(-0.25 * distance2 / self.grid_tuning_width**2)

    def update(self, step_index, position_cm, heading, speed_cm_s):
        dt_ms = DT.to_decimal(u.ms)
        phase = 2.0 * jnp.pi * (step_index * dt_ms % THETA_PERIOD_MS) / THETA_PERIOD_MS
        speed_ratio = speed_cm_s / REFERENCE_SPEED.to_decimal(u.cm / u.second)

        old_hd_rate = self._hd_rates(self.hd_u.value)
        hd_recurrent = jnp.real(
            jnp.fft.ifft(jnp.fft.fft(old_hd_rate) * self.hd_kernel_fft)
        )
        hd_distance = wrap_angle(self.hd_angles - heading)
        theta_hd = 1.0 + self.hd_theta_gain * speed_ratio * jnp.cos(phase)
        sensory = (
            self.hd_sensory_gain
            * theta_hd
            * jnp.exp(-0.25 * (hd_distance / self.hd_tuning_width) ** 2)
        )
        hd_target = hd_recurrent - self.hd_adaptation.value + sensory
        hd_decay = jnp.exp(-dt_ms / self.tau_hd_ms)
        hd_adapt_decay = jnp.exp(-dt_ms / self.tau_hd_adapt_ms)
        new_hd_u = hd_decay * self.hd_u.value + (1.0 - hd_decay) * hd_target
        new_hd_rate = self._hd_rates(new_hd_u)
        new_hd_adaptation = (
            hd_adapt_decay * self.hd_adaptation.value
            + (1.0 - hd_adapt_decay) * self.hd_adapt_gain * new_hd_rate
        )

        animal_phase = self._physical_to_phase(position_cm)
        direction_vectors = jnp.stack(
            [jnp.cos(self.hd_angles), jnp.sin(self.hd_angles)], axis=1
        )
        projected_directions = direction_vectors @ self.phase_basis.T
        phase_offsets = 2.0 * jnp.pi * self.shift_fraction * projected_directions
        shifted_centers = wrap_angle(animal_phase[:, None, :] + phase_offsets[None, :, :])
        conjunctive_fields = self._toroidal_gaussian(shifted_centers)
        hd_weights = new_hd_rate**4
        hd_weights = hd_weights / (jnp.sum(hd_weights) + 1e-8)
        shifted_input = jnp.sum(conjunctive_fields * hd_weights[None, :, None], axis=1)

        position_centers = animal_phase[:, None, :]
        position_input = self._toroidal_gaussian(position_centers)[:, 0, :]
        theta_grid = 1.0 + self.grid_theta_gain * speed_ratio * jnp.cos(phase - 0.18 * jnp.pi)
        external_grid = (
            self.grid_sensory_gain * position_input
            + self.grid_coupling_gain * speed_ratio * theta_grid * shifted_input
        )

        old_grid_rate = self._grid_rates(self.grid_u.value)
        grid_2d = old_grid_rate.reshape(-1, self.n_grid, self.n_grid)
        grid_recurrent = jnp.real(
            jnp.fft.ifft2(jnp.fft.fft2(grid_2d) * self.grid_kernel_fft)
        ).reshape(-1, self.n_cell)
        grid_target = grid_recurrent - self.grid_adaptation.value + external_grid
        grid_decay = jnp.exp(-dt_ms / self.tau_grid_ms)
        grid_adapt_decay = jnp.exp(-dt_ms / self.tau_grid_adapt_ms)
        new_grid_u = grid_decay * self.grid_u.value + (1.0 - grid_decay) * grid_target
        new_grid_rate = self._grid_rates(new_grid_u)
        new_grid_adaptation = (
            grid_adapt_decay * self.grid_adaptation.value
            + (1.0 - grid_adapt_decay) * self.grid_adapt_gain * new_grid_rate
        )

        self.hd_u.value = new_hd_u
        self.hd_rate.value = new_hd_rate
        self.hd_adaptation.value = new_hd_adaptation
        self.grid_u.value = new_grid_u
        self.grid_rate.value = new_grid_rate
        self.grid_adaptation.value = new_grid_adaptation

        hd_center = jnp.angle(jnp.sum(new_hd_rate * jnp.exp(1j * self.hd_angles)))
        grid_phase_x = jnp.angle(
            jnp.sum(new_grid_rate * jnp.exp(1j * self.phase_x)[None, :], axis=1)
        )
        grid_phase_y = jnp.angle(
            jnp.sum(new_grid_rate * jnp.exp(1j * self.phase_y)[None, :], axis=1)
        )
        grid_center = jnp.stack([grid_phase_x, grid_phase_y], axis=1)
        return hd_center, grid_center, new_hd_rate, phase


def make_protocol(kind):
    dt_s = DT.to_decimal(u.second)
    if kind == "straight":
        duration_s = 7.0
        speed = np.full(round(duration_s / dt_s), 30.0)
        heading = np.zeros_like(speed)
    elif kind == "speed":
        duration_s = 9.0
        speed = np.repeat([15.0, 30.0, 45.0], round(3.0 / dt_s))
        heading = np.zeros_like(speed)
    elif kind == "turn":
        duration_s = 9.0
        speed = np.full(round(duration_s / dt_s), 25.0)
        heading = np.arange(speed.size) * dt_s * (2.0 * np.pi / 8.0)
        heading = np.angle(np.exp(1j * heading))
    else:
        raise ValueError(f"unknown protocol: {kind}")

    velocity = speed[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    position = np.cumsum(velocity * dt_s, axis=0)
    return {
        "step": np.arange(speed.size, dtype=np.int32),
        "time_s": (np.arange(speed.size) + 1) * dt_s,
        "position_cm": position,
        "heading_rad": heading,
        "speed_cm_s": speed,
    }


def run_protocol(kind, *, adaptation_scale=1.0, theta_scale=1.0, coupling_scale=1.0):
    protocol = make_protocol(kind)
    brainstate.random.seed(SEED)
    network = DirectionGridNetwork(
        grid_scales_cm=GRID_SCALES.to_decimal(u.cm),
        adaptation_scale=adaptation_scale,
        theta_scale=theta_scale,
        coupling_scale=coupling_scale,
    )

    def step(index, position, heading, speed):
        with brainstate.environ.context(i=index, t=index * DT):
            return network.update(index, position, heading, speed)

    with brainstate.environ.context(dt=DT):
        hd_center, grid_phase, hd_rate, theta_phase = brainstate.transform.for_loop(
            step,
            jnp.asarray(protocol["step"]),
            jnp.asarray(protocol["position_cm"]),
            jnp.asarray(protocol["heading_rad"]),
            jnp.asarray(protocol["speed_cm_s"]),
        )
    jax.block_until_ready(grid_phase)
    protocol.update(
        hd_center=np.asarray(hd_center),
        grid_phase=np.asarray(grid_phase),
        hd_rate=np.asarray(hd_rate),
        theta_phase=np.asarray(theta_phase),
        final_grid_rate=np.asarray(network.grid_rate.value),
    )
    return protocol


def phase_to_displacement(grid_phase, position_cm):
    basis = np.asarray([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    inverse_basis = np.linalg.inv(basis)
    projected = position_cm @ basis.T
    animal_phase = (
        2.0 * np.pi * projected[:, None, :] / GRID_SCALES.to_decimal(u.cm)[None, :, None]
    )
    phase_error = circular_difference(grid_phase, animal_phase)
    physical = np.einsum("ij,tmj->tmi", inverse_basis, phase_error)
    return physical * GRID_SCALES.to_decimal(u.cm)[None, :, None] / (2.0 * np.pi)


def cycle_metrics(run, warmup_s=1.0):
    samples_per_cycle = round(THETA_PERIOD_MS / DT.to_decimal(u.ms))
    displacement = phase_to_displacement(run["grid_phase"], run["position_cm"])
    first_cycle = int(np.ceil(warmup_s * 1000.0 / THETA_PERIOD_MS))
    n_cycles = len(run["time_s"]) // samples_per_cycle
    metrics = []
    for cycle in range(first_cycle, n_cycles):
        sl = slice(cycle * samples_per_cycle, (cycle + 1) * samples_per_cycle)
        heading = circular_mean(run["heading_rad"][sl])
        early = slice(sl.start + 5, sl.start + 20)
        early_hd = circular_mean(run["hd_center"][early])
        hd_offsets = circular_difference(run["hd_center"][sl], early_hd)
        hd_peak_index = int(np.argmax(np.abs(hd_offsets[15:90])) + 15)
        internal_direction = run["hd_center"][sl][hd_peak_index]
        hd_sweep_angle = hd_offsets[hd_peak_index]

        start = np.mean(displacement[early], axis=0)
        local = displacement[sl] - start[None, :, :]
        distance = np.linalg.norm(local, axis=2)
        peak_index = np.argmax(distance[15:90], axis=0) + 15
        endpoint = np.stack([local[peak_index[m], m] for m in range(local.shape[1])])
        sweep_length = np.linalg.norm(endpoint, axis=1)
        sweep_direction = np.arctan2(endpoint[:, 1], endpoint[:, 0])
        sweep_angle = circular_difference(sweep_direction, heading)
        alignment = np.cos(circular_difference(sweep_direction, internal_direction))
        metrics.append(
            (
                cycle,
                np.mean(run["speed_cm_s"][sl]),
                heading,
                hd_sweep_angle,
                internal_direction,
                sweep_length,
                sweep_direction,
                sweep_angle,
                alignment,
                endpoint,
            )
        )
    return {
        "cycle": np.asarray([m[0] for m in metrics]),
        "speed_cm_s": np.asarray([m[1] for m in metrics]),
        "heading_rad": np.asarray([m[2] for m in metrics]),
        "hd_sweep_angle_rad": np.asarray([m[3] for m in metrics]),
        "internal_direction_rad": np.asarray([m[4] for m in metrics]),
        "sweep_length_cm": np.asarray([m[5] for m in metrics]),
        "sweep_direction_rad": np.asarray([m[6] for m in metrics]),
        "sweep_angle_rad": np.asarray([m[7] for m in metrics]),
        "alignment_cosine": np.asarray([m[8] for m in metrics]),
        "endpoint_cm": np.asarray([m[9] for m in metrics]),
    }


def alternation_score(angles):
    signs = np.sign(np.asarray(angles))
    valid = signs != 0
    signs = signs[valid]
    if signs.size < 2:
        return np.nan
    return float(np.mean(signs[1:] != signs[:-1]))


def shuffled_alternation(angles, n_shuffle=1000):
    rng = np.random.default_rng(SHUFFLE_SEED)
    angles = np.asarray(angles)
    return np.asarray([alternation_score(rng.permutation(angles)) for _ in range(n_shuffle)])


def single_cell_metrics(turn_run, warmup_s=1.0):
    samples_per_cycle = round(THETA_PERIOD_MS / DT.to_decimal(u.ms))
    start = round(warmup_s / DT.to_decimal(u.second))
    rates = turn_run["hd_rate"][start:]
    heading = turn_run["heading_rad"][start:]
    n_cycle = len(rates) // samples_per_cycle
    cycle_rate = rates[: n_cycle * samples_per_cycle].reshape(
        n_cycle, samples_per_cycle, rates.shape[1]
    ).sum(axis=1)
    centered = cycle_rate - cycle_rate.mean(axis=0, keepdims=True)
    lag1 = np.sum(centered[1:] * centered[:-1], axis=0)
    lag2 = np.sum(centered[2:] * centered[:-2], axis=0)
    skipping = (lag2 - lag1) / (np.abs(lag2) + np.abs(lag1) + 1e-9)

    preferred = np.linspace(-np.pi, np.pi, rates.shape[1], endpoint=False)
    relative = circular_difference(heading[:, None], preferred[None, :])
    resultant = np.abs(np.sum(rates * np.exp(1j * relative), axis=0)) / (
        np.sum(rates, axis=0) + 1e-9
    )
    tuning_width = np.sqrt(np.maximum(-2.0 * np.log(np.clip(resultant, 1e-9, 1.0)), 0.0))

    phase = turn_run["theta_phase"][start:]
    phase_coding = np.full(rates.shape[1], np.nan)
    phase_slope = np.full(rates.shape[1], np.nan)
    for cell in range(rates.shape[1]):
        mask = (np.abs(relative[:, cell]) < 0.75) & (rates[:, cell] > np.quantile(rates[:, cell], 0.55))
        if np.sum(mask) < 20:
            continue
        x = relative[mask, cell]
        z = np.sum(rates[mask, cell] * np.exp(1j * phase[mask]))
        center = np.angle(z)
        y = np.unwrap(circular_difference(phase[mask], center))
        order = np.argsort(x)
        fit = linregress(x[order], y[order])
        phase_coding[cell] = fit.rvalue
        phase_slope[cell] = fit.slope
    return {
        "preferred_direction_rad": preferred,
        "theta_skipping_index": skipping,
        "directional_tuning_width_rad": tuning_width,
        "turn_phase_correlation": phase_coding,
        "turn_phase_slope": phase_slope,
    }


def summarize_condition(metrics):
    shuffled = shuffled_alternation(metrics["sweep_angle_rad"][:, 0])
    return {
        "alternation": alternation_score(metrics["sweep_angle_rad"][:, 0]),
        "shuffle_mean": float(np.nanmean(shuffled)),
        "shuffle_95": [float(x) for x in np.nanpercentile(shuffled, [2.5, 97.5])],
        "shuffle_p_upper": float((1 + np.sum(shuffled >= alternation_score(metrics["sweep_angle_rad"][:, 0]))) / (1 + len(shuffled))),
        "mean_alignment": float(np.mean(metrics["alignment_cosine"][:, 0])),
        "mean_sweep_length_cm": [float(x) for x in np.mean(metrics["sweep_length_cm"], axis=0)],
        "mean_abs_sweep_angle_deg": float(np.degrees(np.mean(np.abs(metrics["sweep_angle_rad"][:, 0])))),
    }


def save_figures(baseline, baseline_cycles, condition_summary, speed_summary, adaptation_summary, cells):
    displacement = phase_to_displacement(baseline["grid_phase"], baseline["position_cm"])
    samples_per_cycle = round(THETA_PERIOD_MS / DT.to_decimal(u.ms))
    cycle0 = int(baseline_cycles["cycle"][5])
    window = slice(cycle0 * samples_per_cycle, (cycle0 + 8) * samples_per_cycle)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(
        baseline["hd_rate"][window].T,
        aspect="auto",
        origin="lower",
        extent=[0, 8, -180, 180],
        cmap="magma",
    )
    ax.set(xlabel="Theta cycles", ylabel="Preferred direction (deg)", title="Direction-ring activity")
    fig.colorbar(im, ax=ax, label="Rate (a.u.)")

    ax = axes[0, 1]
    relative_hd = np.degrees(circular_difference(baseline["hd_center"][window], baseline["heading_rad"][window]))
    brainmass.viz.plot_timeseries(relative_hd, ax=ax, color="#1565c0", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlabel="Time within 8 cycles (ms)", ylabel="Internal - head direction (deg)", title="Alternating internal direction")

    ax = axes[0, 2]
    for index in range(10):
        endpoint = baseline_cycles["endpoint_cm"][index + 3, 0]
        color = "#c62828" if baseline_cycles["sweep_angle_rad"][index + 3, 0] > 0 else "#1565c0"
        ax.plot([0, endpoint[0]], [0, endpoint[1]], color=color, alpha=0.75)
    ax.set_aspect("equal")
    ax.set(xlabel="Forward displacement (cm)", ylabel="Lateral displacement (cm)", title="Decoded grid sweeps")

    ax = axes[1, 0]
    shuffled = shuffled_alternation(baseline_cycles["sweep_angle_rad"][:, 0])
    ax.hist(shuffled, bins=24, color="#bdbdbd", edgecolor="white")
    ax.axvline(alternation_score(baseline_cycles["sweep_angle_rad"][:, 0]), color="#c62828", linewidth=2, label="Observed")
    ax.set(xlabel="Left-right alternation score", ylabel="Shuffles", title="Cycle-order control")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.scatter(
        np.degrees(baseline_cycles["internal_direction_rad"] - baseline_cycles["heading_rad"]),
        np.degrees(baseline_cycles["sweep_angle_rad"][:, 0]),
        s=18,
        alpha=0.65,
        color="#00897b",
    )
    ax.plot([-90, 90], [-90, 90], color="black", linewidth=0.8)
    ax.set(xlabel="Internal direction offset (deg)", ylabel="Grid sweep angle (deg)", title="Direction-position alignment")

    ax = axes[1, 2]
    mean_length = np.mean(baseline_cycles["sweep_length_cm"], axis=0)
    ax.plot(GRID_SCALES.to_decimal(u.cm), mean_length, marker="o", color="#6a1b9a")
    ax.set(xlabel="Grid scale (cm)", ylabel="Mean sweep length (cm)", title="Sweep length scales with module")
    fig.savefig(OUT / "theta_sweeps_main.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(speed_summary["speed_cm_s"], speed_summary["length_cm"], marker="o", color="#1565c0")
    ax.set(xlabel="Running speed (cm/s)", ylabel="Sweep length (cm)", title="Speed modulation")
    ax = axes[0, 1]
    ax.plot(speed_summary["speed_cm_s"], speed_summary["angle_deg"], marker="o", color="#ef6c00")
    ax.set(xlabel="Running speed (cm/s)", ylabel="Absolute sweep angle (deg)", title="Sweep angle")
    ax = axes[0, 2]
    ax.plot(speed_summary["speed_cm_s"], speed_summary["alternation"], marker="o", color="#00897b")
    ax.set(xlabel="Running speed (cm/s)", ylabel="Alternation score", ylim=(0, 1.05), title="Alternation reliability")

    ax = axes[1, 0]
    ax.plot(adaptation_summary["scale"], adaptation_summary["length_cm"], marker="o", label="Length")
    ax2 = ax.twinx()
    ax2.plot(adaptation_summary["scale"], adaptation_summary["alternation"], marker="s", color="#c62828", label="Alternation")
    ax.set(xlabel="Adaptation scale", ylabel="Sweep length (cm)", title="Adaptation dependence")
    ax2.set(ylabel="Alternation score", ylim=(0, 1.05))

    ax = axes[1, 1]
    names = list(condition_summary)
    ax.bar(np.arange(len(names)) - 0.18, [condition_summary[n]["alternation"] for n in names], width=0.36, label="Alternation", color="#1565c0")
    ax.bar(np.arange(len(names)) + 0.18, [condition_summary[n]["mean_alignment"] for n in names], width=0.36, label="Alignment", color="#ef6c00")
    ax.set_xticks(np.arange(len(names)), names, rotation=20)
    ax.set(ylabel="Score", ylim=(-0.2, 1.05), title="Matched mechanism controls")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    ax.scatter(np.degrees(cells["directional_tuning_width_rad"]), cells["theta_skipping_index"], s=24, alpha=0.7, color="#6a1b9a")
    ax.set(xlabel="Directional tuning width (deg)", ylabel="Theta-skipping index", title="Single-cell expression")
    fig.savefig(OUT / "theta_sweeps_mechanisms.png", dpi=180)
    plt.close(fig)


def main():
    baseline = run_protocol("straight")
    condition_runs = {
        "baseline": baseline,
        "no_adaptation": run_protocol("straight", adaptation_scale=0.0),
        "no_theta": run_protocol("straight", theta_scale=0.0),
        "no_coupling": run_protocol("straight", coupling_scale=0.0),
    }
    condition_cycles = {name: cycle_metrics(run) for name, run in condition_runs.items()}
    condition_summary = {name: summarize_condition(cycles) for name, cycles in condition_cycles.items()}

    speed_run = run_protocol("speed")
    speed_cycles = cycle_metrics(speed_run)
    speed_levels = np.asarray([15.0, 30.0, 45.0])
    speed_summary = {
        "speed_cm_s": speed_levels,
        "length_cm": np.asarray([np.mean(speed_cycles["sweep_length_cm"][np.isclose(speed_cycles["speed_cm_s"], speed), 0]) for speed in speed_levels]),
        "angle_deg": np.asarray([np.degrees(np.mean(np.abs(speed_cycles["sweep_angle_rad"][np.isclose(speed_cycles["speed_cm_s"], speed), 0]))) for speed in speed_levels]),
        "alternation": np.asarray([alternation_score(speed_cycles["sweep_angle_rad"][np.isclose(speed_cycles["speed_cm_s"], speed), 0]) for speed in speed_levels]),
    }

    adaptation_levels = np.asarray([0.0, 0.5, 1.0, 1.5])
    adaptation_cycles = [cycle_metrics(run_protocol("straight", adaptation_scale=float(level))) for level in adaptation_levels]
    adaptation_summary = {
        "scale": adaptation_levels,
        "length_cm": np.asarray([np.mean(cycles["sweep_length_cm"][:, 0]) for cycles in adaptation_cycles]),
        "angle_deg": np.asarray([np.degrees(np.mean(np.abs(cycles["sweep_angle_rad"][:, 0]))) for cycles in adaptation_cycles]),
        "alternation": np.asarray([alternation_score(cycles["sweep_angle_rad"][:, 0]) for cycles in adaptation_cycles]),
    }

    turn_run = run_protocol("turn")
    cells = single_cell_metrics(turn_run)
    finite = np.isfinite(cells["turn_phase_correlation"])
    width_skipping = spearmanr(cells["directional_tuning_width_rad"], cells["theta_skipping_index"])
    metrics = {
        "artifact_version": ARTIFACT_VERSION,
        "seed": SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "dt_ms": DT.to_decimal(u.ms),
        "theta_frequency_hz": THETA_FREQUENCY.to_decimal(u.Hz),
        "grid_scales_cm": GRID_SCALES.to_decimal(u.cm).tolist(),
        "conditions": condition_summary,
        "grid_scale_length_spearman": float(spearmanr(GRID_SCALES.to_decimal(u.cm), np.mean(condition_cycles["baseline"]["sweep_length_cm"], axis=0)).statistic),
        "speed_length_spearman": float(spearmanr(speed_summary["speed_cm_s"], speed_summary["length_cm"]).statistic),
        "adaptation_alternation_spearman": float(spearmanr(adaptation_summary["scale"], adaptation_summary["alternation"]).statistic),
        "tuning_width_skipping_spearman": {"rho": float(width_skipping.statistic), "p": float(width_skipping.pvalue)},
        "turn_phase_coding": {
            "n_cells": int(np.sum(finite)),
            "median_abs_correlation": float(np.nanmedian(np.abs(cells["turn_phase_correlation"]))),
            "median_slope_rad_per_rad": float(np.nanmedian(cells["turn_phase_slope"])),
        },
    }

    np.savez_compressed(
        OUT / "theta_sweeps_evidence.npz",
        baseline_time_s=baseline["time_s"],
        baseline_position_cm=baseline["position_cm"],
        baseline_heading_rad=baseline["heading_rad"],
        baseline_hd_center_rad=baseline["hd_center"],
        baseline_grid_phase_rad=baseline["grid_phase"],
        baseline_hd_rate=baseline["hd_rate"],
        baseline_cycle_index=condition_cycles["baseline"]["cycle"],
        baseline_sweep_length_cm=condition_cycles["baseline"]["sweep_length_cm"],
        baseline_sweep_angle_rad=condition_cycles["baseline"]["sweep_angle_rad"],
        baseline_alignment_cosine=condition_cycles["baseline"]["alignment_cosine"],
        speed_levels_cm_s=speed_summary["speed_cm_s"],
        speed_sweep_length_cm=speed_summary["length_cm"],
        speed_sweep_angle_deg=speed_summary["angle_deg"],
        speed_alternation=speed_summary["alternation"],
        adaptation_scale=adaptation_summary["scale"],
        adaptation_sweep_length_cm=adaptation_summary["length_cm"],
        adaptation_sweep_angle_deg=adaptation_summary["angle_deg"],
        adaptation_alternation=adaptation_summary["alternation"],
        cell_tuning_width_rad=cells["directional_tuning_width_rad"],
        cell_theta_skipping=cells["theta_skipping_index"],
        cell_turn_phase_correlation=cells["turn_phase_correlation"],
        cell_turn_phase_slope=cells["turn_phase_slope"],
    )
    (OUT / "theta_sweeps_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_figures(
        baseline,
        condition_cycles["baseline"],
        condition_summary,
        speed_summary,
        adaptation_summary,
        cells,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
