"""A unit-aware E/I cortical wave encountering a circular silent patch.

The simulation keeps one excitatory and one inhibitory LIF neuron at every
location on a rectangular sheet. Local spikes travel through explicit
BrainEvent CSR matrices. BrainState maps independent lesion/inhibition
conditions while one ``for_loop`` advances all conditions through time.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
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
from matplotlib.colors import BoundaryNorm, ListedColormap


@dataclass(frozen=True)
class ExperimentConfig:
    nx: int = 34
    ny: int = 22
    spacing: object = field(default_factory=lambda: 1.0 * u.mm)
    dt: object = field(default_factory=lambda: 0.25 * u.ms)
    duration: object = field(default_factory=lambda: 130.0 * u.ms)
    obstacle_x: object = field(default_factory=lambda: 16.0 * u.mm)
    obstacle_y: object = field(default_factory=lambda: 9.5 * u.mm)
    lesion_radii_mm: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.5)
    inhibition_gains: tuple[float, ...] = (0.64, 0.68, 0.72, 0.76, 0.78, 0.80)
    seed: int = 23


# Neuron and synapse parameters. R=1 ohm makes each mA of current contribute
# one mV to the LIF driving term, matching the BrainPy-State COBA convention.
V_REST_E = -62.0 * u.mV
V_REST_I = -62.0 * u.mV
V_RESET = -64.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
E_EXC = 0.0 * u.mV
E_INH = -80.0 * u.mV
R_MEMBRANE = 1.0 * u.ohm
TAU_E = 16.0 * u.ms
TAU_I = 10.0 * u.ms
TAU_REF_E = 3.0 * u.ms
TAU_REF_I = 2.0 * u.ms
TAU_SYN_EXC = 3.5 * u.ms
TAU_SYN_INH = 6.0 * u.ms
BACKGROUND_E = 9.5 * u.mA
BACKGROUND_I = 3.5 * u.mA
SPARK_CURRENT = 95.0 * u.mA
SPARK_DURATION = 3.0 * u.ms
LESION_CLAMP = -120.0 * u.mA

EE_WEIGHT = 1.25 * u.siemens
EI_WEIGHT = 0.55 * u.siemens
IE_WEIGHT = 0.45 * u.siemens

OUTCOME_NAMES = ("crosses", "bends", "splits", "dies")
CROSSES, BENDS, SPLITS, DIES = range(4)
DISPLAY_WINDOW_MS = 6.0


def sheet_coordinates(config: ExperimentConfig):
    """Return flattened x/y coordinates with physical length units."""
    x = u.math.arange(config.nx) * config.spacing
    y = u.math.arange(config.ny) * config.spacing
    x_grid, y_grid = u.math.meshgrid(x, y, indexing="xy")
    return x_grid.flatten(), y_grid.flatten()


def circular_lesion_mask(x, y, radius, center_x, center_y):
    """True at sheet locations silenced by the circular lesion."""
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def make_local_csr(
    nx: int,
    ny: int,
    radius_cells: float,
    weight,
    *,
    include_self: bool,
):
    """Construct a row-oriented local projection on a rectangular lattice."""
    offsets: list[tuple[int, int]] = []
    reach = int(np.ceil(radius_cells))
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if not include_self and dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                offsets.append((dx, dy))

    indices: list[int] = []
    indptr = [0]
    for y in range(ny):
        for x in range(nx):
            for dx, dy in offsets:
                tx, ty = x + dx, y + dy
                if 0 <= tx < nx and 0 <= ty < ny:
                    indices.append(ty * nx + tx)
            indptr.append(len(indices))

    n_sites = nx * ny
    return brainevent.CSR(
        (
            u.math.asarray(weight, dtype=brainstate.environ.dftype()),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(n_sites, n_sites),
    )


class CorticalSheet(brainstate.nn.Module):
    """Spatial E/I sheet with event-driven local conductance projections."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.n_sites = config.nx * config.ny
        self.x, self.y = sheet_coordinates(config)
        self.obstacle_distance = u.math.sqrt(
            (self.x - config.obstacle_x) ** 2
            + (self.y - config.obstacle_y) ** 2
        )
        self.spark_mask = self.x <= config.spacing

        brainstate.random.seed(config.seed)
        self.exc = brainpy.state.LIFRef(
            self.n_sites,
            R=R_MEMBRANE,
            tau=TAU_E,
            V_rest=V_REST_E,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau_ref=TAU_REF_E,
            V_initializer=braintools.init.Constant(V_REST_E),
        )
        self.inh = brainpy.state.LIFRef(
            self.n_sites,
            R=R_MEMBRANE,
            tau=TAU_I,
            V_rest=V_REST_I,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau_ref=TAU_REF_I,
            V_initializer=braintools.init.Constant(V_REST_I),
        )

        # Excitation is nearest-neighbor; inhibition reaches farther, producing
        # a compact propagating front followed by a refractory/inhibitory wake.
        self.ee_conn = make_local_csr(
            config.nx, config.ny, 1.5, EE_WEIGHT, include_self=False
        )
        self.ei_conn = make_local_csr(
            config.nx, config.ny, 1.5, EI_WEIGHT, include_self=True
        )
        self.ie_conn = make_local_csr(
            config.nx, config.ny, 2.25, IE_WEIGHT, include_self=True
        )

        self.ee_syn = brainpy.state.Expon(self.n_sites, tau=TAU_SYN_EXC)
        self.ei_syn = brainpy.state.Expon(self.n_sites, tau=TAU_SYN_EXC)
        self.ie_syn = brainpy.state.Expon(self.n_sites, tau=TAU_SYN_INH)
        self.ee_out = brainpy.state.COBA(E=E_EXC)
        self.ei_out = brainpy.state.COBA(E=E_EXC)
        self.ie_out = brainpy.state.COBA(E=E_INH)
        self.exc.add_current_input("ee", self.ee_out)
        self.exc.add_current_input("ie", self.ie_out)
        self.inh.add_current_input("ei", self.ei_out)

    def update(self, t, lesion_radius, inhibition_gain):
        """Advance one step for one lesion/inhibition condition."""
        with brainstate.environ.context(t=t):
            active = self.obstacle_distance > lesion_radius
            previous_exc = (self.exc.get_spike() != 0.0) & active
            previous_inh = (self.inh.get_spike() != 0.0) & active

            ee_drive = brainevent.BinaryArray(previous_exc) @ self.ee_conn
            ei_drive = brainevent.BinaryArray(previous_exc) @ self.ei_conn
            ie_drive = brainevent.BinaryArray(previous_inh) @ self.ie_conn
            self.ee_out.bind_cond(self.ee_syn(ee_drive))
            self.ei_out.bind_cond(self.ei_syn(ei_drive))
            self.ie_out.bind_cond(self.ie_syn(ie_drive * inhibition_gain))

            spark = self.spark_mask & active & (t < SPARK_DURATION)
            silent = ~active
            exc_current = (
                BACKGROUND_E
                + spark.astype(jnp.float32) * SPARK_CURRENT
                + silent.astype(jnp.float32) * LESION_CLAMP
            )
            inh_current = (
                BACKGROUND_I
                + silent.astype(jnp.float32) * LESION_CLAMP
            )
            exc_spike = (self.exc(exc_current) != 0.0) & active
            inh_spike = (self.inh(inh_current) != 0.0) & active
            return exc_spike, inh_spike


def condition_grid(config: ExperimentConfig):
    radii = jnp.asarray(config.lesion_radii_mm, dtype=jnp.float32) * u.mm
    gains = jnp.asarray(config.inhibition_gains, dtype=jnp.float32)
    radius_grid, gain_grid = u.math.meshgrid(radii, gains, indexing="ij")
    return radius_grid.flatten(), gain_grid.flatten()


def simulate_sweep(config: ExperimentConfig):
    """Simulate all radius/gain conditions with mapped independent State."""
    with brainstate.environ.context(dt=config.dt, precision=32):
        sheet = CorticalSheet(config)
        radii, inhibition_gains = condition_grid(config)
        brainstate.nn.vmap_init_all_states(sheet, axis_size=radii.shape[0])

        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = vmap2(
            sheet.update,
            in_axes=(None, 0, 0),
            out_axes=(0, 0),
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )
        times = u.math.arange(0.0 * u.ms, config.duration, config.dt)

        @brainstate.transform.jit
        def run():
            return brainstate.transform.for_loop(
                lambda t: mapped_step(t, radii, inhibition_gains),
                times,
            )

        exc_spikes, inh_spikes = run()
    return sheet, times, radii, inhibition_gains, exc_spikes, inh_spikes


def classify_outcome(
    radius_mm: float,
    reach_fraction: float,
    upper_fraction: float,
    lower_fraction: float,
    *,
    reach_threshold: float = 0.12,
    route_threshold: float = 0.15,
) -> int:
    """Classify propagation from downstream reach and bypass-arm activity."""
    if reach_fraction < reach_threshold:
        return DIES
    if radius_mm < 0.5:
        return CROSSES
    upper = upper_fraction >= route_threshold
    lower = lower_fraction >= route_threshold
    if upper and lower:
        return SPLITS
    return BENDS


def summarize_conditions(
    config: ExperimentConfig,
    radii,
    gains,
    exc_spikes,
):
    """Compute host-side propagation metrics and categorical outcomes."""
    spikes = np.asarray(exc_spikes, dtype=bool)  # [time, condition, site]
    radius_mm = np.asarray(radii.to_decimal(u.mm))
    gain_values = np.asarray(gains)
    x, y = sheet_coordinates(config)
    x_mm = np.asarray(x.to_decimal(u.mm))
    y_mm = np.asarray(y.to_decimal(u.mm))
    cx = float(config.obstacle_x.to_decimal(u.mm))
    cy = float(config.obstacle_y.to_decimal(u.mm))
    width = float((config.nx - 1) * config.spacing.to_decimal(u.mm))

    ever_active = spikes.any(axis=0)
    summaries: list[dict[str, float | int | str]] = []
    for condition, (radius, gain) in enumerate(zip(radius_mm, gain_values)):
        right = x_mm >= width - 3.0
        # A narrow transect through the obstacle center records which physical
        # corridor actually carries the front past the disk.
        route_x = np.abs(x_mm - cx) <= 1.0
        upper = route_x & (y_mm > cy + radius)
        lower = route_x & (y_mm < cy - radius)

        reach = float(ever_active[condition, right].mean())
        upper_fraction = float(ever_active[condition, upper].mean()) if upper.any() else 0.0
        lower_fraction = float(ever_active[condition, lower].mean()) if lower.any() else 0.0
        outcome = classify_outcome(radius, reach, upper_fraction, lower_fraction)
        active_sites = np.flatnonzero(ever_active[condition])
        max_x = float(x_mm[active_sites].max()) if active_sites.size else 0.0
        summaries.append(
            {
                "condition": condition,
                "radius_mm": float(radius),
                "inhibition_gain": float(gain),
                "reach_fraction": reach,
                "upper_route_fraction": upper_fraction,
                "lower_route_fraction": lower_fraction,
                "max_x_mm": max_x,
                "outcome_code": outcome,
                "outcome": OUTCOME_NAMES[outcome],
            }
        )
    return summaries


def write_summary(output_dir: Path, summaries: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "outcomes.csv"
    with csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return csv_path


def _smoothed_activity(spikes: np.ndarray, window_steps: int) -> np.ndarray:
    cumulative = np.concatenate(
        [np.zeros((1, spikes.shape[1]), dtype=np.int32), np.cumsum(spikes, axis=0)],
        axis=0,
    )
    starts = np.maximum(np.arange(spikes.shape[0]) + 1 - window_steps, 0)
    return cumulative[1:] - cumulative[starts]


def _story_frame_steps(
    config: ExperimentConfig,
    times_ms: np.ndarray,
    baseline_spikes: np.ndarray,
) -> np.ndarray:
    """Choose frames when the unobstructed front visits five x positions."""
    x, _ = sheet_coordinates(config)
    x_mm = np.asarray(x.to_decimal(u.mm))
    activity = _smoothed_activity(
        baseline_spikes,
        max(1, int(round(DISPLAY_WINDOW_MS / config.dt.to_decimal(u.ms)))),
    )
    targets = np.linspace(3.0, (config.nx - 1) - 3.0, 5)
    steps = []
    floor = 0
    for target in targets:
        nearest_x = np.unique(x_mm)[np.abs(np.unique(x_mm) - target).argmin()]
        band = np.isclose(x_mm, nearest_x)
        score = activity[:, band].sum(axis=1)
        arrivals = np.flatnonzero((score > 0) & (np.arange(score.size) >= floor))
        step = int(arrivals[0]) if arrivals.size else int(score.argmax())
        steps.append(step)
        floor = min(step + 1, score.size - 1)
    return np.asarray(steps, dtype=int)


def plot_wave_storyboard(
    output_dir: Path,
    config: ExperimentConfig,
    times,
    radii,
    gains,
    exc_spikes,
    summaries: list[dict],
):
    spikes = np.asarray(exc_spikes, dtype=bool)
    radius_mm = np.asarray(radii.to_decimal(u.mm))
    gain_values = np.asarray(gains)
    times_ms = np.asarray(times.to_decimal(u.ms))

    viable_gains = []
    for gain in config.inhibition_gains:
        at_gain = [item for item in summaries if np.isclose(item["inhibition_gain"], gain)]
        intact_crosses = any(
            item["radius_mm"] == 0.0 and item["outcome"] == "crosses" for item in at_gain
        )
        obstacle_survives = any(
            item["radius_mm"] > 0.0 and item["outcome"] in {"bends", "splits"}
            for item in at_gain
        )
        if intact_crosses and obstacle_survives:
            viable_gains.append(gain)
    target_gain = max(viable_gains) if viable_gains else config.inhibition_gains[0]
    baseline = int(np.flatnonzero((radius_mm == 0.0) & np.isclose(gain_values, target_gain))[0])
    obstacle_candidates = [
        item
        for item in summaries
        if np.isclose(item["inhibition_gain"], target_gain)
        and item["radius_mm"] > 0.0
        and item["outcome"] == "bends"
    ]
    if not obstacle_candidates:
        obstacle_candidates = [
            item
            for item in summaries
            if np.isclose(item["inhibition_gain"], target_gain)
            and item["radius_mm"] > 0.0
            and item["outcome"] == "splits"
        ]
    if not obstacle_candidates:
        obstacle_candidates = [
            item
            for item in summaries
            if np.isclose(item["inhibition_gain"], target_gain) and item["radius_mm"] > 0.0
        ]
    obstacle = int(obstacle_candidates[-1]["condition"])

    frame_steps = _story_frame_steps(config, times_ms, spikes[:, baseline])
    window_steps = max(
        1, int(round(DISPLAY_WINDOW_MS / config.dt.to_decimal(u.ms)))
    )
    baseline_activity = _smoothed_activity(spikes[:, baseline], window_steps)
    obstacle_activity = _smoothed_activity(spikes[:, obstacle], window_steps)
    vmax = max(
        1,
        np.percentile(
            np.concatenate(
                [baseline_activity[frame_steps].ravel(), obstacle_activity[frame_steps].ravel()]
            ),
            99,
        ),
    )

    fig, axes = plt.subplots(
        2,
        len(frame_steps),
        figsize=(14.5, 5.6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    extent = (-0.5, config.nx - 0.5, -0.5, config.ny - 0.5)
    rows = (
        (baseline, baseline_activity, f"intact sheet: inhibition={target_gain:.2f}"),
        (
            obstacle,
            obstacle_activity,
            f"silent patch: r={radius_mm[obstacle]:.1f} mm, {summaries[obstacle]['outcome']}",
        ),
    )
    image = None
    for row, (condition, activity, label) in enumerate(rows):
        for col, step in enumerate(frame_steps):
            ax = axes[row, col]
            image = ax.imshow(
                activity[step].reshape(config.ny, config.nx),
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            radius_cells = radius_mm[condition] / config.spacing.to_decimal(u.mm)
            if radius_cells > 0.0:
                circle = plt.Circle(
                    (
                        config.obstacle_x.to_decimal(u.mm) / config.spacing.to_decimal(u.mm),
                        config.obstacle_y.to_decimal(u.mm) / config.spacing.to_decimal(u.mm),
                    ),
                    radius_cells,
                    facecolor="#d9dde2",
                    edgecolor="#1f2933",
                    linewidth=1.0,
                    zorder=3,
                )
                ax.add_patch(circle)
            if row == 0:
                ax.set_title(f"{times_ms[step]:.1f} ms", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{label}\ny (mm)")
            if row == 1:
                ax.set_xlabel("x (mm)")
            ax.set_xticks(np.linspace(0, config.nx - 1, 5, dtype=int))
            ax.set_yticks(np.linspace(0, config.ny - 1, 5, dtype=int))

    fig.suptitle(
        f"Excitatory spikes in the preceding {DISPLAY_WINDOW_MS:g} ms",
        fontsize=14,
    )
    cbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("spikes per site")
    path = output_dir / "wave_storyboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, obstacle


def plot_phase_map(output_dir: Path, config: ExperimentConfig, summaries: list[dict]):
    nr = len(config.lesion_radii_mm)
    ng = len(config.inhibition_gains)
    phase = np.asarray([item["outcome_code"] for item in summaries]).reshape(nr, ng)
    reach = np.asarray([item["reach_fraction"] for item in summaries]).reshape(nr, ng)
    cmap = ListedColormap(["#2f7d6d", "#e6a23c", "#b85c8a", "#343a40"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    image = ax.imshow(phase, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    for row in range(nr):
        for col in range(ng):
            code = phase[row, col]
            text_color = "white" if code in (CROSSES, DIES) else "#1d242b"
            ax.text(
                col,
                row,
                f"{OUTCOME_NAMES[code]}\n{reach[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    ax.set_xticks(np.arange(ng), [f"{gain:.2f}" for gain in config.inhibition_gains])
    ax.set_yticks(np.arange(nr), [f"{radius:.1f}" for radius in config.lesion_radii_mm])
    ax.set_xlabel("inhibitory conductance gain")
    ax.set_ylabel("silent-patch radius (mm)")
    ax.set_title("Wave outcome phase map\nnumber in each cell = right-edge reach fraction")
    cbar = fig.colorbar(image, ax=ax, ticks=range(4), fraction=0.045, pad=0.04)
    cbar.ax.set_yticklabels(OUTCOME_NAMES)
    fig.tight_layout()
    path = output_dir / "phase_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_metrics(output_dir: Path, config: ExperimentConfig, summaries: list[dict]):
    nr = len(config.lesion_radii_mm)
    ng = len(config.inhibition_gains)
    values = lambda key: np.asarray([item[key] for item in summaries]).reshape(nr, ng)
    path = output_dir / "phase_metrics.npz"
    np.savez_compressed(
        path,
        lesion_radii_mm=np.asarray(config.lesion_radii_mm),
        inhibition_gains=np.asarray(config.inhibition_gains),
        outcome_code=values("outcome_code"),
        reach_fraction=values("reach_fraction"),
        upper_route_fraction=values("upper_route_fraction"),
        lower_route_fraction=values("lower_route_fraction"),
        max_x_mm=values("max_x_mm"),
    )
    return path


def run_experiment(config: ExperimentConfig, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet, times, radii, gains, exc_spikes, inh_spikes = simulate_sweep(config)
    summaries = summarize_conditions(config, radii, gains, exc_spikes)
    csv_path = write_summary(output_dir, summaries)
    metrics_path = save_metrics(output_dir, config, summaries)
    phase_path = plot_phase_map(output_dir, config, summaries)
    story_path, obstacle_condition = plot_wave_storyboard(
        output_dir, config, times, radii, gains, exc_spikes, summaries
    )

    counts = {name: 0 for name in OUTCOME_NAMES}
    for item in summaries:
        counts[item["outcome"]] += 1
    example = summaries[obstacle_condition]
    print(
        f"simulated {len(summaries)} conditions, {sheet.n_sites} E + "
        f"{sheet.n_sites} I neurons, {times.shape[0]} steps"
    )
    print("outcomes: " + ", ".join(f"{name}={counts[name]}" for name in OUTCOME_NAMES))
    print(
        "storyboard obstacle: "
        f"r={example['radius_mm']:.1f} mm, inhibition={example['inhibition_gain']:.2f}, "
        f"outcome={example['outcome']}"
    )
    print(f"wrote {story_path}, {phase_path}, {csv_path}, and {metrics_path}")
    return summaries


def quick_config() -> ExperimentConfig:
    """Smaller condition bank for a fast smoke run."""
    return replace(
        ExperimentConfig(),
        nx=24,
        ny=16,
        duration=90.0 * u.ms,
        obstacle_x=11.0 * u.mm,
        obstacle_y=7.0 * u.mm,
        lesion_radii_mm=(0.0, 2.5, 4.5),
        inhibition_gains=(0.75, 1.05, 1.45),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="run a 3 x 3 smoke sweep")
    args = parser.parse_args()
    run_experiment(quick_config() if args.quick else ExperimentConfig(), args.output)


if __name__ == "__main__":
    main()
