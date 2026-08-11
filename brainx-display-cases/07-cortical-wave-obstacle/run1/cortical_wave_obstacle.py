"""A sparse E/I cortical wave interacting with a circular silent patch.

The simulation uses BrainPy-State point neurons and synapses, BrainEvent CSR
communication, BrainState state-aware mapping and time evolution, and
BrainUnit quantities. Running this file writes a snapshot/phase-map figure and
the numerical phase-map summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib
import numpy as np
from brainstate.util import filter as state_filter
from braintools.conn import Grid2d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch


# Sheet geometry and integration.
NX, NY = 28, 18
DX = 0.25 * u.mm
DT = 0.2 * u.ms
DURATION = 90.0 * u.ms

# LIF and conductance parameters.
V_REST = -65.0 * u.mV
V_RESET = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
MEMBRANE_RESISTANCE = 100.0 * u.Mohm
TAU_E = 12.0 * u.ms
TAU_I = 8.0 * u.ms
TAU_REF = 2.0 * u.ms
E_EXC = 0.0 * u.mV
E_INH = -78.0 * u.mV
TAU_SYN_E = 3.0 * u.ms
TAU_SYN_I = 5.0 * u.ms

# Each local event is multiplied by one of these conductances.
W_EE = 1.75 * u.nS
W_EI = 1.20 * u.nS
W_IE = 2.00 * u.nS
W_II = 0.45 * u.nS

# The resting drive stays subthreshold. A brief extra current at the left edge
# initiates a wave without maintaining it.
BASE_E = 0.130 * u.nA
BASE_I = 0.110 * u.nA
SPARK = 0.65 * u.nA
SPARK_START = 15.0 * u.ms
SPARK_STOP = 19.0 * u.ms

PATCH_RADII = jnp.array([0.35, 0.55, 0.75, 0.95, 1.15]) * u.mm
INHIBITION_SCALES = jnp.array([0.40, 0.48, 0.56, 0.64, 0.72])


def sheet_coordinates():
    """Return flattened physical coordinates in row-major grid order."""
    x = u.math.arange(NX) * DX
    y = u.math.arange(NY) * DX
    x_grid, y_grid = u.math.meshgrid(x, y, indexing="xy")
    return x_grid.reshape(-1), y_grid.reshape(-1)


def local_csr():
    """Build a nonperiodic eight-neighbor sheet and export it to CSR."""
    result = Grid2d(
        connectivity="moore",
        weight=braintools.init.Constant(1.0),
        periodic=False,
    )(pre_size=(NY, NX), post_size=(NY, NX))
    indptr, indices, order = brainevent.coo2csr(
        result.pre_indices,
        result.post_indices,
        shape=result.shape,
    )
    return brainevent.CSR(
        (result.weights[order], indices, indptr),
        shape=result.shape,
    )


class SparseCOBAProjection(brainstate.nn.Module):
    """One explicit BrainEvent communication path into a BrainPy COBA input."""

    def __init__(self, connection, post, tau, reversal, label):
        super().__init__()
        self.connection = connection
        self.syn = brainpy.state.Expon(NX * NY, tau=tau)
        self.out = brainpy.state.COBA(E=reversal)
        post.add_current_input(label, self.out)

    def update(self, spikes, weight):
        event_input = brainevent.BinaryArray(spikes) @ self.connection
        self.out.bind_cond(self.syn(event_input * weight))


class CorticalSheet(brainstate.nn.Module):
    """Paired excitatory and inhibitory LIF neurons on a 2D sheet."""

    def __init__(self):
        super().__init__()
        self.x, self.y = sheet_coordinates()
        self.patch_x = (NX - 1) * DX * 0.58
        self.patch_y = (NY - 1) * DX * 0.50
        self.patch_distance = u.math.sqrt(
            (self.x - self.patch_x) ** 2 + (self.y - self.patch_y) ** 2
        )
        self.spark_mask = (self.x <= DX) & (
            u.math.abs(self.y - self.patch_y) <= 1.55 * u.mm
        )

        initializer = braintools.init.Constant(V_REST)
        self.exc = brainpy.state.LIFRef(
            NX * NY,
            R=MEMBRANE_RESISTANCE,
            tau=TAU_E,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau_ref=TAU_REF,
            V_initializer=initializer,
        )
        self.inh = brainpy.state.LIFRef(
            NX * NY,
            R=MEMBRANE_RESISTANCE,
            tau=TAU_I,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau_ref=TAU_REF,
            V_initializer=initializer,
        )

        connection = local_csr()
        self.ee = SparseCOBAProjection(
            connection, self.exc, TAU_SYN_E, E_EXC, "ee"
        )
        self.ei = SparseCOBAProjection(
            connection, self.inh, TAU_SYN_E, E_EXC, "ei"
        )
        self.ie = SparseCOBAProjection(
            connection, self.exc, TAU_SYN_I, E_INH, "ie"
        )
        self.ii = SparseCOBAProjection(
            connection, self.inh, TAU_SYN_I, E_INH, "ii"
        )

        # A small fixed spatial bias breaks exact top/bottom symmetry without
        # adding condition-specific noise to the phase sweep.
        phase = self.x / (0.73 * u.mm) + self.y / (0.51 * u.mm)
        self.tonic_bias = 0.010 * u.nA * jnp.sin(phase)

    def update(self, t, patch_radius, inhibition_scale):
        with brainstate.environ.context(t=t):
            active = self.patch_distance >= patch_radius
            exc_spikes = (self.exc.get_spike() != 0.0) & active
            inh_spikes = (self.inh.get_spike() != 0.0) & active

            self.ee(exc_spikes, W_EE)
            self.ei(exc_spikes, W_EI)
            self.ie(inh_spikes, W_IE * inhibition_scale)
            self.ii(inh_spikes, W_II * inhibition_scale)

            spark_on = (t >= SPARK_START) & (t < SPARK_STOP)
            exc_current = active * (BASE_E + self.tonic_bias)
            exc_current = exc_current + active * self.spark_mask * spark_on * SPARK
            inh_current = active * BASE_I

            next_exc = (self.exc(exc_current) != 0.0) & active
            self.inh(inh_current)
            return next_exc


def run_sweep():
    """Run the intact case and every patch/inhibition pair in one state batch."""
    radius_grid, inhibition_grid = u.math.meshgrid(
        PATCH_RADII, INHIBITION_SCALES, indexing="ij"
    )
    sweep_radii = radius_grid.reshape(-1)
    sweep_inhibition = inhibition_grid.reshape(-1)

    # A negative radius leaves every neuron active and provides the intact
    # comparison under the same integration and transform path.
    radii = u.math.concatenate((jnp.array([-1.0]) * u.mm, sweep_radii))
    inhibition = jnp.concatenate((jnp.array([0.40]), sweep_inhibition))

    with brainstate.environ.context(dt=DT):
        sheet = CorticalSheet()
        brainstate.nn.vmap_init_all_states(sheet, axis_size=radii.shape[0])
        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = brainstate.transform.vmap2(
            sheet.update,
            in_axes=(None, 0, 0),
            out_axes=0,
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )
        times = u.math.arange(0.0 * u.ms, DURATION, DT)

        @brainstate.transform.jit
        def simulate():
            return brainstate.transform.for_loop(
                lambda t: mapped_step(t, radii, inhibition), times
            )

        spikes = simulate()

    return sheet, times, spikes, radius_grid, inhibition_grid


def classify_outcomes(sheet, times, sweep_spikes, radius_grid):
    """Classify each condition from downstream and two-flank spike counts."""
    spikes = np.asarray(sweep_spikes, dtype=bool)
    times_ms = np.asarray(times.to_decimal(u.ms))
    x_mm = np.asarray(sheet.x.to_decimal(u.mm))
    y_mm = np.asarray(sheet.y.to_decimal(u.mm))
    patch_x = float(sheet.patch_x.to_decimal(u.mm))
    patch_y = float(sheet.patch_y.to_decimal(u.mm))
    radii_mm = np.asarray(radius_grid.to_decimal(u.mm)).reshape(-1)

    late = times_ms >= 30.0
    labels = []
    statistics = []
    for condition, radius in enumerate(radii_mm):
        activity = spikes[late, condition]
        right_mask = x_mm >= (NX - 4) * DX.to_decimal(u.mm)
        near_patch_x = np.abs(x_mm - patch_x) <= radius + 0.55
        upper_mask = near_patch_x & (y_mm >= patch_y + radius)
        lower_mask = near_patch_x & (y_mm <= patch_y - radius)

        right_count = int(activity[:, right_mask].sum())
        upper_count = int(activity[:, upper_mask].sum())
        lower_count = int(activity[:, lower_mask].sum())
        weaker_flank = min(upper_count, lower_count)
        stronger_flank = max(upper_count, lower_count)

        if right_count < 4:
            label = "dies"
        elif weaker_flank >= 3 and weaker_flank / max(stronger_flank, 1) >= 0.32:
            label = "splits"
        else:
            label = "bends"
        labels.append(label)
        statistics.append((right_count, upper_count, lower_count))

    shape = radius_grid.shape
    return np.asarray(labels, dtype=object).reshape(shape), np.asarray(statistics).reshape(
        shape + (3,)
    )


def activity_frame(spikes, times, target_time, window=4.0 * u.ms):
    """Count spikes in the interval ending at target_time for one condition."""
    t_ms = np.asarray(times.to_decimal(u.ms))
    stop = int(np.searchsorted(t_ms, target_time.to_decimal(u.ms), side="right"))
    width = max(1, int(round(window / DT)))
    start = max(0, stop - width)
    return np.asarray(spikes[start:stop]).sum(axis=0).reshape(NY, NX)


def plot_results(
    output_path, sheet, times, spikes, radius_grid, inhibition_grid, outcomes
):
    """Render intact/lesioned wave snapshots and the outcome phase map."""
    spike_array = np.asarray(spikes, dtype=bool)
    snapshot_times = (17.0, 34.0, 48.0, 62.0) * u.ms
    ref_radius_index = 2
    ref_inhibition_index = 1
    n_inhibition = inhibition_grid.shape[1]
    reference_condition = 1 + ref_radius_index * n_inhibition + ref_inhibition_index

    intact_frames = [activity_frame(spike_array[:, 0], times, t) for t in snapshot_times]
    lesion_frames = [
        activity_frame(spike_array[:, reference_condition], times, t)
        for t in snapshot_times
    ]
    vmax = max(1.0, np.percentile(np.stack(intact_frames + lesion_frames), 99))

    fig = plt.figure(figsize=(13.2, 8.8), facecolor="#f7f7f4")
    grid = fig.add_gridspec(
        3, 5, width_ratios=(1, 1, 1, 1, 1.18), height_ratios=(1, 1, 1.04),
        wspace=0.08, hspace=0.32
    )
    heat_cmap = ListedColormap(
        ["#f7f7f4", "#c9ded8", "#4e9b8f", "#164f4a", "#0b2726"]
    )
    extent = (
        -0.5 * DX.to_decimal(u.mm),
        (NX - 0.5) * DX.to_decimal(u.mm),
        -0.5 * DX.to_decimal(u.mm),
        (NY - 0.5) * DX.to_decimal(u.mm),
    )

    for row, (frames, title) in enumerate(
        ((intact_frames, "Intact sheet"), (lesion_frames, "Silent circular patch"))
    ):
        for column, (frame, time) in enumerate(zip(frames, snapshot_times)):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(
                frame,
                origin="lower",
                extent=extent,
                cmap=heat_cmap,
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            if row == 1:
                radius = PATCH_RADII[ref_radius_index].to_decimal(u.mm)
                ax.add_patch(
                    Circle(
                        (
                            sheet.patch_x.to_decimal(u.mm),
                            sheet.patch_y.to_decimal(u.mm),
                        ),
                        radius,
                        facecolor="#f7f7f4",
                        edgecolor="#202522",
                        linewidth=1.1,
                        hatch="////",
                    )
                )
            ax.set_title(f"{time.to_decimal(u.ms):.0f} ms", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d1d3cf")
            if column == 0:
                ax.set_ylabel(title, fontsize=11, labelpad=8)

    colorbar_ax = fig.add_subplot(grid[:2, 4])
    colorbar_ax.axis("off")
    colorbar = fig.colorbar(image, ax=colorbar_ax, fraction=0.12, pad=0.15)
    colorbar.set_label(f"E spikes / {4.0:.0f} ms", fontsize=10)
    colorbar.outline.set_edgecolor("#b5b8b3")
    colorbar_ax.text(
        0.08,
        0.88,
        "Reference lesion",
        transform=colorbar_ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#202522",
    )
    colorbar_ax.text(
        0.08,
        0.80,
        f"radius  {PATCH_RADII[ref_radius_index].to_decimal(u.mm):.2f} mm\n"
        f"inhibition  x{INHIBITION_SCALES[ref_inhibition_index]:.2f}",
        transform=colorbar_ax.transAxes,
        fontsize=10,
        linespacing=1.6,
        color="#4e5551",
    )

    outcome_codes = np.vectorize({"dies": 0, "bends": 1, "splits": 2}.get)(outcomes)
    phase_ax = fig.add_subplot(grid[2, 1:4])
    phase_cmap = ListedColormap(["#343a3a", "#d49a3a", "#2a8278"])
    phase_ax.imshow(outcome_codes, origin="lower", cmap=phase_cmap, vmin=-0.5, vmax=2.5)
    phase_ax.set_xticks(np.arange(INHIBITION_SCALES.size))
    phase_ax.set_xticklabels([f"{value:.2f}" for value in INHIBITION_SCALES])
    phase_ax.set_yticks(np.arange(PATCH_RADII.size))
    phase_ax.set_yticklabels(
        [f"{value:.2f}" for value in PATCH_RADII.to_decimal(u.mm)]
    )
    phase_ax.set_xlabel("Inhibitory conductance scale")
    phase_ax.set_ylabel("Patch radius (mm)")
    phase_ax.set_title("Outcome phase map", fontsize=12, pad=10)
    for row in range(outcomes.shape[0]):
        for column in range(outcomes.shape[1]):
            phase_ax.text(
                column,
                row,
                outcomes[row, column],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if outcome_codes[row, column] in (0, 2) else "#242622",
            )
    phase_ax.set_xticks(np.arange(-0.5, outcomes.shape[1], 1), minor=True)
    phase_ax.set_yticks(np.arange(-0.5, outcomes.shape[0], 1), minor=True)
    phase_ax.grid(which="minor", color="#f7f7f4", linewidth=1.6)
    phase_ax.tick_params(which="minor", bottom=False, left=False)
    phase_ax.legend(
        handles=[
            Patch(facecolor="#343a3a", label="dies"),
            Patch(facecolor="#d49a3a", label="bends"),
            Patch(facecolor="#2a8278", label="splits"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        "A cortical wave meets an obstacle",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#202522",
    )
    fig.text(
        0.07,
        0.945,
        "Sparse E/I conductance dynamics; activity is counted in trailing 4 ms windows",
        ha="left",
        fontsize=10.5,
        color="#5c625e",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_phase_csv(path, radius_grid, inhibition_grid, outcomes, statistics):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "patch_radius_mm",
                "inhibition_scale",
                "outcome",
                "right_spikes",
                "upper_flank_spikes",
                "lower_flank_spikes",
            ]
        )
        radii = radius_grid.to_decimal(u.mm)
        for row in range(outcomes.shape[0]):
            for column in range(outcomes.shape[1]):
                writer.writerow(
                    [
                        f"{radii[row, column]:.3f}",
                        f"{inhibition_grid[row, column]:.3f}",
                        outcomes[row, column],
                        *statistics[row, column],
                    ]
                )


def validate_results(sheet, spikes, radius_grid, outcomes):
    """Check the scientific invariants that define this demonstration."""
    spike_array = np.asarray(spikes, dtype=bool)
    expected_conditions = 1 + PATCH_RADII.size * INHIBITION_SCALES.size
    if spike_array.shape != (int(round(DURATION / DT)), expected_conditions, NX * NY):
        raise RuntimeError(f"Unexpected spike array shape: {spike_array.shape}")

    x_mm = np.asarray(sheet.x.to_decimal(u.mm))
    intact_reached_right = spike_array[:, 0, x_mm >= (NX - 4) * DX.to_decimal(u.mm)].any()
    if not intact_reached_right:
        raise RuntimeError("The intact wave did not reach the right-edge readout zone.")

    distances_mm = np.asarray(sheet.patch_distance.to_decimal(u.mm))
    for condition, radius in enumerate(radius_grid.to_decimal(u.mm).reshape(-1), start=1):
        if spike_array[:, condition, distances_mm < radius].any():
            raise RuntimeError(f"Silent-patch spike detected in condition {condition}.")

    observed = set(outcomes.reshape(-1))
    required = {"bends", "splits", "dies"}
    if not required.issubset(observed):
        raise RuntimeError(f"Phase map is missing outcomes: {sorted(required - observed)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for the PNG phase map and CSV summary.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    brainstate.random.seed(17)
    sheet, times, spikes, radius_grid, inhibition_grid = run_sweep()
    outcomes, statistics = classify_outcomes(
        sheet, times, spikes[:, 1:], radius_grid
    )
    validate_results(sheet, spikes, radius_grid, outcomes)

    figure_path = args.output_dir / "cortical_wave_obstacle.png"
    csv_path = args.output_dir / "phase_map.csv"
    plot_results(
        figure_path,
        sheet,
        times,
        spikes,
        radius_grid,
        inhibition_grid,
        outcomes,
    )
    write_phase_csv(
        csv_path, radius_grid, inhibition_grid, outcomes, statistics
    )

    unique, counts = np.unique(outcomes, return_counts=True)
    summary = ", ".join(f"{label}: {count}" for label, count in zip(unique, counts))
    print(f"Simulated {spikes.shape[1]} independent conditions for {DURATION}.")
    print(f"Outcomes: {summary}")
    print(f"Figure: {figure_path}")
    print(f"Phase data: {csv_path}")


if __name__ == "__main__":
    main()
