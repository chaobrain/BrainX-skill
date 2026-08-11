"""A unit-aware cortical E/I wave encountering a silent circular patch.

The network is a phenomenological two-dimensional sheet. Each lattice site has
one excitatory and one inhibitory LIF neuron. Local spikes travel through an
explicit BrainEvent CSR graph into BrainPy-State conductance synapses. A short
current pulse at the left edge launches the wave.

The complete condition transition is vectorized with BrainState ``vmap2``;
BrainState ``for_loop`` advances all lesion-radius/inhibition conditions through
time. Run this file to write a wave/phase-map figure and the defining metrics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import brainevent
import brainpy
import brainstate
from brainstate.util import filter as state_filter
import braintools
from braintools.input import Constant
import brainunit as u
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle
import numpy as np


# Sheet and integration geometry.
NX, NY = 32, 24
N_SITE = NX * NY
DX = 0.08 * u.mm
DT = 0.2 * u.ms
DURATION = 100.0 * u.ms

# LIF dynamics.
V_REST = -65.0 * u.mV
V_RESET = -62.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
TAU_MEMBRANE = 15.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
MEMBRANE_RESISTANCE = 100.0 * u.Mohm
BASE_CURRENT_E = 0.11 * u.nA
BASE_CURRENT_I = 0.08 * u.nA

# Local conductance paths. Inhibition strength scales only I -> E.
W_EE = 3.00 * u.nS
W_EI = 1.80 * u.nS
W_IE = 4.80 * u.nS
TAU_EXCITATORY = 4.0 * u.ms
TAU_INHIBITORY = 7.0 * u.ms
E_EXCITATORY = 0.0 * u.mV
E_INHIBITORY = -80.0 * u.mV

# The pulse is generated once and supplied time-major to ``for_loop``.
SPARK_ONSET = 5.0 * u.ms
SPARK_DURATION = 3.0 * u.ms
SPARK_CURRENT = 0.62 * u.nA

# Sweep axes. Radius zero is a matched control at every inhibition strength.
LESION_RADII = jnp.array([0.0, 0.16, 0.28, 0.40, 0.52]) * u.mm
INHIBITION_STRENGTHS = jnp.array([0.90, 1.00, 1.10, 1.20, 1.30])


def sheet_coordinates():
    """Return flattened physical coordinates and the circular-patch center."""
    x = u.math.arange(NX) * DX
    y = u.math.arange(NY) * DX
    x_grid, y_grid = u.math.meshgrid(x, y, indexing="xy")
    center = (0.56 * x[-1], 0.50 * y[-1])
    return x_grid.reshape(-1), y_grid.reshape(-1), center


def local_event_connectivity():
    """Create one nonperiodic eight-neighbor topology and export it to CSR."""
    pattern = braintools.conn.Grid2d(
        connectivity="moore",
        weight=braintools.init.Constant(1.0),
        periodic=False,
    )
    result = pattern(pre_size=(NY, NX), post_size=(NY, NX))
    indptr, indices, order = brainevent.coo2csr(
        result.pre_indices,
        result.post_indices,
        shape=result.shape,
    )
    return brainevent.CSR(
        (result.weights[order], indices, indptr),
        shape=result.shape,
    )


class CorticalSheet(brainstate.nn.Module):
    """Excitatory and inhibitory LIF sheets with local event communication."""

    def __init__(self, connectivity, edge_mask):
        super().__init__()
        initializer = braintools.init.Constant(V_REST)
        neuron_parameters = dict(
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            R=MEMBRANE_RESISTANCE,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_initializer=initializer,
        )
        self.excitatory = brainpy.state.LIFRef(N_SITE, **neuron_parameters)
        self.inhibitory = brainpy.state.LIFRef(N_SITE, **neuron_parameters)

        self.ee_synapse = brainpy.state.Expon(N_SITE, tau=TAU_EXCITATORY)
        self.ei_synapse = brainpy.state.Expon(N_SITE, tau=TAU_EXCITATORY)
        self.ie_synapse = brainpy.state.Expon(N_SITE, tau=TAU_INHIBITORY)

        self.ee_output = brainpy.state.COBA(E=E_EXCITATORY)
        self.ei_output = brainpy.state.COBA(E=E_EXCITATORY)
        self.ie_output = brainpy.state.COBA(E=E_INHIBITORY)
        self.excitatory.add_current_input("local_excitation", self.ee_output)
        self.excitatory.add_current_input("local_inhibition", self.ie_output)
        self.inhibitory.add_current_input("local_excitation", self.ei_output)

        self.connectivity = connectivity
        self.edge_mask = edge_mask

    def init_state(self):
        self.excitatory_spike = brainstate.ShortTermState(
            jnp.zeros(N_SITE, dtype=bool)
        )
        self.inhibitory_spike = brainstate.ShortTermState(
            jnp.zeros(N_SITE, dtype=bool)
        )

    def update(self, t, spark_current, active_mask, inhibition_strength):
        """Advance one independent condition by one simulation time step."""
        with brainstate.environ.context(t=t):
            exc_events = brainevent.BinaryArray(self.excitatory_spike.value)
            inh_events = brainevent.BinaryArray(self.inhibitory_spike.value)

            local_excitation = exc_events @ self.connectivity
            local_inhibition = inh_events @ self.connectivity
            self.ee_output.bind_cond(self.ee_synapse(local_excitation * W_EE))
            self.ei_output.bind_cond(self.ei_synapse(local_excitation * W_EI))
            self.ie_output.bind_cond(
                self.ie_synapse(local_inhibition * W_IE * inhibition_strength)
            )

            exc_current = BASE_CURRENT_E + self.edge_mask * spark_current
            exc_spike = (self.excitatory(exc_current) != 0.0) & active_mask
            inh_spike = (self.inhibitory(BASE_CURRENT_I) != 0.0) & active_mask
            self.excitatory_spike.value = exc_spike
            self.inhibitory_spike.value = inh_spike
            return exc_spike, inh_spike


def sweep_conditions(x, y, center):
    """Build the Cartesian lesion/inhibition sweep without stripping units."""
    radii = u.math.repeat(LESION_RADII, INHIBITION_STRENGTHS.shape[0])
    inhibition = jnp.tile(INHIBITION_STRENGTHS, LESION_RADII.shape[0])
    distance = u.math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    active = (radii[:, None] == 0.0 * u.mm) | (distance[None, :] > radii[:, None])
    return radii, inhibition, active


def simulate():
    """Run every sweep condition in one state-aware mapped time loop."""
    brainstate.random.seed(7)
    x, y, center = sheet_coordinates()
    radii, inhibition, active_masks = sweep_conditions(x, y, center)
    edge_mask = x <= 2.0 * DX

    with brainstate.environ.context(dt=DT, precision=32):
        spark = Constant([
            (0.0 * u.nA, SPARK_ONSET),
            (SPARK_CURRENT, SPARK_DURATION),
            (0.0 * u.nA, DURATION - SPARK_ONSET - SPARK_DURATION),
        ])()
        times = u.math.arange(0.0 * u.ms, DURATION, brainstate.environ.get_dt())
        network = CorticalSheet(local_event_connectivity(), edge_mask)
        brainstate.nn.vmap_init_all_states(network, axis_size=radii.shape[0])

        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = brainstate.transform.vmap2(
            network.update,
            in_axes=(None, None, 0, 0),
            out_axes=0,
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )

        @brainstate.transform.jit
        def run():
            def step(t, spark_current):
                return mapped_step(
                    t,
                    spark_current,
                    active_masks,
                    inhibition,
                )

            return brainstate.transform.for_loop(step, times, spark)

        excitatory_spikes, inhibitory_spikes = run()

    return {
        "times": times,
        "excitatory_spikes": excitatory_spikes,
        "inhibitory_spikes": inhibitory_spikes,
        "radii": radii,
        "inhibition": inhibition,
        "active_masks": active_masks,
        "x": x,
        "y": y,
        "center": center,
    }


def _peak_fraction(spikes, region, window_steps):
    if not np.any(region):
        return 0.0
    counts = spikes[:, region].sum(axis=1).astype(float)
    rolling = np.convolve(counts, np.ones(window_steps), mode="same")
    return float(rolling.max() / (window_steps * region.sum()))


def _arrival_ms(spikes, region, window_steps, threshold_fraction=0.006):
    if not np.any(region):
        return np.nan
    counts = spikes[:, region].sum(axis=1).astype(float)
    rolling = np.convolve(counts, np.ones(window_steps), mode="same")
    threshold = max(2.0, threshold_fraction * window_steps * region.sum())
    candidates = np.flatnonzero(rolling >= threshold)
    return float(candidates[0] * DT.to_decimal(u.ms)) if candidates.size else np.nan


def measure_outcomes(result):
    """Derive causal propagation observables and matched-control phase labels."""
    spikes = np.asarray(result["excitatory_spikes"], dtype=bool)
    x_mm = np.asarray(result["x"].to_decimal(u.mm))
    y_mm = np.asarray(result["y"].to_decimal(u.mm))
    center_x = result["center"][0].to_decimal(u.mm)
    center_y = result["center"][1].to_decimal(u.mm)
    radii_mm = np.asarray(result["radii"].to_decimal(u.mm))
    inhibition = np.asarray(result["inhibition"])
    active = np.asarray(result["active_masks"], dtype=bool)
    width = x_mm.max()
    dx_mm = DX.to_decimal(u.mm)
    window_steps = max(1, int(round(3.0 * u.ms / DT)))

    source = x_mm <= 2.0 * dx_mm
    pre_obstacle = (x_mm >= center_x - 0.42) & (x_mm <= center_x - 0.22)
    far_edge = x_mm >= width - 3.0 * dx_mm

    records = []
    for condition in range(spikes.shape[1]):
        radius = radii_mm[condition]
        condition_spikes = spikes[:, condition]
        corridor_x = np.abs(x_mm - center_x) <= radius + 1.5 * dx_mm
        upper = corridor_x & (y_mm >= center_y + max(radius * 0.65, dx_mm))
        lower = corridor_x & (y_mm <= center_y - max(radius * 0.65, dx_mm))
        wake = (
            (x_mm >= center_x + radius)
            & (x_mm <= center_x + radius + 0.32)
            & (np.abs(y_mm - center_y) <= max(0.14, radius * 0.45))
        )
        upper &= active[condition]
        lower &= active[condition]
        wake &= active[condition]

        records.append({
            "condition": condition,
            "radius_mm": float(radius),
            "inhibition_strength": float(inhibition[condition]),
            "source_arrival_ms": _arrival_ms(
                condition_spikes, source, window_steps, threshold_fraction=0.05
            ),
            "pre_arrival_ms": _arrival_ms(condition_spikes, pre_obstacle, window_steps),
            "upper_arrival_ms": _arrival_ms(condition_spikes, upper, window_steps),
            "lower_arrival_ms": _arrival_ms(condition_spikes, lower, window_steps),
            "wake_arrival_ms": _arrival_ms(condition_spikes, wake, window_steps),
            "far_arrival_ms": _arrival_ms(condition_spikes, far_edge, window_steps),
            "far_peak_fraction": _peak_fraction(condition_spikes, far_edge, window_steps),
            "upper_peak_fraction": _peak_fraction(condition_spikes, upper, window_steps),
            "lower_peak_fraction": _peak_fraction(condition_spikes, lower, window_steps),
            "wake_peak_fraction": _peak_fraction(condition_spikes, wake, window_steps),
        })

    n_inhibition = INHIBITION_STRENGTHS.shape[0]
    for record in records:
        control = records[record["condition"] % n_inhibition]
        denominator = max(control["far_peak_fraction"], 1e-9)
        transmission = record["far_peak_fraction"] / denominator
        record["transmission_ratio"] = transmission
        record["control_far_peak_fraction"] = control["far_peak_fraction"]

        source_to_target = (
            np.isfinite(record["source_arrival_ms"])
            and np.isfinite(record["pre_arrival_ms"])
            and np.isfinite(record["far_arrival_ms"])
            and record["source_arrival_ms"] < record["pre_arrival_ms"]
            < record["far_arrival_ms"]
        )
        control_propagates = (
            np.isfinite(control["source_arrival_ms"])
            and np.isfinite(control["pre_arrival_ms"])
            and np.isfinite(control["far_arrival_ms"])
            and control["far_peak_fraction"] >= 0.005
            and control["source_arrival_ms"] < control["pre_arrival_ms"]
            < control["far_arrival_ms"]
        )
        if record["radius_mm"] == 0.0:
            label = "control" if control_propagates else "control fails"
        elif not control_propagates:
            label = "control fails"
        elif not source_to_target or transmission < 0.25:
            label = "dies"
        else:
            upper = record["upper_peak_fraction"]
            lower = record["lower_peak_fraction"]
            flank_max = max(upper, lower, 1e-9)
            balance = min(upper, lower) / flank_max
            flank_arrivals = [
                value
                for value in [record["upper_arrival_ms"], record["lower_arrival_ms"]]
                if np.isfinite(value)
            ]
            route_order = (
                bool(flank_arrivals)
                and record["pre_arrival_ms"] < min(flank_arrivals)
                < record["far_arrival_ms"]
            )
            wake_delay = (
                record["wake_arrival_ms"] - min(flank_arrivals)
                if flank_arrivals and np.isfinite(record["wake_arrival_ms"])
                else np.inf
            )
            if not route_order:
                label = "dies"
            else:
                separation_time = TAU_MEMBRANE.to_decimal(u.ms)
                label = (
                    "splits"
                    if balance >= 0.55 and wake_delay >= separation_time
                    else "bends"
                )
        record["outcome"] = label
    return records


def write_metrics(records, path):
    fields = list(records[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _snapshot_steps(record, n_steps):
    flank_arrivals = [record["upper_arrival_ms"], record["lower_arrival_ms"]]
    finite_flank_arrivals = [value for value in flank_arrivals if np.isfinite(value)]
    arrivals = [
        record["source_arrival_ms"],
        record["pre_arrival_ms"],
        min(finite_flank_arrivals) if finite_flank_arrivals else np.nan,
        record["far_arrival_ms"],
    ]
    fallback = [7.0, 28.0, 48.0, 72.0]
    steps = []
    for arrival, default in zip(arrivals, fallback):
        value = default if not np.isfinite(arrival) else arrival + 1.5
        steps.append(min(n_steps - 1, int(round(value / DT.to_decimal(u.ms)))))
    return steps


def plot_summary(result, records, path):
    spikes = np.asarray(result["excitatory_spikes"], dtype=bool)
    radii_axis = np.asarray(LESION_RADII.to_decimal(u.mm))
    inhibition_axis = np.asarray(INHIBITION_STRENGTHS)
    n_inhibition = inhibition_axis.size

    demo_radius_index = min(2, radii_axis.size - 1)
    demo_inhibition_index = int(np.argmin(np.abs(inhibition_axis - 1.05)))
    demo_condition = demo_radius_index * n_inhibition + demo_inhibition_index
    demo_record = records[demo_condition]
    radius = demo_record["radius_mm"]
    steps = _snapshot_steps(demo_record, spikes.shape[0])
    accumulation = max(1, int(round(3.0 * u.ms / DT)))

    fig = plt.figure(figsize=(14, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.18))
    extent = [
        -0.5 * DX.to_decimal(u.mm),
        (NX - 0.5) * DX.to_decimal(u.mm),
        -0.5 * DX.to_decimal(u.mm),
        (NY - 0.5) * DX.to_decimal(u.mm),
    ]
    center = (
        result["center"][0].to_decimal(u.mm),
        result["center"][1].to_decimal(u.mm),
    )

    for column, step in enumerate(steps):
        axis = fig.add_subplot(grid[0, column])
        start = max(0, step - accumulation + 1)
        activity = spikes[start : step + 1, demo_condition].sum(axis=0)
        image = axis.imshow(
            activity.reshape(NY, NX),
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0,
            vmax=max(1, accumulation // 3),
            interpolation="nearest",
            aspect="equal",
        )
        axis.add_patch(Circle(center, radius, facecolor="white", edgecolor="cyan", lw=1.4))
        axis.set_title(f"{step * DT.to_decimal(u.ms):.1f} ms")
        axis.set_xlabel("x (mm)")
        if column == 0:
            axis.set_ylabel("y (mm)")
        else:
            axis.set_yticklabels([])
    colorbar = fig.colorbar(image, ax=fig.axes[:4], shrink=0.75, pad=0.01)
    colorbar.set_label("E spikes / neuron (3 ms)")

    label_order = ["control", "bends", "splits", "dies", "control fails"]
    label_value = {label: value for value, label in enumerate(label_order)}
    categorical = np.array(
        [label_value[record["outcome"]] for record in records], dtype=int
    ).reshape(radii_axis.size, inhibition_axis.size)
    transmission = np.array(
        [record["transmission_ratio"] for record in records]
    ).reshape(radii_axis.size, inhibition_axis.size)

    phase_axis = fig.add_subplot(grid[1, :2])
    phase_cmap = ListedColormap(["#b9bdc6", "#1f9d8a", "#e8a23a", "#c84b52", "#252a34"])
    phase_axis.imshow(
        categorical,
        origin="lower",
        cmap=phase_cmap,
        norm=BoundaryNorm(np.arange(-0.5, len(label_order) + 0.5), len(label_order)),
        aspect="auto",
    )
    for row in range(radii_axis.size):
        for column in range(inhibition_axis.size):
            label = records[row * n_inhibition + column]["outcome"]
            phase_axis.text(column, row, label, ha="center", va="center", fontsize=8)
    phase_axis.set_title("Outcome phase map")
    phase_axis.set_xlabel("I -> E conductance multiplier")
    phase_axis.set_ylabel("silent-patch radius (mm)")
    phase_axis.set_xticks(np.arange(inhibition_axis.size), [f"{v:.2f}" for v in inhibition_axis])
    phase_axis.set_yticks(np.arange(radii_axis.size), [f"{v:.2f}" for v in radii_axis])

    metric_axis = fig.add_subplot(grid[1, 2:])
    metric_image = metric_axis.imshow(
        np.clip(transmission, 0.0, 1.2),
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.2,
        aspect="auto",
    )
    for row in range(radii_axis.size):
        for column in range(inhibition_axis.size):
            metric_axis.text(
                column,
                row,
                f"{transmission[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if transmission[row, column] < 0.65 else "black",
                fontsize=9,
            )
    metric_axis.set_title("Far-edge transmission / matched control")
    metric_axis.set_xlabel("I -> E conductance multiplier")
    metric_axis.set_ylabel("silent-patch radius (mm)")
    metric_axis.set_xticks(np.arange(inhibition_axis.size), [f"{v:.2f}" for v in inhibition_axis])
    metric_axis.set_yticks(np.arange(radii_axis.size), [f"{v:.2f}" for v in radii_axis])
    fig.colorbar(metric_image, ax=metric_axis, shrink=0.8, pad=0.02, label="transmission ratio")

    fig.suptitle(
        "A cortical wave meets a silent patch\n"
        f"example: radius {radius:.2f} mm, inhibition {demo_record['inhibition_strength']:.2f}, "
        f"outcome = {demo_record['outcome']}",
        fontsize=15,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = simulate()
    records = measure_outcomes(result)
    figure_path = output_dir / "cortical_wave_obstacle.png"
    metrics_path = output_dir / "phase_metrics.csv"
    plot_summary(result, records, figure_path)
    write_metrics(records, metrics_path)

    counts = {label: 0 for label in ["control", "bends", "splits", "dies", "control fails"]}
    for record in records:
        counts[record["outcome"]] += 1
    print(f"figure: {figure_path}")
    print(f"metrics: {metrics_path}")
    print("outcomes: " + ", ".join(f"{label}={count}" for label, count in counts.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for the PNG and CSV outputs (default: outputs)",
    )
    main(parser.parse_args().output_dir)
