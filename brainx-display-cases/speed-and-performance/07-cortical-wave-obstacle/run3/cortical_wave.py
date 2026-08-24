"""A cortical spike wave encountering a circular silent patch.

This is a phenomenological point-neuron demonstration, not a fit to a
particular cortical preparation.  It uses BrainPy-State for E/I LIF dynamics,
BrainEvent CSR matrices for sparse spike communication, BrainState for the
stateful time/parameter transforms, and BrainUnit for physical quantities.
"""

from __future__ import annotations

from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from braintools.conn import Grid2d
from braintools.input import Constant
from brainstate.util import filter as state_filter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Patch


# Sheet and protocol
NX, NY = 44, 28
PITCH = 0.20 * u.mm
DT = 0.20 * u.ms
DURATION = 150.0 * u.ms
SPARK_DURATION = 3.0 * u.ms
SPARK_CURRENT = 1.40 * u.nA
LESION_CENTER = (4.5 * u.mm, 2.35 * u.mm)
LESION_RADII = jnp.asarray([0.0, 0.45, 0.90, 1.35, 1.80, 2.20]) * u.mm
INHIBITION_GAINS = jnp.asarray([0.0, 0.60, 1.20, 1.60, 1.90, 2.10])

# LIF and conductance parameters
V_REST = -65.0 * u.mV
V_RESET = -65.0 * u.mV
V_THRESHOLD_E = -52.0 * u.mV
V_THRESHOLD_I = -52.0 * u.mV
E_EXC = 0.0 * u.mV
E_INH = -80.0 * u.mV
R_MEMBRANE = 100.0 * u.Mohm
TAU_MEMBRANE = 15.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
TAU_EXC = 3.0 * u.ms
TAU_INH = 6.0 * u.ms
W_EE = 6.5 * u.nS
W_EI = 3.2 * u.nS
W_IE = 1.0 * u.nS


def sheet_coordinates():
    """Return flattened cell-center coordinates with explicit length units."""
    x = u.math.arange(NX) * PITCH
    y = u.math.arange(NY) * PITCH
    xx, yy = u.math.meshgrid(x, y)
    return u.math.reshape(xx, (-1,)), u.math.reshape(yy, (-1,))


def local_grid_csr():
    """Build open-boundary, eight-neighbor topology and store it as CSR."""
    result = Grid2d(connectivity="moore", periodic=False)(
        pre_size=(NY, NX),
        post_size=(NY, NX),
    )
    indptr, indices, _ = brainevent.coo2csr(
        result.pre_indices,
        result.post_indices,
        shape=result.shape,
    )
    return brainevent.CSR((jnp.asarray(1.0), indices, indptr), shape=result.shape)


class CorticalSheet(brainstate.nn.Module):
    """One excitatory and one inhibitory LIF neuron at every sheet location."""

    def __init__(self, connectivity):
        super().__init__()
        size = NX * NY
        initializer = braintools.init.Constant(V_REST)
        self.exc = brainpy.state.LIFRef(
            size,
            R=R_MEMBRANE,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_rest=V_REST,
            V_th=V_THRESHOLD_E,
            V_reset=V_RESET,
            V_initializer=initializer,
        )
        self.inh = brainpy.state.LIFRef(
            size,
            R=R_MEMBRANE,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_rest=V_REST,
            V_th=V_THRESHOLD_I,
            V_reset=V_RESET,
            V_initializer=initializer,
        )
        self.local = connectivity

        self.ee_syn = brainpy.state.Expon(size, tau=TAU_EXC)
        self.ei_syn = brainpy.state.Expon(size, tau=TAU_EXC)
        self.ie_syn = brainpy.state.Expon(size, tau=TAU_INH)
        self.ee_out = brainpy.state.COBA(E=E_EXC)
        self.ei_out = brainpy.state.COBA(E=E_EXC)
        self.ie_out = brainpy.state.COBA(E=E_INH)
        self.exc.add_current_input("ee", self.ee_out)
        self.exc.add_current_input("ie", self.ie_out)
        self.inh.add_current_input("ei", self.ei_out)

    def update(self, t, spark_current, active, inhibition_gain):
        """Advance one condition by one dt; lesion cells cannot emit spikes."""
        with brainstate.environ.context(t=t):
            previous_exc = (self.exc.get_spike() != 0.0) & active
            previous_inh = (self.inh.get_spike() != 0.0) & active

            exc_fanout = brainevent.BinaryArray(previous_exc) @ self.local
            inh_fanout = brainevent.BinaryArray(previous_inh) @ self.local
            self.ee_out.bind_cond(self.ee_syn(exc_fanout * W_EE))
            self.ei_out.bind_cond(self.ei_syn(exc_fanout * W_EI))
            self.ie_out.bind_cond(self.ie_syn(inh_fanout * W_IE * inhibition_gain))

            exc_spike = (self.exc(spark_current) != 0.0) & active
            inh_spike = (self.inh(0.0 * u.nA) != 0.0) & active
            return exc_spike, inh_spike


def make_conditions(x, y):
    """Create the radius/gain grid and one silent-patch mask per condition."""
    radius_grid, gain_grid = u.math.meshgrid(LESION_RADII, INHIBITION_GAINS)
    radii = u.math.reshape(radius_grid, (-1,))
    gains = jnp.reshape(gain_grid, (-1,))
    distance = u.math.sqrt(
        (x[None, :] - LESION_CENTER[0]) ** 2
        + (y[None, :] - LESION_CENTER[1]) ** 2
    )
    active = (radii[:, None] == 0.0 * u.mm) | (distance > radii[:, None])
    return radii, gains, active


def simulate():
    """Run the complete lesion/inhibition sweep in one mapped stateful loop."""
    brainstate.random.seed(17)
    x, y = sheet_coordinates()
    radii, gains, active = make_conditions(x, y)

    with brainstate.environ.context(dt=DT):
        net = CorticalSheet(local_grid_csr())
        protocol = Constant(
            [
                (SPARK_CURRENT, SPARK_DURATION),
                (0.0 * u.nA, DURATION - SPARK_DURATION),
            ]
        )()
        times = u.math.arange(0.0 * u.ms, DURATION, brainstate.environ.get_dt())
        edge = x <= 0.4 * u.mm
        spark = protocol[:, None] * edge[None, :]

        # vmap2 is BrainState's filter-based stateful vmap.  Each condition owns
        # a lane of dynamical State while topology and biophysical constants are
        # shared.  The mapping owns conditions; for_loop owns physical time.
        brainstate.nn.vmap_init_all_states(net, axis_size=radii.shape[0])
        dynamics = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = brainstate.transform.vmap2(
            net.update,
            in_axes=(None, None, 0, 0),
            out_axes=0,
            state_in_axes={0: dynamics},
            state_out_axes={0: dynamics},
            unexpected_out_state_mapping="raise",
        )

        @brainstate.transform.jit
        def run():
            def step(t, spark_at_t):
                return mapped_step(t, spark_at_t, active, gains)

            return brainstate.transform.for_loop(step, times, spark)

        exc_spikes, inh_spikes = run()

    return {
        "times": times,
        "x": x,
        "y": y,
        "radii": radii,
        "gains": gains,
        "active": active,
        "exc_spikes": exc_spikes,
        "inh_spikes": inh_spikes,
    }


def first_active_step(activity, minimum=2):
    hits = np.flatnonzero(activity >= minimum)
    return int(hits[0]) if hits.size else activity.shape[0]


def measure_and_classify(result):
    """Derive phase labels from matched controls and ordered route observables."""
    spikes = np.asarray(result["exc_spikes"], dtype=bool)
    x_mm = np.asarray(result["x"].to_decimal(u.mm))
    y_mm = np.asarray(result["y"].to_decimal(u.mm))
    unique_radii = np.asarray(LESION_RADII.to_decimal(u.mm))
    unique_gains = np.asarray(INHIBITION_GAINS)
    n_gain, n_radius = unique_gains.size, unique_radii.size
    n_steps, n_conditions, _ = spikes.shape
    assert n_conditions == n_gain * n_radius

    cx = LESION_CENTER[0].to_decimal(u.mm)
    cy = LESION_CENTER[1].to_decimal(u.mm)
    target_mask = x_mm >= x_mm.max() - 0.40
    target_activity = spikes[:, :, target_mask].sum(axis=2)
    target_count = target_activity.sum(axis=0).reshape(n_gain, n_radius)
    target_first = np.asarray(
        [first_active_step(target_activity[:, i]) for i in range(n_conditions)]
    ).reshape(n_gain, n_radius)

    upper_count = np.zeros((n_gain, n_radius))
    lower_count = np.zeros_like(upper_count)
    control_upper = np.zeros_like(upper_count)
    control_lower = np.zeros_like(lower_count)
    route_first = np.full_like(upper_count, n_steps, dtype=int)
    for gi in range(n_gain):
        control_spikes = spikes[:, gi * n_radius]
        for ri, radius in enumerate(unique_radii):
            ci = gi * n_radius + ri
            x_band = np.abs(x_mm - cx) <= radius + 0.45
            upper = x_band & (y_mm >= cy + radius) & (y_mm <= cy + radius + 0.65)
            lower = x_band & (y_mm <= cy - radius) & (y_mm >= cy - radius - 0.65)
            upper_activity = spikes[:, ci, upper].sum(axis=1)
            lower_activity = spikes[:, ci, lower].sum(axis=1)
            upper_count[gi, ri] = upper_activity.sum()
            lower_count[gi, ri] = lower_activity.sum()
            control_upper[gi, ri] = control_spikes[:, upper].sum()
            control_lower[gi, ri] = control_spikes[:, lower].sum()
            route_first[gi, ri] = min(
                first_active_step(upper_activity), first_active_step(lower_activity)
            )

    control_target = target_count[:, :1]
    reach_ratio = target_count / np.maximum(control_target, 1.0)
    control_route = control_upper + control_lower
    route_strength = (upper_count + lower_count) / np.maximum(control_route, 1.0)
    upper_ratio = upper_count / np.maximum(control_upper, 1.0)
    lower_ratio = lower_count / np.maximum(control_lower, 1.0)
    route_balance = upper_count / np.maximum(upper_count + lower_count, 1.0)
    order_margin_ms = np.where(
        (target_first < n_steps) & (route_first < n_steps),
        (target_first - route_first) * DT.to_decimal(u.ms),
        np.nan,
    )
    control_ok = control_target[:, 0] > 0

    # 0 = matched control itself fails, 1 = lesion dies, 2 = one-sided bend,
    # 3 = two-sided split/rejoin, 4 = no-lesion control crosses the sheet.
    phase = np.zeros((n_gain, n_radius), dtype=np.int8)
    for gi in range(n_gain):
        phase[gi, 0] = 4 if control_ok[gi] else 0
        for ri in range(1, n_radius):
            ordered = route_first[gi, ri] < target_first[gi, ri] < n_steps
            survives = control_ok[gi] and reach_ratio[gi, ri] >= 0.20 and ordered
            if not survives:
                phase[gi, ri] = 1 if control_ok[gi] else 0
                continue
            upper_relative = upper_ratio[gi, ri]
            lower_relative = lower_ratio[gi, ri]
            balanced = 0.25 <= route_balance[gi, ri] <= 0.75
            two_routes = upper_relative >= 0.15 and lower_relative >= 0.15 and balanced
            phase[gi, ri] = 3 if two_routes else 2

    return {
        "phase": phase,
        "reach_ratio": reach_ratio,
        "route_strength": route_strength,
        "upper_ratio": upper_ratio,
        "lower_ratio": lower_ratio,
        "route_balance": route_balance,
        "order_margin_ms": order_margin_ms,
        "target_count": target_count,
        "upper_count": upper_count,
        "lower_count": lower_count,
        "control_ok": control_ok,
        "radii_mm": unique_radii,
        "gains": unique_gains,
    }


def choose_snapshot_condition(metrics):
    """Prefer a split, then a bend, at a middle inhibition and large lesion."""
    phase = metrics["phase"]
    for code in (3, 2, 1):
        candidates = np.argwhere(phase[:, 1:] == code)
        if candidates.size:
            candidates[:, 1] += 1
            score = np.abs(candidates[:, 0] - 2.0) - 0.15 * candidates[:, 1]
            return tuple(candidates[np.argmin(score)])
    return 0, 2


def plot_snapshots(result, metrics, output):
    spikes = np.asarray(result["exc_spikes"], dtype=bool)
    times_ms = np.asarray(result["times"].to_decimal(u.ms))
    gi, ri = choose_snapshot_condition(metrics)
    n_radius = metrics["radii_mm"].size
    lesion_index = gi * n_radius + ri
    control_index = gi * n_radius
    radius = metrics["radii_mm"][ri]
    outcome = {0: "control failure", 1: "dies", 2: "bends", 3: "splits", 4: "crosses"}[
        int(metrics["phase"][gi, ri])
    ]
    snapshot_times = np.asarray([8.0, 40.0, 72.0, 108.0, 144.0])
    window_steps = max(1, int(round(4.0 / DT.to_decimal(u.ms))))
    extent = [
        -0.5 * PITCH.to_decimal(u.mm),
        (NX - 0.5) * PITCH.to_decimal(u.mm),
        -0.5 * PITCH.to_decimal(u.mm),
        (NY - 0.5) * PITCH.to_decimal(u.mm),
    ]

    fig, axes = plt.subplots(2, snapshot_times.size, figsize=(14.0, 5.4), sharex=True, sharey=True)
    vmax = 1
    images = []
    for row, condition in enumerate((control_index, lesion_index)):
        for col, time_ms in enumerate(snapshot_times):
            stop = int(np.argmin(np.abs(times_ms - time_ms))) + 1
            start = max(0, stop - window_steps)
            activity = spikes[start:stop, condition].sum(axis=0).reshape(NY, NX)
            vmax = max(vmax, int(activity.max()))
            images.append((row, col, activity))
    for row, col, activity in images:
        ax = axes[row, col]
        im = ax.imshow(
            activity,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        if row == 1:
            ax.add_patch(
                Circle(
                    (LESION_CENTER[0].to_decimal(u.mm), LESION_CENTER[1].to_decimal(u.mm)),
                    radius,
                    facecolor="none",
                    edgecolor="cyan",
                    linewidth=1.3,
                )
            )
        if row == 0:
            ax.set_title(f"{snapshot_times[col]:.0f} ms")
        if col == 0:
            ax.set_ylabel("control\ny (mm)" if row == 0 else f"lesion: {outcome}\ny (mm)")
        if row == 1:
            ax.set_xlabel("x (mm)")
    fig.suptitle(
        f"Excitatory spikes in the preceding 4 ms  |  inhibition = {metrics['gains'][gi]:.2f}, "
        f"patch radius = {radius:.2f} mm",
        fontsize=12,
    )
    colorbar_axis = fig.add_axes([0.925, 0.20, 0.012, 0.56])
    cbar = fig.colorbar(im, cax=colorbar_axis)
    cbar.set_label("spikes per site")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.12, top=0.84, wspace=0.08, hspace=0.15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_phase_map(metrics, output):
    radii = metrics["radii_mm"]
    gains = metrics["gains"]
    extent = [
        radii[0] - 0.225,
        radii[-1] + 0.225,
        gains[0] - 0.225,
        gains[-1] + 0.225,
    ]
    colors = ["#8b8b8b", "#202020", "#e68632", "#2f9e8f", "#3f6fb6"]
    labels = ["control fails", "dies", "bends", "splits / rejoins", "control crosses"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(
        metrics["phase"], origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm
    )
    axes[0].set_title("Outcome")
    axes[0].legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, labels)],
        loc="upper right",
        fontsize=7,
        frameon=True,
    )

    panels = [
        ("reach_ratio", "Downstream / matched control", "viridis", 0.0, 1.2),
        ("upper_ratio", "Upper route / matched control", "magma", 0.0, 1.2),
        ("lower_ratio", "Lower route / matched control", "magma", 0.0, 1.2),
        ("route_balance", "Upper-route fraction", "coolwarm", 0.0, 1.0),
        ("order_margin_ms", "Target minus route arrival (ms)", "cividis", 0.0, 125.0),
    ]
    for ax, (key, title, cm, vmin, vmax) in zip(axes[1:], panels):
        image = ax.imshow(
            metrics[key],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cm,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.78)
    for ax in axes:
        ax.set_xlabel("silent-patch radius (mm)")
        ax.set_xticks(radii)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("inhibitory conductance multiplier")
    axes[0].set_yticks(gains)
    for ax in axes[1:]:
        ax.set_yticks([])
    fig.suptitle("Wave fate from matched-control reach and ordered upper/lower routes", fontsize=12)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_metrics(metrics, output):
    np.savez_compressed(output, **metrics)


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    result = simulate()
    metrics = measure_and_classify(result)

    lesion_spikes = np.asarray(result["exc_spikes"], dtype=bool)
    lesion_active = np.asarray(result["active"], dtype=bool)
    assert not np.any(lesion_spikes & ~lesion_active[None, :, :])
    assert np.any(metrics["control_ok"]), "The matched control never crossed the sheet."

    plot_snapshots(result, metrics, output_dir / "wave_snapshots.png")
    plot_phase_map(metrics, output_dir / "phase_map.png")
    save_metrics(metrics, output_dir / "phase_metrics.npz")

    names = np.asarray(["control fails", "dies", "bends", "splits", "control crosses"])
    values, counts = np.unique(metrics["phase"], return_counts=True)
    summary = ", ".join(f"{names[value]}={count}" for value, count in zip(values, counts))
    print(f"Completed {metrics['phase'].size} conditions: {summary}")
    print("Saved results/wave_snapshots.png, results/phase_map.png, results/phase_metrics.npz")


if __name__ == "__main__":
    main()
