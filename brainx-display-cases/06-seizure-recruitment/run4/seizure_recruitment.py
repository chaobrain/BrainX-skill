"""Phenomenological seizure-like recruitment across three brain regions.

The script sweeps coupling strength, edge delay, and perturbation size with a
BrainState vmap over complete BrainMass simulations. It saves the exact
continuous evidence used to classify sustained regional recruitment.
"""

from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


DT = 0.1 * u.ms
DURATION = 200.0 * u.ms
RECOVERY_TAU = 20.0 * u.ms
MAX_DELAY = 12.0 * u.ms
STIMULUS_START = 20.0 * u.ms
STIMULUS_DURATION = 5.0 * u.ms

COUPLING_STRENGTHS = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8]) * u.UNITLESS
PROPAGATION_DELAYS = jnp.asarray([2.0, 6.0, 10.0]) * u.ms
PERTURBATION_SIZES = jnp.asarray([0.4, 0.6, 0.8, 1.0]) * u.UNITLESS

RECRUITMENT_THRESHOLD = 0.5 * u.UNITLESS
MINIMUM_BURST_DURATION = 1.0 * u.ms
REGION_LABELS = np.asarray(["Region 1 (focus)", "Region 2", "Region 3"])

# W[target, source]: a directed nearest-neighbor chain, 1 -> 2 -> 3.
CONNECTIVITY = jnp.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

LOCAL_CASE = (0.2, 6.0, 0.8)
RECRUITED_CASE = (0.6, 6.0, 0.8)
ARTIFACT_VERSION = "1.0"


def steps_for(duration):
    """Convert a time Quantity into an exact number of integration steps."""
    steps = duration / DT
    rounded = int(round(float(steps)))
    if not np.isclose(float(steps), rounded):
        raise ValueError(f"{duration} is not an integer multiple of dt={DT}")
    return rounded


N_STEPS = steps_for(DURATION)
STIMULUS_START_STEP = steps_for(STIMULUS_START)
STIMULUS_STOP_STEP = steps_for(STIMULUS_START + STIMULUS_DURATION)
MINIMUM_BURST_STEPS = steps_for(MINIMUM_BURST_DURATION)


class DelayedRegionalChain(brainstate.nn.Module):
    """Three excitable neural masses with a fixed-capacity delay history."""

    def __init__(self):
        super().__init__()
        self.node = brainmass.FitzHughNagumoStep(
            in_size=3,
            tau=RECOVERY_TAU,
            init_V=braintools.init.Constant(0.0),
            init_w=braintools.init.Constant(0.0),
        )
        self.history = brainstate.nn.Delay(
            jnp.zeros(3),
            time=MAX_DELAY,
            init=braintools.init.Constant(0.0),
        )

    def update(self, stimulus, coupling_strength, propagation_delay):
        self.history.update(self.node.V.value)
        delay_steps = jnp.rint(propagation_delay / DT).astype(jnp.int32)
        delayed = self.history.retrieve_at_step(delay_steps)
        sources = jnp.broadcast_to(delayed, CONNECTIVITY.shape)
        coupled_input = brainmass.additive_coupling(
            sources,
            CONNECTIVITY,
            coupling_strength,
        )
        self.node(coupled_input + stimulus)
        return self.node.V.value


def run_condition(coupling_strength, propagation_delay, perturbation_size):
    """Run one complete independent condition and return [time, region]."""
    coupling_strength = coupling_strength.to_decimal(u.UNITLESS)
    perturbation_size = perturbation_size.to_decimal(u.UNITLESS)
    with brainstate.environ.context(dt=DT):
        model = DelayedRegionalChain()
        brainstate.nn.init_all_states(model)

        def step(index):
            pulse_is_on = (index >= STIMULUS_START_STEP) & (
                index < STIMULUS_STOP_STEP
            )
            stimulus = jnp.where(
                pulse_is_on,
                jnp.asarray([perturbation_size, 0.0, 0.0]),
                jnp.zeros(3),
            )
            with brainstate.environ.context(i=index, t=index * DT):
                return model(stimulus, coupling_strength, propagation_delay)

        return brainstate.transform.for_loop(step, jnp.arange(N_STEPS))


def classify_recruitment(activity):
    """Apply the frozen sustained-event predicate to [condition, time, region]."""
    threshold = RECRUITMENT_THRESHOLD.to_decimal(u.UNITLESS)
    window_floor = jax.lax.reduce_window(
        activity,
        jnp.inf,
        jax.lax.min,
        (1, MINIMUM_BURST_STEPS, 1),
        (1, 1, 1),
        "VALID",
    )
    qualifying = window_floor >= threshold
    recruited = jnp.any(qualifying, axis=1)
    first_start = jnp.argmax(qualifying, axis=1)
    onset = jnp.where(recruited, first_start, jnp.nan) * DT
    max_window_floor = jnp.max(window_floor, axis=1)
    peak_activity = jnp.max(activity, axis=1)
    return recruited, onset, max_window_floor, peak_activity


def verify_delay_phase():
    """Lock the insert-then-retrieve convention with a 3 ms impulse check."""
    with brainstate.environ.context(dt=1.0 * u.ms):
        delay = brainstate.nn.Delay(jnp.zeros(()), time=3.0 * u.ms)
        brainstate.nn.init_all_states(delay)

        def delayed_impulse(value):
            delay.update(value)
            return delay.retrieve_at_step(jnp.asarray(3, dtype=jnp.int32))

        observed = brainstate.transform.for_loop(
            delayed_impulse,
            jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
        )
    expected = jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0])
    if not bool(jnp.array_equal(observed, expected)):
        raise AssertionError(f"delay phase check failed: observed {observed}")


def condition_index(k_grid, delay_grid, size_grid, condition):
    k, delay_ms, size = condition
    matches = np.isclose(k_grid, k) & np.isclose(delay_grid, delay_ms)
    matches &= np.isclose(size_grid, size)
    indices = np.flatnonzero(matches)
    if len(indices) != 1:
        raise AssertionError(f"condition {condition} matched {len(indices)} rows")
    return int(indices[0])


def plot_results(
    output_path,
    times_ms,
    activity,
    onset_ms,
    recruited_count_grid,
    k_grid,
    delay_grid,
    size_grid,
):
    local_index = condition_index(k_grid, delay_grid, size_grid, LOCAL_CASE)
    recruited_index = condition_index(
        k_grid, delay_grid, size_grid, RECRUITED_CASE
    )
    colors = ["#b8322a", "#14756f", "#355c9a"]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)
    for ax, index, title in (
        (axes[0, 0], local_index, "Local: weak coupling"),
        (axes[0, 1], recruited_index, "Recruited: stronger coupling"),
    ):
        for region, (label, color) in enumerate(zip(REGION_LABELS, colors)):
            ax.plot(times_ms, activity[index, :, region], label=label, color=color)
        ax.axvspan(
            STIMULUS_START.to_decimal(u.ms),
            (STIMULUS_START + STIMULUS_DURATION).to_decimal(u.ms),
            color="#d9a441",
            alpha=0.18,
            linewidth=0,
        )
        ax.axhline(
            RECRUITMENT_THRESHOLD.to_decimal(u.UNITLESS),
            color="#555555",
            linestyle=":",
            linewidth=1,
        )
        ax.set(title=title, xlabel="Time (ms)", ylabel="Regional activity V")
        ax.set_xlim(10, 70)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")

    onset_ax = axes[0, 2]
    k_match = np.isclose(k_grid, RECRUITED_CASE[0])
    size_match = np.isclose(size_grid, RECRUITED_CASE[2])
    onset_rows = np.flatnonzero(k_match & size_match)
    onset_order = np.argsort(delay_grid[onset_rows])
    onset_rows = onset_rows[onset_order]
    plotted_delays = delay_grid[onset_rows]
    for region, (label, color) in enumerate(zip(REGION_LABELS, colors)):
        onset_ax.plot(
            plotted_delays,
            onset_ms[onset_rows, region],
            marker="o",
            color=color,
            label=label,
        )
    onset_ax.set(
        title="Recruitment timing",
        xlabel="Edge delay (ms)",
        ylabel="Sustained-burst onset (ms)",
        xticks=plotted_delays,
    )

    count_cmap = ListedColormap(["#eeeeea", "#d9a441", "#2a8c82", "#315a88"])
    count_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], count_cmap.N)
    coupling_raw = np.asarray(COUPLING_STRENGTHS.to_decimal(u.UNITLESS))
    size_raw = np.asarray(PERTURBATION_SIZES.to_decimal(u.UNITLESS))
    delay_raw = np.asarray(PROPAGATION_DELAYS.to_decimal(u.ms))
    for delay_index, ax in enumerate(axes[1]):
        image = ax.imshow(
            recruited_count_grid[:, delay_index, :].T,
            origin="lower",
            aspect="auto",
            cmap=count_cmap,
            norm=count_norm,
        )
        for k_index in range(coupling_raw.size):
            for size_index in range(size_raw.size):
                count = recruited_count_grid[k_index, delay_index, size_index]
                ax.text(k_index, size_index, str(count), ha="center", va="center")
        ax.set(
            title=f"Edge delay {delay_raw[delay_index]:g} ms",
            xlabel="Coupling strength k",
            xticks=np.arange(coupling_raw.size),
            xticklabels=[f"{value:.1f}" for value in coupling_raw],
            yticks=np.arange(size_raw.size),
            yticklabels=[f"{value:.1f}" for value in size_raw],
        )
    axes[1, 0].set_ylabel("Perturbation size")
    colorbar = fig.colorbar(image, ax=axes[1, :], ticks=[0, 1, 2, 3], shrink=0.82)
    colorbar.set_label("Regions recruited")
    fig.suptitle("Seizure-like recruitment in a delayed regional chain", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    brainstate.random.seed(0)
    brainstate.environ.set(dt=DT)
    verify_delay_phase()

    coupling_raw = np.asarray(COUPLING_STRENGTHS.to_decimal(u.UNITLESS))
    delay_raw = np.asarray(PROPAGATION_DELAYS.to_decimal(u.ms))
    size_raw = np.asarray(PERTURBATION_SIZES.to_decimal(u.UNITLESS))
    kk, dd, ss = np.meshgrid(coupling_raw, delay_raw, size_raw, indexing="ij")
    grid_shape = kk.shape

    condition_tags = np.asarray(
        ["grid"] * kk.size + ["no_coupling_control", "no_stimulus_control"]
    )
    flat_k = jnp.asarray(np.concatenate([kk.ravel(), [0.0, 0.6]]))
    flat_delay_ms = jnp.asarray(np.concatenate([dd.ravel(), [6.0, 6.0]]))
    flat_size = jnp.asarray(np.concatenate([ss.ravel(), [0.8, 0.0]]))

    # FHN activity and input are documented dimensionless model variables;
    # conversion to raw arrays occurs inside run_condition at that API boundary.
    activity = brainstate.transform.vmap(run_condition)(
        flat_k * u.UNITLESS,
        flat_delay_ms * u.ms,
        flat_size * u.UNITLESS,
    )
    recruited, onset, max_window_floor, peak_activity = classify_recruitment(
        activity
    )

    activity_np = np.asarray(activity)
    recruited_np = np.asarray(recruited)
    onset_ms = np.asarray(onset.to_decimal(u.ms))
    window_floor_np = np.asarray(max_window_floor)
    peak_np = np.asarray(peak_activity)
    recruited_count_grid = recruited_np[: kk.size].sum(axis=1).reshape(grid_shape)

    local_index = condition_index(
        np.asarray(flat_k), np.asarray(flat_delay_ms), np.asarray(flat_size), LOCAL_CASE
    )
    recruited_index = condition_index(
        np.asarray(flat_k),
        np.asarray(flat_delay_ms),
        np.asarray(flat_size),
        RECRUITED_CASE,
    )
    no_coupling_index = kk.size
    no_stimulus_index = kk.size + 1

    if not np.array_equal(recruited_np[local_index], [True, False, False]):
        raise AssertionError("the frozen local example no longer remains local")
    if not np.array_equal(recruited_np[recruited_index], [True, True, True]):
        raise AssertionError("the frozen recruited example no longer recruits all regions")
    recruited_onsets = onset_ms[recruited_index]
    if not np.all(np.diff(recruited_onsets) > 0):
        raise AssertionError("recruitment does not follow Region 1 -> 2 -> 3")
    if not np.array_equal(
        recruited_np[no_coupling_index], [True, False, False]
    ):
        raise AssertionError("the no-coupling control should remain local")
    if np.any(recruited_np[no_stimulus_index]):
        raise AssertionError("the no-stimulus control should remain quiescent")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    bundle_path = output_dir / "seizure_recruitment_results.npz"
    figure_path = output_dir / "seizure_recruitment.png"
    times_ms = np.arange(N_STEPS) * DT.to_decimal(u.ms)

    np.savez_compressed(
        bundle_path,
        artifact_version=ARTIFACT_VERSION,
        model="brainmass.FitzHughNagumoStep",
        model_scope="phenomenological regional neural mass",
        region_labels=REGION_LABELS,
        connectivity=np.asarray(CONNECTIVITY),
        condition_tags=condition_tags,
        coupling_strength=np.asarray(flat_k),
        coupling_unit="dimensionless",
        propagation_delay_ms=np.asarray(flat_delay_ms),
        propagation_delay_unit="ms",
        perturbation_size=np.asarray(flat_size),
        perturbation_unit="dimensionless FHN input",
        dt_ms=DT.to_decimal(u.ms),
        recovery_tau_ms=RECOVERY_TAU.to_decimal(u.ms),
        duration_ms=DURATION.to_decimal(u.ms),
        stimulus_start_ms=STIMULUS_START.to_decimal(u.ms),
        stimulus_duration_ms=STIMULUS_DURATION.to_decimal(u.ms),
        recruitment_threshold=RECRUITMENT_THRESHOLD.to_decimal(u.UNITLESS),
        minimum_burst_duration_ms=MINIMUM_BURST_DURATION.to_decimal(u.ms),
        recruited=recruited_np,
        onset_ms=onset_ms,
        max_sustained_window_floor=window_floor_np,
        peak_activity=peak_np,
        representative_condition_indices=np.asarray(
            [local_index, recruited_index, no_coupling_index, no_stimulus_index]
        ),
        representative_activity=activity_np[
            [local_index, recruited_index, no_coupling_index, no_stimulus_index]
        ],
        representative_time_ms=times_ms,
        seed=0,
        integration_method="FitzHughNagumoStep default exp_euler",
        coupling_method="brainmass.additive_coupling",
        delay_prehistory="zeros",
        delay_phase="insert current source, then retrieve d completed steps earlier",
    )
    plot_results(
        figure_path,
        times_ms,
        activity_np,
        onset_ms,
        recruited_count_grid,
        np.asarray(flat_k),
        np.asarray(flat_delay_ms),
        np.asarray(flat_size),
    )

    print(f"Local example recruited: {recruited_np[local_index].tolist()}")
    print(
        "Recruited example onset (ms): "
        f"{np.round(recruited_onsets, 1).tolist()}"
    )
    print(
        "Controls recruited: "
        f"no coupling={recruited_np[no_coupling_index].tolist()}, "
        f"no stimulus={recruited_np[no_stimulus_index].tolist()}"
    )
    print(f"Saved {bundle_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
