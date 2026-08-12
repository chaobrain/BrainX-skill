"""Explore when a focal Epileptor burst recruits a three-region chain."""

from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DT = 0.2 * u.ms
DURATION = 500.0 * u.ms
MAX_DELAY = 30.0 * u.ms
PULSE_START = 100.0 * u.ms
PULSE_DURATION = 40.0 * u.ms

COUPLING_STRENGTHS = jnp.asarray([0.0, 2.0, 5.0, 10.0])
PROPAGATION_DELAYS = jnp.asarray([0.0, 10.0, 30.0]) * u.ms
PERTURBATION_SIZES = jnp.asarray([1.0, 1.5, 2.0, 2.5])

# W[target, source]: focus -> neighbor 1 -> neighbor 2.
CONNECTIVITY = jnp.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
REGION_NAMES = np.asarray(["focus", "neighbor 1", "neighbor 2"])
X0 = jnp.asarray([-2.2, -2.4, -2.4])

# Exploratory burst rule: at least 2 ms above x1=0 within a 40 ms window.
EVENT_THRESHOLD = 0.0
EVENT_WINDOW = 40.0 * u.ms
MIN_ACTIVE_DURATION = 2.0 * u.ms

OUTPUT_DIR = Path("results")


def steps_for(duration):
    """Convert a time quantity to a static step count at an explicit boundary."""
    return int(round(duration.to_decimal(u.ms) / DT.to_decimal(u.ms)))


class DrivenEpileptorChain(brainstate.nn.Module):
    """Three Epileptor regions with a fixed-capacity delayed coupling path."""

    def __init__(self):
        super().__init__()
        self.node = brainmass.EpileptorStep(
            in_size=3,
            x0=X0,
            Kvf=1.0,
            Ks=0.2,
            init_x1=braintools.init.Constant(-1.5),
        )
        self.history = brainstate.nn.Delay(
            jnp.full(3, -1.5),
            time=MAX_DELAY,
            init=braintools.init.Constant(-1.5),
        )

    def update(self, stimulus, coupling_strength, propagation_delay):
        self.history.update(self.node.x1.value)
        delay_steps = jnp.rint(propagation_delay / DT).astype(jnp.int32)
        delayed = self.history.retrieve_at_step(delay_steps)
        sources = jnp.broadcast_to(delayed, CONNECTIVITY.shape)
        network_input = brainmass.diffusive_coupling(
            sources,
            self.node.x1.value,
            CONNECTIVITY,
            coupling_strength,
        )
        self.node(network_input + stimulus)
        return self.node.x1.value


def run_condition(coupling_strength, propagation_delay, perturbation_size):
    """Run one independent condition; BrainState vmap owns condition isolation."""
    with brainstate.environ.context(dt=DT):
        model = DrivenEpileptorChain()
        brainstate.nn.init_all_states(model)

        def step(index):
            time = index * DT
            pulse_on = (time >= PULSE_START) & (
                time < PULSE_START + PULSE_DURATION
            )
            stimulus = jnp.asarray([1.0, 0.0, 0.0]) * jnp.where(
                pulse_on,
                perturbation_size,
                0.0,
            )
            with brainstate.environ.context(i=index, t=time):
                return model(stimulus, coupling_strength, propagation_delay)

        return brainstate.transform.for_loop(
            step,
            jnp.arange(steps_for(DURATION)),
        )


def classify_events(activity):
    """Return recruitment flags, onsets, and continuous event evidence."""
    window_steps = steps_for(EVENT_WINDOW)
    minimum_active_steps = steps_for(MIN_ACTIVE_DURATION)
    above = activity >= EVENT_THRESHOLD
    window_hits = jax.lax.reduce_window(
        above.astype(jnp.int32),
        0,
        jax.lax.add,
        (1, window_steps, 1),
        (1, 1, 1),
        "VALID",
    )
    qualifying = window_hits >= minimum_active_steps
    recruited = jnp.any(qualifying, axis=1)
    first_window = jnp.argmax(qualifying, axis=1)
    indices = jnp.arange(activity.shape[1])[None, :, None]
    inside_first_window = (
        (indices >= first_window[:, None, :])
        & (indices < first_window[:, None, :] + window_steps)
    )
    first_crossing = jnp.argmax(above & inside_first_window, axis=1)
    onset_ms = jnp.where(
        recruited,
        (first_crossing + 1) * DT.to_decimal(u.ms),
        jnp.nan,
    )
    max_active_ms = (
        jnp.max(window_hits, axis=1) * DT.to_decimal(u.ms)
    )
    return recruited, onset_ms, max_active_ms


def condition_table():
    """Flatten the grid and append matched causal controls."""
    coupling, delay, perturbation = u.math.meshgrid(
        COUPLING_STRENGTHS,
        PROPAGATION_DELAYS,
        PERTURBATION_SIZES,
        indexing="ij",
    )
    flat_coupling = coupling.reshape(-1)
    flat_delay = delay.reshape(-1)
    flat_perturbation = perturbation.reshape(-1)

    # Controls use the same mapped simulation and classification path.
    control_coupling = jnp.asarray([0.0, 10.0])
    control_delay = jnp.asarray([10.0, 10.0]) * u.ms
    control_perturbation = jnp.asarray([2.0, 0.0])
    return (
        jnp.concatenate([flat_coupling, control_coupling]),
        u.math.concatenate([flat_delay, control_delay]),
        jnp.concatenate([flat_perturbation, control_perturbation]),
        np.asarray(
            ["grid"] * flat_coupling.size + ["no coupling", "no pulse"]
        ),
        coupling.shape,
    )


def verify_delay_phase():
    """Lock insertion-before-retrieval to an exact integer delay."""
    test_delay = 1.0 * u.ms
    delay_steps = steps_for(test_delay)
    with brainstate.environ.context(dt=DT):
        history = brainstate.nn.Delay(
            jnp.zeros(()),
            time=test_delay,
            init=braintools.init.Constant(0.0),
        )
        brainstate.nn.init_all_states(history)

        def delayed_impulse(value):
            history.update(value)
            return history.retrieve_at_step(delay_steps)

        observed = brainstate.transform.for_loop(
            delayed_impulse,
            jnp.concatenate([jnp.ones(1), jnp.zeros(delay_steps + 1)]),
        )
    expected = jnp.zeros(delay_steps + 2).at[delay_steps].set(1.0)
    if not bool(jnp.array_equal(observed, expected)):
        raise AssertionError("Delay insertion/retrieval phase is not the declared convention.")


def validate_results(activity, recruited, onset_ms, condition_type):
    """Check numeric validity, controls, and the claimed routed outcome."""
    if not np.isfinite(activity).all():
        raise AssertionError("At least one mapped trajectory is non-finite.")
    no_coupling = np.flatnonzero(condition_type == "no coupling")
    no_pulse = np.flatnonzero(condition_type == "no pulse")
    if no_coupling.size != 1 or not np.array_equal(
        recruited[no_coupling[0]], np.asarray([True, False, False])
    ):
        raise AssertionError("The no-coupling control did not remain focal.")
    if no_pulse.size != 1 or recruited[no_pulse[0]].any():
        raise AssertionError("The no-pulse control produced a classified event.")
    routed = (
        recruited.all(axis=1)
        & (onset_ms[:, 0] < onset_ms[:, 1])
        & (onset_ms[:, 1] < onset_ms[:, 2])
    )
    if not routed.any():
        raise AssertionError("No condition showed strict routed recruitment.")


def plot_results(
    activity,
    recruited,
    onset_ms,
    coupling,
    delay_ms,
    perturbation,
    grid_shape,
):
    grid_count = int(np.prod(grid_shape))
    grid_recruited = recruited[:grid_count].reshape(grid_shape + (3,))
    display_pulse_index = 2
    displayed_neighbors = grid_recruited[
        :, :, display_pulse_index, 1:
    ].sum(axis=-1)

    grid_indices = np.arange(grid_count)
    matched_display = np.isclose(delay_ms[:grid_count], 10.0) & np.isclose(
        perturbation[:grid_count], 2.0
    )
    local_candidates = grid_indices[
        recruited[:grid_count, 0]
        & ~recruited[:grid_count, 1:].any(axis=1)
        & matched_display
    ]
    recruited_candidates = grid_indices[
        recruited[:grid_count].all(axis=1)
        & (onset_ms[:grid_count, 0] < onset_ms[:grid_count, 1])
        & (onset_ms[:grid_count, 1] < onset_ms[:grid_count, 2])
        & matched_display
    ]
    if not local_candidates.size or not recruited_candidates.size:
        raise RuntimeError(
            "The sampled grid did not contain both local and routed-recruitment cases."
        )
    local_index = int(local_candidates[0])
    recruited_index = int(recruited_candidates[0])

    times_ms = (np.arange(activity.shape[1]) + 1) * DT.to_decimal(u.ms)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    image = axes[0, 0].imshow(
        displayed_neighbors.T,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap="viridis",
    )
    axes[0, 0].set_xticks(np.arange(COUPLING_STRENGTHS.size))
    axes[0, 0].set_xticklabels(np.asarray(COUPLING_STRENGTHS))
    axes[0, 0].set_yticks(np.arange(PROPAGATION_DELAYS.size))
    axes[0, 0].set_yticklabels(PROPAGATION_DELAYS.to_decimal(u.ms))
    axes[0, 0].set_xlabel("coupling strength")
    axes[0, 0].set_ylabel("delay (ms)")
    axes[0, 0].set_title(
        "Neighbors recruited "
        f"(pulse = {float(PERTURBATION_SIZES[display_pulse_index]):.1f})"
    )
    fig.colorbar(image, ax=axes[0, 0], ticks=[0, 1, 2])

    onset_grid = onset_ms[:grid_count].reshape(grid_shape + (3,))
    timing_k_index = int(
        np.flatnonzero(np.isclose(np.asarray(COUPLING_STRENGTHS), 5.0))[0]
    )
    timing_pulse_index = int(
        np.flatnonzero(np.isclose(np.asarray(PERTURBATION_SIZES), 2.0))[0]
    )
    timing_onsets = onset_grid[timing_k_index, :, timing_pulse_index]
    axes[0, 1].plot(
        np.asarray(PROPAGATION_DELAYS.to_decimal(u.ms)),
        timing_onsets[:, 1] - timing_onsets[:, 0],
        marker="o",
        label="focus to neighbor 1",
    )
    axes[0, 1].plot(
        np.asarray(PROPAGATION_DELAYS.to_decimal(u.ms)),
        timing_onsets[:, 2] - timing_onsets[:, 1],
        marker="s",
        label="neighbor 1 to neighbor 2",
    )
    axes[0, 1].set_xlabel("propagation delay (ms)")
    axes[0, 1].set_ylabel("recruitment lag (ms)")
    axes[0, 1].set_title("Delay shifts recruitment timing (k = 5, pulse = 2)")
    axes[0, 1].legend()

    for axis, index, title in (
        (axes[1, 0], local_index, "Local burst"),
        (axes[1, 1], recruited_index, "Routed recruitment"),
    ):
        for region, name in enumerate(REGION_NAMES):
            axis.plot(times_ms, activity[index, :, region], label=name)
        axis.axvspan(
            PULSE_START.to_decimal(u.ms),
            (PULSE_START + PULSE_DURATION).to_decimal(u.ms),
            color="#d95f02",
            alpha=0.16,
        )
        axis.axhline(EVENT_THRESHOLD, color="#555555", linewidth=0.8)
        axis.set_xlim(70.0, 230.0)
        axis.set_xlabel("time (ms)")
        axis.set_ylabel("Epileptor fast state x1")
        axis.set_title(
            f"{title}: k={coupling[index]:.2f}, delay={delay_ms[index]:g} ms, "
            f"pulse={perturbation[index]:.1f}"
        )
        axis.legend(loc="upper right")

    output_path = OUTPUT_DIR / "seizure_recruitment.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, local_index, recruited_index


def main():
    brainstate.random.seed(0)
    brainstate.environ.set(dt=DT)
    verify_delay_phase()
    coupling, delay, perturbation, condition_type, grid_shape = condition_table()

    simulate_sweep = brainstate.transform.vmap(run_condition)
    activity = simulate_sweep(coupling, delay, perturbation)
    recruited, onset_ms, max_active_ms = classify_events(activity)

    activity_np = np.asarray(activity)
    recruited_np = np.asarray(recruited)
    onset_ms_np = np.asarray(onset_ms)
    max_active_ms_np = np.asarray(max_active_ms)
    coupling_np = np.asarray(coupling)
    delay_ms_np = np.asarray(delay.to_decimal(u.ms))
    perturbation_np = np.asarray(perturbation)
    validate_results(
        activity_np,
        recruited_np,
        onset_ms_np,
        condition_type,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    bundle_path = OUTPUT_DIR / "seizure_recruitment.npz"
    np.savez_compressed(
        bundle_path,
        activity_x1=activity_np,
        recruited=recruited_np,
        onset_ms=onset_ms_np,
        max_active_duration_in_window_ms=max_active_ms_np,
        coupling_strength=coupling_np,
        propagation_delay_ms=delay_ms_np,
        perturbation_size=perturbation_np,
        condition_type=condition_type,
        coupling_axis=np.asarray(COUPLING_STRENGTHS),
        delay_axis_ms=np.asarray(PROPAGATION_DELAYS.to_decimal(u.ms)),
        perturbation_axis=np.asarray(PERTURBATION_SIZES),
        grid_shape=np.asarray(grid_shape),
        region_names=REGION_NAMES,
        connectivity=np.asarray(CONNECTIVITY),
        dt_ms=DT.to_decimal(u.ms),
        duration_ms=DURATION.to_decimal(u.ms),
        pulse_start_ms=PULSE_START.to_decimal(u.ms),
        pulse_duration_ms=PULSE_DURATION.to_decimal(u.ms),
        x0=np.asarray(X0),
        event_threshold=EVENT_THRESHOLD,
        event_window_ms=EVENT_WINDOW.to_decimal(u.ms),
        minimum_active_duration_ms=MIN_ACTIVE_DURATION.to_decimal(u.ms),
        monitor_phase="post-update; sample 0 is dt",
        connectivity_convention="W[target, source]",
        coupling_kernel="diffusive coupling of delayed x1",
        model="brainmass.EpileptorStep",
        artifact_schema_version="1",
        analysis_status=(
            "exploratory demonstration; event rule and sampled regimes are "
            "outcome-calibrated"
        ),
    )

    figure_path, local_index, recruited_index = plot_results(
        activity_np,
        recruited_np,
        onset_ms_np,
        coupling_np,
        delay_ms_np,
        perturbation_np,
        grid_shape,
    )

    route = (
        recruited_np.all(axis=1)
        & (onset_ms_np[:, 0] < onset_ms_np[:, 1])
        & (onset_ms_np[:, 1] < onset_ms_np[:, 2])
    )
    print(f"simulated conditions: {activity_np.shape[0]}")
    print(f"strict routed recruitment cases: {route.sum()}")
    print(
        "local example:",
        f"k={coupling_np[local_index]:.2f},",
        f"delay={delay_ms_np[local_index]:g} ms,",
        f"pulse={perturbation_np[local_index]:.1f},",
        f"onsets={onset_ms_np[local_index]} ms",
    )
    print(
        "recruited example:",
        f"k={coupling_np[recruited_index]:.2f},",
        f"delay={delay_ms_np[recruited_index]:g} ms,",
        f"pulse={perturbation_np[recruited_index]:.1f},",
        f"onsets={onset_ms_np[recruited_index]} ms",
    )
    print(f"saved numeric bundle: {bundle_path}")
    print(f"saved figure: {figure_path}")


if __name__ == "__main__":
    main()
