"""Map when a focal Epileptor burst stays local or recruits a regional chain."""

from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


DT = 0.1 * u.ms
DURATION = 1200.0 * u.ms
STIMULUS_START = 400.0 * u.ms
STIMULUS_DURATION = 40.0 * u.ms
MAX_DELAY = 12.0 * u.ms

REGION_LABELS = np.asarray(["focus", "neighbor 1", "neighbor 2"])
CONNECTIVITY = jnp.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
FOCUS_MASK = jnp.asarray([1.0, 0.0, 0.0])

# Epileptor state, coupling, and input are dimensionless; time parameters retain units.
COUPLING_VALUES = jnp.asarray([0.0, 0.25, 0.5, 0.75])
DELAY_VALUES = jnp.asarray([2.0, 6.0, 10.0]) * u.ms
PERTURBATION_VALUES = jnp.asarray([0.4, 0.8, 1.2])
LFP_THRESHOLD = 0.0
MINIMUM_BURST_DURATION = 20.0 * u.ms

OUTPUT_DIR = Path("results")
RESULT_PATH = OUTPUT_DIR / "seizure_recruitment.npz"
FIGURE_PATH = OUTPUT_DIR / "seizure_recruitment.png"


class DelayedEpileptorChain(brainstate.nn.Module):
    """Three Epileptor regions with a fixed-capacity delayed coupling path."""

    def __init__(self):
        super().__init__()
        self.node = brainmass.EpileptorStep(
            in_size=3,
            x0=jnp.asarray([-2.05, -2.4, -2.4]),
            Kvf=1.0,
            Ks=0.2,
        )
        self.history = brainstate.nn.Delay(
            jnp.zeros(3),
            time=MAX_DELAY,
            init=braintools.init.Constant(0.0),
        )

    def update(self, stimulus, coupling, delay):
        self.history.update(self.node.x1.value)
        delay_steps = jnp.rint(delay / DT).astype(jnp.int32)
        delayed = self.history.retrieve_at_step(delay_steps)
        sources = jnp.broadcast_to(delayed, CONNECTIVITY.shape)
        network_input = brainmass.diffusive_coupling(
            sources,
            self.node.x1.value,
            CONNECTIVITY,
            coupling,
        )
        self.node(network_input + stimulus)
        return self.node.lfp(), self.node.x1.value


def run_condition(coupling, delay, perturbation):
    """Run one independent condition; the caller maps this complete operation."""

    with brainstate.environ.context(dt=DT):
        model = DelayedEpileptorChain()
        brainstate.nn.init_all_states(model)
        n_steps = int(DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms))
        indices = jnp.arange(n_steps)
        start_step = int(STIMULUS_START.to_decimal(u.ms) / DT.to_decimal(u.ms))
        stop_step = int(
            (STIMULUS_START + STIMULUS_DURATION).to_decimal(u.ms)
            / DT.to_decimal(u.ms)
        )

        def step(i):
            active = (i >= start_step) & (i < stop_step)
            stimulus = jnp.where(
                active,
                perturbation * FOCUS_MASK,
                jnp.zeros(3),
            )
            with brainstate.environ.context(i=i, t=i * DT):
                return model(stimulus, coupling, delay)

        return brainstate.transform.for_loop(step, indices)


def delay_phase_check():
    """Verify that delay d is observed exactly d completed updates later."""

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
        raise RuntimeError(f"delay phase check failed: observed {observed}")


def classify(lfp, x1):
    """Return sustained LFP recruitment, post-update onset, and peak x1."""

    lfp = u.get_magnitude(lfp)
    x1 = u.get_magnitude(x1)
    ictal = lfp < LFP_THRESHOLD
    minimum_steps = int(
        MINIMUM_BURST_DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms)
    )
    window_hits = jax.lax.reduce_window(
        ictal.astype(jnp.int32),
        0,
        jax.lax.add,
        (1, minimum_steps, 1),
        (1, 1, 1),
        "VALID",
    )
    qualifying_windows = window_hits == minimum_steps
    recruited = jnp.any(qualifying_windows, axis=1)
    first_start = jnp.argmax(qualifying_windows, axis=1)
    onset_ms = jnp.where(
        recruited,
        (first_start + 1) * DT.to_decimal(u.ms),
        jnp.nan,
    )
    peak_x1 = jnp.max(x1, axis=1)
    return recruited, onset_ms, peak_x1


def make_conditions():
    """Flatten the grid and append matched mechanism controls."""

    coupling, delay, perturbation = u.math.meshgrid(
        COUPLING_VALUES,
        DELAY_VALUES,
        PERTURBATION_VALUES,
        indexing="ij",
    )
    shape = coupling.shape
    flat_coupling = coupling.reshape(-1)
    flat_delay = delay.reshape(-1)
    flat_perturbation = perturbation.reshape(-1)

    # Same high-drive/high-coupling setting, with one proposed mechanism removed.
    flat_coupling = u.math.concatenate(
        [flat_coupling, jnp.asarray([0.0, 0.75])]
    )
    flat_delay = u.math.concatenate(
        [flat_delay, jnp.asarray([6.0, 6.0]) * u.ms]
    )
    flat_perturbation = u.math.concatenate(
        [flat_perturbation, jnp.asarray([1.2, 0.0])]
    )
    tags = np.asarray(["grid"] * int(np.prod(shape)) + ["no_coupling", "no_perturbation"])
    return flat_coupling, flat_delay, flat_perturbation, tags, shape


def choose_examples(tags, coupling, recruited, onsets):
    grid = tags == "grid"
    coupling_raw = np.asarray(coupling)
    recruited_np = np.asarray(recruited)
    onset_np = np.asarray(onsets)
    source_only = recruited_np[:, 0] & ~recruited_np[:, 1] & ~recruited_np[:, 2]
    local_candidates = np.flatnonzero(grid & (coupling_raw > 0.0) & source_only)

    routed = (
        np.all(recruited_np, axis=1)
        & (onset_np[:, 0] < onset_np[:, 1])
        & (onset_np[:, 1] < onset_np[:, 2])
    )
    recruited_candidates = np.flatnonzero(grid & routed)
    if local_candidates.size == 0 or recruited_candidates.size == 0:
        raise RuntimeError(
            "the sampled grid did not contain both a local and strictly routed recruited case"
        )
    return int(local_candidates[-1]), int(recruited_candidates[0])


def plot_results(
    lfp,
    coupling,
    delay,
    perturbation,
    recruited,
    onsets,
    grid_shape,
    local_index,
    recruited_index,
):
    times_ms = (np.arange(lfp.shape[1]) + 1) * DT.to_decimal(u.ms)
    lfp = np.asarray(u.get_magnitude(lfp))
    coupling_raw = np.asarray(coupling)
    delay_ms = np.asarray(delay.to_decimal(u.ms))
    perturbation_raw = np.asarray(perturbation)
    n_grid = int(np.prod(grid_shape))
    recruited_grid = np.asarray(recruited[:n_grid]).reshape(grid_shape + (3,))
    neighbor_count = recruited_grid[..., 1:].sum(axis=-1)
    regime_code = np.where(recruited_grid[..., 0], neighbor_count, -1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, condition_index, title in (
        (axes[0, 0], local_index, "Local burst"),
        (axes[0, 1], recruited_index, "Routed recruitment"),
    ):
        for region, label in enumerate(REGION_LABELS):
            ax.plot(times_ms, lfp[condition_index, :, region], label=label, linewidth=1.0)
            onset = onsets[condition_index, region]
            if np.isfinite(onset):
                ax.axvline(onset, color=f"C{region}", linewidth=0.8, alpha=0.6)
        ax.set_title(
            f"{title}: k={coupling_raw[condition_index]:.2f}, "
            f"delay={delay_ms[condition_index]:.0f} ms, "
            f"pulse={perturbation_raw[condition_index]:.1f}"
        )
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Epileptor LFP proxy")
        ax.set_xlim(250.0, DURATION.to_decimal(u.ms))
    axes[0, 0].legend(frameon=False, loc="upper right")

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02,
        0.98,
        "Recruitment predicate\n"
        f"post-update LFP < {LFP_THRESHOLD:.1f} for {MINIMUM_BURST_DURATION.to_decimal(u.ms):.0f} ms\n\n"
        "Vertical lines: sustained-burst onset\n"
        "Map: N=no burst, L=local, 1/2=neighbors recruited\n"
        "Connectivity: focus -> neighbor 1 -> neighbor 2",
        ha="left",
        va="top",
        fontsize=10,
    )

    for pulse_index, ax in enumerate(axes[1]):
        values = regime_code[:, :, pulse_index].T
        ax.imshow(values, origin="lower", vmin=-1, vmax=2, cmap="viridis", aspect="auto")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                label = {-1: "N", 0: "L", 1: "1", 2: "2"}[values[row, col]]
                ax.text(col, row, label, ha="center", va="center", color="white")
        ax.set_xticks(np.arange(len(COUPLING_VALUES)))
        ax.set_xticklabels([f"{v:.2f}" for v in COUPLING_VALUES])
        ax.set_yticks(np.arange(len(DELAY_VALUES)))
        ax.set_yticklabels([f"{v:.0f}" for v in DELAY_VALUES.to_decimal(u.ms)])
        ax.set_xlabel("coupling k (dimensionless)")
        ax.set_ylabel("delay (ms)")
        ax.set_title(
            f"Recruitment outcome, pulse={PERTURBATION_VALUES[pulse_index]:.1f}"
        )

    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main():
    brainstate.random.seed(0)
    delay_phase_check()
    coupling, delay, perturbation, tags, grid_shape = make_conditions()
    run_sweep = brainstate.transform.vmap(run_condition)
    lfp, x1 = run_sweep(coupling, delay, perturbation)
    recruited, onsets, peak_x1 = classify(lfp, x1)

    recruited_np = np.asarray(recruited)
    if not (
        recruited_np[-2, 0]
        and not recruited_np[-2, 1:].any()
        and not recruited_np[-1].any()
    ):
        raise RuntimeError("mechanism controls failed")

    local_index, recruited_index = choose_examples(
        tags, coupling, recruited, onsets
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_results(
        lfp,
        coupling,
        delay,
        perturbation,
        recruited,
        np.asarray(onsets),
        grid_shape,
        local_index,
        recruited_index,
    )

    n_grid = int(np.prod(grid_shape))
    example_indices = jnp.asarray([local_index, recruited_index])
    recruited_grid = np.asarray(recruited[:n_grid]).reshape(grid_shape + (3,))
    neighbor_count = recruited_grid[..., 1:].sum(axis=-1)
    np.savez_compressed(
        RESULT_PATH,
        schema_version=np.asarray("1.0"),
        model=np.asarray("brainmass.EpileptorStep"),
        region_labels=REGION_LABELS,
        connectivity=np.asarray(CONNECTIVITY),
        grid_shape=np.asarray(grid_shape),
        coupling_axis=np.asarray(COUPLING_VALUES),
        delay_axis_ms=np.asarray(DELAY_VALUES.to_decimal(u.ms)),
        perturbation_axis=np.asarray(PERTURBATION_VALUES),
        condition_tag=tags,
        condition_coupling=np.asarray(coupling),
        condition_delay_ms=np.asarray(delay.to_decimal(u.ms)),
        condition_perturbation=np.asarray(perturbation),
        recruited=np.asarray(recruited),
        onset_ms=np.asarray(onsets),
        peak_x1=np.asarray(peak_x1),
        neighbor_count=neighbor_count,
        regime_code=np.where(recruited_grid[..., 0], neighbor_count, -1),
        regime_code_labels=np.asarray(["-1=no burst", "0=local", "1/2=neighbors recruited"]),
        example_indices=np.asarray(example_indices),
        example_lfp=np.asarray(u.get_magnitude(lfp[example_indices])),
        example_x1=np.asarray(u.get_magnitude(x1[example_indices])),
        example_time_ms=(np.arange(lfp.shape[1]) + 1) * DT.to_decimal(u.ms),
        dt_ms=np.asarray(DT.to_decimal(u.ms)),
        duration_ms=np.asarray(DURATION.to_decimal(u.ms)),
        stimulus_start_ms=np.asarray(STIMULUS_START.to_decimal(u.ms)),
        stimulus_duration_ms=np.asarray(STIMULUS_DURATION.to_decimal(u.ms)),
        max_delay_ms=np.asarray(MAX_DELAY.to_decimal(u.ms)),
        x0=np.asarray([-2.05, -2.4, -2.4]),
        Kvf=np.asarray(1.0),
        Ks=np.asarray(0.2),
        lfp_threshold=np.asarray(LFP_THRESHOLD),
        minimum_burst_duration_ms=np.asarray(
            MINIMUM_BURST_DURATION.to_decimal(u.ms)
        ),
        monitor_phase=np.asarray("post-update; sample 0 is dt"),
        delay_prehistory=np.asarray("0.0"),
        seed=np.asarray(0),
    )

    print(f"local example condition: {local_index}")
    print(f"recruited example condition: {recruited_index}")
    print(f"saved {RESULT_PATH}")
    print(f"saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
