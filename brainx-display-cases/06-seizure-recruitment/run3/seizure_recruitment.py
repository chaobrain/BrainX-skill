"""Map local versus propagating seizure recruitment in a regional chain.

The model uses three healthy Epileptor neural masses. A finite perturbation is
applied only to region 1, and delayed directed coupling can recruit regions 2
and 3. Coupling strength and perturbation amplitude are dimensionless in the
Epileptor equations; all experimental and propagation times carry BrainUnit
units until plotting and serialization boundaries.
"""

from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


ARTIFACT_VERSION = "seizure-recruitment-v1"
OUTPUT_DIR = Path(__file__).with_name("outputs")

N_REGIONS = 3
DT = 0.1 * u.ms
DURATION = 1500.0 * u.ms
STIMULUS_START = 100.0 * u.ms
STIMULUS_STOP = 300.0 * u.ms
MAX_DELAY = 10.0 * u.ms

# W[target, source]: region 1 -> region 2 -> region 3.
CONNECTIVITY = jnp.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
COUPLING_STRENGTHS = jnp.asarray([0.0, 0.25, 0.40, 0.50, 0.70])
PROPAGATION_DELAYS_MS = jnp.asarray([2.0, 6.0, 10.0])
PERTURBATION_SIZES = jnp.asarray([1.0, 1.5, 2.0])

# Fixed before evaluating the sweep. The Epileptor reference case identifies
# positive x1 as seizure-like activity; here it must persist for at least 1 ms.
EVENT_THRESHOLD = 0.0
MINIMUM_EVENT_DURATION = 1.0 * u.ms


def _steps(duration: u.Quantity["time"]) -> int:
    return int(round(float(duration.to_decimal(u.ms) / DT.to_decimal(u.ms))))


class DrivenEpileptorChain(brainstate.nn.Module):
    """Three regional Epileptor masses with fixed-capacity delayed coupling."""

    def __init__(self):
        super().__init__()
        self.node = brainmass.EpileptorStep(
            in_size=N_REGIONS,
            x0=jnp.full(N_REGIONS, -2.4),
            Kvf=1.0,
            Ks=0.2,
        )
        self.history = brainstate.nn.Delay(
            jnp.zeros(N_REGIONS),
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
        self.node.update(network_input + stimulus)
        return self.node.x1.value, self.node.lfp()


def run_condition(coupling_strength, propagation_delay, perturbation_size):
    """Run one independent stateful condition; vmap owns this condition axis."""

    with brainstate.environ.context(dt=DT):
        model = DrivenEpileptorChain()
        brainstate.nn.init_all_states(model)
        indices = jnp.arange(_steps(DURATION))
        stimulus_start = _steps(STIMULUS_START)
        stimulus_stop = _steps(STIMULUS_STOP)

        def step(index):
            pulse_on = (index >= stimulus_start) & (index < stimulus_stop)
            stimulus = jnp.asarray(
                [jnp.where(pulse_on, perturbation_size, 0.0), 0.0, 0.0]
            )
            with brainstate.environ.context(i=index, t=index * DT):
                return model(stimulus, coupling_strength, propagation_delay)

        return brainstate.transform.for_loop(step, indices)


def classify_events(x1):
    """Apply the fixed sustained-event predicate to [condition, time, region]."""

    minimum_steps = _steps(MINIMUM_EVENT_DURATION)

    def classify_one(activity):
        above = activity > EVENT_THRESHOLD
        window_hits = jax.lax.reduce_window(
            above.astype(jnp.int32),
            0,
            jax.lax.add,
            (minimum_steps, 1),
            (1, 1),
            "VALID",
        )
        qualifying = window_hits == minimum_steps
        recruited = jnp.any(qualifying, axis=0)
        first_start = jnp.argmax(qualifying, axis=0)
        # Monitors are post-update, so sample zero is observed at one dt.
        onset_steps = jnp.where(recruited, first_start + 1, jnp.nan)
        return (
            recruited,
            onset_steps * DT,
            jnp.max(window_hits, axis=0),
            jnp.max(activity, axis=0),
        )

    return jax.vmap(classify_one)(x1)


def regime_codes(recruited, onset_ms):
    """Encode no event, local, partial, ordered, or unordered recruitment."""

    count = recruited.sum(axis=1)
    ordered = (
        recruited.all(axis=1)
        & (onset_ms[:, 0] < onset_ms[:, 1])
        & (onset_ms[:, 1] < onset_ms[:, 2])
    )
    codes = np.full(recruited.shape[0], 4, dtype=np.int8)
    codes[count == 0] = 0
    codes[(recruited[:, 0]) & (count == 1)] = 1
    codes[(recruited[:, 0]) & (count == 2)] = 2
    codes[ordered] = 3
    return codes, count.astype(np.int8)


def check_delay_convention():
    """Lock insertion/retrieval order with a three-step impulse check."""

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
        raise AssertionError(f"delay phase check failed: {observed}")


def build_conditions():
    """Return the ordered sweep grid followed by two matched controls."""

    amplitude, delay_ms, coupling = jnp.meshgrid(
        PERTURBATION_SIZES,
        PROPAGATION_DELAYS_MS,
        COUPLING_STRENGTHS,
        indexing="ij",
    )
    grid_shape = amplitude.shape
    coupling = coupling.reshape(-1)
    delay_ms = delay_ms.reshape(-1)
    amplitude = amplitude.reshape(-1)
    n_grid = coupling.size

    # Controls use the same mapped path as the intervention conditions.
    coupling = jnp.concatenate([coupling, jnp.asarray([0.0, 0.70])])
    delay_ms = jnp.concatenate([delay_ms, jnp.asarray([6.0, 6.0])])
    amplitude = jnp.concatenate([amplitude, jnp.asarray([2.0, 0.0])])
    tags = np.asarray(["grid"] * n_grid + ["no_coupling", "no_drive"])
    return coupling, delay_ms, amplitude, tags, grid_shape, n_grid


def _grid_index(amplitude_index, delay_index, coupling_index, grid_shape):
    return int(np.ravel_multi_index(
        (amplitude_index, delay_index, coupling_index), grid_shape
    ))


def validate_results(regimes, recruited, onset_ms, grid_shape, n_grid):
    local_index = _grid_index(2, 1, 1, grid_shape)  # size=2, delay=6 ms, k=0.25
    spread_index = _grid_index(2, 1, 3, grid_shape)  # size=2, delay=6 ms, k=0.50
    no_coupling_index = n_grid
    no_drive_index = n_grid + 1

    if regimes[local_index] != 1:
        raise AssertionError("calibrated local case no longer remains local")
    if regimes[spread_index] != 3:
        raise AssertionError("calibrated spreading case lost ordered recruitment")
    if regimes[no_coupling_index] != 1:
        raise AssertionError("no-coupling control should burst only in region 1")
    if regimes[no_drive_index] != 0:
        raise AssertionError("no-drive control should remain quiescent")
    if not np.all(np.diff(onset_ms[spread_index]) > 0.0):
        raise AssertionError("recruitment onsets are not strictly route ordered")
    if not np.array_equal(recruited[local_index], [True, False, False]):
        raise AssertionError("local case has an unexpected recruitment mask")
    return local_index, spread_index


def save_bundle(
    x1,
    lfp,
    coupling,
    delay_ms,
    amplitude,
    tags,
    grid_shape,
    recruited,
    onset_ms,
    max_window_hits,
    max_x1,
    regimes,
    recruited_count,
):
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_path = OUTPUT_DIR / "seizure_recruitment_results.npz"
    time_ms = (np.arange(x1.shape[1]) + 1) * float(DT.to_decimal(u.ms))
    np.savez_compressed(
        result_path,
        artifact_version=ARTIFACT_VERSION,
        model="brainmass.EpileptorStep",
        interpretation="phenomenological demonstration; not patient calibrated",
        condition_tag=tags,
        coupling_strength=np.asarray(coupling),
        propagation_delay_ms=np.asarray(delay_ms),
        perturbation_size=np.asarray(amplitude),
        grid_shape=np.asarray(grid_shape),
        grid_coupling_strength=np.asarray(COUPLING_STRENGTHS),
        grid_propagation_delay_ms=np.asarray(PROPAGATION_DELAYS_MS),
        grid_perturbation_size=np.asarray(PERTURBATION_SIZES),
        connectivity=np.asarray(CONNECTIVITY),
        x0=np.full(N_REGIONS, -2.4),
        Kvf=1.0,
        Ks=0.2,
        dt_ms=float(DT.to_decimal(u.ms)),
        duration_ms=float(DURATION.to_decimal(u.ms)),
        stimulus_start_ms=float(STIMULUS_START.to_decimal(u.ms)),
        stimulus_stop_ms=float(STIMULUS_STOP.to_decimal(u.ms)),
        event_threshold=EVENT_THRESHOLD,
        minimum_event_duration_ms=float(
            MINIMUM_EVENT_DURATION.to_decimal(u.ms)
        ),
        time_ms=time_ms,
        x1=x1,
        lfp=lfp,
        recruited=recruited,
        onset_ms=onset_ms,
        maximum_event_window_hits=max_window_hits,
        maximum_x1=max_x1,
        regime_code=regimes,
        regime_labels=np.asarray(
            ["no burst", "local", "partial", "ordered recruitment", "unordered"]
        ),
        recruited_region_count=recruited_count,
    )
    return result_path, time_ms


def plot_results(
    x1,
    time_ms,
    onset_ms,
    recruited_count,
    grid_shape,
    n_grid,
    local_index,
    spread_index,
):
    figure_path = OUTPUT_DIR / "seizure_recruitment.png"
    colors = ListedColormap(["#f2f2ed", "#e9c46a", "#ef8354", "#b7094c"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colors.N)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), constrained_layout=True)

    count_grid = recruited_count[:n_grid].reshape(grid_shape)
    for amplitude_index, ax in enumerate(axes[0]):
        image = ax.imshow(
            count_grid[amplitude_index],
            origin="lower",
            aspect="auto",
            cmap=colors,
            norm=norm,
        )
        for delay_index in range(PROPAGATION_DELAYS_MS.size):
            for coupling_index in range(COUPLING_STRENGTHS.size):
                ax.text(
                    coupling_index,
                    delay_index,
                    str(count_grid[amplitude_index, delay_index, coupling_index]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        ax.set_title(
            f"Perturbation size {float(PERTURBATION_SIZES[amplitude_index]):.1f}"
        )
        ax.set_xticks(
            np.arange(COUPLING_STRENGTHS.size),
            [f"{value:.2g}" for value in np.asarray(COUPLING_STRENGTHS)],
        )
        ax.set_yticks(
            np.arange(PROPAGATION_DELAYS_MS.size),
            [f"{value:.0f}" for value in np.asarray(PROPAGATION_DELAYS_MS)],
        )
        ax.set_xlabel("Coupling strength")
    axes[0, 0].set_ylabel("Propagation delay (ms)")
    colorbar = fig.colorbar(image, ax=axes[0], ticks=[0, 1, 2, 3], shrink=0.82)
    colorbar.set_label("Regions with sustained burst")

    region_colors = ["#00798c", "#e76f51", "#7b2cbf"]
    for ax, condition_index, title in (
        (axes[1, 0], local_index, "Local: k = 0.25, delay = 6 ms"),
        (axes[1, 1], spread_index, "Recruited: k = 0.50, delay = 6 ms"),
    ):
        for region in range(N_REGIONS):
            ax.plot(
                time_ms,
                x1[condition_index, :, region],
                color=region_colors[region],
                linewidth=1.0,
                label=f"Region {region + 1}",
            )
            if np.isfinite(onset_ms[condition_index, region]):
                ax.axvline(
                    onset_ms[condition_index, region],
                    color=region_colors[region],
                    linewidth=0.8,
                    alpha=0.7,
                )
        ax.axhline(EVENT_THRESHOLD, color="#222222", linewidth=0.8, linestyle="--")
        ax.axvspan(
            float(STIMULUS_START.to_decimal(u.ms)),
            float(STIMULUS_STOP.to_decimal(u.ms)),
            color="#d9d9d9",
            alpha=0.45,
            linewidth=0,
        )
        ax.set_xlim(50.0, 400.0)
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Fast Epileptor state x1")
    axes[1, 0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")

    onset_grid = onset_ms[:n_grid].reshape(grid_shape + (N_REGIONS,))
    amplitude_index = 2
    for coupling_index in (3, 4):
        latency = (
            onset_grid[amplitude_index, :, coupling_index, 2]
            - onset_grid[amplitude_index, :, coupling_index, 0]
        )
        axes[1, 2].plot(
            np.asarray(PROPAGATION_DELAYS_MS),
            latency,
            marker="o",
            linewidth=1.5,
            label=f"k = {float(COUPLING_STRENGTHS[coupling_index]):.2f}",
        )
    axes[1, 2].set_title("Region 3 recruitment latency\n(perturbation size 2.0)")
    axes[1, 2].set_xlabel("Propagation delay (ms)")
    axes[1, 2].set_ylabel("Region 3 onset - region 1 onset (ms)")
    axes[1, 2].set_xticks(np.asarray(PROPAGATION_DELAYS_MS))
    axes[1, 2].legend(frameon=False)
    axes[1, 2].grid(axis="y", color="#dddddd", linewidth=0.6)

    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def main():
    brainstate.random.seed(0)
    brainstate.environ.set(dt=DT)
    check_delay_convention()
    coupling, delay_ms, amplitude, tags, grid_shape, n_grid = build_conditions()

    mapped_rollout = brainstate.transform.vmap(run_condition)
    (x1, lfp) = mapped_rollout(coupling, delay_ms * u.ms, amplitude)
    recruited, onset, max_window_hits, max_x1 = classify_events(x1)

    x1 = np.asarray(jax.device_get(u.get_magnitude(x1)))
    lfp = np.asarray(jax.device_get(u.get_magnitude(lfp)))
    recruited = np.asarray(jax.device_get(recruited))
    onset_ms = np.asarray(jax.device_get(onset.to_decimal(u.ms)))
    max_window_hits = np.asarray(jax.device_get(max_window_hits))
    max_x1 = np.asarray(jax.device_get(max_x1))
    if not np.isfinite(x1).all() or not np.isfinite(lfp).all():
        raise AssertionError("simulation produced non-finite activity")

    regimes, recruited_count = regime_codes(recruited, onset_ms)
    local_index, spread_index = validate_results(
        regimes, recruited, onset_ms, grid_shape, n_grid
    )
    result_path, time_ms = save_bundle(
        x1,
        lfp,
        coupling,
        delay_ms,
        amplitude,
        tags,
        grid_shape,
        recruited,
        onset_ms,
        max_window_hits,
        max_x1,
        regimes,
        recruited_count,
    )
    figure_path = plot_results(
        x1,
        time_ms,
        onset_ms,
        recruited_count,
        grid_shape,
        n_grid,
        local_index,
        spread_index,
    )

    print("Local case onset (ms):", onset_ms[local_index])
    print("Recruitment case onset (ms):", onset_ms[spread_index])
    print("No-coupling control:", regimes[n_grid], "(1 = local)")
    print("No-drive control:", regimes[n_grid + 1], "(0 = no burst)")
    print("Saved:", result_path)
    print("Saved:", figure_path)


if __name__ == "__main__":
    main()
