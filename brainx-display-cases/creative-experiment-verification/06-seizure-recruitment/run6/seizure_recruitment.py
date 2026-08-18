"""Map local and propagating seizure-like bursts in a three-region chain.

This is a deterministic, phenomenological Epileptor experiment. Region 0
receives a finite perturbation; directed diffusive coupling can then recruit
regions 1 and 2 after a configurable propagation delay.
"""

from __future__ import annotations

import json
from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_VERSION = "1.1"
REGION_LABELS = np.asarray(["Focus", "Neighbor 1", "Neighbor 2"])

# BrainUnit owns all physical protocol values. Epileptor states and their
# coupling inputs are dimensionless, so those quantities use u.UNITLESS.
DT = 0.2 * u.ms
DURATION = 1200.0 * u.ms
FAST_TIME_CONSTANT = 10.0 * u.ms
ULTRASLOW_TIME_CONSTANT = 100.0 * u.ms
PERMITTIVITY_TIME_CONSTANT = 500.0 * u.ms
STIMULUS_START = 100.0 * u.ms
STIMULUS_DURATION = 100.0 * u.ms
MAX_DELAY = 40.0 * u.ms
MINIMUM_BURST_DURATION = 20.0 * u.ms

COUPLING_STRENGTHS = jnp.asarray([0.0, 5.0, 10.0, 20.0]) * u.UNITLESS
PROPAGATION_DELAYS = jnp.asarray([5.0, 20.0, 40.0]) * u.ms
PERTURBATION_SIZES = jnp.asarray([4.0, 7.0, 10.0]) * u.UNITLESS

# W[target, source]: Focus -> Neighbor 1 -> Neighbor 2.
CONNECTIVITY = jnp.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

N_STEPS = int(DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms))
STIMULUS_START_STEP = int(STIMULUS_START.to_decimal(u.ms) / DT.to_decimal(u.ms))
STIMULUS_STOP_STEP = int(
    (STIMULUS_START + STIMULUS_DURATION).to_decimal(u.ms)
    / DT.to_decimal(u.ms)
)
MINIMUM_BURST_STEPS = int(
    MINIMUM_BURST_DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms)
)


class DelayedEpileptorChain(brainstate.nn.Module):
    """Three healthy Epileptor regions with fixed-capacity delayed coupling."""

    def __init__(self) -> None:
        super().__init__()
        self.regions = brainmass.EpileptorStep(
            in_size=3,
            x0=jnp.full(3, -2.4),
            Kvf=1.0,
            Ks=0.0,
            tt=1.0 * u.ms / FAST_TIME_CONSTANT,
            r=FAST_TIME_CONSTANT / PERMITTIVITY_TIME_CONSTANT,
            tau=ULTRASLOW_TIME_CONSTANT / FAST_TIME_CONSTANT,
            init_x1=braintools.init.Constant(-1.5),
        )
        self.history = brainstate.nn.Delay(
            jnp.full(3, -1.5),
            time=MAX_DELAY,
            init=braintools.init.Constant(-1.5),
        )

    def update(self, stimulus, coupling_strength, propagation_delay):
        self.history.update(self.regions.x1.value)
        delay_steps = jnp.rint(propagation_delay / DT).astype(jnp.int32)
        delayed = self.history.retrieve_at_step(delay_steps)
        sources = jnp.broadcast_to(delayed, CONNECTIVITY.shape)
        coupling = brainmass.diffusive_coupling(
            sources,
            self.regions.x1.value,
            CONNECTIVITY,
            coupling_strength,
        )
        self.regions(coupling + stimulus)
        return self.regions.x1.value, self.regions.lfp()


def run_condition(coupling_strength, propagation_delay, perturbation_size):
    """Run one complete independent condition with a BrainState time loop."""
    with brainstate.environ.context(dt=DT):
        model = DelayedEpileptorChain()
        brainstate.nn.init_all_states(model)

        def step(index):
            stimulated = (index >= STIMULUS_START_STEP) & (
                index < STIMULUS_STOP_STEP
            )
            stimulus = jnp.asarray([stimulated, False, False]) * perturbation_size
            with brainstate.environ.context(i=index, t=index * DT):
                return model(stimulus, coupling_strength, propagation_delay)

        return brainstate.transform.for_loop(step, jnp.arange(N_STEPS))


def classify_bursts(x1):
    """Apply the fixed sustained-burst and routed-propagation predicate."""
    above = x1 > 0.0
    window_hits = jax.lax.reduce_window(
        above.astype(jnp.int32),
        0,
        jax.lax.add,
        (1, MINIMUM_BURST_STEPS, 1),
        (1, 1, 1),
        "VALID",
    )
    qualifying = window_hits == MINIMUM_BURST_STEPS
    recruited = jnp.any(qualifying, axis=1)
    first_start = jnp.argmax(qualifying, axis=1)
    onset_ms = jnp.where(
        recruited,
        (first_start + 1) * DT.to_decimal(u.ms),
        jnp.nan,
    )
    max_above_in_window_ms = (
        jnp.max(window_hits, axis=1) * DT.to_decimal(u.ms)
    )
    routed = (
        jnp.all(recruited, axis=1)
        & (onset_ms[:, 0] < onset_ms[:, 1])
        & (onset_ms[:, 1] < onset_ms[:, 2])
    )
    return recruited, onset_ms, max_above_in_window_ms, routed


def condition_table():
    """Build the flattened grid and matched causal controls."""
    coupling_grid_raw, delay_grid_raw, perturbation_grid_raw = jnp.meshgrid(
        COUPLING_STRENGTHS.to_decimal(u.UNITLESS),
        PROPAGATION_DELAYS.to_decimal(u.ms),
        PERTURBATION_SIZES.to_decimal(u.UNITLESS),
        indexing="ij",
    )
    coupling_grid = coupling_grid_raw * u.UNITLESS
    delay_grid = delay_grid_raw * u.ms
    perturbation_grid = perturbation_grid_raw * u.UNITLESS
    flat_coupling = coupling_grid.reshape(-1)
    flat_delay = delay_grid.reshape(-1)
    flat_perturbation = perturbation_grid.reshape(-1)

    # Controls use the strongest, shortest-delay grid condition and change
    # exactly one causal input: coupling or focal perturbation.
    flat_coupling = u.math.concatenate(
        [flat_coupling, jnp.asarray([0.0, 20.0]) * u.UNITLESS]
    )
    flat_delay = u.math.concatenate(
        [flat_delay, jnp.asarray([5.0, 5.0]) * u.ms]
    )
    flat_perturbation = u.math.concatenate(
        [flat_perturbation, jnp.asarray([10.0, 0.0]) * u.UNITLESS]
    )
    tags = np.asarray(
        ["grid"] * coupling_grid.size + ["no_coupling", "no_stimulus"]
    )
    return flat_coupling, flat_delay, flat_perturbation, tags


def save_results(
    output_dir,
    coupling,
    delay,
    perturbation,
    tags,
    x1,
    lfp,
    recruited,
    onset_ms,
    max_above_in_window_ms,
    routed,
):
    recruited_count = recruited.sum(axis=1)
    regime_label = np.where(
        routed,
        "routed_recruitment",
        np.where(
            recruited_count == 1,
            "local_burst",
            np.where(recruited_count == 0, "no_sustained_burst", "partial_recruitment"),
        ),
    )
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "code_version": f"seizure_recruitment.py:{ARTIFACT_VERSION}",
        "model": "brainmass.EpileptorStep",
        "model_scale": "regional neural mass",
        "regional_x0": [-2.4, -2.4, -2.4],
        "epileptor_input_gains": {"Kvf": 1.0, "Ks": 0.0},
        "coupling": "directed diffusive x1 coupling",
        "coupling_strength_unit": "1",
        "connectivity_convention": "W[target, source]",
        "dt_ms": float(DT.to_decimal(u.ms)),
        "duration_ms": float(DURATION.to_decimal(u.ms)),
        "fast_time_constant_ms": float(FAST_TIME_CONSTANT.to_decimal(u.ms)),
        "ultraslow_time_constant_ms": float(
            ULTRASLOW_TIME_CONSTANT.to_decimal(u.ms)
        ),
        "permittivity_time_constant_ms": float(
            PERMITTIVITY_TIME_CONSTANT.to_decimal(u.ms)
        ),
        "stimulus_start_ms": float(STIMULUS_START.to_decimal(u.ms)),
        "stimulus_duration_ms": float(STIMULUS_DURATION.to_decimal(u.ms)),
        "perturbation_size_unit": "1",
        "maximum_delay_ms": float(MAX_DELAY.to_decimal(u.ms)),
        "delay_prehistory_x1": -1.5,
        "delay_phase": "insert current x1, then retrieve step offset",
        "monitor_phase": "post-update; sample zero is at one dt",
        "burst_observable": "x1",
        "burst_direction": "x1 > threshold",
        "burst_threshold": 0.0,
        "minimum_burst_duration_ms": float(
            MINIMUM_BURST_DURATION.to_decimal(u.ms)
        ),
        "route_predicate": "Focus onset < Neighbor 1 onset < Neighbor 2 onset",
        "integration_method": "Epileptor default exp_euler",
        "seed": None,
        "calibration_status": "deterministic phenomenological demonstration",
    }
    np.savez_compressed(
        output_dir / "seizure_recruitment_results.npz",
        metadata_json=np.asarray(json.dumps(metadata, indent=2)),
        region_labels=REGION_LABELS,
        condition_tags=tags,
        coupling_strength=np.asarray(u.get_magnitude(coupling)),
        propagation_delay_ms=np.asarray(delay.to_decimal(u.ms)),
        perturbation_size=np.asarray(u.get_magnitude(perturbation)),
        times_ms=(np.arange(N_STEPS) + 1) * DT.to_decimal(u.ms),
        connectivity=np.asarray(CONNECTIVITY),
        x1=np.asarray(x1),
        lfp=np.asarray(lfp),
        recruited=np.asarray(recruited),
        recruited_count=np.asarray(recruited_count),
        regime_label=np.asarray(regime_label),
        onset_ms=np.asarray(onset_ms),
        max_above_in_window_ms=np.asarray(max_above_in_window_ms),
        routed_propagation=np.asarray(routed),
    )


def select_examples(tags, recruited, routed):
    is_grid = tags == "grid"
    local_candidates = np.flatnonzero(
        is_grid & recruited[:, 0] & (recruited.sum(axis=1) == 1)
    )
    routed_candidates = np.flatnonzero(is_grid & routed)
    if local_candidates.size == 0 or routed_candidates.size == 0:
        raise RuntimeError(
            "The fixed grid did not produce both a local and an ordered "
            "recruitment example; inspect continuous evidence before changing "
            "the predeclared predicate."
        )
    return int(local_candidates[-1]), int(routed_candidates[0])


def plot_results(
    output_dir,
    coupling,
    delay,
    perturbation,
    tags,
    x1,
    recruited,
    onset_ms,
    routed,
):
    local_index, routed_index = select_examples(tags, recruited, routed)
    grid_shape = (
        COUPLING_STRENGTHS.shape[0],
        PROPAGATION_DELAYS.shape[0],
        PERTURBATION_SIZES.shape[0],
    )
    grid_count = recruited[tags == "grid"].sum(axis=1).reshape(grid_shape)
    coupling_values = np.asarray(COUPLING_STRENGTHS.to_decimal(u.UNITLESS))
    delay_values = np.asarray(PROPAGATION_DELAYS.to_decimal(u.ms))
    perturbation_values = np.asarray(
        PERTURBATION_SIZES.to_decimal(u.UNITLESS)
    )
    times_ms = (np.arange(N_STEPS) + 1) * DT.to_decimal(u.ms)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    heatmap = None
    for perturbation_index, axis in enumerate(axes[0]):
        heatmap = axis.imshow(
            grid_count[:, :, perturbation_index].T,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=3,
            cmap="RdYlBu_r",
        )
        axis.set_xticks(np.arange(coupling_values.size), coupling_values)
        axis.set_yticks(np.arange(delay_values.size), delay_values)
        axis.set_xlabel("Coupling strength (1)")
        axis.set_ylabel("Delay (ms)")
        axis.set_title(f"Perturbation {perturbation_values[perturbation_index]:g}")
    colorbar = fig.colorbar(heatmap, ax=axes[0], shrink=0.82, pad=0.02)
    colorbar.set_label("Regions with sustained burst")
    colorbar.set_ticks([0, 1, 2, 3])

    colors = ["#c43d3d", "#16827d", "#d09418"]
    for axis, condition_index, title in (
        (axes[1, 0], local_index, "Burst remains local"),
        (axes[1, 1], routed_index, "Neighbors are recruited"),
    ):
        for region_index, (label, color) in enumerate(zip(REGION_LABELS, colors)):
            axis.plot(
                times_ms,
                x1[condition_index, :, region_index],
                label=label,
                color=color,
                linewidth=1.1,
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.axvspan(
            STIMULUS_START.to_decimal(u.ms),
            (STIMULUS_START + STIMULUS_DURATION).to_decimal(u.ms),
            color="#dddddd",
            zorder=-1,
        )
        axis.set_xlim(0.0, DURATION.to_decimal(u.ms))
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Fast activity x1 (1)")
        axis.set_title(title)
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper right")

    routed_grid = np.flatnonzero((tags == "grid") & routed)
    for condition_index, color in zip(routed_grid, ["#16827d", "#d09418", "#c43d3d"]):
        axes[1, 2].plot(
            np.arange(3),
            onset_ms[condition_index],
            color=color,
            marker="o",
            linewidth=1.5,
            label=f"{delay[condition_index].to_decimal(u.ms):g} ms delay",
        )
    axes[1, 2].set_xticks(np.arange(3), REGION_LABELS, rotation=15)
    axes[1, 2].set_ylabel("Sustained-burst onset (ms)")
    axes[1, 2].set_title("Delay shifts recruitment timing")
    axes[1, 2].grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[1, 2].legend(frameon=False, fontsize=8)

    local_description = (
        f"k={u.get_magnitude(coupling[local_index]):g}, "
        f"delay={delay[local_index].to_decimal(u.ms):g} ms, "
        f"perturbation={u.get_magnitude(perturbation[local_index]):g}"
    )
    routed_description = (
        f"k={u.get_magnitude(coupling[routed_index]):g}, "
        f"delay={delay[routed_index].to_decimal(u.ms):g} ms, "
        f"perturbation={u.get_magnitude(perturbation[routed_index]):g}"
    )
    axes[1, 0].set_title(f"Burst remains local\n{local_description}")
    axes[1, 1].set_title(f"Neighbors are recruited\n{routed_description}")
    fig.suptitle("Seizure-like recruitment in a directed three-region chain")
    fig.savefig(output_dir / "seizure_recruitment.png", dpi=180)
    plt.close(fig)
    return local_index, routed_index


def verify_delay_phase():
    """Lock the insert-then-retrieve convention with a unit-aware impulse."""
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
    np.testing.assert_array_equal(
        np.asarray(observed), np.asarray([0.0, 0.0, 0.0, 1.0, 0.0])
    )


def main():
    output_dir = Path(__file__).with_name("outputs")
    output_dir.mkdir(exist_ok=True)
    verify_delay_phase()

    coupling, delay, perturbation, tags = condition_table()
    # vmap owns independent parameter conditions; for_loop inside each mapped
    # call owns time and all Epileptor/Delay State transitions.
    x1, lfp = brainstate.transform.vmap(run_condition)(
        coupling, delay, perturbation
    )
    x1 = jax.block_until_ready(x1)
    lfp = jax.block_until_ready(lfp)
    recruited, onset_ms, max_above_in_window_ms, routed = classify_bursts(x1)
    recruited, onset_ms, max_above_in_window_ms, routed = jax.block_until_ready(
        (recruited, onset_ms, max_above_in_window_ms, routed)
    )

    recruited_np = np.asarray(recruited)
    onset_np = np.asarray(onset_ms)
    routed_np = np.asarray(routed)
    grid_mask = tags == "grid"
    no_coupling_index = int(np.flatnonzero(tags == "no_coupling")[0])
    no_stimulus_index = int(np.flatnonzero(tags == "no_stimulus")[0])
    assert np.any(grid_mask & (recruited_np.sum(axis=1) == 1))
    assert np.any(grid_mask & routed_np)
    np.testing.assert_array_equal(
        recruited_np[no_coupling_index], np.asarray([True, False, False])
    )
    np.testing.assert_array_equal(
        recruited_np[no_stimulus_index], np.asarray([False, False, False])
    )
    save_results(
        output_dir,
        coupling,
        delay,
        perturbation,
        tags,
        x1,
        lfp,
        recruited_np,
        onset_np,
        np.asarray(max_above_in_window_ms),
        routed_np,
    )
    local_index, routed_index = plot_results(
        output_dir,
        coupling,
        delay,
        perturbation,
        tags,
        np.asarray(x1),
        recruited_np,
        onset_np,
        routed_np,
    )

    print(f"Mapped {np.count_nonzero(tags == 'grid')} grid conditions + 2 controls")
    print(
        "Local example: "
        f"k={coupling[local_index]}, delay={delay[local_index]}, "
        f"perturbation={perturbation[local_index]}, "
        f"recruited={recruited_np[local_index].tolist()}"
    )
    print(
        "Recruited example: "
        f"k={coupling[routed_index]}, delay={delay[routed_index]}, "
        f"perturbation={perturbation[routed_index]}, "
        f"onsets_ms={onset_np[routed_index].tolist()}"
    )
    for control in ("no_coupling", "no_stimulus"):
        index = int(np.flatnonzero(tags == control)[0])
        print(f"{control}: recruited={recruited_np[index].tolist()}")
    print(f"Saved {output_dir / 'seizure_recruitment.png'}")
    print(f"Saved {output_dir / 'seizure_recruitment_results.npz'}")


if __name__ == "__main__":
    main()
