"""Map local and propagating seizure-like bursts across four brain regions.

The regional dynamics are phenomenological FitzHugh-Nagumo neural masses.  A
brief input is applied only to region 0; directed nearest-neighbor coupling can
then recruit regions 1-3 after a finite conduction delay.
"""

from __future__ import annotations

import csv
from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs")

N_REGIONS = 4
DT = 0.1 * u.ms
REGIONAL_TIME_CONSTANT = 12.5 * u.ms
MAX_DELAY = 12.0 * u.ms
SIMULATION_DURATION = 120.0 * u.ms
STIMULUS_ONSET = 10.0 * u.ms
STIMULUS_DURATION = 20.0 * u.ms

# FHN activity, its drive, and the coupling gain are dimensionless variables.
COUPLING_STRENGTHS = u.math.asarray([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
PROPAGATION_DELAYS = jnp.asarray([1.0, 4.0, 8.0, 12.0]) * u.ms
PERTURBATION_SIZES = u.math.asarray([0.3, 0.5, 0.7])
RECRUITMENT_THRESHOLD = 0.5

# BrainMass uses W[target, source].  This is a directed chain 0 -> 1 -> 2 -> 3.
CONNECTIVITY = jnp.eye(N_REGIONS, k=-1)


def quantity_to_steps(value: u.Quantity["time"]) -> int:
    """Convert a time quantity to an integer count at the fixed model dt."""
    return int(round(float(value / DT)))


N_STEPS = quantity_to_steps(SIMULATION_DURATION)
STIMULUS_START_STEP = quantity_to_steps(STIMULUS_ONSET)
STIMULUS_STOP_STEP = quantity_to_steps(STIMULUS_ONSET + STIMULUS_DURATION)


class DelayedRegionalChain(brainstate.nn.Module):
    """Four excitable BrainMass regions with a shared edge delay."""

    def __init__(self) -> None:
        super().__init__()
        self.node = brainmass.FitzHughNagumoStep(
            in_size=N_REGIONS,
            tau=REGIONAL_TIME_CONSTANT,
            init_V=braintools.init.Constant(0.0),
            init_w=braintools.init.Constant(0.0),
        )
        self.history = brainstate.nn.Delay(
            jnp.zeros(N_REGIONS),
            time=MAX_DELAY,
            init=braintools.init.Constant(0.0),
        )

    def update(self, stimulus, coupling_strength, propagation_delay):
        self.history.update(self.node.V.value)
        delay_steps = jnp.rint(propagation_delay / DT).astype(jnp.int32)
        delayed_activity = self.history.retrieve_at_step(delay_steps)
        sources = jnp.broadcast_to(delayed_activity, CONNECTIVITY.shape)
        regional_input = brainmass.additive_coupling(
            sources,
            CONNECTIVITY,
            coupling_strength,
        )
        return self.node(regional_input + stimulus)


def run_condition(coupling_strength, propagation_delay, perturbation_size):
    """Run one independent regional chain and return time x region activity."""
    with brainstate.environ.context(dt=DT):
        model = DelayedRegionalChain()
        brainstate.nn.init_all_states(model)

        def step(index):
            stimulus = jnp.where(
                (index >= STIMULUS_START_STEP) & (index < STIMULUS_STOP_STEP),
                jnp.asarray([perturbation_size, 0.0, 0.0, 0.0]),
                jnp.zeros(N_REGIONS),
            )
            with brainstate.environ.context(i=index, t=index * DT):
                return model(stimulus, coupling_strength, propagation_delay)

        return brainstate.transform.for_loop(step, jnp.arange(N_STEPS))


def make_parameter_grid():
    """Flatten the physical coordinates for one stateful vmap call."""
    delay_ms = PROPAGATION_DELAYS.to_decimal(u.ms)
    coupling, delay, perturbation = jnp.meshgrid(
        COUPLING_STRENGTHS,
        delay_ms,
        PERTURBATION_SIZES,
        indexing="ij",
    )
    return coupling, delay * u.ms, perturbation


def recruitment_metrics(traces):
    """Return continuous peaks plus thresholded recruitment and onset times."""
    peaks = jnp.max(traces, axis=1)
    above_threshold = traces >= RECRUITMENT_THRESHOLD
    was_recruited = jnp.any(above_threshold, axis=1)
    first_crossing = jnp.argmax(above_threshold, axis=1)
    onset_ms = jnp.where(
        was_recruited,
        first_crossing * DT.to_decimal(u.ms),
        jnp.nan,
    )
    return peaks, was_recruited, onset_ms


def verify_delay_phase() -> None:
    """Lock the insert-then-retrieve convention to an exact three-step delay."""
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


def write_long_form_csv(
    path,
    condition_types,
    coupling,
    delay,
    perturbation,
    peaks,
    recruited,
    onset_ms,
):
    """Store every continuous boundary observable with its physical coordinates."""
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "condition_type",
                "coupling_dimensionless",
                "delay_ms",
                "perturbation_dimensionless",
                "region",
                "peak_activity_dimensionless",
                "recruited",
                "onset_ms",
            ]
        )
        for condition in range(coupling.size):
            for region in range(N_REGIONS):
                writer.writerow(
                    [
                        condition_types[condition],
                        float(coupling[condition]),
                        float(delay[condition]),
                        float(perturbation[condition]),
                        region,
                        float(peaks[condition, region]),
                        bool(recruited[condition, region]),
                        float(onset_ms[condition, region]),
                    ]
                )


def condition_index(coupling_grid, delay_grid, perturbation_grid, k, delay_ms, amplitude):
    matches = (
        jnp.isclose(coupling_grid, k)
        & jnp.isclose(delay_grid.to_decimal(u.ms), delay_ms)
        & jnp.isclose(perturbation_grid, amplitude)
    )
    flat_matches = matches.reshape(-1)
    if not bool(jnp.any(flat_matches)):
        raise ValueError("requested condition is absent from the parameter grid")
    return int(jnp.argmax(flat_matches))


def plot_results(
    path,
    traces,
    coupling_grid,
    delay_grid,
    perturbation_grid,
    recruited,
    onset_ms,
):
    """Plot representative traces, onset timing, and the full regime sweep."""
    coupling_flat = coupling_grid.reshape(-1)
    delay_flat = delay_grid.reshape(-1)
    perturbation_flat = perturbation_grid.reshape(-1)
    local_index = condition_index(
        coupling_grid, delay_grid, perturbation_grid, 0.25, 4.0, 0.5
    )
    recruited_index = condition_index(
        coupling_grid, delay_grid, perturbation_grid, 0.55, 4.0, 0.5
    )
    times_ms = np.arange(N_STEPS) * DT.to_decimal(u.ms)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = ["#b3261e", "#1769aa", "#2e7d32", "#7b3f98"]
    for axis, index, title in (
        (axes[0, 0], local_index, "Local event: k=0.25, delay=4 ms, input=0.5"),
        (axes[0, 1], recruited_index, "Recruited chain: k=0.55, delay=4 ms, input=0.5"),
    ):
        for region, color in enumerate(colors):
            axis.plot(times_ms, traces[index, :, region], color=color, label=f"Region {region}")
        axis.axhline(RECRUITMENT_THRESHOLD, color="#555555", linestyle="--", linewidth=1)
        axis.axvspan(
            STIMULUS_ONSET.to_decimal(u.ms),
            (STIMULUS_ONSET + STIMULUS_DURATION).to_decimal(u.ms),
            color="#f2c14e",
            alpha=0.2,
        )
        axis.set_title(title)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Fast population activity V")
        axis.set_xlim(0.0, 80.0)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)

    timing_axis = axes[0, 2]
    high_coupling = (
        jnp.isclose(coupling_flat, 0.55) & jnp.isclose(perturbation_flat, 0.5)
    )
    for region, color in zip(range(1, N_REGIONS), colors[1:]):
        timing_axis.plot(
            delay_flat.to_decimal(u.ms)[high_coupling],
            onset_ms[high_coupling, region],
            marker="o",
            color=color,
            label=f"Region {region}",
        )
    timing_axis.set_title("Delay shifts recruitment onset")
    timing_axis.set_xlabel("Per-edge delay (ms)")
    timing_axis.set_ylabel("First threshold crossing (ms)")
    timing_axis.legend(frameon=False, fontsize=8)

    n_coupling = COUPLING_STRENGTHS.size
    n_delay = PROPAGATION_DELAYS.size
    n_perturbation = PERTURBATION_SIZES.size
    recruited_grid = recruited.reshape(n_coupling, n_delay, n_perturbation, N_REGIONS)
    neighbor_fraction = jnp.mean(recruited_grid[..., 1:], axis=-1)
    image = None
    for amplitude_index, axis in enumerate(axes[1]):
        image = axis.imshow(
            neighbor_fraction[:, :, amplitude_index],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.set_title(
            f"Input size {float(PERTURBATION_SIZES[amplitude_index]):.1f}"
        )
        axis.set_xlabel("Delay (ms)")
        axis.set_ylabel("Coupling k")
        axis.set_xticks(
            np.arange(n_delay),
            labels=[f"{value:g}" for value in PROPAGATION_DELAYS.to_decimal(u.ms)],
        )
        axis.set_yticks(
            np.arange(n_coupling),
            labels=[f"{float(value):.2f}" for value in COUPLING_STRENGTHS],
        )
    fig.colorbar(image, ax=axes[1].tolist(), label="Fraction of neighboring regions recruited")
    fig.suptitle("Seizure-like recruitment in an excitable regional mass chain", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    verify_delay_phase()
    coupling_grid, delay_grid, perturbation_grid = make_parameter_grid()
    n_grid_conditions = coupling_grid.size

    # Run mechanism controls as extra lanes in the same stateful mapping:
    # no coupling with stimulation, and strong coupling without stimulation.
    coupling = jnp.concatenate(
        [coupling_grid.reshape(-1), jnp.asarray([0.0, 0.55])]
    )
    delay = jnp.concatenate(
        [delay_grid.reshape(-1).to_decimal(u.ms), jnp.asarray([4.0, 4.0])]
    ) * u.ms
    perturbation = jnp.concatenate(
        [perturbation_grid.reshape(-1), jnp.asarray([0.5, 0.0])]
    )
    run_sweep = brainstate.transform.vmap(run_condition)
    traces = run_sweep(coupling, delay, perturbation)
    jax.block_until_ready(traces)
    peaks, recruited, onset_ms = recruitment_metrics(traces)
    grid_recruited = recruited[:n_grid_conditions]
    grid_onset_ms = onset_ms[:n_grid_conditions]

    if not bool(jnp.all(grid_recruited[:, 0])):
        raise AssertionError("the perturbation did not trigger region 0 in every condition")

    local_index = condition_index(
        coupling_grid, delay_grid, perturbation_grid, 0.25, 4.0, 0.5
    )
    spread_index = condition_index(
        coupling_grid, delay_grid, perturbation_grid, 0.55, 4.0, 0.5
    )
    if bool(jnp.any(grid_recruited[local_index, 1:])):
        raise AssertionError("the declared local control recruited a neighbor")
    if not bool(jnp.all(grid_recruited[spread_index])):
        raise AssertionError("the declared spread case failed to recruit the chain")
    spread_onsets = grid_onset_ms[spread_index]
    if not bool(jnp.all(jnp.diff(spread_onsets) > 0.0)):
        raise AssertionError("recruitment did not follow source-to-neighbor order")
    no_coupling_index = n_grid_conditions
    no_perturbation_index = n_grid_conditions + 1
    if bool(jnp.any(recruited[no_coupling_index, 1:])):
        raise AssertionError("neighbors recruited without inter-region coupling")
    if bool(jnp.any(recruited[no_perturbation_index])):
        raise AssertionError("an event appeared without the initiating perturbation")

    OUTPUT_DIR.mkdir(exist_ok=True)
    flat_coupling = np.asarray(coupling)
    flat_delay_ms = np.asarray(delay.to_decimal(u.ms))
    flat_perturbation = np.asarray(perturbation)
    traces_np = np.asarray(traces)
    peaks_np = np.asarray(peaks)
    recruited_np = np.asarray(recruited)
    onset_ms_np = np.asarray(onset_ms)
    condition_types = np.asarray(
        ["grid"] * n_grid_conditions + ["no_coupling", "no_perturbation"]
    )
    np.savez_compressed(
        OUTPUT_DIR / "seizure_recruitment_data.npz",
        model="FitzHughNagumoStep",
        condition_type=condition_types,
        coupling=flat_coupling,
        delay_ms=flat_delay_ms,
        perturbation=flat_perturbation,
        time_ms=np.arange(N_STEPS) * DT.to_decimal(u.ms),
        traces=traces_np,
        peaks=peaks_np,
        recruited=recruited_np,
        onset_ms=onset_ms_np,
        threshold=RECRUITMENT_THRESHOLD,
        dt_ms=DT.to_decimal(u.ms),
        regional_time_constant_ms=REGIONAL_TIME_CONSTANT.to_decimal(u.ms),
        maximum_delay_ms=MAX_DELAY.to_decimal(u.ms),
        simulation_duration_ms=SIMULATION_DURATION.to_decimal(u.ms),
        stimulus_onset_ms=STIMULUS_ONSET.to_decimal(u.ms),
        stimulus_duration_ms=STIMULUS_DURATION.to_decimal(u.ms),
        connectivity=np.asarray(CONNECTIVITY),
    )
    write_long_form_csv(
        OUTPUT_DIR / "seizure_recruitment_metrics.csv",
        condition_types,
        flat_coupling,
        flat_delay_ms,
        flat_perturbation,
        peaks_np,
        recruited_np,
        onset_ms_np,
    )
    plot_results(
        OUTPUT_DIR / "seizure_recruitment.png",
        traces_np[:n_grid_conditions],
        coupling_grid,
        delay_grid,
        perturbation_grid,
        recruited_np[:n_grid_conditions],
        onset_ms_np[:n_grid_conditions],
    )

    local_peaks = peaks_np[local_index]
    spread_peaks = peaks_np[spread_index]
    print("Local case peaks:", np.round(local_peaks, 3))
    print("Recruited case peaks:", np.round(spread_peaks, 3))
    print("Recruited case onset times (ms):", np.round(onset_ms_np[spread_index], 1))
    print("Mechanism controls passed: no coupling; no perturbation")
    print(f"Wrote results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
