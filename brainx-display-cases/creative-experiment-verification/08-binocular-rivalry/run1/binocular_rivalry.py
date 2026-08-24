"""Binocular rivalry in two competing Wong-Wang visual populations.

Both populations receive the same continuous input. Slow activity-dependent
adaptation weakens the current winner, while independent OU current noise
perturbs when the next escape from that winner occurs.

The parameter regime is a phenomenological, outcome-calibrated demonstration;
the fixed seed below was not used while selecting the displayed regime.
"""

from pathlib import Path

import brainmass
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs")
SEED = 314159

# Protocol and physical parameters. Wong-Wang gating variables are dimensionless;
# currents therefore remain in nA when multiplied by a gating variable.
DT = 1.0 * u.ms
DURATION = 30.0 * u.second
BURN_IN = 5.0 * u.second
BINOCULAR_DRIVE = 0.010 * u.nA
RIVALRY_COUPLING = 0.020 * u.nA
ADAPTATION_TAU = 1.5 * u.second
NOISE_TAU = 50.0 * u.ms
RATE_SCALE = 40.0 * u.Hz
PERCEPT_THRESHOLD = 4.0 * u.Hz

ADAPTATION_LEVELS = jnp.asarray([0.000, 0.025, 0.050, 0.075]) * u.nA
NOISE_LEVELS = jnp.asarray([0.0000, 0.0025, 0.0035, 0.0045, 0.0060]) * u.nA
REPLICATES = 8


def simulate_observer(adaptation_strength, noise_sigma):
    """Simulate one observer and return fixed-shape, time-major observables."""
    model = brainmass.WongWangStep(in_size=1)
    brainstate.nn.init_all_states(model)

    adaptation_vertical = brainstate.HiddenState(jnp.zeros(1) * u.nA)
    adaptation_horizontal = brainstate.HiddenState(jnp.zeros(1) * u.nA)
    noise_vertical = brainstate.HiddenState(jnp.zeros(1) * u.nA)
    noise_horizontal = brainstate.HiddenState(jnp.zeros(1) * u.nA)
    percept = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def step(index):
        del index
        innovations = brainstate.random.randn(2)
        ou_scale = u.math.sqrt(2.0 * DT / NOISE_TAU)
        noise_vertical.value += (
            -noise_vertical.value / NOISE_TAU * DT
            + noise_sigma * ou_scale * innovations[0]
        )
        noise_horizontal.value += (
            -noise_horizontal.value / NOISE_TAU * DT
            + noise_sigma * ou_scale * innovations[1]
        )

        current_vertical, current_horizontal = model.compute_inputs(
            coherence=0.0,
            noise_1_val=noise_vertical.value,
            noise_2_val=noise_horizontal.value,
        )
        rate_vertical = model.phi(
            current_vertical
            + BINOCULAR_DRIVE
            - RIVALRY_COUPLING * model.S2.value
            - adaptation_vertical.value
        )
        rate_horizontal = model.phi(
            current_horizontal
            + BINOCULAR_DRIVE
            - RIVALRY_COUPLING * model.S1.value
            - adaptation_horizontal.value
        )

        adaptation_vertical.value += DT / ADAPTATION_TAU * (
            adaptation_strength * rate_vertical / RATE_SCALE
            - adaptation_vertical.value
        )
        adaptation_horizontal.value += DT / ADAPTATION_TAU * (
            adaptation_strength * rate_horizontal / RATE_SCALE
            - adaptation_horizontal.value
        )
        model.S1.value = u.math.clip(
            brainstate.nn.exp_euler_step(
                model.dS1_dt, model.S1.value, rate_vertical
            ),
            0.0,
            1.0,
        )
        model.S2.value = u.math.clip(
            brainstate.nn.exp_euler_step(
                model.dS2_dt, model.S2.value, rate_horizontal
            ),
            0.0,
            1.0,
        )

        rate_difference = rate_vertical[0] - rate_horizontal[0]
        percept.value = jnp.where(
            rate_difference > PERCEPT_THRESHOLD,
            1,
            jnp.where(
                rate_difference < -PERCEPT_THRESHOLD, -1, percept.value
            ),
        )
        return (
            rate_vertical[0].to_decimal(u.Hz),
            rate_horizontal[0].to_decimal(u.Hz),
            percept.value,
            (
                adaptation_vertical.value[0]
                - adaptation_horizontal.value[0]
            ).to_decimal(u.nA),
        )

    n_steps = int(DURATION / DT)
    with brainstate.environ.context(dt=DT):
        return brainstate.transform.for_loop(step, jnp.arange(n_steps))


def build_cohort():
    """Return flattened factorial conditions for independent observers."""
    adaptation, noise, replicate = jnp.meshgrid(
        ADAPTATION_LEVELS.to_decimal(u.nA),
        NOISE_LEVELS.to_decimal(u.nA),
        jnp.arange(REPLICATES),
        indexing="ij",
    )
    return adaptation.reshape(-1), noise.reshape(-1), replicate.reshape(-1)


def complete_dominance_durations(percept):
    """Return uncensored run durations after removing initial undecided samples."""
    decided = percept[percept != 0]
    if decided.size < 2:
        return np.asarray([], dtype=float)
    edges = np.flatnonzero(
        np.r_[True, decided[1:] != decided[:-1], True]
    )
    run_steps = np.diff(edges)
    # The first and last runs touch analysis-window boundaries and are censored.
    return run_steps[1:-1] * float(DT.to_decimal(u.second))


def analyze(percepts, adaptation_na, noise_na):
    """Reduce trajectories to observer- and condition-level rivalry metrics."""
    burn_steps = int(BURN_IN / DT)
    analyzed = percepts[:, burn_steps:]
    n_observers = analyzed.shape[0]

    switch_count = np.zeros(n_observers, dtype=int)
    mean_duration_s = np.full(n_observers, np.nan)
    vertical_fraction = np.full(n_observers, np.nan)
    complete_runs = []

    for observer, trace in enumerate(analyzed):
        decided = trace[trace != 0]
        if decided.size:
            switch_count[observer] = np.count_nonzero(
                decided[1:] != decided[:-1]
            )
            vertical_fraction[observer] = np.mean(decided == 1)
        durations = complete_dominance_durations(trace)
        complete_runs.append(durations)
        if durations.size:
            mean_duration_s[observer] = np.mean(durations)

    n_adaptation = ADAPTATION_LEVELS.size
    n_noise = NOISE_LEVELS.size
    median_duration_s = np.full((n_adaptation, n_noise), np.nan)
    switch_rate_per_min = np.zeros((n_adaptation, n_noise))
    locked_fraction = np.zeros((n_adaptation, n_noise))
    undecided_fraction = np.zeros((n_adaptation, n_noise))
    analysis_minutes = float((DURATION - BURN_IN).to_decimal(u.second)) / 60.0

    adaptation_values = np.asarray(ADAPTATION_LEVELS.to_decimal(u.nA))
    noise_values = np.asarray(NOISE_LEVELS.to_decimal(u.nA))
    for adaptation_index, adaptation_value in enumerate(adaptation_values):
        for noise_index, noise_value in enumerate(noise_values):
            selected = np.flatnonzero(
                np.isclose(adaptation_na, adaptation_value)
                & np.isclose(noise_na, noise_value)
            )
            durations = [complete_runs[index] for index in selected]
            nonempty = [values for values in durations if values.size]
            if nonempty:
                median_duration_s[adaptation_index, noise_index] = np.median(
                    np.concatenate(nonempty)
                )
            switch_rate_per_min[adaptation_index, noise_index] = (
                np.mean(switch_count[selected]) / analysis_minutes
            )
            decided = np.isfinite(vertical_fraction[selected])
            locked_fraction[adaptation_index, noise_index] = np.mean(
                decided & (switch_count[selected] == 0)
            )
            undecided_fraction[adaptation_index, noise_index] = np.mean(
                ~decided
            )

    return {
        "switch_count": switch_count,
        "mean_complete_duration_s": mean_duration_s,
        "vertical_fraction": vertical_fraction,
        "median_duration_s": median_duration_s,
        "switch_rate_per_min": switch_rate_per_min,
        "locked_fraction": locked_fraction,
        "undecided_fraction": undecided_fraction,
    }


def save_results(
    adaptation_na,
    noise_na,
    replicate,
    rates_vertical,
    rates_horizontal,
    percepts,
    adaptation_difference,
    metrics,
):
    """Save summaries and one auditable representative trajectory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    target_adaptation = 0.050
    target_noise = 0.0035
    representative = np.flatnonzero(
        np.isclose(adaptation_na, target_adaptation)
        & np.isclose(noise_na, target_noise)
    )[0]
    times_s = (
        np.arange(rates_vertical.shape[1]) + 1
    ) * float(DT.to_decimal(u.second))

    np.savez_compressed(
        OUTPUT_DIR / "binocular_rivalry_results.npz",
        schema_version=np.asarray("1.0"),
        model=np.asarray("brainmass.WongWangStep"),
        seed=np.asarray(SEED),
        adaptation_nA=adaptation_na,
        noise_sigma_nA=noise_na,
        replicate=replicate,
        switch_count=metrics["switch_count"],
        mean_complete_duration_s=metrics["mean_complete_duration_s"],
        vertical_fraction=metrics["vertical_fraction"],
        median_duration_s=metrics["median_duration_s"],
        switch_rate_per_min=metrics["switch_rate_per_min"],
        locked_fraction=metrics["locked_fraction"],
        undecided_fraction=metrics["undecided_fraction"],
        adaptation_axis_nA=np.asarray(ADAPTATION_LEVELS.to_decimal(u.nA)),
        noise_axis_nA=np.asarray(NOISE_LEVELS.to_decimal(u.nA)),
        representative_time_s=times_s,
        representative_vertical_rate_Hz=rates_vertical[representative],
        representative_horizontal_rate_Hz=rates_horizontal[representative],
        representative_percept=percepts[representative],
        representative_adaptation_difference_nA=(
            adaptation_difference[representative]
        ),
        dt_ms=np.asarray(DT.to_decimal(u.ms)),
        duration_s=np.asarray(DURATION.to_decimal(u.second)),
        burn_in_s=np.asarray(BURN_IN.to_decimal(u.second)),
        binocular_drive_nA=np.asarray(BINOCULAR_DRIVE.to_decimal(u.nA)),
        rivalry_coupling_nA=np.asarray(
            RIVALRY_COUPLING.to_decimal(u.nA)
        ),
        adaptation_tau_s=np.asarray(ADAPTATION_TAU.to_decimal(u.second)),
        noise_tau_ms=np.asarray(NOISE_TAU.to_decimal(u.ms)),
        percept_threshold_Hz=np.asarray(
            PERCEPT_THRESHOLD.to_decimal(u.Hz)
        ),
        monitor_phase=np.asarray("post-update"),
        calibration=np.asarray("phenomenological, outcome-calibrated"),
    )
    return representative, times_s


def plot_results(
    representative,
    times_s,
    rates_vertical,
    rates_horizontal,
    percepts,
    metrics,
):
    """Plot an observer trace and the adaptation-by-noise cohort summaries."""
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    plot_window = (times_s >= 5.0) & (times_s <= 17.0)
    axes[0].plot(
        times_s[plot_window],
        rates_vertical[representative, plot_window],
        color="#167d78",
        label="Vertical",
        linewidth=1.1,
    )
    axes[0].plot(
        times_s[plot_window],
        rates_horizontal[representative, plot_window],
        color="#d1495b",
        label="Horizontal",
        linewidth=1.1,
    )
    axes[0].fill_between(
        times_s[plot_window],
        0,
        1,
        where=percepts[representative, plot_window] == 1,
        color="#167d78",
        alpha=0.10,
        transform=axes[0].get_xaxis_transform(),
    )
    axes[0].fill_between(
        times_s[plot_window],
        0,
        1,
        where=percepts[representative, plot_window] == -1,
        color="#d1495b",
        alpha=0.10,
        transform=axes[0].get_xaxis_transform(),
    )
    axes[0].set(
        title="Continuous rivalry in one observer",
        xlabel="Time (s)",
        ylabel="Population rate (Hz)",
    )
    axes[0].legend(frameon=False)

    adaptation_labels = [
        f"{value:.3f}"
        for value in np.asarray(ADAPTATION_LEVELS.to_decimal(u.nA))
    ]
    noise_labels = [
        f"{value:.4f}"
        for value in np.asarray(NOISE_LEVELS.to_decimal(u.nA))
    ]

    duration_image = axes[1].imshow(
        metrics["median_duration_s"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set(
        title="Median complete dominance (s)",
        xlabel="OU noise sigma (nA)",
        ylabel="Adaptation strength (nA)",
        xticks=np.arange(len(noise_labels)),
        yticks=np.arange(len(adaptation_labels)),
        xticklabels=noise_labels,
        yticklabels=adaptation_labels,
    )
    axes[1].tick_params(axis="x", rotation=35)
    figure.colorbar(duration_image, ax=axes[1], fraction=0.046, pad=0.04)

    switch_image = axes[2].imshow(
        metrics["switch_rate_per_min"],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes[2].set(
        title="Switches per minute",
        xlabel="OU noise sigma (nA)",
        ylabel="Adaptation strength (nA)",
        xticks=np.arange(len(noise_labels)),
        yticks=np.arange(len(adaptation_labels)),
        xticklabels=noise_labels,
        yticklabels=adaptation_labels,
    )
    axes[2].tick_params(axis="x", rotation=35)
    figure.colorbar(switch_image, ax=axes[2], fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "binocular_rivalry.png", dpi=180)
    plt.close(figure)


def print_summary(metrics, n_observers):
    """Print the scientific interpretation next to concise cohort evidence."""
    print(f"Simulated {n_observers} observers with a stateful vmap rollout.")
    print("Rows: adaptation strength (nA); columns: OU noise sigma (nA)")
    print("Median complete dominance duration (s):")
    print(np.array2string(metrics["median_duration_s"], precision=3))
    print("Switches per minute:")
    print(np.array2string(metrics["switch_rate_per_min"], precision=1))
    print("Locked-observer fraction (decided but no switch after burn-in):")
    print(np.array2string(metrics["locked_fraction"], precision=2))
    print("Undecided-observer fraction:")
    print(np.array2string(metrics["undecided_fraction"], precision=2))
    print()
    print("Interpretation:")
    print(
        "- Mutual competition makes one population suppress the other, "
        "creating a dominant percept."
    )
    print(
        "- Adaptation accumulates in the active population and erodes its "
        "advantage; stronger adaptation usually shortens dominance."
    )
    print(
        "- Noise controls escape timing. More noise increases the hazard of "
        "an early switch and broadens individual dominance durations."
    )
    print(
        "- At exactly zero noise, perfect symmetry remains undecided: "
        "adaptation alone cannot choose a winner from equal activity. Weak "
        "noise breaks symmetry, but some decided runs do not switch within "
        "25 s."
    )
    print(
        "- Dominance intervals touching an analysis-window boundary are "
        "reported as censored, not assigned an artificial finite duration."
    )
    print(
        "- At the strongest noise, random escapes dominate and the systematic "
        "effect of adaptation becomes comparatively small."
    )


def main():
    brainstate.random.seed(SEED)
    adaptation_na, noise_na, replicate = build_cohort()
    run_cohort = brainstate.transform.vmap(
        simulate_observer,
        in_axes=(0, 0),
        out_axes=0,
    )
    outputs = run_cohort(adaptation_na * u.nA, noise_na * u.nA)
    jax.block_until_ready(outputs)
    rates_vertical, rates_horizontal, percepts, adaptation_difference = (
        np.asarray(value) for value in outputs
    )

    adaptation_na = np.asarray(adaptation_na)
    noise_na = np.asarray(noise_na)
    replicate = np.asarray(replicate)
    metrics = analyze(percepts, adaptation_na, noise_na)
    representative, times_s = save_results(
        adaptation_na,
        noise_na,
        replicate,
        rates_vertical,
        rates_horizontal,
        percepts,
        adaptation_difference,
        metrics,
    )
    plot_results(
        representative,
        times_s,
        rates_vertical,
        rates_horizontal,
        percepts,
        metrics,
    )
    print_summary(metrics, adaptation_na.size)
    print(f"Saved {OUTPUT_DIR / 'binocular_rivalry_results.npz'}")
    print(f"Saved {OUTPUT_DIR / 'binocular_rivalry.png'}")


if __name__ == "__main__":
    main()
