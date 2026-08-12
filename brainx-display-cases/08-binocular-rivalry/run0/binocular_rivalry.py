"""Population-level binocular rivalry with adaptation and stochastic switching."""

from pathlib import Path
import json

import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


DT = 1.0 * u.ms
DURATION = 40.0 * u.second
TRANSIENT = 5.0 * u.second
TAU_ADAPTATION = 1.2 * u.second
TAU_NOISE = 50.0 * u.ms
BACKGROUND_INPUT = 0.3255 * u.nA
SENSORY_COUPLING = 0.2243 * u.pA / u.Hz
SENSORY_RATE = 30.0 * u.Hz
SELF_EXCITATION = 0.2609 * u.nA
MUTUAL_INHIBITION = 0.15 * u.nA
DOMINANCE_THRESHOLD = 0.15
SEED = 23
REPETITIONS = 12

ADAPTATION_LEVELS = jnp.linspace(0.18, 0.30, 5) * u.nA
NOISE_LEVELS = jnp.asarray([0.005, 0.0125, 0.025, 0.0375, 0.05]) * u.nA


class RivalryMassStep(brainstate.nn.Module):
    """Wong-Wang competing populations augmented with slow fatigue."""

    def __init__(self):
        super().__init__()
        self.competition = brainmass.WongWangStep(
            in_size=1,
            J_N11=SELF_EXCITATION,
            J_N22=SELF_EXCITATION,
            J_N12=MUTUAL_INHIBITION,
            J_N21=MUTUAL_INHIBITION,
            J_A_ext=SENSORY_COUPLING,
            mu_0=SENSORY_RATE,
            I_0=BACKGROUND_INPUT,
        )
        self.fluctuations = brainmass.OUProcess(
            in_size=2,
            sigma=1.0 * u.nA,
            tau=TAU_NOISE,
        )

    def init_state(self):
        self.adaptation = brainstate.HiddenState(jnp.zeros(2) * u.nA)

    def update(self, adaptation_strength, noise_sigma):
        dt = brainstate.environ.get_dt()
        adaptation = self.adaptation.value
        noise = self.fluctuations.update() * (noise_sigma / (1.0 * u.nA))

        input_1, input_2 = self.competition.compute_inputs(
            coherence=0.0,
            noise_1_val=noise[0] - adaptation[0],
            noise_2_val=noise[1] - adaptation[1],
        )
        rate_1 = self.competition.phi(input_1)
        rate_2 = self.competition.phi(input_2)
        self.competition.S1.value = jnp.clip(
            brainstate.nn.exp_euler_step(
                self.competition.dS1_dt,
                self.competition.S1.value,
                rate_1,
            ),
            0.0,
            1.0,
        )
        self.competition.S2.value = jnp.clip(
            brainstate.nn.exp_euler_step(
                self.competition.dS2_dt,
                self.competition.S2.value,
                rate_2,
            ),
            0.0,
            1.0,
        )
        activity = jnp.concatenate(
            (self.competition.S1.value, self.competition.S2.value)
        )
        adaptation = adaptation + dt / TAU_ADAPTATION * (
            adaptation_strength * activity - adaptation
        )

        self.adaptation.value = adaptation
        return activity, adaptation


def observer_parameters():
    adaptation, noise = jnp.meshgrid(
        u.get_magnitude(ADAPTATION_LEVELS),
        u.get_magnitude(NOISE_LEVELS),
        indexing="ij",
    )
    adaptation = jnp.repeat(adaptation.reshape(-1), REPETITIONS) * u.nA
    noise = jnp.repeat(noise.reshape(-1), REPETITIONS) * u.nA
    repetitions = np.tile(
        np.arange(REPETITIONS),
        ADAPTATION_LEVELS.shape[0] * NOISE_LEVELS.shape[0],
    )
    return adaptation, noise, repetitions


def simulate():
    adaptation_strength, noise_sigma, repetitions = observer_parameters()
    num_observers = adaptation_strength.shape[0]
    num_steps = int(DURATION / DT)

    brainstate.random.seed(SEED)
    model = RivalryMassStep()
    brainstate.nn.vmap_init_all_states(model, axis_size=num_observers)

    observer_states = (
        model.competition.S1,
        model.competition.S2,
        model.adaptation,
        model.fluctuations.x,
    )
    observer_step = brainstate.transform.vmap(
        model.update,
        in_axes=(0, 0),
        out_axes=0,
        in_states=observer_states,
        out_states=observer_states,
    )

    with brainstate.environ.context(dt=DT):
        activity, adaptation = brainstate.transform.for_loop(
            lambda _i: observer_step(adaptation_strength, noise_sigma),
            jnp.arange(num_steps),
        )

    assert activity.shape == (num_steps, num_observers, 2)
    assert adaptation.shape == (num_steps, num_observers, 2)
    assert model.competition.S1.value.shape == (num_observers, 1)
    assert model.fluctuations.x.value.shape == (num_observers, 2)
    return activity, adaptation, adaptation_strength, noise_sigma, repetitions


def dominance_statistics(activity):
    transient_steps = int(TRANSIENT / DT)
    difference = np.asarray(activity[transient_steps:, :, 0] - activity[transient_steps:, :, 1])
    analysis_seconds = float((DURATION - TRANSIENT).to_decimal(u.second))
    dt_seconds = float(DT.to_decimal(u.second))

    num_observers = difference.shape[1]
    interval_seconds = np.empty(num_observers)
    switches_per_minute = np.empty(num_observers)
    vertical_fraction = np.empty(num_observers)
    completed_mean_seconds = np.full(num_observers, np.nan)
    completed_count = np.zeros(num_observers, dtype=int)

    for observer in range(num_observers):
        trace = difference[:, observer]
        percept = np.where(
            trace > DOMINANCE_THRESHOLD,
            1,
            np.where(trace < -DOMINANCE_THRESHOLD, -1, 0),
        )
        decided = np.flatnonzero(percept)
        if decided.size == 0:
            interval_seconds[observer] = analysis_seconds
            switches_per_minute[observer] = 0.0
            vertical_fraction[observer] = 0.5
            continue

        percept[: decided[0]] = percept[decided[0]]
        for index in range(1, percept.size):
            if percept[index] == 0:
                percept[index] = percept[index - 1]

        changes = np.flatnonzero(percept[1:] != percept[:-1]) + 1
        edges = np.concatenate(([0], changes, [percept.size]))
        durations = np.diff(edges) * dt_seconds
        completed = durations[1:-1]

        interval_seconds[observer] = analysis_seconds / (changes.size + 1)
        switches_per_minute[observer] = changes.size / analysis_seconds * 60.0
        vertical_fraction[observer] = np.mean(percept == 1)
        completed_count[observer] = completed.size
        if completed.size:
            completed_mean_seconds[observer] = np.mean(completed)

    return {
        "difference": difference,
        "interval_seconds": interval_seconds,
        "switches_per_minute": switches_per_minute,
        "vertical_fraction": vertical_fraction,
        "completed_mean_seconds": completed_mean_seconds,
        "completed_count": completed_count,
    }


def condition_means(values):
    shape = (ADAPTATION_LEVELS.shape[0], NOISE_LEVELS.shape[0], REPETITIONS)
    return np.asarray(values).reshape(shape).mean(axis=2)


def save_results(
    output_dir,
    activity,
    adaptation,
    observer_adaptation,
    observer_noise,
    repetitions,
    stats,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_grid = condition_means(stats["interval_seconds"])
    switch_grid = condition_means(stats["switches_per_minute"])
    vertical_grid = condition_means(stats["vertical_fraction"])
    example_index = (
        (ADAPTATION_LEVELS.shape[0] // 2) * NOISE_LEVELS.shape[0]
        + NOISE_LEVELS.shape[0] // 2
    ) * REPETITIONS
    sample_every = 10
    times = (
        (np.arange(activity.shape[0]) + 1)
        * float(DT.to_decimal(u.second))
    )

    metadata = {
        "schema_version": "1.0",
        "model": "RivalryMassStep",
        "population_labels": ["vertical", "horizontal"],
        "monitor_phase": "post-update",
        "activity_unit": "normalized population activity (dimensionless)",
        "current_unit": "nA",
        "time_unit": "second",
        "dt_ms": float(DT.to_decimal(u.ms)),
        "duration_s": float(DURATION.to_decimal(u.second)),
        "transient_s": float(TRANSIENT.to_decimal(u.second)),
        "competition_model": "brainmass.WongWangStep",
        "tau_adaptation_s": float(TAU_ADAPTATION.to_decimal(u.second)),
        "tau_noise_ms": float(TAU_NOISE.to_decimal(u.ms)),
        "background_input_nA": float(BACKGROUND_INPUT.to_decimal(u.nA)),
        "sensory_coupling_pA_per_Hz": float(
            SENSORY_COUPLING.to_decimal(u.pA / u.Hz)
        ),
        "sensory_rate_Hz": float(SENSORY_RATE.to_decimal(u.Hz)),
        "self_excitation_nA": float(SELF_EXCITATION.to_decimal(u.nA)),
        "mutual_inhibition_nA": float(MUTUAL_INHIBITION.to_decimal(u.nA)),
        "integration": "BrainState exponential Euler",
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "repetitions_per_condition": REPETITIONS,
        "seed": SEED,
        "calibration": "phenomenological, parameters fixed before the saved sweep",
    }
    np.savez_compressed(
        output_dir / "binocular_rivalry_results.npz",
        metadata_json=json.dumps(metadata),
        adaptation_levels_nA=np.asarray(ADAPTATION_LEVELS.to_decimal(u.nA)),
        noise_levels_nA=np.asarray(NOISE_LEVELS.to_decimal(u.nA)),
        observer_adaptation_nA=np.asarray(observer_adaptation.to_decimal(u.nA)),
        observer_noise_nA=np.asarray(observer_noise.to_decimal(u.nA)),
        observer_repetition=repetitions,
        observer_interval_s=stats["interval_seconds"],
        observer_switches_per_min=stats["switches_per_minute"],
        observer_vertical_fraction=stats["vertical_fraction"],
        observer_completed_mean_s=stats["completed_mean_seconds"],
        observer_completed_count=stats["completed_count"],
        condition_interval_s=interval_grid,
        condition_switches_per_min=switch_grid,
        condition_vertical_fraction=vertical_grid,
        example_time_s=times[::sample_every],
        example_activity=np.asarray(activity[::sample_every, example_index]),
        example_adaptation_nA=np.asarray(
            adaptation[::sample_every, example_index].to_decimal(u.nA)
        ),
    )
    return interval_grid, switch_grid, example_index, times


def plot_results(output_dir, activity, interval_grid, example_index, times):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    trace_stop = int(20.0 / float(DT.to_decimal(u.second)))
    axes[0].plot(
        times[:trace_stop],
        np.asarray(activity[:trace_stop, example_index, 0]),
        color="#2878b5",
        label="Vertical",
    )
    axes[0].plot(
        times[:trace_stop],
        np.asarray(activity[:trace_stop, example_index, 1]),
        color="#d1495b",
        label="Horizontal",
    )
    axes[0].set(xlabel="Time (s)", ylabel="Normalized activity", title="Continuous rivalry")
    axes[0].legend(frameon=False)

    image = axes[1].imshow(
        interval_grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[1].set_xticks(
        np.arange(NOISE_LEVELS.shape[0]),
        labels=[f"{value:.4f}" for value in NOISE_LEVELS.to_decimal(u.nA)],
    )
    axes[1].set_yticks(
        np.arange(ADAPTATION_LEVELS.shape[0]),
        labels=[f"{value:.3f}" for value in ADAPTATION_LEVELS.to_decimal(u.nA)],
    )
    axes[1].set(
        xlabel="OU noise amplitude (nA)",
        ylabel="Adaptation strength (nA)",
        title="Mean dominance interval (s)",
    )
    fig.colorbar(image, ax=axes[1], label="Seconds")
    fig.savefig(output_dir / "binocular_rivalry.png", dpi=180)
    plt.close(fig)


def report(interval_grid, switch_grid, stats):
    adaptation_effect = interval_grid.mean(axis=1)
    noise_effect = interval_grid.mean(axis=0)
    alternating = np.mean(stats["switches_per_minute"] >= 2.0)
    print(f"Simulated {stats['interval_seconds'].size} independent observers.")
    print(f"Observers with at least 2 switches/min: {alternating:.1%}")
    print("Mean dominance interval by adaptation strength:")
    for strength, duration in zip(ADAPTATION_LEVELS.to_decimal(u.nA), adaptation_effect):
        print(f"  {strength:.3f} nA: {duration:.2f} s")
    print("Mean dominance interval by noise amplitude:")
    for sigma, duration in zip(NOISE_LEVELS.to_decimal(u.nA), noise_effect):
        print(f"  {sigma:.4f} nA: {duration:.2f} s")
    print(
        "Fastest grid condition: "
        f"{switch_grid.max():.1f} switches/min; "
        f"slowest: {switch_grid.min():.1f} switches/min."
    )


def main():
    output_dir = Path(__file__).with_name("results")
    activity, adaptation, observer_adaptation, observer_noise, repetitions = simulate()
    stats = dominance_statistics(activity)
    interval_grid, switch_grid, example_index, times = save_results(
        output_dir,
        activity,
        adaptation,
        observer_adaptation,
        observer_noise,
        repetitions,
        stats,
    )
    plot_results(output_dir, activity, interval_grid, example_index, times)
    report(interval_grid, switch_grid, stats)
    print(f"Saved figure and numeric results in {output_dir}")


if __name__ == "__main__":
    main()
