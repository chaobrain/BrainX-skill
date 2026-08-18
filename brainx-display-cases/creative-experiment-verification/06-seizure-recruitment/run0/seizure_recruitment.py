"""Phenomenological seizure-like recruitment across three brain regions."""

from pathlib import Path

import brainmass
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


DT = 0.1 * u.ms
DURATION = 160.0 * u.ms
REGIONAL_TAU = 20.0 * u.ms
STIMULUS_START = 20.0 * u.ms
STIMULUS_STOP = 40.0 * u.ms
MAX_PROPAGATION_DELAY = 12.0 * u.ms
BURST_THRESHOLD = 0.5 * u.UNITLESS
MIN_BURST_DURATION = 2.0 * u.ms

N_REGIONS = 3
REGION_LABELS = ("Focus", "Neighbor 1", "Neighbor 2")
CONNECTIVITY = jnp.asarray(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)
FOCUS_MASK = jnp.asarray([1.0, 0.0, 0.0])

COUPLING_VALUES = jnp.linspace(0.0, 0.5, 11) * u.UNITLESS
DELAY_VALUES = jnp.asarray([1.0, 4.0, 8.0, 12.0]) * u.ms
PERTURBATION_VALUES = jnp.asarray([0.0, 0.2, 0.3, 0.4, 0.5]) * u.UNITLESS

N_STEPS = int(round(DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms)))
MAX_DELAY_STEPS = int(
    round(MAX_PROPAGATION_DELAY.to_decimal(u.ms) / DT.to_decimal(u.ms))
) + 1
MIN_BURST_STEPS = int(
    round(MIN_BURST_DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms))
)


class DelayedRegionalNetwork(brainstate.nn.Module):
    """Three excitable neural masses with a fixed-size delayed history."""

    def __init__(self):
        super().__init__()
        self.node = brainmass.FitzHughNagumoStep(
            in_size=N_REGIONS,
            tau=REGIONAL_TAU,
            init_V=braintools.init.Constant(0.0),
            init_w=braintools.init.Constant(0.0),
        )
        self.history = None
        self.pointer = None

    def init_state(self, batch_size=None):
        del batch_size
        self.history = brainstate.HiddenState(
            jnp.zeros((MAX_DELAY_STEPS, N_REGIONS))
        )
        self.pointer = brainstate.ShortTermState(
            jnp.asarray(0, dtype=jnp.int32)
        )

    def update(self, stimulus, coupling, delay_steps):
        delayed_activity = self.history.value[
            (self.pointer.value - delay_steps) % MAX_DELAY_STEPS
        ]
        source_by_target = jnp.broadcast_to(
            delayed_activity, CONNECTIVITY.shape
        )
        regional_input = brainmass.additive_coupling(
            source_by_target,
            CONNECTIVITY,
            coupling,
        )
        self.node.update(regional_input + stimulus)
        self.history.value = self.history.value.at[self.pointer.value].set(
            self.node.V.value
        )
        self.pointer.value = (self.pointer.value + 1) % MAX_DELAY_STEPS
        return self.node.V.value


def simulate_condition(coupling, delay, perturbation):
    """Run one independent condition with BrainState State carrying time."""
    model = DelayedRegionalNetwork()
    delay_steps = jnp.rint(delay / DT).astype(jnp.int32)
    indices = jnp.arange(N_STEPS)

    with brainstate.environ.context(dt=DT):
        brainstate.nn.init_all_states(model)

        def step(index):
            time = index * DT
            pulse_on = (time >= STIMULUS_START) & (time < STIMULUS_STOP)
            stimulus = jnp.where(
                pulse_on,
                FOCUS_MASK * perturbation,
                jnp.zeros(N_REGIONS),
            )
            with brainstate.environ.context(i=index, t=time):
                return model.update(stimulus, coupling, delay_steps)

        return brainstate.transform.for_loop(step, indices)


def analyze_trajectory(activity):
    """Return sustained-burst onsets, peaks, and ordered recruitment count."""
    active = activity > BURST_THRESHOLD.to_decimal(u.UNITLESS)
    cumulative = jnp.concatenate(
        [jnp.zeros((1, N_REGIONS), dtype=jnp.int32),
         jnp.cumsum(active, axis=0, dtype=jnp.int32)],
        axis=0,
    )
    window_counts = (
        cumulative[MIN_BURST_STEPS:] - cumulative[:-MIN_BURST_STEPS]
    )
    sustained = window_counts == MIN_BURST_STEPS
    has_burst = jnp.any(sustained, axis=0)
    onset_index = jnp.argmax(sustained, axis=0)
    onset_ms = jnp.where(
        has_burst,
        (onset_index + 1) * DT.to_decimal(u.ms),
        jnp.inf,
    )

    focus_started = jnp.isfinite(onset_ms[0])
    neighbor_1_followed = (
        focus_started
        & jnp.isfinite(onset_ms[1])
        & (onset_ms[1] > onset_ms[0])
    )
    neighbor_2_followed = (
        neighbor_1_followed
        & jnp.isfinite(onset_ms[2])
        & (onset_ms[2] > onset_ms[1])
    )
    recruitment_count = (
        focus_started.astype(jnp.int32)
        + neighbor_1_followed.astype(jnp.int32)
        + neighbor_2_followed.astype(jnp.int32)
    )
    return onset_ms, jnp.max(activity, axis=0), recruitment_count


def summarize_condition(coupling, delay, perturbation):
    return analyze_trajectory(
        simulate_condition(coupling, delay, perturbation)
    )


def run_sweep():
    perturbation, delay, coupling = u.math.meshgrid(
        PERTURBATION_VALUES,
        DELAY_VALUES,
        COUPLING_VALUES,
        indexing="ij",
    )
    mapped_summary = brainstate.transform.vmap(summarize_condition)
    onset_ms, peak_v, recruited = mapped_summary(
        coupling.flatten(),
        delay.flatten(),
        perturbation.flatten(),
    )
    grid_shape = perturbation.shape
    return (
        onset_ms.reshape(grid_shape + (N_REGIONS,)),
        peak_v.reshape(grid_shape + (N_REGIONS,)),
        recruited.reshape(grid_shape),
    )


def representative_cases():
    couplings = jnp.asarray([0.2, 0.4]) * u.UNITLESS
    delays = jnp.asarray([4.0, 4.0]) * u.ms
    perturbations = jnp.asarray([0.5, 0.5]) * u.UNITLESS
    traces = brainstate.transform.vmap(simulate_condition)(
        couplings, delays, perturbations
    )
    onsets, peaks, recruited = brainstate.transform.vmap(analyze_trajectory)(
        traces
    )
    return traces, onsets, peaks, recruited


def plot_results(traces, onsets, recruited_grid, output_path):
    times = (jnp.arange(N_STEPS) + 1) * DT
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    for case_index, title in enumerate(
        ("Local burst", "Sequential recruitment")
    ):
        brainmass.viz.plot_timeseries(
            traces[case_index],
            ts=times,
            labels=REGION_LABELS,
            ax=axes[0, case_index],
        )
        axes[0, case_index].axhline(
            BURST_THRESHOLD.to_decimal(u.UNITLESS),
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axes[0, case_index].set_title(title)
        axes[0, case_index].set_ylabel("Fast population activity V")
        axes[0, case_index].set_xlabel("Time (ms)")

    image = None
    for axis, perturbation_index in zip(axes[1], (2, 4)):
        image = axis.imshow(
            np.asarray(recruited_grid[perturbation_index]),
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=3,
            cmap="viridis",
            extent=[
                COUPLING_VALUES[0].to_decimal(u.UNITLESS),
                COUPLING_VALUES[-1].to_decimal(u.UNITLESS),
                DELAY_VALUES[0].to_decimal(u.ms),
                DELAY_VALUES[-1].to_decimal(u.ms),
            ],
            interpolation="nearest",
        )
        axis.set_title(
            "Perturbation = "
            f"{PERTURBATION_VALUES[perturbation_index].to_decimal(u.UNITLESS):.1f}"
        )
        axis.set_xlabel("Coupling strength (dimensionless)")
        axis.set_ylabel("Propagation delay (ms)")

    colorbar = fig.colorbar(image, ax=axes[1].tolist(), ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(
        ["No burst", "Focus only", "+ Neighbor 1", "+ Neighbor 2"]
    )
    fig.suptitle("Seizure-like burst recruitment in a three-region chain")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_results(output_path, onset_ms, peak_v, recruited, case_data):
    traces, case_onsets, case_peaks, case_recruited = case_data
    np.savez_compressed(
        output_path,
        coupling=np.asarray(COUPLING_VALUES.to_decimal(u.UNITLESS)),
        propagation_delay_ms=np.asarray(DELAY_VALUES.to_decimal(u.ms)),
        perturbation=np.asarray(
            PERTURBATION_VALUES.to_decimal(u.UNITLESS)
        ),
        onset_ms=np.asarray(onset_ms),
        peak_v=np.asarray(peak_v),
        recruitment_count=np.asarray(recruited),
        representative_traces=np.asarray(traces),
        representative_onset_ms=np.asarray(case_onsets),
        representative_peak_v=np.asarray(case_peaks),
        representative_recruitment_count=np.asarray(case_recruited),
        dt_ms=DT.to_decimal(u.ms),
        regional_tau_ms=REGIONAL_TAU.to_decimal(u.ms),
        burst_threshold=BURST_THRESHOLD.to_decimal(u.UNITLESS),
        minimum_burst_duration_ms=MIN_BURST_DURATION.to_decimal(u.ms),
    )


def format_onsets(values):
    return ", ".join(
        "not recruited" if not np.isfinite(value) else f"{value:.1f} ms"
        for value in np.asarray(values)
    )


def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    onset_ms, peak_v, recruited = run_sweep()
    cases = representative_cases()
    traces, case_onsets, case_peaks, case_recruited = cases

    assert int(case_recruited[0]) == 1, "local case recruited a neighbor"
    assert int(case_recruited[1]) == 3, "propagating case did not recruit both neighbors"
    assert np.all(np.diff(np.asarray(case_onsets[1])) > 0), (
        "propagating onsets are not ordered"
    )

    no_stimulus = np.asarray(recruited[0])
    no_coupling = np.asarray(recruited[:, :, 0])
    assert np.all(no_stimulus == 0), "activity appeared without stimulation"
    assert np.all(no_coupling <= 1), "a neighbor activated without coupling"

    figure_path = output_dir / "seizure_recruitment.png"
    results_path = output_dir / "seizure_recruitment_results.npz"
    plot_results(traces, case_onsets, recruited, figure_path)
    save_results(results_path, onset_ms, peak_v, recruited, cases)

    print("Local case onsets:      ", format_onsets(case_onsets[0]))
    print("Recruited case onsets:  ", format_onsets(case_onsets[1]))
    print("Saved figure:           ", figure_path)
    print("Saved sweep data:        ", results_path)


if __name__ == "__main__":
    main()
