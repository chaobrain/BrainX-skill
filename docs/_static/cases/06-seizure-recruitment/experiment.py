"""Map when a focal seizure-like burst recruits neighboring regions.

This is a phenomenological four-region FitzHugh-Nagumo neural-mass model.  A
finite pulse is applied only to region 0, and directed nearest-neighbor
coupling determines whether that event remains focal or propagates.
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


DT = 0.1 * u.ms
DURATION = 80.0 * u.ms
PULSE_START = 5.0 * u.ms
PULSE_DURATION = 5.0 * u.ms
MAX_DELAY = 8.0 * u.ms
N_REGIONS = 4

# W[target, source]: a directed chain from the stimulated focus outwards.
CONNECTIVITY = jnp.asarray(
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

# Fixed before evaluating the sweep: a sustained positive excursion defines a
# recruited regional event, rather than a single noisy threshold crossing.
RECRUITMENT_THRESHOLD = 0.5
MIN_RECRUITMENT_DURATION = 1.0 * u.ms

COUPLING_VALUES = jnp.asarray([0.15, 0.25, 0.35, 0.45, 0.55])
DELAY_VALUES = jnp.asarray([1.0, 3.0, 5.0, 7.0]) * u.ms
PULSE_VALUES = jnp.asarray([0.35, 0.50, 0.65, 0.80])


class DrivenRegionalChain(brainstate.nn.Module):
    """Excitable neural masses with package-owned delayed coupling history."""

    def __init__(self):
        super().__init__()
        self.node = brainmass.FitzHughNagumoStep(
            in_size=N_REGIONS,
            init_V=braintools.init.Constant(0.0),
            init_w=braintools.init.Constant(0.0),
        )
        self.history = brainstate.nn.Delay(
            jnp.zeros(N_REGIONS),
            time=MAX_DELAY,
            init=braintools.init.Constant(0.0),
        )

    def update(self, stimulus, coupling, delay):
        self.history.update(self.node.V.value)
        delay_steps = jnp.rint(delay / DT).astype(jnp.int32)
        delayed = self.history.retrieve_at_step(delay_steps)
        sources = jnp.broadcast_to(delayed, CONNECTIVITY.shape)
        regional_input = brainmass.additive_coupling(
            sources, CONNECTIVITY, coupling
        )
        return self.node(regional_input + stimulus)


def run_condition(coupling, delay, pulse_size):
    """Run one independently initialized condition and return V[time, region]."""
    with brainstate.environ.context(dt=DT):
        model = DrivenRegionalChain()
        brainstate.nn.init_all_states(model)
        times = u.math.arange(0.0 * u.ms, DURATION, DT)
        indices = jnp.arange(times.shape[0])

        def step(index, time):
            pulse_on = (time >= PULSE_START) & (
                time < PULSE_START + PULSE_DURATION
            )
            stimulus = jnp.where(
                pulse_on,
                jnp.asarray([pulse_size, 0.0, 0.0, 0.0]),
                jnp.zeros(N_REGIONS),
            )
            with brainstate.environ.context(i=index, t=time):
                return model(stimulus, coupling, delay)

        return brainstate.transform.for_loop(step, indices, times)


def sustained_recruitment(activity):
    """Return recruited flags and first sustained-event onsets for each region."""
    minimum_steps = int(MIN_RECRUITMENT_DURATION / DT)
    above = activity >= RECRUITMENT_THRESHOLD
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
    onsets = jnp.where(recruited, first_start, jnp.nan) * DT
    return recruited, onsets


def sweep_conditions():
    """Vectorize complete rollouts over coupling, delay, and pulse size."""
    kk, dd, pp = u.math.meshgrid(
        COUPLING_VALUES,
        DELAY_VALUES,
        PULSE_VALUES,
        indexing="ij",
    )
    run_sweep = brainstate.transform.vmap(run_condition)
    flat_activity = run_sweep(kk.reshape(-1), dd.reshape(-1), pp.reshape(-1))
    activity = flat_activity.reshape(kk.shape + flat_activity.shape[1:])

    classify_sweep = brainstate.transform.vmap(sustained_recruitment)
    recruited, onsets = classify_sweep(flat_activity)
    return kk, dd, pp, activity, recruited.reshape(kk.shape + (N_REGIONS,)), onsets.reshape(
        kk.shape + (N_REGIONS,)
    )


def choose_examples(recruited, onsets):
    """Select sampled local and strictly ordered propagation cases."""
    counts = jnp.sum(recruited, axis=-1)
    ordered_route = jnp.all(u.math.diff(onsets, axis=-1) > 0.0 * u.ms, axis=-1)
    local_candidates = jnp.argwhere(counts == 1, size=counts.size, fill_value=-1)
    spread_candidates = jnp.argwhere(
        (counts == N_REGIONS) & ordered_route,
        size=counts.size,
        fill_value=-1,
    )
    local = local_candidates[0]
    spread = spread_candidates[0]
    if bool(jnp.any(local < 0)) or bool(jnp.any(spread < 0)):
        raise RuntimeError(
            "The sampled grid did not contain both local and fully recruited regimes."
        )
    return tuple(map(int, local)), tuple(map(int, spread))


def plot_results(activity, recruited, onsets, local_index, spread_index, output):
    """Show representative traces and the sampled recruitment regime map."""
    times_ms = np.asarray(u.math.arange(0.0 * u.ms, DURATION, DT).to_decimal(u.ms))
    delay_index = spread_index[1]
    pulse_index = spread_index[2]
    counts = np.asarray(jnp.sum(recruited[:, delay_index], axis=-1))
    distal_onset_ms = np.asarray(
        onsets[:, :, pulse_index, -1].to_decimal(u.ms)
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, index, title in zip(
        axes[0],
        (local_index, spread_index),
        ("Local burst", "Neighbor recruitment"),
    ):
        traces = np.asarray(activity[index])
        onset_ms = np.asarray(onsets[index].to_decimal(u.ms))
        for region in range(N_REGIONS):
            ax.plot(times_ms, traces[:, region], label=f"Region {region}")
            if np.isfinite(onset_ms[region]):
                ax.axvline(onset_ms[region], color=f"C{region}", alpha=0.25)
        k = float(COUPLING_VALUES[index[0]])
        delay = float(DELAY_VALUES[index[1]].to_decimal(u.ms))
        pulse = float(PULSE_VALUES[index[2]])
        ax.set_title(f"{title}\nk={k:.2f}, delay={delay:.0f} ms, pulse={pulse:.2f}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Fast population activity V")
        ax.axhline(RECRUITMENT_THRESHOLD, color="0.3", ls="--", lw=1)
    axes[0, 0].legend(frameon=False, ncol=2)

    count_image = axes[1, 0].imshow(
        counts.T,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=N_REGIONS,
        cmap="viridis",
    )
    axes[1, 0].set_xticks(range(COUPLING_VALUES.shape[0]))
    axes[1, 0].set_xticklabels(
        [f"{float(value):.2f}" for value in COUPLING_VALUES]
    )
    axes[1, 0].set_yticks(range(PULSE_VALUES.shape[0]))
    axes[1, 0].set_yticklabels([f"{float(value):.2f}" for value in PULSE_VALUES])
    axes[1, 0].set_xlabel("Coupling strength k")
    axes[1, 0].set_ylabel("Pulse size")
    axes[1, 0].set_title(
        f"Recruitment extent at delay="
        f"{float(DELAY_VALUES[delay_index].to_decimal(u.ms)):.0f} ms"
    )
    fig.colorbar(
        count_image,
        ax=axes[1, 0],
        label="Number recruited",
        ticks=range(N_REGIONS + 1),
    )

    onset_image = axes[1, 1].imshow(
        distal_onset_ms,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes[1, 1].set_xticks(range(DELAY_VALUES.shape[0]))
    axes[1, 1].set_xticklabels(
        [f"{float(value.to_decimal(u.ms)):.0f}" for value in DELAY_VALUES]
    )
    axes[1, 1].set_yticks(range(COUPLING_VALUES.shape[0]))
    axes[1, 1].set_yticklabels(
        [f"{float(value):.2f}" for value in COUPLING_VALUES]
    )
    axes[1, 1].set_xlabel("Propagation delay (ms)")
    axes[1, 1].set_ylabel("Coupling strength k")
    axes[1, 1].set_title(
        f"Region 3 onset at pulse={float(PULSE_VALUES[pulse_index]):.2f}"
    )
    fig.colorbar(onset_image, ax=axes[1, 1], label="Onset (ms)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def print_case(label, index, recruited, onsets, activity):
    flags = np.asarray(recruited[index])
    onset_ms = np.asarray(onsets[index].to_decimal(u.ms))
    peaks = np.asarray(jnp.max(activity[index], axis=0))
    onset_text = [f"{value:.1f} ms" if np.isfinite(value) else "not recruited" for value in onset_ms]
    print(
        f"{label}: k={float(COUPLING_VALUES[index[0]]):.2f}, "
        f"delay={float(DELAY_VALUES[index[1]].to_decimal(u.ms)):.1f} ms, "
        f"pulse={float(PULSE_VALUES[index[2]]):.2f}"
    )
    print(f"  recruited={flags.tolist()}")
    print(f"  onsets={onset_text}")
    print(f"  peak V={np.round(peaks, 3).tolist()}")


def main():
    brainstate.random.seed(0)
    brainstate.environ.set(dt=DT)
    kk, dd, pp, activity, recruited, onsets = sweep_conditions()
    del kk, dd, pp
    local_index, spread_index = choose_examples(recruited, onsets)

    output = Path(__file__).with_name("seizure_recruitment.png")
    plot_results(activity, recruited, onsets, local_index, spread_index, output)
    print_case("LOCAL", local_index, recruited, onsets, activity)
    print_case("SPREAD", spread_index, recruited, onsets, activity)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
