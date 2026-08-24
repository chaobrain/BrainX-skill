"""Prior bias in a noisy two-choice spiking decision circuit.

The model uses two recurrently excited LIF populations that inhibit one
another.  Signed sensory evidence drives the two populations oppositely; a
small prior current only favors population A.  Independent trials and
conditions are vectorized with BrainState, while simulation time is handled by
one state-aware loop and the complete rollout is JIT compiled.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.transform import vmap2
from brainstate.util import filter as util_filter


DT = 1.0 * u.ms
DURATION = 600.0 * u.ms
NUM_NEURONS = 24
NUM_TRIALS = 64
DECISION_BOUND = 0.8
EVIDENCE_MA = np.asarray(
    [-1.2, -0.5, -0.2, -0.08, 0.0, 0.08, 0.2, 0.5, 1.2],
    dtype=np.float32,
)
PRIOR_MA = np.asarray([0.0, 0.16], dtype=np.float32)

V_REST = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
V_RESET = -60.0 * u.mV
BASELINE = 10.6 * u.mA
PRIVATE_NOISE_STD = 2.0 * u.mA
SHARED_NOISE_STD = 2.2 * u.mA

DYNAMICAL_STATE = util_filter.Any(
    util_filter.OfType(brainstate.HiddenState),
    util_filter.OfType(brainstate.ShortTermState),
)

RESULTS_DIR = Path("results")
FIGURE_PATH = RESULTS_DIR / "prior_bias_decision.png"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def lif_population(size: int):
    return brainpy.state.LIFRef(
        size,
        R=1.0 * u.ohm,
        tau=20.0 * u.ms,
        tau_ref=3.0 * u.ms,
        V_rest=V_REST,
        V_th=V_THRESHOLD,
        V_reset=V_RESET,
        V_initializer=braintools.init.Constant(V_REST),
    )


class TwoChoiceCircuit(brainstate.nn.Module):
    """Two noisy choice populations with explicit recurrent projections."""

    def __init__(self):
        super().__init__()
        self.a = lif_population(NUM_NEURONS)
        self.b = lif_population(NUM_NEURONS)

        self.a_to_a = self._projection(self.a, 1.1 * u.mS, 0.0 * u.mV)
        self.b_to_b = self._projection(self.b, 1.1 * u.mS, 0.0 * u.mV)
        self.a_to_b = self._projection(self.b, 1.8 * u.mS, -80.0 * u.mV)
        self.b_to_a = self._projection(self.a, 1.8 * u.mS, -80.0 * u.mV)

    @staticmethod
    def _projection(post, weight, reversal):
        return brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                NUM_NEURONS,
                NUM_NEURONS,
                conn_num=0.35,
                conn_weight=weight,
            ),
            syn=brainpy.state.Expon.desc(NUM_NEURONS, tau=7.0 * u.ms),
            out=brainpy.state.COBA.desc(E=reversal),
            post=post,
        )

    def update(self, t, evidence, prior):
        with brainstate.environ.context(t=t):
            previous_a = self.a.get_spike() != 0.0
            previous_b = self.b.get_spike() != 0.0
            self.a_to_a(previous_a)
            self.b_to_b(previous_b)
            self.a_to_b(previous_a)
            self.b_to_a(previous_b)

            private_a = brainstate.random.normal(
                0.0, 1.0, size=(NUM_NEURONS,)
            ) * PRIVATE_NOISE_STD
            private_b = brainstate.random.normal(
                0.0, 1.0, size=(NUM_NEURONS,)
            ) * PRIVATE_NOISE_STD
            shared_a = brainstate.random.normal() * SHARED_NOISE_STD
            shared_b = brainstate.random.normal() * SHARED_NOISE_STD
            spikes_a = self.a(BASELINE + evidence + prior + shared_a + private_a)
            spikes_b = self.b(BASELINE - evidence + shared_b + private_b)
            return u.math.mean(spikes_a), u.math.mean(spikes_b)


def condition_grid():
    """Return flattened [bias, evidence, trial] currents for vmapped lanes."""
    evidence = np.broadcast_to(
        EVIDENCE_MA[None, :, None],
        (PRIOR_MA.size, EVIDENCE_MA.size, NUM_TRIALS),
    )
    prior = np.broadcast_to(
        PRIOR_MA[:, None, None],
        (PRIOR_MA.size, EVIDENCE_MA.size, NUM_TRIALS),
    )
    return (
        jnp.asarray(evidence.reshape(-1)) * u.mA,
        jnp.asarray(prior.reshape(-1)) * u.mA,
    )


def build_rollout(circuit, num_lanes: int):
    mapped_step = vmap2(
        circuit.update,
        in_axes=(None, 0, 0),
        out_axes=0,
        state_in_axes={0: DYNAMICAL_STATE},
        state_out_axes={0: DYNAMICAL_STATE},
        unexpected_out_state_mapping="raise",
    )
    times = u.math.arange(0.0 * u.ms, DURATION, DT)

    def rollout(evidence, prior):
        def step(t):
            return mapped_step(t, evidence, prior)

        return brainstate.transform.for_loop(step, times)

    brainstate.nn.vmap_init_all_states(circuit, axis_size=num_lanes)
    return brainstate.transform.jit(rollout), times


def synchronize(value):
    """Wait for asynchronous device work before stopping a host timer."""
    return jax.block_until_ready(value)


def dynamical_snapshot(circuit):
    """Capture immutable values without replacing the vmapped State objects."""
    return [(state, state.value) for state in circuit.states(DYNAMICAL_STATE).values()]


def restore_snapshot(snapshot):
    for state, initial_value in snapshot:
        state.value = initial_value


def run_and_benchmark(compiled_rollout, circuit, evidence, prior, repeats=5):
    initial_state = dynamical_snapshot(circuit)

    brainstate.random.seed(20260811)
    restore_snapshot(initial_state)
    started = time.perf_counter()
    spikes = compiled_rollout(evidence, prior)
    synchronize(spikes)
    compile_and_run_s = time.perf_counter() - started

    elapsed = []
    for _ in range(repeats):
        restore_snapshot(initial_state)
        started = time.perf_counter()
        timed_spikes = compiled_rollout(evidence, prior)
        synchronize(timed_spikes)
        elapsed.append(time.perf_counter() - started)

    return spikes, compile_and_run_s, np.asarray(elapsed)


def decode(spikes_a, spikes_b):
    """Decode first-passage choices from cumulative spikes per neuron."""
    difference = np.cumsum(spikes_a - spikes_b, axis=0)
    crossed_a = difference >= DECISION_BOUND
    crossed_b = difference <= -DECISION_BOUND
    crossed = crossed_a | crossed_b
    first_index = np.argmax(crossed, axis=0)
    decided = np.any(crossed, axis=0)

    lane_index = np.arange(difference.shape[1])
    first_value = difference[first_index, lane_index]
    final_choice_a = difference[-1] > 0.0
    choice_a = np.where(decided, first_value > 0.0, final_choice_a)
    decision_time_ms = np.where(
        decided,
        (first_index + 1) * DT.to_decimal(u.ms),
        DURATION.to_decimal(u.ms),
    )

    shape = (PRIOR_MA.size, EVIDENCE_MA.size, NUM_TRIALS)
    return (
        difference.reshape((difference.shape[0],) + shape),
        choice_a.reshape(shape),
        decision_time_ms.reshape(shape),
    )


def wilson_interval(successes, n, z=1.96):
    p = successes / n
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return center - radius, center + radius


def make_figure(times_ms, spikes_a, spikes_b, difference, choices, elapsed):
    figure = plt.figure(figsize=(13.2, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(0.85, 1.45, 1.15))
    ax_circuit = figure.add_subplot(grid[0, 0])
    ax_examples = figure.add_subplot(grid[0, 1:])
    ax_probability = figure.add_subplot(grid[1, :2])
    ax_speed = figure.add_subplot(grid[1, 2])

    # Compact circuit schematic.
    ax_circuit.set_xlim(0, 1)
    ax_circuit.set_ylim(0, 1)
    ax_circuit.axis("off")
    colors = {"a": "#c43b3b", "b": "#237b74", "ink": "#202428"}
    circle_a = plt.Circle((0.30, 0.55), 0.15, color=colors["a"], alpha=0.95)
    circle_b = plt.Circle((0.72, 0.55), 0.15, color=colors["b"], alpha=0.95)
    ax_circuit.add_patch(circle_a)
    ax_circuit.add_patch(circle_b)
    for x, label in ((0.30, "A"), (0.72, "B")):
        ax_circuit.text(
            x,
            0.55,
            label,
            color="white",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )
    inhibitory_style = dict(arrowstyle="-[", color=colors["ink"], lw=1.7)
    ax_circuit.annotate(
        "", (0.56, 0.55), (0.46, 0.55), arrowprops=inhibitory_style
    )
    ax_circuit.annotate(
        "", (0.46, 0.47), (0.56, 0.47), arrowprops=inhibitory_style
    )
    ax_circuit.annotate(
        "",
        (0.21, 0.67),
        (0.18, 0.45),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=-1.5",
            color=colors["a"],
            lw=1.6,
        ),
    )
    ax_circuit.annotate(
        "",
        (0.81, 0.67),
        (0.84, 0.45),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=1.5",
            color=colors["b"],
            lw=1.6,
        ),
    )
    ax_circuit.annotate(
        "",
        (0.17, 0.75),
        (0.08, 0.88),
        arrowprops=dict(arrowstyle="->", color=colors["a"], lw=1.8),
    )
    ax_circuit.annotate(
        "",
        (0.84, 0.75),
        (0.93, 0.88),
        arrowprops=dict(arrowstyle="->", color=colors["b"], lw=1.8),
    )
    ax_circuit.text(0.50, 0.95, "signed evidence", ha="center", va="top", fontsize=10)
    ax_circuit.annotate(
        "prior",
        (0.25, 0.36),
        (0.08, 0.20),
        arrowprops=dict(arrowstyle="->", color="#725aa6", lw=2),
        color="#725aa6",
        fontsize=10,
    )
    ax_circuit.text(
        0.50,
        0.12,
        "recurrent excitation + mutual inhibition",
        ha="center",
        fontsize=9,
    )
    ax_circuit.set_title("Noisy competing circuit", loc="left", weight="bold")

    # Ambiguous-evidence examples, where the prior should matter most.
    zero_evidence = int(np.argmin(np.abs(EVIDENCE_MA)))
    for bias_index, linestyle in enumerate(("--", "-")):
        final_values = difference[-1, bias_index, zero_evidence]
        order = np.argsort(final_values)
        trial_ids = order[[0, 1, -2, -1]]
        for trial_index in trial_ids:
            trace = difference[:, bias_index, zero_evidence, trial_index]
            ax_examples.plot(
                times_ms,
                trace,
                color="#777777" if bias_index == 0 else "#725aa6",
                alpha=0.35 if bias_index == 0 else 0.55,
                lw=1.2,
                ls=linestyle,
            )
    ax_examples.axhline(DECISION_BOUND, color=colors["a"], lw=1, alpha=0.7)
    ax_examples.axhline(-DECISION_BOUND, color=colors["b"], lw=1, alpha=0.7)
    ax_examples.axhline(0.0, color="#9aa0a6", lw=0.8)
    ax_examples.set(xlabel="time (ms)", ylabel="cumulative A - B spikes / neuron")
    ax_examples.set_title("Ambiguous evidence: choices unfold", loc="left", weight="bold")
    ax_examples.plot([], [], "--", color="#777777", label="unbiased")
    ax_examples.plot([], [], "-", color="#725aa6", label=f"prior +{PRIOR_MA[1]:.2f} mA to A")
    ax_examples.legend(frameon=False, ncol=2, loc="upper left")

    labels = ("unbiased", f"prior +{PRIOR_MA[1]:.2f} mA")
    plot_colors = ("#4f5962", "#725aa6")
    for bias_index, (label, color) in enumerate(zip(labels, plot_colors)):
        successes = choices[bias_index].sum(axis=1)
        probability = successes / NUM_TRIALS
        low, high = wilson_interval(successes, NUM_TRIALS)
        ax_probability.plot(EVIDENCE_MA, probability, marker="o", lw=2.2, color=color, label=label)
        ax_probability.fill_between(EVIDENCE_MA, low, high, color=color, alpha=0.13)
    ax_probability.axvline(0.0, color="#9aa0a6", lw=0.8)
    ax_probability.axhline(0.5, color="#9aa0a6", lw=0.8)
    ax_probability.set(
        xlabel="signed evidence current (mA; positive favors A)",
        ylabel="P(choice A)",
        ylim=(-0.02, 1.02),
    )
    ax_probability.set_title(
        "A small prior shifts mainly uncertain decisions",
        loc="left",
        weight="bold",
    )
    ax_probability.legend(frameon=False, loc="upper left")

    lane_steps = spikes_a.shape[0] * spikes_a.shape[1]
    simulated_seconds = lane_steps * DT.to_decimal(u.second)
    real_time_factor = simulated_seconds / elapsed
    ax_speed.bar(np.arange(elapsed.size), real_time_factor, color="#d59b2d", width=0.72)
    ax_speed.axhline(np.median(real_time_factor), color="#202428", ls="--", lw=1.2)
    ax_speed.set(
        xlabel="steady-state repeat",
        ylabel="simulated seconds / wall second",
        xticks=np.arange(elapsed.size),
        xticklabels=np.arange(1, elapsed.size + 1),
    )
    ax_speed.set_title("Compiled simulation speed", loc="left", weight="bold")
    ax_speed.text(
        0.03,
        0.96,
        (
            f"median {np.median(real_time_factor):,.0f}x real time\n"
            f"{spikes_a.shape[1]:,} parallel trials"
        ),
        transform=ax_speed.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0),
    )

    for axis in (ax_examples, ax_probability, ax_speed):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e7e9eb", lw=0.8)

    figure.suptitle(
        "Prior bias under ambiguous evidence",
        x=0.01,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    brainstate.random.seed(20260811)
    evidence, prior = condition_grid()

    with brainstate.environ.context(dt=DT):
        circuit = TwoChoiceCircuit()
        compiled_rollout, times = build_rollout(circuit, evidence.shape[0])
        spikes, compile_and_run_s, elapsed = run_and_benchmark(
            compiled_rollout, circuit, evidence, prior
        )

    spikes_a = np.asarray(spikes[0])
    spikes_b = np.asarray(spikes[1])
    difference, choices, decision_times_ms = decode(spikes_a, spikes_b)
    times_ms = np.asarray(times.to_decimal(u.ms))
    make_figure(times_ms, spikes_a, spikes_b, difference, choices, elapsed)

    probabilities = choices.mean(axis=2)
    probability_shift = probabilities[1] - probabilities[0]
    ambiguous_shift = probability_shift[EVIDENCE_MA.size // 2]
    weak_shift = np.mean(np.abs(probability_shift[np.abs(EVIDENCE_MA) <= 0.081]))
    strong_shift = np.mean(np.abs(probability_shift[np.abs(EVIDENCE_MA) >= 0.5]))
    simulated_seconds = spikes_a.shape[0] * spikes_a.shape[1] * DT.to_decimal(u.second)
    real_time_factor = simulated_seconds / elapsed

    summary = {
        "evidence_mA": [round(float(value), 2) for value in EVIDENCE_MA],
        "prior_mA": [round(float(value), 2) for value in PRIOR_MA],
        "trials_per_condition": NUM_TRIALS,
        "choice_a_probability": probabilities.tolist(),
        "probability_shift_by_evidence": probability_shift.tolist(),
        "median_decision_time_ms": np.median(decision_times_ms, axis=2).tolist(),
        "ambiguous_probability_shift": float(ambiguous_shift),
        "mean_weak_evidence_probability_shift": float(weak_shift),
        "mean_strong_evidence_probability_shift": float(strong_shift),
        "compile_and_first_run_seconds": compile_and_run_s,
        "steady_run_seconds": elapsed.tolist(),
        "median_realtime_factor": float(np.median(real_time_factor)),
        "device": str(jax.devices()[0]),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Figure: {FIGURE_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"P(A) shift at zero evidence: {ambiguous_shift:+.3f}")
    print(f"Mean shift, weak vs. strong evidence: {weak_shift:.3f} vs. {strong_shift:.3f}")
    print(f"Compile + first run: {compile_and_run_s:.3f} s")
    print(f"Median steady run: {np.median(elapsed):.3f} s")
    print(f"Median simulated/wall time: {np.median(real_time_factor):,.0f}x")


if __name__ == "__main__":
    main()
