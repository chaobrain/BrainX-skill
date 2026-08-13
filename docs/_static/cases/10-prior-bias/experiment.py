"""Prior bias in a noisy, mutually inhibitory two-choice circuit.

The simulation batches independent evidence/bias/trial lanes with BrainState's
state-aware vmap2, advances time with for_loop, and compiles the complete
rollout with jit. BrainUnit quantities remain attached through the dynamics.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "brainx-mpl"))

import brainpy
import brainstate
import brainunit as u
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter


DT = 0.5 * u.ms
DURATION = 450.0 * u.ms
N_NEURONS = 32
N_TRIALS = 128
EVIDENCE_NA = np.array([-0.030, -0.020, -0.012, -0.006, 0.0,
                        0.006, 0.012, 0.020, 0.030])
PRIOR_NA = np.array([0.0, 0.006])
SEED = 20260811


class TwoChoiceCircuit(brainstate.nn.Module):
    """Two recurrently excited LIF populations with cross-inhibition."""

    def __init__(self, n_neurons: int):
        super().__init__()
        neuron = dict(
            R=100.0 * u.Mohm,
            tau=20.0 * u.ms,
            tau_ref=2.0 * u.ms,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        self.choice_a = brainpy.state.LIFRef(n_neurons, **neuron)
        self.choice_b = brainpy.state.LIFRef(n_neurons, **neuron)
        self.n_neurons = n_neurons

        self.baseline = 0.112 * u.nA
        self.self_excitation = 0.0030 * u.nA / u.Hz
        self.mutual_inhibition = 0.0022 * u.nA / u.Hz
        self.rate_tau = 20.0 * u.ms
        self.shared_noise = 0.065 * u.nA
        self.private_noise = 0.050 * u.nA

    def init_state(self):
        self.rate_a = brainstate.HiddenState(0.0 * u.Hz)
        self.rate_b = brainstate.HiddenState(0.0 * u.Hz)

    def update(self, evidence: u.Quantity, prior: u.Quantity):
        dt = brainstate.environ.get_dt()
        signed_input = evidence + prior

        shared = brainstate.random.randn(2) * self.shared_noise
        private = brainstate.random.randn(2, self.n_neurons) * self.private_noise
        current_a = (
            self.baseline
            + self.self_excitation * self.rate_a.value
            - self.mutual_inhibition * self.rate_b.value
            + signed_input
            + shared[0]
            + private[0]
        )
        current_b = (
            self.baseline
            + self.self_excitation * self.rate_b.value
            - self.mutual_inhibition * self.rate_a.value
            - signed_input
            + shared[1]
            + private[1]
        )

        spikes_a = self.choice_a(current_a)
        spikes_b = self.choice_b(current_b)
        instantaneous_a = u.math.mean(spikes_a) / dt
        instantaneous_b = u.math.mean(spikes_b) / dt
        self.rate_a.value = (
            self.rate_a.value
            + (instantaneous_a - self.rate_a.value) * dt / self.rate_tau
        )
        self.rate_b.value = (
            self.rate_b.value
            + (instantaneous_b - self.rate_b.value) * dt / self.rate_tau
        )
        return u.math.asarray([self.rate_a.value, self.rate_b.value])


def condition_inputs():
    """Return flattened [bias, evidence, trial] inputs with current units."""
    evidence_na = np.tile(np.repeat(EVIDENCE_NA, N_TRIALS), len(PRIOR_NA))
    prior_na = np.repeat(PRIOR_NA, len(EVIDENCE_NA) * N_TRIALS)
    return jnp.asarray(evidence_na) * u.nA, jnp.asarray(prior_na) * u.nA


def analyze(activity_hz: np.ndarray):
    """Decode each trial from mean population activity in the final 50 ms."""
    tail_steps = int(50.0 / DT.to_decimal(u.ms))
    final_rates = activity_hz[-tail_steps:].mean(axis=0)
    choose_a = final_rates[:, 0] > final_rates[:, 1]
    choice_grid = choose_a.reshape(len(PRIOR_NA), len(EVIDENCE_NA), N_TRIALS)
    return choose_a, choice_grid.mean(axis=2)


def validate_results(activity_hz: np.ndarray, probabilities: np.ndarray):
    expected_shape = (
        int(DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms)),
        len(PRIOR_NA) * len(EVIDENCE_NA) * N_TRIALS,
        2,
    )
    if activity_hz.shape != expected_shape:
        raise RuntimeError(f"Unexpected activity shape {activity_hz.shape}; expected {expected_shape}")
    if not np.isfinite(activity_hz).all() or not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
        raise RuntimeError("Simulation produced non-finite activity or invalid probabilities")

    shift = np.abs(probabilities[1] - probabilities[0])
    ambiguous = np.abs(EVIDENCE_NA) <= 0.006
    strong = np.abs(EVIDENCE_NA) >= 0.020
    if shift[ambiguous].mean() <= shift[strong].mean():
        raise RuntimeError("This run did not show a larger prior effect under ambiguous evidence")
    if probabilities[:, 0].max() >= 0.1 or probabilities[:, -1].min() <= 0.9:
        raise RuntimeError("Strong evidence did not reliably control the circuit's choice")


def plot_results(
    activity_hz: np.ndarray,
    choose_a: np.ndarray,
    probabilities: np.ndarray,
    first_seconds: float,
    steady_seconds: float,
    output: Path,
):
    time_ms = np.arange(activity_hz.shape[0]) * DT.to_decimal(u.ms)
    zero_index = int(np.flatnonzero(EVIDENCE_NA == 0.0)[0])
    fig = plt.figure(figsize=(14.2, 6.4), facecolor="#f7f7f4")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.45, 1.1, 0.9), wspace=0.38, hspace=0.34)
    trajectory_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    color_a, color_b = "#d4553d", "#167d82"

    for bias_index, axis in enumerate(trajectory_axes):
        start = (bias_index * len(EVIDENCE_NA) + zero_index) * N_TRIALS
        trial_indices = start + np.arange(8)
        decision_variable = activity_hz[:, trial_indices, 0] - activity_hz[:, trial_indices, 1]
        for local, trial in enumerate(trial_indices):
            axis.plot(
                time_ms,
                decision_variable[:, local],
                color=color_a if choose_a[trial] else color_b,
                alpha=0.78,
                linewidth=1.15,
            )
        axis.axhline(0.0, color="#3b3b3b", linewidth=0.8, alpha=0.55)
        axis.set_xlim(0.0, time_ms[-1])
        axis.set_ylabel("rate A - B (Hz)")
        label = "unbiased" if bias_index == 0 else f"prior = +{PRIOR_NA[1]:.3f} nA"
        axis.set_title(f"Zero evidence, {label}", loc="left", fontsize=10.5, weight="bold")
        axis.grid(alpha=0.17, linewidth=0.7)
    trajectory_axes[1].set_xlabel("time (ms)")
    trajectory_axes[0].tick_params(labelbottom=False)

    probability_axis = fig.add_subplot(grid[:, 1])
    curve_colors = ["#3f4650", "#d19a22"]
    labels = ["no prior", f"A prior (+{PRIOR_NA[1]:.3f} nA)"]
    for curve, color, label in zip(probabilities, curve_colors, labels):
        probability_axis.plot(EVIDENCE_NA, curve, "o-", color=color, label=label, linewidth=2.2, markersize=5.5)
    probability_axis.axhline(0.5, color="#3b3b3b", linewidth=0.8, alpha=0.45)
    probability_axis.axvline(0.0, color="#3b3b3b", linewidth=0.8, alpha=0.45)
    probability_axis.set(xlabel="evidence for A (nA)", ylabel="P(choice A)", ylim=(-0.03, 1.03))
    probability_axis.set_title("Choice probability", loc="left", fontsize=11, weight="bold")
    probability_axis.legend(frameon=False, loc="lower right")
    probability_axis.grid(alpha=0.17, linewidth=0.7)

    speed_axis = fig.add_subplot(grid[:, 2])
    n_decisions = activity_hz.shape[1]
    throughputs = np.array([n_decisions / first_seconds, n_decisions / steady_seconds])
    bars = speed_axis.bar(
        ["first call\n+ compile", "compiled\nsteady"],
        throughputs,
        color=["#8b8e91", "#34715b"],
        width=0.62,
    )
    speed_axis.set_ylabel("simulated decisions / wall second")
    speed_axis.set_title("Measured simulation speed", loc="left", fontsize=11, weight="bold")
    speed_axis.grid(axis="y", alpha=0.2, linewidth=0.7)
    speed_axis.set_ylim(0.0, throughputs.max() * 1.22)
    for bar, speed, seconds in zip(bars, throughputs, [first_seconds, steady_seconds]):
        speed_axis.text(
            bar.get_x() + bar.get_width() / 2,
            speed + throughputs.max() * 0.035,
            f"{speed:,.0f}/s\n({seconds:.3f} s)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("A small prior shifts choices most when sensory evidence is ambiguous", x=0.04, ha="left", fontsize=16, weight="bold")
    fig.text(0.04, 0.015, f"{N_NEURONS} LIF neurons per choice | {N_TRIALS} trials per condition | dt = {DT.to_decimal(u.ms):g} ms", fontsize=9, color="#55585b")
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    output = Path(__file__).with_name("prior_bias_results.png")
    evidence, prior = condition_inputs()
    n_lanes = evidence.shape[0]
    times = u.math.arange(0.0 * u.ms, DURATION, DT)

    brainstate.random.seed(SEED)
    with brainstate.environ.context(dt=DT):
        circuit = TwoChoiceCircuit(N_NEURONS)
        brainstate.nn.vmap_init_all_states(circuit, axis_size=n_lanes)
        dynamic_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = vmap2(
            circuit.update,
            in_axes=(0, 0),
            out_axes=0,
            state_in_axes={0: dynamic_state},
            state_out_axes={0: dynamic_state},
            unexpected_out_state_mapping="raise",
        )

        def rollout(evidence_input, prior_input):
            brainstate.nn.vmap_init_all_states(circuit, axis_size=n_lanes)

            def step(current_time):
                with brainstate.environ.context(t=current_time):
                    return mapped_step(evidence_input, prior_input)

            return brainstate.transform.for_loop(
                step,
                times,
            )

        compiled_rollout = brainstate.transform.jit(rollout)
        start = time.perf_counter()
        activity = compiled_rollout(evidence, prior)
        activity_hz = activity.to_decimal(u.Hz)
        jax.block_until_ready(activity_hz)
        first_seconds = time.perf_counter() - start

        steady_timings = []
        for _ in range(3):
            start = time.perf_counter()
            benchmark_activity = compiled_rollout(evidence, prior)
            jax.block_until_ready(benchmark_activity.to_decimal(u.Hz))
            steady_timings.append(time.perf_counter() - start)

    activity_hz = np.asarray(activity_hz)
    choose_a, probabilities = analyze(activity_hz)
    validate_results(activity_hz, probabilities)
    steady_seconds = float(np.median(steady_timings))
    plot_results(activity_hz, choose_a, probabilities, first_seconds, steady_seconds, output)

    bias_shift = probabilities[1] - probabilities[0]
    ambiguous = np.abs(EVIDENCE_NA) <= 0.006
    strong = np.abs(EVIDENCE_NA) >= 0.020
    print(f"Saved figure: {output}")
    print("P(choice A), no prior: " + np.array2string(probabilities[0], precision=3))
    print("P(choice A), A prior:  " + np.array2string(probabilities[1], precision=3))
    print(f"Mean |prior shift|, ambiguous evidence: {np.mean(np.abs(bias_shift[ambiguous])):.3f}")
    print(f"Mean |prior shift|, strong evidence:    {np.mean(np.abs(bias_shift[strong])):.3f}")
    print(f"First call: {first_seconds:.3f} s; compiled median: {steady_seconds:.3f} s")


if __name__ == "__main__":
    main()
