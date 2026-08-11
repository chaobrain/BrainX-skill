"""Noisy two-choice spiking circuit: prior bias under ambiguous evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import brainpy
import brainstate
import braintools
import brainunit as u
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter


DT = 1.0 * u.ms
BASELINE_END = 50.0 * u.ms
EVIDENCE_START = 130.0 * u.ms
TRIAL_END = 430.0 * u.ms

N_NEURONS = 24
N_TRIALS = 64
EVIDENCE_LEVELS = jnp.array([-0.3, -0.15, -0.075, 0.0, 0.075, 0.15, 0.3]) * u.mA
PRIOR_LEVELS = jnp.array([0.0, 0.28]) * u.mA
DECISION_THRESHOLD = 0.35  # excess spikes per neuron


class TwoChoiceCircuit(brainstate.nn.Module):
    """Two LIF populations with self-excitation and cross-inhibition."""

    def __init__(self, n_neurons: int = N_NEURONS):
        super().__init__()
        self.n_neurons = n_neurons
        neuron_args = dict(
            tau=20.0 * u.ms,
            tau_ref=4.0 * u.ms,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        self.choice_a = brainpy.state.LIFRef(n_neurons, **neuron_args)
        self.choice_b = brainpy.state.LIFRef(n_neurons, **neuron_args)

        self.recurrent_a = self._projection(
            self.choice_a, 0.035 * u.mS, 5.0 * u.ms, 0.0 * u.mV
        )
        self.recurrent_b = self._projection(
            self.choice_b, 0.035 * u.mS, 5.0 * u.ms, 0.0 * u.mV
        )
        self.a_inhibits_b = self._projection(
            self.choice_b, 0.11 * u.mS, 8.0 * u.ms, -80.0 * u.mV
        )
        self.b_inhibits_a = self._projection(
            self.choice_a, 0.11 * u.mS, 8.0 * u.ms, -80.0 * u.mV
        )

        self.baseline_current = 13.2 * u.mA
        self.noise_scale = 2.6 * u.mA

    def _projection(self, post, weight, tau, reversal):
        return brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                self.n_neurons,
                self.n_neurons,
                conn_num=1.0,
                conn_weight=weight,
            ),
            syn=brainpy.state.Expon.desc(self.n_neurons, tau=tau),
            out=brainpy.state.COBA.desc(E=reversal),
            post=post,
        )

    def update(self, t, evidence, prior):
        with brainstate.environ.context(t=t):
            spike_a = self.choice_a.get_spike() != 0.0
            spike_b = self.choice_b.get_spike() != 0.0

            self.recurrent_a(spike_a)
            self.recurrent_b(spike_b)
            self.a_inhibits_b(spike_a)
            self.b_inhibits_a(spike_b)

            prior_on = (t >= BASELINE_END) & (t < EVIDENCE_START)
            evidence_on = t >= EVIDENCE_START
            noise_a = brainstate.random.normal(
                0.0 * u.mA, self.noise_scale, self.n_neurons
            )
            noise_b = brainstate.random.normal(
                0.0 * u.mA, self.noise_scale, self.n_neurons
            )
            drive_a = (
                self.baseline_current
                + prior * prior_on
                + evidence * evidence_on
                + noise_a
            )
            drive_b = self.baseline_current - evidence * evidence_on + noise_b

            spikes_a = self.choice_a(drive_a)
            spikes_b = self.choice_b(drive_b)
            return u.math.mean(spikes_a), u.math.mean(spikes_b)


def condition_lanes():
    shape = (PRIOR_LEVELS.shape[0], EVIDENCE_LEVELS.shape[0], N_TRIALS)
    evidence = u.math.broadcast_to(EVIDENCE_LEVELS[None, :, None], shape).flatten()
    prior = u.math.broadcast_to(PRIOR_LEVELS[:, None, None], shape).flatten()
    return evidence, prior, shape


def snapshot_states(model):
    return {path: state.value for path, state in model.states().items()}


def restore_states(model, snapshot):
    unexpected, missing = brainstate.nn.assign_state_values(model, snapshot)
    if unexpected or missing:
        raise RuntimeError(
            f"State restore mismatch: unexpected={unexpected}, missing={missing}"
        )


def analyze(rates, shape, dt_ms):
    evidence_step = int(round(EVIDENCE_START.to_decimal(u.ms) / dt_ms))
    rate_data = np.stack([np.asarray(population) for population in rates], axis=-1)
    rate_data = rate_data[evidence_step:]
    accumulation = np.cumsum(rate_data[..., 0] - rate_data[..., 1], axis=0)

    crossed = np.abs(accumulation) >= DECISION_THRESHOLD
    reached = crossed.any(axis=0)
    first_crossing = crossed.argmax(axis=0)
    lanes = np.arange(accumulation.shape[1])
    threshold_choice = accumulation[first_crossing, lanes] > 0.0
    final_choice = accumulation[-1] > 0.0
    choices_a = np.where(reached, threshold_choice, final_choice)
    ties = (~reached) & (accumulation[-1] == 0.0)

    choices_a = choices_a.reshape(shape)
    ties = ties.reshape(shape)
    probabilities = choices_a.mean(axis=-1) + 0.5 * ties.mean(axis=-1)

    decision_ms = np.where(reached, (first_crossing + 1) * dt_ms, np.nan)
    decision_ms = decision_ms.reshape(shape)
    median_decision_ms = np.nanmedian(decision_ms, axis=-1)
    threshold_fraction = reached.reshape(shape).mean(axis=-1)
    return (
        accumulation.reshape((accumulation.shape[0],) + shape),
        probabilities,
        median_decision_ms,
        threshold_fraction,
    )


def make_figure(
    accumulation,
    probabilities,
    compile_seconds,
    steady_seconds,
    simulated_seconds,
    output_path,
):
    evidence_ma = np.asarray(EVIDENCE_LEVELS.to_decimal(u.mA))
    time_ms = np.arange(accumulation.shape[0]) * DT.to_decimal(u.ms)
    colors = ("#45474B", "#D1493F")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.25),
        gridspec_kw={"width_ratios": (1.25, 1.0, 0.8)},
    )
    fig.patch.set_facecolor("#F7F7F4")
    for ax in axes:
        ax.set_facecolor("#F7F7F4")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9D9D3", linewidth=0.7, alpha=0.8)

    neutral_index = int(np.flatnonzero(evidence_ma == 0.0)[0])
    for prior_index, color in enumerate(colors):
        label = "Unbiased" if prior_index == 0 else "Small prior for A"
        for trial in range(7):
            axes[0].plot(
                time_ms,
                accumulation[:, prior_index, neutral_index, trial],
                color=color,
                alpha=0.30,
                linewidth=1.0,
                label=label if trial == 0 else None,
            )
    axes[0].axhline(DECISION_THRESHOLD, color="#222222", linestyle="--", linewidth=0.8)
    axes[0].axhline(-DECISION_THRESHOLD, color="#222222", linestyle="--", linewidth=0.8)
    axes[0].set(
        title="Choices unfolding at zero evidence",
        xlabel="Time after evidence onset (ms)",
        ylabel="Accumulated A - B spikes / neuron",
        xlim=(0.0, time_ms[-1]),
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    for prior_index, color in enumerate(colors):
        label = "Unbiased" if prior_index == 0 else "Small prior for A"
        probability = probabilities[prior_index]
        interval = 1.96 * np.sqrt(probability * (1.0 - probability) / N_TRIALS)
        yerr = np.vstack(
            [np.minimum(interval, probability), np.minimum(interval, 1.0 - probability)]
        )
        axes[1].errorbar(
            evidence_ma,
            probability,
            yerr=yerr,
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=color,
            label=label,
            capsize=3,
        )
    axes[1].axhline(0.5, color="#777777", linewidth=0.8, linestyle=":")
    axes[1].axvline(0.0, color="#777777", linewidth=0.8, linestyle=":")
    axes[1].set(
        title="Prior shifts ambiguous choices",
        xlabel="Evidence for choice A (mA)",
        ylabel="P(choice A)",
        ylim=(-0.03, 1.03),
        xticks=evidence_ma,
    )
    axes[1].tick_params(axis="x", labelrotation=35)
    for tick in axes[1].get_xticklabels():
        tick.set_horizontalalignment("right")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    throughput = np.array(
        [simulated_seconds / compile_seconds, simulated_seconds / steady_seconds]
    )
    bars = axes[2].bar(
        ["First call\n(compile + run)", "Compiled\nmedian run"],
        throughput,
        color=("#8A8D91", "#317A67"),
        width=0.65,
    )
    axes[2].set(
        title="Measured simulation speed",
        ylabel="Simulated seconds / wall second",
    )
    axes[2].bar_label(
        bars,
        labels=(f"{compile_seconds:.2f}s wall", f"{steady_seconds:.3f}s wall"),
        padding=4,
        fontsize=8,
    )

    fig.suptitle(
        "A small prior matters most when sensory evidence is ambiguous",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    brainstate.random.seed(20260811)

    evidence_lanes, prior_lanes, condition_shape = condition_lanes()
    n_lanes = evidence_lanes.shape[0]

    with brainstate.environ.context(dt=DT):
        circuit = TwoChoiceCircuit()
        brainstate.nn.vmap_init_all_states(circuit, axis_size=n_lanes)
        initial_states = snapshot_states(circuit)
        initial_key = brainstate.random.get_key()
        times = u.math.arange(0.0 * u.ms, TRIAL_END, brainstate.environ.get_dt())

        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = vmap2(
            circuit.update,
            in_axes=(None, 0, 0),
            out_axes=0,
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )

        def rollout(evidence, prior):
            return brainstate.transform.for_loop(
                lambda t: mapped_step(t, evidence, prior), times
            )

        compiled_rollout = brainstate.transform.jit(rollout)

        start = time.perf_counter()
        first_rates = compiled_rollout(evidence_lanes, prior_lanes)
        jax.block_until_ready(first_rates)
        compile_seconds = time.perf_counter() - start

        timings = []
        rates = first_rates
        for _ in range(5):
            restore_states(circuit, initial_states)
            brainstate.random.set_key(initial_key)
            start = time.perf_counter()
            rates = compiled_rollout(evidence_lanes, prior_lanes)
            jax.block_until_ready(rates)
            timings.append(time.perf_counter() - start)

    steady_seconds = float(np.median(timings))
    dt_ms = DT.to_decimal(u.ms)
    accumulation, probabilities, median_decision_ms, threshold_fraction = analyze(
        rates, condition_shape, dt_ms
    )
    simulated_seconds = (
        TRIAL_END.to_decimal(u.second) * n_lanes
    )
    figure_path = output_dir / "prior_bias_decision.png"
    make_figure(
        accumulation,
        probabilities,
        compile_seconds,
        steady_seconds,
        simulated_seconds,
        figure_path,
    )

    evidence_ma = np.asarray(EVIDENCE_LEVELS.to_decimal(u.mA))
    summary = {
        "evidence_mA": [round(float(value), 3) for value in evidence_ma],
        "prior_mA": [
            round(float(value), 3)
            for value in np.asarray(PRIOR_LEVELS.to_decimal(u.mA))
        ],
        "choice_a_probability": probabilities.tolist(),
        "median_decision_time_ms": median_decision_ms.tolist(),
        "threshold_crossing_fraction": threshold_fraction.tolist(),
        "trials_per_condition": N_TRIALS,
        "neurons_per_choice": N_NEURONS,
        "time_step_ms": float(dt_ms),
        "first_call_seconds": compile_seconds,
        "compiled_median_seconds": steady_seconds,
        "compiled_simulated_seconds_per_wall_second": simulated_seconds
        / steady_seconds,
        "compiled_condition_steps_per_second": len(times) * n_lanes
        / steady_seconds,
        "jax_device": jax.devices()[0].device_kind,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    neutral = int(np.flatnonzero(evidence_ma == 0.0)[0])
    print(f"Figure: {figure_path}")
    print(f"Summary: {summary_path}")
    print(
        "P(A) at zero evidence: "
        f"unbiased={probabilities[0, neutral]:.3f}, "
        f"biased={probabilities[1, neutral]:.3f}"
    )
    print(
        "Compiled speed: "
        f"{simulated_seconds / steady_seconds:,.1f} simulated s / wall s "
        f"({steady_seconds:.3f} s median wall time)"
    )


if __name__ == "__main__":
    main()
