"""Learn and reverse a two-tone temporal-order discrimination with BrainX.

The circuit has two sensory LIF neurons, two order-selective LIF neurons, and
two output LIF neurons. A decaying sensory trace makes an order neuron fire
only when the other tone arrives second. BrainEvent then communicates those
binary detector spikes through a plastic dense readout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import brainevent
import brainpy
import brainstate
import brainunit as u
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    """Physical and learning parameters for the temporal-order task."""

    dt: u.Quantity = field(default_factory=lambda: 1.0 * u.ms)
    trial_duration: u.Quantity = field(default_factory=lambda: 100.0 * u.ms)
    first_tone_onset: u.Quantity = field(default_factory=lambda: 12.0 * u.ms)
    tone_gap: u.Quantity = field(default_factory=lambda: 24.0 * u.ms)
    tone_duration: u.Quantity = field(default_factory=lambda: 8.0 * u.ms)
    axonal_delay: u.Quantity = field(default_factory=lambda: 2.0 * u.ms)
    sensory_trace_tau: u.Quantity = field(default_factory=lambda: 35.0 * u.ms)
    readout_synapse_tau: u.Quantity = field(default_factory=lambda: 6.0 * u.ms)
    plasticity_trace_tau: u.Quantity = field(default_factory=lambda: 20.0 * u.ms)
    membrane_tau: u.Quantity = field(default_factory=lambda: 8.0 * u.ms)
    refractory_period: u.Quantity = field(default_factory=lambda: 4.0 * u.ms)
    membrane_resistance: u.Quantity = field(default_factory=lambda: 100.0 * u.Mohm)
    resting_voltage: u.Quantity = field(default_factory=lambda: -65.0 * u.mV)
    sensory_threshold: u.Quantity = field(default_factory=lambda: -50.0 * u.mV)
    detector_threshold: u.Quantity = field(default_factory=lambda: -50.0 * u.mV)
    output_threshold: u.Quantity = field(default_factory=lambda: -45.0 * u.mV)
    tone_current: u.Quantity = field(default_factory=lambda: 450.0 * u.pA)
    detector_current: u.Quantity = field(default_factory=lambda: 3.0 * u.nA)
    readout_current: u.Quantity = field(default_factory=lambda: 2.0 * u.nA)
    learning_rate: float = 0.06
    weight_min: float = 0.05
    weight_max: float = 0.95

    @property
    def steps_per_trial(self) -> int:
        return int(round(self.trial_duration.to_decimal(u.ms) / self.dt.to_decimal(u.ms)))


class TrialBatch(NamedTuple):
    sensory_drive: jax.Array
    teacher_events: jax.Array
    response_mask: jax.Array
    times: u.Quantity


class PhaseResult(NamedTuple):
    sensory_spikes: jax.Array
    detector_spikes: jax.Array
    output_spikes: jax.Array
    weight_history: jax.Array


class EvaluationResult(NamedTuple):
    accuracy: float
    predictions: np.ndarray
    detector_spikes: np.ndarray
    output_spikes: np.ndarray


def _lif_population(size: int, threshold: u.Quantity, cfg: ExperimentConfig):
    return brainpy.state.LIFRef(
        size,
        R=cfg.membrane_resistance,
        tau=cfg.membrane_tau,
        V_rest=cfg.resting_voltage,
        V_th=threshold,
        V_reset=cfg.resting_voltage,
        tau_ref=cfg.refractory_period,
        V_initializer=braintools.init.Constant(cfg.resting_voltage),
    )


class FixedEventDelay(brainstate.nn.Module):
    """Fixed-grid delay line for boolean events with optional batch axes."""

    def __init__(
        self,
        feature_size: int,
        delay: u.Quantity,
        dt: u.Quantity,
        state_batch_size: int | None,
    ):
        super().__init__()
        delay_steps = delay.to_decimal(u.ms) / dt.to_decimal(u.ms)
        self.delay_steps = int(round(delay_steps))
        if not np.isclose(delay_steps, self.delay_steps):
            raise ValueError("FixedEventDelay requires delay to be an integer multiple of dt")
        state_prefix = () if state_batch_size is None else (state_batch_size,)
        self.history = brainstate.ShortTermState(
            jnp.zeros(
                state_prefix + (self.delay_steps + 1, feature_size),
                dtype=bool,
            )
        )

    def reset(self) -> None:
        self.history.value = jnp.zeros_like(self.history.value)

    def update(self, events: jax.Array) -> jax.Array:
        events = jnp.asarray(events, dtype=bool)
        history = jnp.concatenate(
            [events[..., None, :], self.history.value[..., :-1, :]], axis=-2
        )
        self.history.value = history
        return history[..., self.delay_steps, :]


class TemporalOrderCircuit(brainstate.nn.Module):
    """Small spiking circuit with a persistent, plastic detector readout."""

    # Binary delayed sensory events [A, B] become second-tone events [B, A].
    _CROSS_CONNECTIVITY = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    def __init__(
        self,
        cfg: ExperimentConfig,
        initial_weights: jax.Array | None = None,
        state_batch_size: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.sensory = _lif_population(2, cfg.sensory_threshold, cfg)
        self.detectors = _lif_population(2, cfg.detector_threshold, cfg)
        self.outputs = _lif_population(2, cfg.output_threshold, cfg)
        self.sensory_delay = FixedEventDelay(
            feature_size=2,
            delay=cfg.axonal_delay,
            dt=cfg.dt,
            state_batch_size=state_batch_size,
        )

        if initial_weights is None:
            initial_weights = jnp.array(
                [[0.52, 0.48], [0.52, 0.48]], dtype=jnp.float32
            )
        initial_weights = jnp.asarray(initial_weights, dtype=jnp.float32)
        state_prefix = () if state_batch_size is None else (state_batch_size,)
        trace_shape = state_prefix + (2,)

        self.weights = brainstate.LongTermState(initial_weights)
        self.sensory_trace = brainstate.ShortTermState(
            jnp.zeros(trace_shape, dtype=jnp.float32)
        )
        self.readout_trace = brainstate.ShortTermState(
            jnp.zeros(trace_shape, dtype=jnp.float32)
        )
        self.teacher_trace = brainstate.ShortTermState(
            jnp.zeros(trace_shape, dtype=jnp.float32)
        )

    def reset_traces(self) -> None:
        self.sensory_delay.reset()
        self.sensory_trace.value = jnp.zeros_like(self.sensory_trace.value)
        self.readout_trace.value = jnp.zeros_like(self.readout_trace.value)
        self.teacher_trace.value = jnp.zeros_like(self.teacher_trace.value)

    def _advance(
        self,
        sensory_drive: jax.Array,
        teacher_event: jax.Array,
        learn: bool,
    ):
        dt = brainstate.environ.get_dt()
        sensory_decay = u.math.exp(-dt / self.cfg.sensory_trace_tau)
        readout_decay = u.math.exp(-dt / self.cfg.readout_synapse_tau)
        teacher_decay = u.math.exp(-dt / self.cfg.plasticity_trace_tau)

        self.sensory(sensory_drive.astype(jnp.float32) * self.cfg.tone_current)
        sensory_spikes = self.sensory.get_spike() != 0.0

        # Update-before-retrieve makes index d exactly d completed steps old.
        delayed_spikes = self.sensory_delay(sensory_spikes)

        old_sensory_trace = self.sensory_trace.value * sensory_decay
        second_tone_events = (
            brainevent.BinaryArray(delayed_spikes) @ self._CROSS_CONNECTIVITY
        )
        order_gate = old_sensory_trace * second_tone_events
        self.sensory_trace.value = (
            old_sensory_trace + delayed_spikes.astype(jnp.float32)
        )

        self.detectors(order_gate * self.cfg.detector_current)
        detector_spikes = self.detectors.get_spike() != 0.0

        readout_events = brainevent.BinaryArray(detector_spikes) @ self.weights.value
        self.readout_trace.value = (
            self.readout_trace.value * readout_decay + readout_events
        )
        self.outputs(self.readout_trace.value * self.cfg.readout_current)
        output_spikes = self.outputs.get_spike() != 0.0

        self.teacher_trace.value = (
            self.teacher_trace.value * teacher_decay + teacher_event
        )
        if learn:
            self.weights.value = brainevent.update_dense_on_binary_pre(
                weight=self.weights.value,
                pre_spike=detector_spikes,
                post_trace=self.teacher_trace.value * self.cfg.learning_rate,
                w_min=self.cfg.weight_min,
                w_max=self.cfg.weight_max,
            )

        return sensory_spikes, detector_spikes, output_spikes, self.weights.value

    def learn_step(self, sensory_drive: jax.Array, teacher_event: jax.Array):
        return self._advance(sensory_drive, teacher_event, learn=True)

    def evaluate_step(self, sensory_drive: jax.Array):
        zero_teacher = jnp.zeros_like(self.teacher_trace.value)
        return self._advance(sensory_drive, zero_teacher, learn=False)


def encode_trials(
    orders: jax.Array,
    labels: jax.Array,
    gaps: u.Quantity,
    cfg: ExperimentConfig,
) -> TrialBatch:
    """Vectorize unit-aware tone and teaching-event construction over trials."""

    local_times = u.math.arange(0.0 * u.ms, cfg.trial_duration, cfg.dt)

    def encode_one(order, label, gap):
        first_onset = cfg.first_tone_onset
        second_onset = first_onset + gap
        first_active = (local_times >= first_onset) & (
            local_times < first_onset + cfg.tone_duration
        )
        second_active = (local_times >= second_onset) & (
            local_times < second_onset + cfg.tone_duration
        )

        first_channel = jnp.where(order == 0, 0, 1)
        second_channel = 1 - first_channel
        sensory = jnp.zeros((cfg.steps_per_trial, 2), dtype=bool)
        sensory = sensory.at[:, first_channel].set(first_active)
        sensory = sensory.at[:, second_channel].set(second_active)

        teacher_pulse = (local_times >= second_onset) & (
            local_times < second_onset + cfg.dt
        )
        signed_label = jnp.where(
            jnp.arange(2) == label,
            jnp.float32(1.0),
            jnp.float32(-1.0),
        )
        teacher = teacher_pulse[:, None] * signed_label[None, :]
        response = (local_times >= second_onset) & (
            local_times < second_onset + 40.0 * u.ms
        )
        return sensory, teacher, response

    # This is the pure-data vmap; the stateful simulation vmap is in evaluate_batch().
    encode_batch = brainstate.transform.vmap(encode_one, in_axes=(0, 0, 0))
    sensory, teacher, response = encode_batch(orders, labels, gaps)

    trial_offsets = u.math.arange(orders.shape[0]) * cfg.trial_duration
    times = trial_offsets[:, None] + local_times[None, :]
    return TrialBatch(sensory, teacher, response, times)


def run_learning_phase(
    circuit: TemporalOrderCircuit,
    batch: TrialBatch,
) -> PhaseResult:
    """Run trials and within-trial steps as nested State-aware loops."""

    def run_one_trial(trial_times, sensory_drive, teacher_events):
        def step(t, drive, teacher):
            with brainstate.environ.context(t=t):
                return circuit.learn_step(drive, teacher)

        return brainstate.transform.for_loop(
            step, trial_times, sensory_drive, teacher_events
        )

    @brainstate.transform.jit
    def learn_all_trials(times, sensory_drive, teacher_events):
        return brainstate.transform.for_loop(
            run_one_trial, times, sensory_drive, teacher_events
        )

    return PhaseResult(*learn_all_trials(batch.times, batch.sensory_drive, batch.teacher_events))


def evaluate_batch(
    learned_weights: jax.Array,
    orders: jax.Array,
    labels: jax.Array,
    gaps: u.Quantity,
    cfg: ExperimentConfig,
) -> EvaluationResult:
    """Run a vmap-built trial batch with independent neural and trace State."""

    batch = encode_trials(orders, labels, gaps, cfg)
    num_trials = int(orders.shape[0])
    evaluator = TemporalOrderCircuit(
        cfg,
        initial_weights=learned_weights,
        state_batch_size=num_trials,
    )

    with brainstate.environ.context(dt=cfg.dt):
        brainstate.nn.init_all_states(evaluator, batch_size=num_trials)

        local_times = u.math.arange(0.0 * u.ms, cfg.trial_duration, cfg.dt)
        time_major_drive = jnp.swapaxes(batch.sensory_drive, 0, 1)

        def step(t, drive):
            with brainstate.environ.context(t=t):
                return evaluator.evaluate_step(drive)

        @brainstate.transform.jit
        def evaluate_trials(times, sensory_drive):
            return brainstate.transform.for_loop(step, times, sensory_drive)

        _, detector_spikes, output_spikes, _ = evaluate_trials(
            local_times, time_major_drive
        )

    detector_spikes = jnp.swapaxes(detector_spikes, 0, 1)
    output_spikes = jnp.swapaxes(output_spikes, 0, 1)

    response_mask = np.asarray(batch.response_mask)[..., None]
    output_counts = np.asarray(output_spikes).astype(np.int32)
    output_counts = (output_counts * response_mask).sum(axis=1)
    predictions = output_counts.argmax(axis=1)
    accuracy = float(np.mean(predictions == np.asarray(labels)))
    return EvaluationResult(
        accuracy,
        predictions,
        np.asarray(detector_spikes),
        np.asarray(output_spikes),
    )


def _trial_accuracy(result: PhaseResult, batch: TrialBatch, labels: jax.Array) -> np.ndarray:
    mask = np.asarray(batch.response_mask)[..., None]
    counts = (np.asarray(result.output_spikes).astype(np.int32) * mask).sum(axis=1)
    return counts.argmax(axis=1) == np.asarray(labels)


def _alternating_orders(num_trials: int) -> jax.Array:
    return jnp.arange(num_trials, dtype=jnp.int32) % 2


def run_experiment(
    trials_per_phase: int = 30,
    seed: int = 7,
    cfg: ExperimentConfig | None = None,
):
    """Acquire A-before-B/B-before-A, reverse the labels, and relearn."""

    cfg = cfg or ExperimentConfig()
    brainstate.random.seed(seed)
    orders = _alternating_orders(trials_per_phase)
    original_labels = orders
    reversed_labels = 1 - orders
    train_gaps = jnp.full(trials_per_phase, cfg.tone_gap.to_decimal(u.ms)) * u.ms

    evaluation_orders = _alternating_orders(12)
    evaluation_gaps = (
        jnp.array([22.0, 24.0, 26.0, 23.0, 25.0, 27.0] * 2) * u.ms
    )
    evaluation_original_labels = evaluation_orders
    evaluation_reversed_labels = 1 - evaluation_orders

    with brainstate.environ.context(dt=cfg.dt):
        circuit = TemporalOrderCircuit(cfg)
        brainstate.nn.init_all_states(circuit)
        initial_weights = jnp.array(circuit.weights.value)

        initial_eval = evaluate_batch(
            initial_weights,
            evaluation_orders,
            evaluation_original_labels,
            evaluation_gaps,
            cfg,
        )

        acquisition_batch = encode_trials(orders, original_labels, train_gaps, cfg)
        acquisition = run_learning_phase(circuit, acquisition_batch)
        acquired_weights = jnp.array(circuit.weights.value)

        acquired_eval = evaluate_batch(
            acquired_weights,
            evaluation_orders,
            evaluation_original_labels,
            evaluation_gaps,
            cfg,
        )
        switch_eval = evaluate_batch(
            acquired_weights,
            evaluation_orders,
            evaluation_reversed_labels,
            evaluation_gaps,
            cfg,
        )

        # Reverse the teaching rule. Reset trial-scale dynamics but preserve weights.
        brainstate.nn.init_all_states(circuit)
        circuit.reset_traces()
        circuit.weights.value = acquired_weights
        reversal_batch = encode_trials(orders, reversed_labels, train_gaps, cfg)
        reversal = run_learning_phase(circuit, reversal_batch)
        reversed_weights = jnp.array(circuit.weights.value)

        reversed_eval = evaluate_batch(
            reversed_weights,
            evaluation_orders,
            evaluation_reversed_labels,
            evaluation_gaps,
            cfg,
        )

    return {
        "config": cfg,
        "orders": np.asarray(orders),
        "initial_weights": np.asarray(initial_weights),
        "acquired_weights": np.asarray(acquired_weights),
        "reversed_weights": np.asarray(reversed_weights),
        "acquisition": acquisition,
        "reversal": reversal,
        "acquisition_batch": acquisition_batch,
        "reversal_batch": reversal_batch,
        "acquisition_trial_accuracy": _trial_accuracy(
            acquisition, acquisition_batch, original_labels
        ),
        "reversal_trial_accuracy": _trial_accuracy(
            reversal, reversal_batch, reversed_labels
        ),
        "initial_accuracy": initial_eval.accuracy,
        "acquired_accuracy": acquired_eval.accuracy,
        "switch_accuracy": switch_eval.accuracy,
        "reversed_accuracy": reversed_eval.accuracy,
        "acquired_evaluation": acquired_eval,
        "reversed_evaluation": reversed_eval,
    }


def _rolling_mean(values: np.ndarray, window: int = 6) -> np.ndarray:
    values = values.astype(np.float32)
    cumulative = np.cumsum(values)
    result = cumulative.copy()
    result[window:] = cumulative[window:] - cumulative[:-window]
    counts = np.minimum(np.arange(1, values.size + 1), window)
    return result / counts


def plot_results(results: dict, output_path: Path) -> None:
    cfg = results["config"]
    acquisition = results["acquisition"]
    reversal = results["reversal"]
    num_trials = len(results["orders"])
    trial_axis = np.arange(1, 2 * num_trials + 1)

    acquisition_weights = np.asarray(acquisition.weight_history)[:, -1]
    reversal_weights = np.asarray(reversal.weight_history)[:, -1]
    weights = np.concatenate([acquisition_weights, reversal_weights], axis=0)
    accuracy = np.concatenate(
        [
            results["acquisition_trial_accuracy"],
            results["reversal_trial_accuracy"],
        ]
    )

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), constrained_layout=True)

    ax = axes[0]
    for row, label, color in [
        (0, "A->B detector", "#177E89"),
        (1, "B->A detector", "#D1495B"),
    ]:
        ax.plot(
            trial_axis,
            weights[:, row, 0],
            color=color,
            linewidth=2.0,
            label=f"{label} -> output 1",
        )
        ax.plot(
            trial_axis,
            weights[:, row, 1],
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{label} -> output 2",
        )
    ax.axvline(num_trials + 0.5, color="#202124", linestyle=":", linewidth=1.5)
    ax.text(num_trials + 1.2, 0.08, "labels reversed", fontsize=9)
    ax.set(
        ylabel="Readout efficacy",
        ylim=(0.0, 1.0),
        title="Online plasticity remaps temporal-order detectors",
    )
    ax.legend(ncol=2, frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(trial_axis, _rolling_mean(accuracy), color="#355070", linewidth=2.2)
    ax.axvline(num_trials + 0.5, color="#202124", linestyle=":", linewidth=1.5)
    ax.set(
        xlabel="Training trial",
        ylabel="Rolling accuracy",
        ylim=(-0.05, 1.05),
        title="Performance falls at reversal and recovers",
    )
    ax.set_yticks([0.0, 0.5, 1.0])

    ax = axes[2]
    stage_names = ["Untrained", "Learned", "Task switch", "Relearned"]
    stage_values = [
        results["initial_accuracy"],
        results["acquired_accuracy"],
        results["switch_accuracy"],
        results["reversed_accuracy"],
    ]
    colors = ["#A7A9AC", "#177E89", "#D1495B", "#2A9D8F"]
    bars = ax.bar(stage_names, stage_values, color=colors, width=0.68)
    for bar, value in zip(bars, stage_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.0%}",
            ha="center",
            fontsize=9,
        )
    ax.set(
        ylabel="Batched evaluation accuracy",
        ylim=(0.0, 1.12),
        title=f"Independent vmap trials; tone gap {cfg.tone_gap}",
    )
    ax.set_yticks([0.0, 0.5, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(results: dict, output_path: Path) -> None:
    summary = {
        "initial_accuracy": results["initial_accuracy"],
        "acquired_accuracy": results["acquired_accuracy"],
        "accuracy_immediately_after_label_reversal": results["switch_accuracy"],
        "reversed_accuracy": results["reversed_accuracy"],
        "initial_weights": results["initial_weights"].round(4).tolist(),
        "acquired_weights": results["acquired_weights"].round(4).tolist(),
        "reversed_weights": results["reversed_weights"].round(4).tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-per-phase", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("artifacts/temporal_order_reversal.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/temporal_order_summary.json"),
    )
    args = parser.parse_args()

    results = run_experiment(args.trials_per_phase, args.seed)
    plot_results(results, args.figure)
    write_summary(results, args.summary)

    print(f"Untrained accuracy:       {results['initial_accuracy']:.0%}")
    print(f"After acquisition:        {results['acquired_accuracy']:.0%}")
    print(f"At task reversal:         {results['switch_accuracy']:.0%}")
    print(f"After reversal learning:  {results['reversed_accuracy']:.0%}")
    print(f"Figure: {args.figure}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
