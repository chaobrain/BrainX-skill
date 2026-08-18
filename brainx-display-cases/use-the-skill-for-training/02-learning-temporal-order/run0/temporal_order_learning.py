"""Learn and reverse a two-tone temporal order in a small spiking circuit.

Tone A and tone B each drive one sensory LIF neuron. A BrainState delay line
holds the first sensory spike for the inter-tone interval. At the second tone,
the delayed first spike and the direct second spike form one of two distinct
event patterns::

    A then B -> [delayed A, delayed B, direct A, direct B] = [1, 0, 0, 1]
    B then A -> [delayed A, delayed B, direct A, direct B] = [0, 1, 1, 0]

A signed teaching trace potentiates the correct output and depresses the other
through BrainEvent's spike-triggered dense plasticity operator. The experiment
first teaches A-first, then reverses the tones and teaches B-first.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np


DT = 1.0 * u.ms
TONE_DURATION = 5.0 * u.ms
TONE_GAP = 20.0 * u.ms
FIRST_TONE_ONSET = 10.0 * u.ms
TRIAL_DURATION = 70.0 * u.ms

TONE_CURRENT = 250.0 * u.mA
# A learned pair can cross threshold, while either learned feature alone cannot.
EVENT_CURRENT = 40.0 * u.mA
TEACHER_CURRENT = 250.0 * u.mA

MEMBRANE_TAU = 10.0 * u.ms
REFRACTORY_TIME = 6.0 * u.ms
TEACHER_TRACE_TAU = 10.0 * u.ms

LEARNING_RATE = 0.04
INITIAL_WEIGHT = 0.08
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

N_SENSORY = 2
N_FEATURES = 4
N_OUTPUT = 2
ORDER_NAMES = ("A then B", "B then A")
OUTPUT_NAMES = ("A first", "B first")


def _steps(duration) -> int:
    """Convert a time quantity to a fixed number of simulation steps."""
    duration_ms = duration.to_decimal(u.ms)
    dt_ms = DT.to_decimal(u.ms)
    steps = int(round(float(duration_ms / dt_ms)))
    if not np.isclose(steps * dt_ms, duration_ms):
        raise ValueError(f"{duration} must be an integer multiple of DT={DT}.")
    return steps


TONE_STEPS = _steps(TONE_DURATION)
DELAY_STEPS = _steps(TONE_GAP)
FIRST_ONSET_STEP = _steps(FIRST_TONE_ONSET)
SECOND_ONSET_STEP = FIRST_ONSET_STEP + DELAY_STEPS
TRIAL_STEPS = _steps(TRIAL_DURATION)


@dataclass(frozen=True)
class TrialBatch:
    """Time-major inputs for a batch of contiguous trials."""

    tone_current: object
    teacher_gate: jax.Array
    target: jax.Array
    learning: jax.Array
    trial_start: jax.Array
    orders: jax.Array


@dataclass(frozen=True)
class ExperimentResult:
    """Host-side summary of learning and teacher-free probe trials."""

    phase_trials: int
    training_orders: np.ndarray
    training_scores: np.ndarray
    training_weights: np.ndarray
    initial_weights: np.ndarray
    phase1_weights: np.ndarray
    phase2_weights: np.ndarray
    phase1_probe_spikes: np.ndarray
    phase2_probe_spikes: np.ndarray


def _encode_one_trial(order):
    """Encode one order as two tone-current windows and its teaching target."""
    step = jnp.arange(TRIAL_STEPS)
    first_window = (step >= FIRST_ONSET_STEP) & (
        step < FIRST_ONSET_STEP + TONE_STEPS
    )
    second_window = (step >= SECOND_ONSET_STEP) & (
        step < SECOND_ONSET_STEP + TONE_STEPS
    )

    first_tone = jax.nn.one_hot(order, N_SENSORY, dtype=jnp.float32)
    second_tone = 1.0 - first_tone
    tone_mask = (
        first_window[:, None] * first_tone[None, :]
        + second_window[:, None] * second_tone[None, :]
    )
    target = jnp.broadcast_to(first_tone, (TRIAL_STEPS, N_OUTPUT))
    trial_start = step == 0
    return tone_mask, second_window.astype(jnp.float32), target, trial_start


_encode_trials = brainstate.transform.vmap(
    _encode_one_trial,
    in_axes=0,
    out_axes=(0, 0, 0, 0),
)


def build_trial_batch(
    orders: jax.Array,
    learning: jax.Array,
    teacher_enabled: jax.Array,
) -> TrialBatch:
    """Batch trial construction with ``vmap``, then flatten trials into time."""
    orders = jnp.asarray(orders, dtype=jnp.int32)
    learning = jnp.asarray(learning, dtype=bool)
    teacher_enabled = jnp.asarray(teacher_enabled, dtype=bool)
    if not (orders.ndim == learning.ndim == teacher_enabled.ndim == 1):
        raise ValueError("orders, learning, and teacher_enabled must be vectors.")
    if not (orders.shape == learning.shape == teacher_enabled.shape):
        raise ValueError("orders, learning, and teacher_enabled must have one shape.")

    tone_mask, teacher_gate, target, trial_start = _encode_trials(orders)
    teacher_gate = teacher_gate * teacher_enabled[:, None]
    learning_by_step = jnp.broadcast_to(learning[:, None], teacher_gate.shape)

    return TrialBatch(
        tone_current=tone_mask.reshape(-1, N_SENSORY) * TONE_CURRENT,
        teacher_gate=teacher_gate.reshape(-1),
        target=target.reshape(-1, N_OUTPUT),
        learning=learning_by_step.reshape(-1),
        trial_start=trial_start.reshape(-1),
        orders=orders,
    )


class TemporalOrderCircuit(brainstate.nn.Module):
    """Two sensory neurons, an event delay line, and two output neurons."""

    def __init__(self):
        super().__init__()
        neuron_parameters = dict(
            R=1.0 * u.ohm,
            tau=MEMBRANE_TAU,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            tau_ref=REFRACTORY_TIME,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        self.sensory = brainpy.state.LIFRef(N_SENSORY, **neuron_parameters)
        self.output = brainpy.state.LIFRef(N_OUTPUT, **neuron_parameters)
        self.sensory_delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((N_SENSORY,), bool),
            TONE_GAP,
        )
        self.weights = brainstate.LongTermState(
            jnp.full((N_FEATURES, N_OUTPUT), INITIAL_WEIGHT, dtype=jnp.float32)
        )
        self.teacher_trace = brainstate.ShortTermState(
            jnp.zeros(N_OUTPUT, dtype=jnp.float32)
        )

    def update(
        self,
        time,
        tone_current,
        teacher_gate,
        target,
        learning,
        trial_start,
    ):
        with brainstate.environ.context(t=time):
            delayed_spikes = self.sensory_delay.retrieve_at_step(
                # Retrieval precedes insertion of this step's sensory event.
                jnp.asarray(DELAY_STEPS - 1, dtype=jnp.int32)
            )
            sensory_spikes = self.sensory(tone_current) != 0.0
            self.sensory_delay(sensory_spikes)

            features = jnp.concatenate((delayed_spikes, sensory_spikes))
            event_score = brainevent.BinaryArray(features) @ self.weights.value
            output_current = event_score * EVENT_CURRENT
            output_current += teacher_gate * target * TEACHER_CURRENT
            output_spikes = self.output(output_current) != 0.0

            decay = u.math.exp(-brainstate.environ.get_dt() / TEACHER_TRACE_TAU)
            decayed_trace = self.teacher_trace.value * decay
            decayed_trace = jnp.where(
                trial_start,
                jnp.zeros_like(decayed_trace),
                decayed_trace,
            )
            signed_target = 2.0 * target - 1.0
            self.teacher_trace.value = decayed_trace + teacher_gate * signed_target

            self.weights.value = brainevent.update_dense_on_binary_pre(
                weight=self.weights.value,
                pre_spike=features,
                post_trace=(
                    self.teacher_trace.value
                    * LEARNING_RATE
                    * learning.astype(jnp.float32)
                    * teacher_gate
                ),
                w_min=MIN_WEIGHT,
                w_max=MAX_WEIGHT,
            )
            return (
                sensory_spikes,
                features,
                event_score,
                output_spikes,
                self.weights.value,
            )


def run_sequence(circuit: TemporalOrderCircuit, batch: TrialBatch):
    """Run one continuous sequence; State carries dynamics and plasticity."""
    total_steps = batch.tone_current.shape[0]
    times = u.math.arange(0.0 * u.ms, total_steps * DT, DT)

    def run():
        return brainstate.transform.for_loop(
            circuit.update,
            times,
            batch.tone_current,
            batch.teacher_gate,
            batch.target,
            batch.learning,
            batch.trial_start,
        )

    return brainstate.transform.jit(run)()


def ideal_order_features(order):
    """Return the delayed/direct event pattern at the second-tone onset."""
    first_tone = jax.nn.one_hot(order, N_SENSORY, dtype=jnp.float32)
    second_tone = 1.0 - first_tone
    return jnp.concatenate((first_tone, second_tone)).astype(bool)


def _score_one_trial(order, weights):
    return brainevent.BinaryArray(ideal_order_features(order)) @ weights


_score_trial_batch = brainstate.transform.vmap(
    _score_one_trial,
    in_axes=(0, 0),
    out_axes=0,
)


def run_experiment(phase_trials: int = 14) -> ExperimentResult:
    """Teach A->B, reverse to B->A, and run teacher-free probes."""
    if phase_trials < 1:
        raise ValueError("phase_trials must be positive.")

    # Probe trials after each phase receive neither a teaching current nor a
    # plasticity update, so their output spikes are genuine recognition tests.
    orders = jnp.asarray(
        [0] * phase_trials
        + [0, 1]
        + [1] * phase_trials
        + [0, 1],
        dtype=jnp.int32,
    )
    learning = jnp.asarray(
        [True] * phase_trials
        + [False, False]
        + [True] * phase_trials
        + [False, False],
        dtype=bool,
    )
    batch = build_trial_batch(orders, learning, learning)

    with brainstate.environ.context(dt=DT):
        circuit = TemporalOrderCircuit()
        brainstate.nn.init_all_states(circuit)
        initial_weights = np.asarray(circuit.weights.value)
        _, _, _, output_spikes, weight_history = run_sequence(circuit, batch)

    output_by_trial = np.asarray(output_spikes).reshape(
        orders.size, TRIAL_STEPS, N_OUTPUT
    )
    probe_spike_counts = output_by_trial.sum(axis=1)
    end_of_trial_weights = np.asarray(weight_history)[
        TRIAL_STEPS - 1 :: TRIAL_STEPS
    ]

    phase1_end = phase_trials - 1
    phase2_start = phase_trials + 2
    phase2_end = phase2_start + phase_trials - 1
    training_indices = np.r_[
        np.arange(phase_trials),
        np.arange(phase2_start, phase2_start + phase_trials),
    ]
    training_orders = np.asarray(orders)[training_indices]
    training_weights = end_of_trial_weights[training_indices]
    training_scores = np.asarray(
        _score_trial_batch(jnp.asarray(training_orders), jnp.asarray(training_weights))
    )

    phase1_probe_start = phase_trials
    phase2_probe_start = phase2_end + 1
    return ExperimentResult(
        phase_trials=phase_trials,
        training_orders=training_orders,
        training_scores=training_scores,
        training_weights=training_weights,
        initial_weights=initial_weights,
        phase1_weights=end_of_trial_weights[phase1_end],
        phase2_weights=end_of_trial_weights[phase2_end],
        phase1_probe_spikes=probe_spike_counts[
            phase1_probe_start : phase1_probe_start + 2
        ],
        phase2_probe_spikes=probe_spike_counts[
            phase2_probe_start : phase2_probe_start + 2
        ],
    )


def save_figure(result: ExperimentResult, path: str | Path) -> Path:
    """Plot acquisition, reversal, learned weights, and teacher-free probes."""
    import matplotlib.pyplot as plt

    path = Path(path)
    trial = np.arange(1, 2 * result.phase_trials + 1)
    target_index = result.training_orders
    correct = result.training_scores[np.arange(trial.size), target_index]
    competitor = result.training_scores[np.arange(trial.size), 1 - target_index]

    ab_pair = np.array([0, 3])
    ba_pair = np.array([1, 2])
    ab_a_first = result.training_weights[:, ab_pair, 0].sum(axis=1)
    ab_b_first = result.training_weights[:, ab_pair, 1].sum(axis=1)
    ba_a_first = result.training_weights[:, ba_pair, 0].sum(axis=1)
    ba_b_first = result.training_weights[:, ba_pair, 1].sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    ax.plot(trial, correct, color="#147d64", lw=2.0, label="taught output")
    ax.plot(trial, competitor, color="#a33b3b", lw=1.6, label="competing output")
    ax.axvline(result.phase_trials + 0.5, color="0.2", ls="--", lw=1.0)
    ax.set(title="Order preference adapts after reversal", ylabel="event drive")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(trial, ab_a_first, color="#285f9e", lw=2.0, label="AB -> A-first")
    ax.plot(trial, ab_b_first, color="#8eb4df", lw=1.4, label="AB -> B-first")
    ax.plot(trial, ba_b_first, color="#b24b31", lw=2.0, label="BA -> B-first")
    ax.plot(trial, ba_a_first, color="#e1a18f", lw=1.4, label="BA -> A-first")
    ax.axvline(result.phase_trials + 0.5, color="0.2", ls="--", lw=1.0)
    ax.set(title="Coincident event-pair efficacy", ylabel="sum of two weights")
    ax.legend(frameon=False, fontsize=8)

    probe = np.stack((result.phase1_probe_spikes, result.phase2_probe_spikes))
    x = np.arange(4)
    width = 0.34
    flattened = probe.reshape(4, N_OUTPUT)
    ax = axes[1, 0]
    ax.bar(x - width / 2, flattened[:, 0], width, color="#285f9e", label="A-first output")
    ax.bar(x + width / 2, flattened[:, 1], width, color="#b24b31", label="B-first output")
    ax.set_xticks(x, ["AB\nphase 1", "BA\nphase 1", "AB\nphase 2", "BA\nphase 2"])
    ax.set(title="Teacher-free output spikes", ylabel="spikes per probe trial")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    weights = np.stack(
        (result.initial_weights, result.phase1_weights, result.phase2_weights)
    )
    image = ax.imshow(weights.reshape(3 * N_FEATURES, N_OUTPUT), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.axhline(3.5, color="white", lw=1.5)
    ax.axhline(7.5, color="white", lw=1.5)
    ax.set_xticks([0, 1], ["A", "B"])
    ax.set_xlabel("first-tone output")
    ax.set_yticks(
        np.arange(3 * N_FEATURES),
        [
            "initial: delayed A", "delayed B", "direct A", "direct B",
            "after AB: delayed A", "delayed B", "direct A", "direct B",
            "after BA: delayed A", "delayed B", "direct A", "direct B",
        ],
        fontsize=7,
    )
    ax.set_title("Plastic synapses")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("training trial" if ax in axes[0] else ax.get_xlabel())
    fig.suptitle("A delayed-event spiking circuit learns temporal order", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def print_summary(result: ExperimentResult) -> None:
    """Print the teacher-free probe outcome in a compact table."""
    print("Teacher-free output spikes [A-first, B-first]")
    for phase, probes in (("after A->B", result.phase1_probe_spikes), ("after B->A", result.phase2_probe_spikes)):
        print(f"  {phase} training:")
        for order_name, counts in zip(ORDER_NAMES, probes):
            print(f"    {order_name:8s}: {counts.astype(int).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-trials",
        type=int,
        default=14,
        help="teaching trials before and after reversal (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="temporal_order_relearning.png",
        help="summary figure path (default: %(default)s)",
    )
    args = parser.parse_args()

    result = run_experiment(args.phase_trials)
    print_summary(result)
    print(f"Saved figure: {save_figure(result, args.output)}")


if __name__ == "__main__":
    main()
