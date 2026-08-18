"""Learn whether tone A or tone B occurred first, then learn the reverse order.

The circuit contains three two-neuron populations:

* sensory LIF neurons for tones A and B;
* LIF coincidence neurons for the sequences A->B and B->A;
* LIF readout neurons whose labels are "A first" and "B first".

An eligibility trace makes an order neuron fire only when one tone follows the
other inside the temporal-order window.  During feedback, a replay of that
event and the readout spike traces update stored CSR weights with BrainEvent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import brainevent
import brainpy
import brainstate
from brainstate.transform import vmap2
from brainstate.util.filter import OfType
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


DT = 1.0 * u.ms
TRIAL_DURATION = 50.0 * u.ms
FIRST_TONE_TIME = 8.0 * u.ms
INTER_TONE_DELAY = 15.0 * u.ms
TONE_DURATION = 3.0 * u.ms
FEEDBACK_DELAY = 10.0 * u.ms
FEEDBACK_DURATION = 3.0 * u.ms

V_REST = -60.0 * u.mV
V_RESET = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
MEMBRANE_TAU = 10.0 * u.ms
REFRACTORY_TIME = 4.0 * u.ms

TONE_CURRENT = 140.0 * u.mA
DETECTOR_CURRENT = 140.0 * u.mA
READOUT_CURRENT = 160.0 * u.mA
TEACHER_CURRENT = 140.0 * u.mA

ORDER_TRACE_TAU = 25.0 * u.ms
ELIGIBILITY_TAU = 20.0 * u.ms
POST_TRACE_TAU = 10.0 * u.ms
TRACE_THRESHOLD = 0.35
LEARNING_RATE = 0.16
INITIAL_WEIGHT = 0.05
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0

ORDER_NAMES = ("A->B", "B->A")
OUTPUT_NAMES = ("A first", "B first")


def _steps(duration: u.Quantity["time"]) -> int:
    """Convert a unitful duration to a schedule index at the host boundary."""
    return int(round(duration.to_decimal(u.ms) / DT.to_decimal(u.ms)))


STEPS_PER_TRIAL = _steps(TRIAL_DURATION)
FIRST_TONE_STEP = _steps(FIRST_TONE_TIME)
SECOND_TONE_STEP = _steps(FIRST_TONE_TIME + INTER_TONE_DELAY)
TONE_STEPS = _steps(TONE_DURATION)
FEEDBACK_STEP = _steps(
    FIRST_TONE_TIME + INTER_TONE_DELAY + FEEDBACK_DELAY
)
FEEDBACK_STEPS = _steps(FEEDBACK_DURATION)


@dataclass(frozen=True)
class Protocol:
    sensory_current: u.Quantity
    teacher_current: u.Quantity
    target_sign: jnp.ndarray
    learn_gate: jnp.ndarray
    response_gate: jnp.ndarray
    labels: np.ndarray
    orders: np.ndarray


def make_protocol(orders: np.ndarray, *, teach: bool) -> Protocol:
    """Build a time-major stimulus schedule for a sequence of trials.

    ``orders == 0`` means A then B and ``orders == 1`` means B then A.  The
    correct readout reports the identity of the first tone.
    """
    orders = np.asarray(orders, dtype=np.int32)
    labels = orders.copy()
    n_trials = len(orders)
    n_steps = n_trials * STEPS_PER_TRIAL

    sensory_ma = np.zeros((n_steps, 2), dtype=np.float32)
    teacher_ma = np.zeros((n_steps, 2), dtype=np.float32)
    target_sign = np.zeros((n_steps, 2), dtype=np.float32)
    learn_gate = np.zeros(n_steps, dtype=bool)
    response_gate = np.zeros(n_steps, dtype=bool)

    for trial, order in enumerate(orders):
        offset = trial * STEPS_PER_TRIAL
        first, second = (0, 1) if order == 0 else (1, 0)
        sensory_ma[
            offset + FIRST_TONE_STEP : offset + FIRST_TONE_STEP + TONE_STEPS,
            first,
        ] = TONE_CURRENT.to_decimal(u.mA)
        sensory_ma[
            offset + SECOND_TONE_STEP : offset + SECOND_TONE_STEP + TONE_STEPS,
            second,
        ] = TONE_CURRENT.to_decimal(u.mA)
        response_gate[
            offset + SECOND_TONE_STEP : offset + FEEDBACK_STEP
        ] = True

        if teach:
            feedback = slice(
                offset + FEEDBACK_STEP,
                offset + FEEDBACK_STEP + FEEDBACK_STEPS,
            )
            teacher_ma[feedback, labels[trial]] = TEACHER_CURRENT.to_decimal(
                u.mA
            )
            target_sign[feedback] = -1.0
            target_sign[feedback, labels[trial]] = 1.0
            learn_gate[feedback] = True

    return Protocol(
        sensory_current=jnp.asarray(sensory_ma) * u.mA,
        teacher_current=jnp.asarray(teacher_ma) * u.mA,
        target_sign=jnp.asarray(target_sign),
        learn_gate=jnp.asarray(learn_gate),
        response_gate=jnp.asarray(response_gate),
        labels=labels,
        orders=orders,
    )


def make_batched_protocol(orders: np.ndarray) -> tuple[u.Quantity, jnp.ndarray]:
    """Build independent one-trial schedules with a leading trial axis."""
    protocols = [make_protocol(np.asarray([order]), teach=False) for order in orders]
    currents = u.math.stack([protocol.sensory_current for protocol in protocols])
    return currents, protocols[0].response_gate


class TemporalOrderCircuit(brainstate.nn.Module):
    """A small spiking temporal-order circuit with persistent plastic weights."""

    def __init__(
        self,
        *,
        plastic: bool,
        weights: jnp.ndarray | None = None,
        batch_size: int | None = None,
    ):
        super().__init__()
        self.plastic = plastic

        neuron_args = dict(
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau=MEMBRANE_TAU,
            tau_ref=REFRACTORY_TIME,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.sensory = brainpy.state.LIFRef(2, **neuron_args)
        self.detectors = brainpy.state.LIFRef(2, **neuron_args)
        self.outputs = brainpy.state.LIFRef(2, **neuron_args)

        trace_shape = (2,) if batch_size is None else (batch_size, 2)
        self.tone_trace = brainstate.ShortTermState(jnp.zeros(trace_shape))
        self.eligibility = brainstate.ShortTermState(jnp.zeros(trace_shape))
        self.post_trace = brainstate.ShortTermState(jnp.zeros(trace_shape))

        if weights is None:
            weight_value = jnp.full((4,), INITIAL_WEIGHT, dtype=jnp.float32)
        else:
            weight_value = jnp.asarray(weights, dtype=jnp.float32)
        self.weights = brainstate.LongTermState(weight_value)

        # Full 2x2 topology stored in CSR row order:
        # [AB->A-first, AB->B-first, BA->A-first, BA->B-first].
        self.indices = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
        self.indptr = jnp.asarray([0, 2, 4], dtype=jnp.int32)
        self.conn_shape = (2, 2)

    def update(
        self,
        sensory_current,
        teacher_current,
        target_sign,
        learn_gate,
    ):
        sensory_spike = self.sensory(sensory_current) != 0.0

        previous_tone = self.tone_trace.value
        order_event = jnp.stack(
            (
                sensory_spike[1] & (previous_tone[0] > TRACE_THRESHOLD),
                sensory_spike[0] & (previous_tone[1] > TRACE_THRESHOLD),
            )
        )
        detector_drive = u.math.where(
            order_event, DETECTOR_CURRENT, 0.0 * u.mA
        )
        detector_spike = self.detectors(detector_drive) != 0.0

        weight_matrix = brainevent.CSR(
            (self.weights.value, self.indices, self.indptr),
            shape=self.conn_shape,
        )
        efficacy = brainevent.BinaryArray(detector_spike) @ weight_matrix
        output_drive = efficacy * READOUT_CURRENT + teacher_current
        output_spike = self.outputs(output_drive) != 0.0

        dt = brainstate.environ.get_dt()
        tone_decay = u.math.exp(-dt / ORDER_TRACE_TAU)
        eligibility_decay = u.math.exp(-dt / ELIGIBILITY_TAU)
        post_decay = u.math.exp(-dt / POST_TRACE_TAU)
        self.tone_trace.value = (
            previous_tone * tone_decay + sensory_spike.astype(jnp.float32)
        )
        self.eligibility.value = (
            self.eligibility.value * eligibility_decay
            + detector_spike.astype(jnp.float32)
        )
        self.post_trace.value = (
            self.post_trace.value * post_decay
            + output_spike.astype(jnp.float32)
        )

        if self.plastic:
            replay_event = learn_gate & (
                self.eligibility.value > TRACE_THRESHOLD
            )
            signed_post_trace = self.post_trace.value * target_sign * LEARNING_RATE
            self.weights.value = brainevent.update_csr_on_binary_pre(
                weight=self.weights.value,
                indices=self.indices,
                indptr=self.indptr,
                pre_spike=replay_event,
                post_trace=signed_post_trace,
                w_min=WEIGHT_MIN,
                w_max=WEIGHT_MAX,
                shape=self.conn_shape,
            )

        return sensory_spike, detector_spike, output_spike, self.weights.value


def train(orders: np.ndarray):
    """Run all causally dependent learning trials in one State-aware loop."""
    protocol = make_protocol(orders, teach=True)
    circuit = TemporalOrderCircuit(plastic=True)

    with brainstate.environ.context(dt=DT):
        brainstate.nn.init_all_states(circuit)
        times = u.math.arange(
            0.0 * u.ms,
            len(orders) * TRIAL_DURATION,
            brainstate.environ.get_dt(),
        )

        @brainstate.transform.jit
        def run():
            def step(t, sensory, teacher, sign, learn):
                with brainstate.environ.context(t=t):
                    return circuit.update(sensory, teacher, sign, learn)

            return brainstate.transform.for_loop(
                step,
                times,
                protocol.sensory_current,
                protocol.teacher_current,
                protocol.target_sign,
                protocol.learn_gate,
            )

        sensory_spikes, detector_spikes, output_spikes, weight_history = run()

    return (
        protocol,
        circuit,
        sensory_spikes,
        detector_spikes,
        output_spikes,
        weight_history,
    )


def trial_responses(output_spikes, response_gate, labels):
    """Decode pre-feedback spikes; silence counts as an incorrect response."""
    spikes = np.asarray(output_spikes, dtype=bool).reshape(
        len(labels), STEPS_PER_TRIAL, 2
    )
    gate = np.asarray(response_gate, dtype=bool).reshape(
        len(labels), STEPS_PER_TRIAL
    )
    counts = (spikes * gate[..., None]).sum(axis=1)
    predictions = counts.argmax(axis=1)
    responded = counts.max(axis=1) > 0
    correct = responded & (predictions == labels)
    return counts, predictions, correct


def evaluate_batched(learned_weights, orders: np.ndarray):
    """Evaluate independent trials with mapped dynamical State and shared weights."""
    orders = np.asarray(orders, dtype=np.int32)
    batch_size = len(orders)
    sensory_current, response_gate = make_batched_protocol(orders)
    repeated_weights = jnp.broadcast_to(
        jnp.asarray(learned_weights), (batch_size, 4)
    )
    circuit = TemporalOrderCircuit(
        plastic=False,
        weights=repeated_weights,
        batch_size=batch_size,
    )

    with brainstate.environ.context(dt=DT):
        brainstate.nn.init_all_states(circuit, batch_size=batch_size)
        times = u.math.arange(0.0 * u.ms, TRIAL_DURATION, DT)

        def run_one(trial_current):
            def step(t, current):
                with brainstate.environ.context(t=t):
                    zero_current = jnp.zeros(2) * u.mA
                    zero_sign = jnp.zeros(2)
                    return circuit.update(
                        current,
                        zero_current,
                        zero_sign,
                        jnp.asarray(False),
                    )[2]

            output_spikes = brainstate.transform.for_loop(
                step, times, trial_current
            )
            return u.math.sum(
                output_spikes * response_gate[:, None], axis=0
            )

        mapped_run = vmap2(
            run_one,
            in_axes=0,
            out_axes=0,
            state_in_axes={0: OfType(brainstate.State)},
            state_out_axes={0: OfType(brainstate.State)},
        )
        counts = brainstate.transform.jit(mapped_run)(sensory_current)

    counts = np.asarray(counts)
    predictions = counts.argmax(axis=1)
    responded = counts.max(axis=1) > 0
    correct = responded & (predictions == orders)
    return counts, predictions, correct


def plot_results(
    correct,
    trial_weights,
    trials_per_phase,
    eval_orders,
    eval_counts,
    output_path: Path,
):
    trials = np.arange(1, len(correct) + 1)
    rolling_window = 4
    rolling = np.convolve(
        correct.astype(float),
        np.ones(rolling_window) / rolling_window,
        mode="valid",
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)

    axes[0].scatter(
        trials,
        correct.astype(int),
        c=np.where(correct, "tab:green", "tab:red"),
        s=22,
        zorder=3,
    )
    axes[0].plot(
        trials[rolling_window - 1 :],
        rolling,
        color="black",
        linewidth=1.5,
        label="4-trial accuracy",
    )
    axes[0].axvline(trials_per_phase + 0.5, color="0.35", linestyle="--")
    axes[0].set(ylim=(-0.08, 1.08), ylabel="correct before feedback")
    axes[0].set_title("The response is acquired, then the tone order reverses")
    axes[0].text(trials_per_phase / 2, 0.1, "A->B", ha="center")
    axes[0].text(1.5 * trials_per_phase, 0.1, "B->A", ha="center")
    axes[0].legend(loc="lower right")

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    labels = (
        "A->B to A-first",
        "A->B to B-first",
        "B->A to A-first",
        "B->A to B-first",
    )
    for index, (color, label) in enumerate(zip(colors, labels)):
        axes[1].plot(trials, trial_weights[:, index], color=color, label=label)
    axes[1].axvline(trials_per_phase + 0.5, color="0.35", linestyle="--")
    axes[1].set(ylabel="synaptic efficacy", xlabel="training trial", ylim=(0, 1.02))
    axes[1].set_title("Online event-driven plasticity stores each temporal order")
    axes[1].legend(ncol=2, fontsize=8, loc="upper center")

    mean_counts = np.stack(
        [eval_counts[eval_orders == order].mean(axis=0) for order in (0, 1)]
    )
    image = axes[2].imshow(mean_counts, cmap="viridis", vmin=0)
    axes[2].set_xticks([0, 1], OUTPUT_NAMES)
    axes[2].set_yticks([0, 1], ORDER_NAMES)
    axes[2].set(xlabel="readout population", ylabel="held-out order")
    axes[2].set_title("Batched held-out response after both phases")
    for row in range(2):
        for column in range(2):
            axes[2].text(
                column,
                row,
                f"{mean_counts[row, column]:.1f}",
                ha="center",
                va="center",
                color="white" if mean_counts[row, column] > mean_counts.max() / 2 else "black",
            )
    fig.colorbar(image, ax=axes[2], label="mean response-window spikes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(trials_per_phase: int = 18, output_path: Path | None = None):
    if trials_per_phase < 2:
        raise ValueError("trials_per_phase must be at least 2")

    # Trials are deliberately sequential: phase two consumes the weight State
    # left by phase one, so this axis must not be vmapped.
    training_orders = np.concatenate(
        (
            np.zeros(trials_per_phase, dtype=np.int32),
            np.ones(trials_per_phase, dtype=np.int32),
        )
    )
    (
        protocol,
        circuit,
        _sensory_spikes,
        _detector_spikes,
        output_spikes,
        weight_history,
    ) = train(training_orders)
    _, predictions, correct = trial_responses(
        output_spikes, protocol.response_gate, protocol.labels
    )
    trial_weights = np.asarray(weight_history).reshape(
        len(training_orders), STEPS_PER_TRIAL, 4
    )[:, -1]

    eval_orders = np.tile(np.asarray([0, 1], dtype=np.int32), 8)
    eval_counts, eval_predictions, eval_correct = evaluate_batched(
        circuit.weights.value, eval_orders
    )

    if output_path is not None:
        plot_results(
            correct,
            trial_weights,
            trials_per_phase,
            eval_orders,
            eval_counts,
            output_path,
        )

    phase_one_accuracy = correct[:trials_per_phase].mean()
    phase_two_accuracy = correct[trials_per_phase:].mean()
    final_eval_accuracy = eval_correct.mean()
    print(f"phase 1 A->B accuracy: {phase_one_accuracy:.1%}")
    print(f"phase 2 B->A accuracy: {phase_two_accuracy:.1%}")
    print(f"held-out mixed-order accuracy: {final_eval_accuracy:.1%}")
    print("final weights [AB->A, AB->B, BA->A, BA->B]:")
    print(np.asarray(circuit.weights.value))
    if output_path is not None:
        print(f"figure: {output_path}")

    return {
        "training_orders": training_orders,
        "training_predictions": predictions,
        "training_correct": correct,
        "trial_weights": trial_weights,
        "eval_orders": eval_orders,
        "eval_predictions": eval_predictions,
        "eval_correct": eval_correct,
        "eval_counts": eval_counts,
        "final_weights": np.asarray(circuit.weights.value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials-per-phase",
        type=int,
        default=18,
        help="sequential training trials before and after order reversal",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("temporal_order_learning.png"),
        help="path for the summary figure",
    )
    args = parser.parse_args()
    run_experiment(args.trials_per_phase, args.output)


if __name__ == "__main__":
    main()
