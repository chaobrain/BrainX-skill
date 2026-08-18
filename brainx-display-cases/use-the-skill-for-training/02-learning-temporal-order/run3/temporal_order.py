"""Learn two-tone temporal order, reverse the association, and relearn it.

The two sensory neurons represent tones A and B. Two output neurons represent
the currently rewarded order. A supervised, event-triggered plasticity event
arrives after both tones: the recent sensory eligibility trace potentiates the
target output column and depresses the competing column.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/brainx-matplotlib")

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.util.filter import OfType


DT = 1.0 * u.ms
TRIAL_DURATION = 48.0 * u.ms
FIRST_TONE_AT = 5.0 * u.ms
SECOND_TONE_AT = 20.0 * u.ms
TONE_DURATION = 5.0 * u.ms
AXONAL_DELAY = 2.0 * u.ms
AXONAL_DELAY_STEPS = int(round(AXONAL_DELAY / DT))
DECISION_AT = 27.0 * u.ms
TEACHER_AT = 35.0 * u.ms
TEACHER_DURATION = 5.0 * u.ms
PLASTICITY_AT = 40.0 * u.ms

TONE_CURRENT = 32.0 * u.mA
SYNAPTIC_CURRENT = 31.0 * u.mA
TEACHER_CURRENT = 42.0 * u.mA
TAU_ELIGIBILITY = 20.0 * u.ms
TAU_SYNAPSE = 24.0 * u.ms
LEARNING_RATE = 0.10
DEPRESSION_RATIO = 0.75
W_MIN = 0.05
W_MAX = 1.00

V_REST = -60.0 * u.mV
V_RESET = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV

ACQUISITION_TARGETS = np.array([0, 1], dtype=np.int32)  # AB -> 0, BA -> 1
REVERSAL_TARGETS = np.array([1, 0], dtype=np.int32)     # AB -> 1, BA -> 0


class TemporalOrderCircuit(brainstate.nn.Module):
    """Two sensory LIF neurons, two decision LIF neurons, and plastic weights."""

    def __init__(self):
        super().__init__()
        lif_common = dict(
            R=1.0 * u.ohm,
            V_rest=V_REST,
            V_reset=V_RESET,
            V_th=V_THRESHOLD,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.sensory = brainpy.state.LIFRef(
            2,
            tau=8.0 * u.ms,
            tau_ref=3.0 * u.ms,
            **lif_common,
        )
        self.output = brainpy.state.LIFRef(
            2,
            tau=12.0 * u.ms,
            tau_ref=4.0 * u.ms,
            **lif_common,
        )
        self.synapse = brainpy.state.Expon(
            2,
            tau=TAU_SYNAPSE,
            g_initializer=braintools.init.Constant(0.0 * u.mA),
        )
        self.axon = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((2,), jnp.bool_),
            AXONAL_DELAY,
        )
        self.pre_trace = brainstate.ShortTermState(jnp.zeros(2, dtype=jnp.float32))
        self.weight = brainstate.LongTermState(
            jnp.full((2, 2), 0.30, dtype=jnp.float32)
        )

    def update(
        self,
        t,
        tone_drive,
        teacher_drive,
        target_event,
        competitor_event,
        learning_rate,
    ):
        with brainstate.environ.context(t=t):
            sensory_spikes = self.sensory(tone_drive) != 0.0

            # Insert before retrieval: delay step 0 is current, and 2 ms is
            # exactly two completed 1 ms updates earlier.
            self.axon.update(sensory_spikes)
            arrived_spikes = self.axon.retrieve_at_step(
                jnp.asarray(AXONAL_DELAY_STEPS, dtype=jnp.int32)
            )

            efficacy = brainevent.BinaryArray(arrived_spikes) @ self.weight.value
            synaptic_current = self.synapse(efficacy * SYNAPTIC_CURRENT)
            decision_gate = t >= DECISION_AT
            output_spikes = self.output(
                synaptic_current * decision_gate + teacher_drive
            ) != 0.0

            trace_decay = u.math.exp(-brainstate.environ.get_dt() / TAU_ELIGIBILITY)
            self.pre_trace.value = (
                self.pre_trace.value * trace_decay
                + arrived_spikes.astype(jnp.float32)
            )

            # BrainEvent visits only the columns selected by each binary
            # postsynaptic teaching event. The signed eligibility supplied to
            # the second call weakens the old association during reversal.
            self.weight.value = brainevent.update_dense_on_binary_post(
                weight=self.weight.value,
                post_spike=target_event,
                pre_trace=self.pre_trace.value * learning_rate,
                w_min=W_MIN,
                w_max=W_MAX,
            )
            self.weight.value = brainevent.update_dense_on_binary_post(
                weight=self.weight.value,
                post_spike=competitor_event,
                pre_trace=-self.pre_trace.value * learning_rate * DEPRESSION_RATIO,
                w_min=W_MIN,
                w_max=W_MAX,
            )

            return (
                sensory_spikes,
                arrived_spikes,
                output_spikes,
                self.output.V.value,
                self.pre_trace.value,
            )


def make_trial(times, order: int, target: int, *, teach: bool, jitter_steps: int = 0):
    """Create one time-major trial while retaining units on all currents."""
    jitter = jitter_steps * DT
    first_mask = (times >= FIRST_TONE_AT + jitter) & (
        times < FIRST_TONE_AT + jitter + TONE_DURATION
    )
    second_mask = (times >= SECOND_TONE_AT + jitter) & (
        times < SECOND_TONE_AT + jitter + TONE_DURATION
    )

    first_tone = order
    second_tone = 1 - order
    first_code = jax.nn.one_hot(first_tone, 2, dtype=jnp.float32)
    second_code = jax.nn.one_hot(second_tone, 2, dtype=jnp.float32)
    tone_drive = (
        first_mask[:, None] * first_code[None, :]
        + second_mask[:, None] * second_code[None, :]
    ) * TONE_CURRENT

    target_code = jax.nn.one_hot(target, 2, dtype=jnp.float32)
    competitor_code = jax.nn.one_hot(1 - target, 2, dtype=jnp.float32)
    teacher_mask = teach & (times >= TEACHER_AT) & (
        times < TEACHER_AT + TEACHER_DURATION
    )
    teacher_drive = teacher_mask[:, None] * target_code[None, :] * TEACHER_CURRENT

    plasticity_mask = teach & (times == PLASTICITY_AT)
    target_event = plasticity_mask[:, None] & target_code[None, :].astype(bool)
    competitor_event = (
        plasticity_mask[:, None] & competitor_code[None, :].astype(bool)
    )
    return tone_drive, teacher_drive, target_event, competitor_event


def reset_trial_state(circuit: TemporalOrderCircuit):
    """Reset transient State while preserving the LongTermState weights."""
    brainstate.nn.reset_all_states(circuit.sensory)
    brainstate.nn.reset_all_states(circuit.output)
    brainstate.nn.reset_all_states(circuit.synapse)
    brainstate.nn.reset_all_states(circuit.axon)
    circuit.pre_trace.value = jnp.zeros(2, dtype=jnp.float32)


def decision_score(output_spikes, output_voltage, decision_mask):
    """Spike count with a small subthreshold tie-break from peak voltage."""
    spikes = np.asarray(output_spikes)[decision_mask]
    volts_mv = np.asarray(output_voltage.to_decimal(u.mV))[decision_mask]
    spike_count = spikes.sum(axis=0).astype(np.float32)
    subthreshold = (volts_mv.max(axis=0) - V_REST.to_decimal(u.mV)) / 100.0
    return spike_count + subthreshold


def train_and_reverse(acquisition_epochs=16, reversal_epochs=48):
    brainstate.random.seed(7)
    with brainstate.environ.context(dt=DT):
        times = u.math.arange(0.0 * u.ms, TRIAL_DURATION, DT)
        decision_mask = np.asarray(times >= DECISION_AT)
        circuit = TemporalOrderCircuit()
        brainstate.nn.init_all_states(circuit)

        def rollout(tone_drive, teacher_drive, target_event, competitor_event, eta):
            def step(t, tone, teacher, target, competitor):
                return circuit.update(
                    t, tone, teacher, target, competitor, eta
                )

            return brainstate.transform.for_loop(
                step,
                times,
                tone_drive,
                teacher_drive,
                target_event,
                competitor_event,
            )

        run_trial = brainstate.transform.jit(rollout)

        def run_order(order, target, *, teach, eta=0.0, jitter_steps=0):
            reset_trial_state(circuit)
            trial = make_trial(
                times, order, target, teach=teach, jitter_steps=jitter_steps
            )
            monitors = run_trial(*trial, jnp.asarray(eta, dtype=jnp.float32))
            return decision_score(monitors[2], monitors[3], decision_mask), monitors

        def evaluate(targets):
            scores = np.stack(
                [run_order(order, int(targets[order]), teach=False)[0]
                 for order in (0, 1)]
            )
            predictions = scores.argmax(axis=1)
            accuracy = np.mean(predictions == targets)
            margins = scores[np.arange(2), targets] - scores[np.arange(2), 1 - targets]
            return float(accuracy), float(margins.mean()), scores

        epochs = [0]
        accuracy, margin, _ = evaluate(ACQUISITION_TARGETS)
        accuracies = [accuracy]
        margins = [margin]

        for epoch in range(1, acquisition_epochs + 1):
            for order in (0, 1):
                target = int(ACQUISITION_TARGETS[order])
                run_order(order, target, teach=True, eta=LEARNING_RATE)
            accuracy, margin, _ = evaluate(ACQUISITION_TARGETS)
            epochs.append(epoch)
            accuracies.append(accuracy)
            margins.append(margin)

        acquisition_weight = np.asarray(circuit.weight.value).copy()
        acquisition_accuracy, _, acquisition_scores = evaluate(ACQUISITION_TARGETS)
        immediate_reversal_accuracy, immediate_reversal_margin, _ = evaluate(
            REVERSAL_TARGETS
        )

        # Insert a second point at the same boundary so the changed reward
        # contingency appears as a vertical drop rather than a sloped segment.
        epochs.append(acquisition_epochs)
        accuracies.append(immediate_reversal_accuracy)
        margins.append(immediate_reversal_margin)

        for reversal_epoch in range(1, reversal_epochs + 1):
            for order in (0, 1):
                target = int(REVERSAL_TARGETS[order])
                run_order(order, target, teach=True, eta=LEARNING_RATE)
            accuracy, margin, _ = evaluate(REVERSAL_TARGETS)
            epochs.append(acquisition_epochs + reversal_epoch)
            accuracies.append(accuracy)
            margins.append(margin)

        reversal_weight = np.asarray(circuit.weight.value).copy()
        reversal_accuracy, _, reversal_scores = evaluate(REVERSAL_TARGETS)

        vmapped_accuracy, vmapped_scores = evaluate_vmapped(
            circuit,
            rollout,
            times,
            decision_mask,
            REVERSAL_TARGETS,
        )

    results = {
        "epochs": np.asarray(epochs),
        "acquisition_epochs": acquisition_epochs,
        "accuracies": np.asarray(accuracies),
        "margins": np.asarray(margins),
        "acquisition_weight": acquisition_weight,
        "reversal_weight": reversal_weight,
        "acquisition_accuracy": acquisition_accuracy,
        "immediate_reversal_accuracy": immediate_reversal_accuracy,
        "reversal_accuracy": reversal_accuracy,
        "vmapped_accuracy": vmapped_accuracy,
        "acquisition_scores": acquisition_scores,
        "reversal_scores": reversal_scores,
        "vmapped_scores": vmapped_scores,
    }
    return results


def evaluate_vmapped(circuit, rollout, times, decision_mask, targets):
    """Batch complete independent stateful trials with BrainState vmap2."""
    orders = np.tile(np.array([0, 1], dtype=np.int32), 6)
    jitters = np.array([-1, 0, 1, 0, -1, 1] * 2, dtype=np.int32)
    trials = [
        make_trial(
            times,
            int(order),
            int(targets[order]),
            teach=False,
            jitter_steps=int(jitter),
        )
        for order, jitter in zip(orders, jitters)
    ]
    batched_inputs = tuple(u.math.stack(items, axis=0) for items in zip(*trials))

    reset_trial_state(circuit)
    batch_size = len(orders)
    for state in circuit.states().values():
        state.value = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (batch_size,) + leaf.shape),
            state.value,
        )

    state_axes = {0: OfType(brainstate.State)}
    batched_rollout = brainstate.transform.vmap2(
        rollout,
        in_axes=(0, 0, 0, 0, None),
        out_axes=0,
        state_in_axes=state_axes,
        state_out_axes=state_axes,
        unexpected_out_state_mapping="raise",
    )
    run_batch = brainstate.transform.jit(batched_rollout)
    monitors = run_batch(
        *batched_inputs,
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    scores = np.stack(
        [decision_score(monitors[2][i], monitors[3][i], decision_mask)
         for i in range(batch_size)]
    )
    predictions = scores.argmax(axis=1)
    accuracy = float(np.mean(predictions == targets[orders]))
    return accuracy, scores


def plot_results(results, path="temporal_order_relearning.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")

    axes[0, 0].plot(results["epochs"], results["accuracies"], color="black", lw=2)
    reversal_at = results["acquisition_epochs"]
    axes[0, 0].axvline(reversal_at, color="#d95f02", ls="--", lw=1.5, label="reversal")
    axes[0, 0].set(ylim=(-0.05, 1.05), ylabel="accuracy", xlabel="training epoch")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(results["epochs"], results["margins"], color="#0072b2", lw=2)
    axes[0, 1].axhline(0.0, color="0.55", lw=1)
    axes[0, 1].axvline(reversal_at, color="#d95f02", ls="--", lw=1.5)
    axes[0, 1].set(ylabel="target decision margin", xlabel="training epoch")

    labels = ["output 0", "output 1"]
    for ax, matrix, title in (
        (axes[1, 0], results["acquisition_weight"], "after acquisition"),
        (axes[1, 1], results["reversal_weight"], "after reversal"),
    ):
        image = ax.imshow(matrix, vmin=W_MIN, vmax=W_MAX, cmap="viridis", aspect="auto")
        ax.set_xticks([0, 1], labels=labels)
        ax.set_yticks([0, 1], labels=["tone A", "tone B"])
        ax.set_title(title)
        for row in range(2):
            for col in range(2):
                color = "white" if matrix[row, col] < 0.55 else "black"
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=color)

    fig.colorbar(
        image,
        ax=axes[1, :],
        label="synaptic efficacy",
        location="bottom",
        orientation="horizontal",
        shrink=0.55,
        pad=0.10,
    )
    fig.suptitle("Temporal-order learning and reversal")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    results = train_and_reverse()
    figure_path = plot_results(results)

    print("Acquisition scores [AB, BA] x [output 0, output 1]:")
    print(np.array2string(results["acquisition_scores"], precision=3))
    print("Reversal scores [AB, BA] x [output 0, output 1]:")
    print(np.array2string(results["reversal_scores"], precision=3))
    print(
        "Accuracy: "
        f"acquired={results['acquisition_accuracy']:.0%}, "
        f"immediate reversal={results['immediate_reversal_accuracy']:.0%}, "
        f"relearned={results['reversal_accuracy']:.0%}, "
        f"vmapped jittered batch={results['vmapped_accuracy']:.0%}"
    )
    print(f"Saved {figure_path}")

    assert results["acquisition_accuracy"] == 1.0
    assert results["immediate_reversal_accuracy"] <= 0.5
    assert results["reversal_accuracy"] == 1.0
    assert results["vmapped_accuracy"] >= 0.9


if __name__ == "__main__":
    main()
