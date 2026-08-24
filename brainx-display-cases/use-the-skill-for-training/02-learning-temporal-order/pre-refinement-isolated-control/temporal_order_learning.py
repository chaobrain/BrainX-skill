"""Learn A-before-B, reverse the tones, and learn B-before-A.

The model deliberately stays small enough that its mechanism is visible:

* two sensory LIF neurons encode tones A and B;
* two LIF coincidence neurons encode A->B and B->A;
* two output LIF neurons report which tone came first;
* a dense-topology CSR projection is updated online whenever an order neuron
  spikes.  A teaching signal potentiates the correct output and depresses the
  competing output.

Run this file directly to print the decisions and write
``temporal_order_relearning.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# Every physical model quantity is unit-bearing.  The dimensionless constants
# below are event amplitudes, connection weights, and learning rates.
DT = 1.0 * u.ms
TRIAL_DURATION = 100.0 * u.ms
FIRST_TONE_ONSET = 10.0 * u.ms
INTER_TONE_DELAY = 25.0 * u.ms
TONE_DURATION = 6.0 * u.ms

V_REST = -65.0 * u.mV
V_RESET = -65.0 * u.mV
V_THRESHOLD = -55.0 * u.mV
MEMBRANE_TAU = 10.0 * u.ms
REFRACTORY_PERIOD = 3.0 * u.ms
MEMBRANE_RESISTANCE = 1.0 * u.ohm

TONE_CURRENT = 40.0 * u.mA
ORDER_CURRENT = 300.0 * u.mA
READOUT_CURRENT = 70.0 * u.mA
ORDER_TRACE_TAU = 30.0 * u.ms
READOUT_SYN_TAU = 8.0 * u.ms

POTENTIATION = 0.035
DEPRESSION = 0.025
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.95

N_ORDERS = 2
N_OUTPUTS = 2
A_FIRST = 0
B_FIRST = 1

brainstate.environ.set(dt=DT, precision=32)


def make_order_templates():
    """Return time points and the two deterministic tone-order spike drives."""
    times = u.math.arange(0.0 * u.ms, TRIAL_DURATION, DT)
    second_onset = FIRST_TONE_ONSET + INTER_TONE_DELAY

    def pulse(onset):
        return (times >= onset) & (times < onset + TONE_DURATION)

    a_early = pulse(FIRST_TONE_ONSET)
    b_late = pulse(second_onset)
    ab = jnp.stack((a_early, b_late), axis=-1)
    ba = jnp.stack((b_late, a_early), axis=-1)
    return times, jnp.stack((ab, ba), axis=0)


def batch_trial_templates(templates, order_codes):
    """Select independent trial sequences with BrainState vectorization."""
    select_one = brainstate.transform.vmap(lambda code: templates[code])
    return select_one(order_codes)


def event_product(spikes, connectivity):
    """Apply one shared event projection to one trial or a batch of trials."""
    communicate_one = lambda events: brainevent.BinaryArray(events) @ connectivity
    if spikes.ndim == 1:
        return communicate_one(spikes)
    return brainstate.transform.vmap(communicate_one)(spikes)


class TemporalOrderCircuit(brainstate.nn.Module):
    """Three-stage spiking circuit with an online plastic CSR readout."""

    def __init__(self):
        super().__init__()
        neuron_parameters = dict(
            R=MEMBRANE_RESISTANCE,
            tau=MEMBRANE_TAU,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau_ref=REFRACTORY_PERIOD,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.sensory = brainpy.state.LIFRef(2, **neuron_parameters)
        self.order = brainpy.state.LIFRef(2, **neuron_parameters)
        self.output = brainpy.state.LIFRef(2, **neuron_parameters)
        # Rows are A->B and B->A detectors; columns are A-first and B-first
        # outputs.  Both rows start biased toward the wrong decision so that
        # each phase visibly has something to learn.
        self.weight_data = brainstate.LongTermState(
            jnp.asarray([0.20, 0.35, 0.35, 0.20], dtype=jnp.float32)
        )
        self.sensory_trace = brainstate.ShortTermState(
            jnp.zeros(2, dtype=jnp.float32)
        )
        self.readout_trace = brainstate.ShortTermState(
            jnp.zeros(2, dtype=jnp.float32)
        )

        # Fully connected 2x2 CSR topology.  Plasticity changes data only.
        self.indices = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
        self.indptr = jnp.asarray([0, 2, 4], dtype=jnp.int32)
        self.crossed_sensory = jnp.asarray(
            [[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32
        )
        self.trace_decay = u.math.exp(-DT / ORDER_TRACE_TAU)
        self.readout_decay = u.math.exp(-DT / READOUT_SYN_TAU)

    @property
    def weight_matrix(self):
        return self.weight_data.value.reshape(N_ORDERS, N_OUTPUTS)

    def init_dynamic_state(self, batch_size: int | None = None):
        """Initialize trial dynamics while preserving learned weights."""
        modules = (self.sensory, self.order, self.output)
        for module in modules:
            if batch_size is None:
                brainstate.nn.init_all_states(module)
            else:
                brainstate.nn.init_all_states(module, batch_size=batch_size)
        shape = (2,) if batch_size is None else (batch_size, 2)
        self.sensory_trace.value = jnp.zeros(shape, dtype=jnp.float32)
        self.readout_trace.value = jnp.zeros(shape, dtype=jnp.float32)

    def update(self, t, tone_events, reset_trace, target, learning: bool):
        """Advance one millisecond and optionally update readout weights."""
        with brainstate.environ.context(t=t):
            trace = jnp.where(
                reset_trace,
                jnp.zeros_like(self.sensory_trace.value),
                self.sensory_trace.value,
            )
            trace = trace * self.trace_decay

            sensory_spikes = self.sensory(
                tone_events.astype(jnp.float32) * TONE_CURRENT
            ) != 0.0

            # [A spike, B spike] -> [B spike, A spike].  Multiplication by the
            # old [A trace, B trace] yields [A->B evidence, B->A evidence].
            crossed_spikes = event_product(sensory_spikes, self.crossed_sensory)
            coincidence = trace * crossed_spikes
            self.sensory_trace.value = (
                trace + sensory_spikes.astype(jnp.float32)
            )

            order_spikes = self.order(coincidence * ORDER_CURRENT) != 0.0
            readout = brainevent.CSR(
                (self.weight_data.value, self.indices, self.indptr),
                shape=(N_ORDERS, N_OUTPUTS),
            )
            output_events = event_product(order_spikes, readout)
            self.readout_trace.value = (
                self.readout_trace.value * self.readout_decay + output_events
            )
            output_spikes = self.output(
                self.readout_trace.value * READOUT_CURRENT
            ) != 0.0

            if learning:
                correct = jax.nn.one_hot(target, N_OUTPUTS, dtype=jnp.float32)
                teaching_trace = (
                    correct * POTENTIATION
                    - (1.0 - correct) * DEPRESSION
                )
                self.weight_data.value = brainevent.update_csr_on_binary_pre(
                    weight=self.weight_data.value,
                    indices=self.indices,
                    indptr=self.indptr,
                    pre_spike=order_spikes,
                    post_trace=teaching_trace,
                    w_min=WEIGHT_MIN,
                    w_max=WEIGHT_MAX,
                    shape=(N_ORDERS, N_OUTPUTS),
                )

            return output_spikes, order_spikes, self.weight_data.value


def train_phase(
    circuit: TemporalOrderCircuit,
    template,
    target: int,
    n_trials: int,
):
    """Train one repeated order as a single state-aware sequence loop."""
    steps_per_trial = template.shape[0]
    total_steps = n_trials * steps_per_trial
    events = jnp.tile(template[None, :, :], (n_trials, 1, 1)).reshape(
        total_steps, 2
    )
    reset_trace = (jnp.arange(total_steps) % steps_per_trial) == 0
    times = u.math.arange(0.0 * u.ms, total_steps * DT, DT)

    circuit.init_dynamic_state()

    @brainstate.transform.jit
    def run():
        def step(t, tone_event, reset):
            return circuit.update(t, tone_event, reset, target, learning=True)

        return brainstate.transform.for_loop(step, times, events, reset_trace)

    output_spikes, order_spikes, weights = run()
    output_counts = output_spikes.reshape(
        n_trials, steps_per_trial, N_OUTPUTS
    ).sum(axis=1)
    detector_counts = order_spikes.reshape(
        n_trials, steps_per_trial, N_ORDERS
    ).sum(axis=1)
    trial_weights = weights.reshape(n_trials, steps_per_trial, 4)[:, -1, :]
    return output_counts, detector_counts, trial_weights.reshape(n_trials, 2, 2)


def evaluate_orders(circuit: TemporalOrderCircuit, times, templates):
    """Run AB and BA as an independent two-trial batch with shared weights."""
    order_codes = jnp.asarray([A_FIRST, B_FIRST], dtype=jnp.int32)
    trials = batch_trial_templates(templates, order_codes)
    time_major_trials = jnp.swapaxes(trials, 0, 1)
    reset_trace = jnp.arange(times.shape[0]) == 0
    targets = jnp.asarray([A_FIRST, B_FIRST], dtype=jnp.int32)

    circuit.init_dynamic_state(batch_size=order_codes.shape[0])

    @brainstate.transform.jit
    def run():
        def step(t, tone_events, reset):
            outputs, detectors, _ = circuit.update(
                t, tone_events, reset, targets, learning=False
            )
            return outputs, detectors

        return brainstate.transform.for_loop(
            step, times, time_major_trials, reset_trace
        )

    output_spikes, detector_spikes = run()
    return output_spikes.sum(axis=0), detector_spikes.sum(axis=0)


def predicted_orders(output_counts):
    """Decode each output population by spike count."""
    return jnp.argmax(output_counts, axis=-1)


def run_experiment(n_trials: int = 18):
    """Run baseline, A-first learning, and reversed B-first learning."""
    times, templates = make_order_templates()
    circuit = TemporalOrderCircuit()

    before_outputs, before_detectors = evaluate_orders(circuit, times, templates)
    phase_ab = train_phase(circuit, templates[A_FIRST], A_FIRST, n_trials)
    after_ab_outputs, after_ab_detectors = evaluate_orders(circuit, times, templates)
    phase_ba = train_phase(circuit, templates[B_FIRST], B_FIRST, n_trials)
    after_ba_outputs, after_ba_detectors = evaluate_orders(circuit, times, templates)

    return {
        "times": times,
        "templates": templates,
        "before_outputs": before_outputs,
        "before_detectors": before_detectors,
        "phase_ab_outputs": phase_ab[0],
        "phase_ab_detectors": phase_ab[1],
        "phase_ab_weights": phase_ab[2],
        "after_ab_outputs": after_ab_outputs,
        "after_ab_detectors": after_ab_detectors,
        "phase_ba_outputs": phase_ba[0],
        "phase_ba_detectors": phase_ba[1],
        "phase_ba_weights": phase_ba[2],
        "after_ba_outputs": after_ba_outputs,
        "after_ba_detectors": after_ba_detectors,
        "final_weights": circuit.weight_matrix,
    }


def _as_numpy(value):
    return np.asarray(value)


def plot_experiment(result, path: str | Path):
    """Plot stimulus timing, learning margins, and probe decisions."""
    path = Path(path)
    times_ms = _as_numpy(result["times"].to_decimal(u.ms))
    templates = _as_numpy(result["templates"])
    ab_weights = _as_numpy(result["phase_ab_weights"])
    ba_weights = _as_numpy(result["phase_ba_weights"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)

    axes[0].step(times_ms, templates[A_FIRST, :, 0], where="post", label="tone A")
    axes[0].step(times_ms, templates[A_FIRST, :, 1], where="post", label="tone B")
    axes[0].set(
        title="A-first stimulus (the second phase swaps these onsets)",
        ylabel="tone drive",
        xlabel="time (ms)",
        ylim=(-0.05, 1.15),
    )
    axes[0].legend(loc="upper right")

    ab_margin = ab_weights[:, A_FIRST, A_FIRST] - ab_weights[:, A_FIRST, B_FIRST]
    ba_margin = ba_weights[:, B_FIRST, B_FIRST] - ba_weights[:, B_FIRST, A_FIRST]
    axes[1].plot(np.arange(1, len(ab_margin) + 1), ab_margin, label="A-first phase")
    axes[1].plot(np.arange(1, len(ba_margin) + 1), ba_margin, label="B-first reversal")
    axes[1].axhline(0.0, color="0.45", linewidth=0.8)
    axes[1].set(
        title="Online plasticity reverses each active readout preference",
        ylabel="correct weight - competing weight",
        xlabel="training trial",
    )
    axes[1].legend(loc="lower right")

    probe_sets = np.stack(
        [
            _as_numpy(result["before_outputs"]),
            _as_numpy(result["after_ab_outputs"]),
            _as_numpy(result["after_ba_outputs"]),
        ]
    )
    x = np.arange(3)
    width = 0.18
    labels = ("baseline", "after A-first", "after reversal")
    for order, order_name in ((A_FIRST, "AB trial"), (B_FIRST, "BA trial")):
        for output, color in ((A_FIRST, "tab:blue"), (B_FIRST, "tab:red")):
            offset = (-1.5 + order * 2 + output) * width
            axes[2].bar(
                x + offset,
                probe_sets[:, order, output],
                width,
                color=color,
                alpha=0.55 + 0.35 * order,
                label=f"{order_name}: output {'A' if output == 0 else 'B'}",
            )
    axes[2].set(
        title="Batched probe decisions from output-population spike counts",
        ylabel="spikes per trial",
        xticks=x,
        xticklabels=labels,
    )
    axes[2].legend(ncol=2, fontsize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_summary(result):
    names = ("A-first", "B-first")
    for label, key in (
        ("baseline", "before_outputs"),
        ("after A-first training", "after_ab_outputs"),
        ("after B-first reversal", "after_ba_outputs"),
    ):
        counts = _as_numpy(result[key]).astype(int)
        predictions = _as_numpy(predicted_orders(result[key])).astype(int)
        print(label)
        for order in (A_FIRST, B_FIRST):
            print(
                f"  {names[order]} probe: spikes={counts[order].tolist()} "
                f"decision={names[predictions[order]]}"
            )
    print("final detector-to-output weights:")
    print(_as_numpy(result["final_weights"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=18, help="trials per phase")
    parser.add_argument(
        "--output", default="temporal_order_relearning.png", help="figure path"
    )
    args = parser.parse_args()

    result = run_experiment(n_trials=args.trials)
    print_summary(result)
    figure = plot_experiment(result, args.output)

    after_ab = _as_numpy(predicted_orders(result["after_ab_outputs"]))
    after_ba = _as_numpy(predicted_orders(result["after_ba_outputs"]))
    if after_ab[A_FIRST] != A_FIRST or after_ba[B_FIRST] != B_FIRST:
        raise RuntimeError("the circuit did not learn both temporal-order decisions")
    print(f"figure saved to {figure}")


if __name__ == "__main__":
    main()
