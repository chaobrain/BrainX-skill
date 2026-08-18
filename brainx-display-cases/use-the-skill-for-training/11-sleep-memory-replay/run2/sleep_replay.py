"""Route learning, sleep replay, suppression, and recall in a BrainX SNN.

The four place ensembles are stimulated in the order A -> B -> C -> D during
learning.  A temporally asymmetric event rule strengthens recurrent synapses
in that direction.  During sleep, an intrinsic slow oscillator periodically
depolarizes ensemble A; there is no external current.  Two state-mapped lanes
receive the same learned weights and intrinsic events, but recurrent
transmission is gated off in the replay-suppression lane.  Plasticity remains
open during sleep, so successful replay can consolidate the learned route.

Run:
    MPLCONFIGDIR=/tmp/matplotlib-cache python sleep_replay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter
from braintools.input import Constant


DT = 1.0 * u.ms
N_PLACES = 4
CELLS_PER_PLACE = 5
N_CELLS = N_PLACES * CELLS_PER_PLACE
PLACE_NAMES = ("A", "B", "C", "D")
CONDITIONS = ("replay", "suppressed")

V_REST = -65.0 * u.mV
V_RESET = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
TAU_MEMBRANE = 15.0 * u.ms
TAU_REFRACTORY = 3.0 * u.ms
TAU_SYNAPSE = 4.0 * u.ms
TAU_STDP = 20.0 * u.ms

PLACE_DRIVE = 0.75 * u.nA
INTRINSIC_DRIVE = 0.82 * u.nA
RECURRENT_SCALE = 1.0 * u.nA
INITIAL_EFFICACY = 0.018
MAX_EFFICACY = 0.32
A_PLUS = 0.030
A_MINUS = 0.032

LEARNING_REPEATS = 8
SLEEP_DURATION = 480.0 * u.ms
SLEEP_INTERVAL = 80.0 * u.ms
SLEEP_PULSE = 4.0 * u.ms
RECALL_DURATION = 120.0 * u.ms
RECALL_PULSE = 4.0 * u.ms
RECALL_DEADLINE = 25.0 * u.ms


def _steps(duration) -> int:
    return int(round(duration.to_decimal(u.ms) / DT.to_decimal(u.ms)))


PLACE_INDEX = jnp.repeat(jnp.arange(N_PLACES), CELLS_PER_PLACE)
PLACE_MASKS = jnp.stack([PLACE_INDEX == place for place in range(N_PLACES)])
ROUTE_EDGE_MASK = (
    jnp.abs(PLACE_INDEX[:, None] - PLACE_INDEX[None, :]) == 1
).astype(jnp.float32)


class PlaceRouteNetwork(brainstate.nn.Module):
    """Four LIF place ensembles with event-driven recurrent STDP."""

    def __init__(self, initial_weight=None):
        super().__init__()
        if initial_weight is None:
            initial_weight = INITIAL_EFFICACY * ROUTE_EDGE_MASK
        self.initial_weight = u.math.asarray(initial_weight)
        self.cells = brainpy.state.LIFRef(
            N_CELLS,
            R=100.0 * u.Mohm,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.recurrent_synapse = brainpy.state.Expon(
            N_CELLS,
            tau=TAU_SYNAPSE,
            g_initializer=braintools.init.Constant(0.0 * u.nA),
        )

    def init_state(self):
        self.weight = brainstate.LongTermState(self.initial_weight)
        self.pre_trace = brainstate.ShortTermState(jnp.zeros(N_CELLS))
        self.post_trace = brainstate.ShortTermState(jnp.zeros(N_CELLS))
        self.sleep_clock = brainstate.ShortTermState(jnp.asarray(0, jnp.int32))
        self.sleep_pulse = brainstate.ShortTermState(jnp.asarray(0, jnp.int32))

    def reset_state(self):
        self.pre_trace.value = jnp.zeros_like(self.pre_trace.value)
        self.post_trace.value = jnp.zeros_like(self.post_trace.value)
        self.sleep_clock.value = jnp.zeros_like(self.sleep_clock.value)
        self.sleep_pulse.value = jnp.zeros_like(self.sleep_pulse.value)

    def _intrinsic_sleep_current(self):
        interval_steps = _steps(SLEEP_INTERVAL)
        pulse_steps = _steps(SLEEP_PULSE)
        next_clock = self.sleep_clock.value + 1
        start = next_clock >= interval_steps
        self.sleep_clock.value = jnp.where(start, 0, next_clock)
        remaining = jnp.where(start, pulse_steps, self.sleep_pulse.value)
        active = remaining > 0
        self.sleep_pulse.value = jnp.maximum(remaining - 1, 0)
        return active * PLACE_MASKS[0] * INTRINSIC_DRIVE

    def _plasticity(self, spikes, plasticity_gate):
        decay = u.math.exp(-brainstate.environ.get_dt() / TAU_STDP)
        old_weight = self.weight.value

        potentiated = brainevent.update_dense_on_binary_post(
            weight=old_weight,
            post_spike=spikes,
            pre_trace=self.pre_trace.value * A_PLUS,
            w_min=0.0,
            w_max=MAX_EFFICACY,
        )
        candidate = brainevent.update_dense_on_binary_pre(
            weight=potentiated,
            pre_spike=spikes,
            post_trace=self.post_trace.value * -A_MINUS,
            w_min=0.0,
            w_max=MAX_EFFICACY,
        )
        candidate = candidate * ROUTE_EDGE_MASK
        self.weight.value = old_weight + plasticity_gate * (candidate - old_weight)
        spike_float = spikes.astype(jnp.float32)
        self.pre_trace.value = decay * self.pre_trace.value + spike_float
        self.post_trace.value = decay * self.post_trace.value + spike_float

    def update(
        self,
        t,
        external_current,
        recurrent_gate,
        plasticity_gate,
        sleep_mode=False,
    ):
        with brainstate.environ.context(t=t):
            previous_spikes = self.cells.get_spike() != 0.0
            recurrent_current = (
                brainevent.BinaryArray(previous_spikes)
                @ (self.weight.value * RECURRENT_SCALE)
            )
            synaptic_current = self.recurrent_synapse(
                recurrent_gate * recurrent_current
            )
            intrinsic_current = (
                self._intrinsic_sleep_current()
                if sleep_mode
                else jnp.zeros(N_CELLS) * u.nA
            )
            spikes = self.cells(
                external_current + intrinsic_current + synaptic_current
            ) != 0.0
            self._plasticity(spikes, plasticity_gate)
            return spikes


def learning_protocol():
    """Build repeated, time-major A -> B -> C -> D current sections."""
    silence = jnp.zeros(N_CELLS) * u.nA
    sections = [(silence, 12.0 * u.ms)]
    for place in range(N_PLACES):
        sections.extend(
            [
                (PLACE_MASKS[place] * PLACE_DRIVE, 4.0 * u.ms),
                (silence, 6.0 * u.ms),
            ]
        )
    sections.append((silence, 18.0 * u.ms))
    one_route = Constant(sections)()
    return u.math.concatenate([one_route] * LEARNING_REPEATS, axis=0)


def recall_protocol():
    """Cue only A, then remove external input for the recall assay."""
    silence = jnp.zeros(N_CELLS) * u.nA
    return Constant(
        [
            (PLACE_MASKS[0] * PLACE_DRIVE, RECALL_PULSE),
            (silence, RECALL_DURATION - RECALL_PULSE),
        ]
    )()


def learn_route():
    with brainstate.environ.context(dt=DT):
        net = PlaceRouteNetwork()
        brainstate.nn.init_all_states(net)
        current = learning_protocol()
        times = u.math.arange(0.0 * u.ms, current.shape[0] * DT, DT)

        @brainstate.transform.jit
        def run_learning():
            return brainstate.transform.for_loop(
                lambda t, drive: net.update(t, drive, 1.0, 1.0),
                times,
                current,
            )

        spikes = run_learning()
    return net.weight.value, spikes


MAPPED_STATE = state_filter.Any(
    state_filter.OfType(brainstate.HiddenState),
    state_filter.OfType(brainstate.ShortTermState),
    state_filter.OfType(brainstate.LongTermState),
)


def sleep_and_recall(learned_weight):
    """Run matched replay/suppression lanes, then the same recall cue."""
    with brainstate.environ.context(dt=DT):
        net = PlaceRouteNetwork(learned_weight)
        brainstate.nn.vmap_init_all_states(net, axis_size=len(CONDITIONS))

        mapped_sleep_step = vmap2(
            net.update,
            in_axes=(None, 0, 0, None, None),
            out_axes=0,
            state_in_axes={0: MAPPED_STATE},
            state_out_axes={0: MAPPED_STATE},
            unexpected_out_state_mapping="raise",
        )
        sleep_times = u.math.arange(0.0 * u.ms, SLEEP_DURATION, DT)
        zero_external_current = jnp.zeros(
            (sleep_times.shape[0], len(CONDITIONS), N_CELLS)
        ) * u.nA
        recurrent_gates = jnp.asarray([1.0, 0.0])

        @brainstate.transform.jit
        def run_sleep():
            return brainstate.transform.for_loop(
                lambda t, drive: mapped_sleep_step(
                    t, drive, recurrent_gates, 1.0, True
                ),
                sleep_times,
                zero_external_current,
            )

        sleep_spikes = run_sleep()
        post_sleep_weight = net.weight.value

        # Start recall with fresh neural/trace State and the two post-sleep
        # weight lanes. This is the exact independent-assay boundary; the
        # selected LIF reset implementation does not preserve its vmapped axis.
        recall_net = PlaceRouteNetwork(learned_weight)
        brainstate.nn.vmap_init_all_states(
            recall_net, axis_size=len(CONDITIONS)
        )
        if recall_net.weight.value.shape != post_sleep_weight.shape:
            raise RuntimeError("post-sleep weight lanes do not match recall lanes")
        recall_net.weight.value = post_sleep_weight

        mapped_recall_step = vmap2(
            recall_net.update,
            in_axes=(None, 0, None, None, None),
            out_axes=0,
            state_in_axes={0: MAPPED_STATE},
            state_out_axes={0: MAPPED_STATE},
            unexpected_out_state_mapping="raise",
        )
        recall_current = recall_protocol()
        recall_times = u.math.arange(0.0 * u.ms, RECALL_DURATION, DT)
        matched_current = u.math.broadcast_to(
            recall_current[:, None, :],
            (recall_current.shape[0], len(CONDITIONS), N_CELLS),
        )

        @brainstate.transform.jit
        def run_recall():
            return brainstate.transform.for_loop(
                lambda t, drive: mapped_recall_step(t, drive, 1.0, 0.0, False),
                recall_times,
                matched_current,
            )

        recall_spikes = run_recall()

    return sleep_spikes, post_sleep_weight, recall_spikes


def place_counts(spikes):
    values = np.asarray(spikes, dtype=bool)
    return np.stack(
        [values[..., np.asarray(PLACE_INDEX) == place].sum(axis=-1) for place in range(N_PLACES)],
        axis=-1,
    )


def first_spike_times(counts, start, stop):
    first = []
    for place in range(N_PLACES):
        active = np.flatnonzero(counts[start:stop, place] > 0)
        first.append(float(start + active[0]) if active.size else np.nan)
    return np.asarray(first) * DT.to_decimal(u.ms)


def decode_sleep(sleep_counts):
    """Classify complete route events from first-spike order in fixed windows."""
    interval = _steps(SLEEP_INTERVAL)
    window = min(interval, _steps(55.0 * u.ms))
    starts = np.arange(
        interval - 1,
        sleep_counts.shape[0] - window + 1,
        interval,
    )
    decoded = []
    for condition, name in enumerate(CONDITIONS):
        events = []
        for event_id, start in enumerate(starts, 1):
            first_ms = first_spike_times(
                sleep_counts[:, condition], start, min(start + window, sleep_counts.shape[0])
            )
            if np.all(np.isfinite(first_ms)):
                slope = float(np.polyfit(np.arange(N_PLACES), first_ms, 1)[0])
                direction = "forward" if slope > 0 else "backward" if slope < 0 else "simultaneous"
            else:
                slope = None
                direction = "incomplete"
            events.append(
                {
                    "event": event_id,
                    "first_spike_ms": [None if not np.isfinite(x) else x for x in first_ms],
                    "slope_ms_per_place": slope,
                    "direction": direction,
                }
            )
        decoded.append({"condition": name, "events": events})
    return decoded


def recall_metrics(recall_counts):
    metrics = []
    for condition, name in enumerate(CONDITIONS):
        first_ms = first_spike_times(
            recall_counts[:, condition], 0, recall_counts.shape[0]
        )
        downstream = first_ms[1:]
        recalled = np.isfinite(downstream)
        ordered_pairs = sum(
            np.isfinite(downstream[i : i + 2]).all()
            and downstream[i] < downstream[i + 1]
            for i in range(N_PLACES - 2)
        )
        completion = float(recalled.mean())
        order = float(ordered_pairs / (N_PLACES - 2))
        propagation_ms = (
            float(first_ms[-1] - first_ms[0])
            if np.isfinite(first_ms[[0, -1]]).all()
            else None
        )
        deadline_ms = RECALL_DEADLINE.to_decimal(u.ms)
        speed = (
            max(0.0, 1.0 - propagation_ms / deadline_ms)
            if propagation_ms is not None
            else 0.0
        )
        score = 100.0 * completion * order * speed
        metrics.append(
            {
                "condition": name,
                "first_spike_ms": [None if not np.isfinite(x) else x for x in first_ms],
                "downstream_places_recalled": int(recalled.sum()),
                "completion": completion,
                "order": order,
                "route_propagation_ms": propagation_ms,
                "deadline_ms": deadline_ms,
                "recall_score_percent": score,
            }
        )
    return metrics


def place_weight_matrix(weight):
    weight = np.asarray(weight)
    return weight.reshape(
        N_PLACES, CELLS_PER_PLACE, N_PLACES, CELLS_PER_PLACE
    ).mean(axis=(1, 3))


def save_figure(
    learned_weight,
    sleep_counts,
    post_sleep_weight,
    recall_counts,
    recall_results,
    output_path,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    route_weight = place_weight_matrix(learned_weight)
    image = axes[0, 0].imshow(route_weight, cmap="magma", vmin=0.0, vmax=MAX_EFFICACY)
    axes[0, 0].set(
        title="Route weights after A -> B -> C -> D learning",
        xlabel="postsynaptic place",
        ylabel="presynaptic place",
        xticks=range(N_PLACES),
        yticks=range(N_PLACES),
        xticklabels=PLACE_NAMES,
        yticklabels=PLACE_NAMES,
    )
    fig.colorbar(image, ax=axes[0, 0], label="mean efficacy")

    t_sleep = np.arange(sleep_counts.shape[0]) * DT.to_decimal(u.ms)
    colors = ("#2166ac", "#b2182b", "#1b7837", "#762a83")
    for place, color in enumerate(colors):
        active = sleep_counts[:, 0, place] > 0
        axes[0, 1].scatter(
            t_sleep[active],
            np.full(active.sum(), place),
            s=18,
            color=color,
            label=PLACE_NAMES[place],
        )
    axes[0, 1].set(
        title="Sleep activity in replay-enabled network",
        xlabel="sleep time (ms)",
        ylabel="place ensemble",
        yticks=range(N_PLACES),
        yticklabels=PLACE_NAMES,
        ylim=(-0.5, N_PLACES - 0.5),
    )
    axes[0, 1].invert_yaxis()

    t_recall = np.arange(recall_counts.shape[0]) * DT.to_decimal(u.ms)
    for condition, name in enumerate(CONDITIONS):
        downstream = recall_counts[:, condition, 1:].sum(axis=1)
        axes[1, 0].step(t_recall, downstream, where="post", label=name)
    axes[1, 0].axvspan(
        0.0,
        RECALL_PULSE.to_decimal(u.ms),
        color="0.85",
        label="A cue",
    )
    axes[1, 0].set(
        title="Downstream route activity during recall",
        xlabel="time after recall cue (ms)",
        ylabel="B+C+D spikes per step",
    )
    axes[1, 0].legend(frameon=False)

    scores = [row["recall_score_percent"] for row in recall_results]
    axes[1, 1].bar(CONDITIONS, scores, color=("#2166ac", "#777777"), width=0.62)
    axes[1, 1].set(
        title="Ordered route recall",
        ylabel="recall score (%)",
        ylim=(0.0, 105.0),
    )
    for index, score in enumerate(scores):
        axes[1, 1].text(index, score + 3.0, f"{score:.0f}%", ha="center")

    fig.suptitle("Memory replay during sleep consolidates a learned route", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_experiment(output_dir="results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    brainstate.random.seed(7)

    learned_weight, learning_spikes = learn_route()
    sleep_spikes, post_sleep_weight, recall_spikes = sleep_and_recall(
        learned_weight
    )

    learning_counts = place_counts(learning_spikes)
    sleep_counts = place_counts(sleep_spikes)
    recall_counts = place_counts(recall_spikes)
    replay_events = decode_sleep(sleep_counts)
    recall_results = recall_metrics(recall_counts)

    complete_replay = [
        event
        for event in replay_events[0]["events"]
        if event["direction"] in ("forward", "backward")
    ]
    suppressed_complete = [
        event
        for event in replay_events[1]["events"]
        if event["direction"] in ("forward", "backward")
    ]
    if not complete_replay:
        raise RuntimeError("the replay-enabled lane produced no complete route events")

    result = {
        "parameters": {
            "dt_ms": DT.to_decimal(u.ms),
            "places": list(PLACE_NAMES),
            "cells_per_place": CELLS_PER_PLACE,
            "learning_repeats": LEARNING_REPEATS,
            "sleep_duration_ms": SLEEP_DURATION.to_decimal(u.ms),
            "sleep_external_current_nA": 0.0,
            "sleep_seed": "network-internal slow oscillator in place A",
            "regime": "phenomenological demonstration; not a biological fit",
        },
        "learning_spikes_by_place": learning_counts.sum(axis=0).tolist(),
        "learned_place_weights": place_weight_matrix(learned_weight).tolist(),
        "post_sleep_place_weights": {
            name: place_weight_matrix(post_sleep_weight[index]).tolist()
            for index, name in enumerate(CONDITIONS)
        },
        "sleep_replay": replay_events,
        "sleep_summary": {
            "replay_complete_events": len(complete_replay),
            "replay_forward_events": sum(event["direction"] == "forward" for event in complete_replay),
            "replay_backward_events": sum(event["direction"] == "backward" for event in complete_replay),
            "suppressed_complete_events": len(suppressed_complete),
            "spikes_by_condition_and_place": {
                name: sleep_counts[:, index].sum(axis=0).tolist()
                for index, name in enumerate(CONDITIONS)
            },
        },
        "recall": recall_results,
    }

    replay_seed_spikes = result["sleep_summary"]["spikes_by_condition_and_place"]["replay"][0]
    suppressed_seed_spikes = result["sleep_summary"]["spikes_by_condition_and_place"]["suppressed"][0]
    if replay_seed_spikes != suppressed_seed_spikes:
        raise RuntimeError("sleep seed dose differs between matched conditions")
    if any(event["direction"] != "forward" for event in complete_replay):
        raise RuntimeError("a complete replay event was not forward")
    if suppressed_complete:
        raise RuntimeError("replay suppression did not block complete route events")
    if recall_results[0]["recall_score_percent"] <= recall_results[1]["recall_score_percent"]:
        raise RuntimeError("replay did not improve the defined recall score")

    json_path = output_dir / "sleep_replay_metrics.json"
    figure_path = output_dir / "sleep_replay_summary.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    save_figure(
        learned_weight,
        sleep_counts,
        post_sleep_weight,
        recall_counts,
        recall_results,
        figure_path,
    )
    return result, figure_path, json_path


def main():
    result, figure_path, json_path = run_experiment()
    summary = result["sleep_summary"]
    print("Sleep replay")
    print(
        f"  replay-enabled: {summary['replay_forward_events']} forward, "
        f"{summary['replay_backward_events']} backward complete events"
    )
    print(f"  replay-suppressed: {summary['suppressed_complete_events']} complete events")
    print("Recall")
    for row in result["recall"]:
        print(
            f"  {row['condition']:10s}: {row['recall_score_percent']:5.1f}% "
            f"(first spikes A-D: {row['first_spike_ms']})"
        )
    print(f"Saved {figure_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
