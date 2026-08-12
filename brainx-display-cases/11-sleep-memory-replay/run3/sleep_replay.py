"""Learn, replay, suppress, and recall a four-place route with BrainX."""

from __future__ import annotations

import json
from pathlib import Path

import brainevent
import brainpy
import brainstate
import brainunit as u
import braintools
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter


DT = 1.0 * u.ms
N_PLACES = 4
CELLS_PER_PLACE = 6
N_CELLS = N_PLACES * CELLS_PER_PLACE
GROUPS = ("replay", "suppressed")


class RouteNetwork(brainstate.nn.Module):
    """Place-cell LIF populations with dense event-driven recurrent STDP."""

    def __init__(self):
        super().__init__()
        self.cells = brainpy.state.LIFRef(
            N_CELLS,
            R=100.0 * u.Mohm,
            tau=12.0 * u.ms,
            V_rest=-65.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-62.0 * u.mV,
            tau_ref=3.0 * u.ms,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        self.recurrent_filter = brainpy.state.Expon(
            N_CELLS,
            tau=6.0 * u.ms,
            g_initializer=braintools.init.Constant(0.0 * u.nA),
        )

        place = jnp.repeat(jnp.arange(N_PLACES), CELLS_PER_PLACE)
        self.plastic_mask = (place[:, None] != place[None, :]).astype(jnp.float32)
        self.trace_decay = u.math.exp(-DT / (18.0 * u.ms))
        self.recurrent_scale = 1.0 * u.nA
        self.a_plus = 0.017
        self.a_minus = 0.014

    def init_state(self):
        self.weight = brainstate.LongTermState(
            jnp.zeros((N_CELLS, N_CELLS), dtype=jnp.float32)
        )
        self.pre_trace = brainstate.ShortTermState(
            jnp.zeros(N_CELLS, dtype=jnp.float32)
        )
        self.post_trace = brainstate.ShortTermState(
            jnp.zeros(N_CELLS, dtype=jnp.float32)
        )

    def update(self, time, external_current, recurrent_gate, plasticity_gate):
        with brainstate.environ.context(t=time):
            previous_spikes = self.cells.get_spike() != 0.0
            event_input = brainevent.BinaryArray(previous_spikes) @ self.weight.value
            recurrent_current = self.recurrent_filter(
                event_input * self.recurrent_scale * recurrent_gate
            )
            spikes = self.cells(external_current + recurrent_current) != 0.0

            weight = brainevent.update_dense_on_binary_post(
                weight=self.weight.value,
                post_spike=spikes,
                pre_trace=self.pre_trace.value * self.a_plus * plasticity_gate,
                w_min=0.0,
                w_max=1.0,
            )
            weight = brainevent.update_dense_on_binary_pre(
                weight=weight,
                pre_spike=spikes,
                post_trace=-self.post_trace.value * self.a_minus * plasticity_gate,
                w_min=0.0,
                w_max=1.0,
            )
            self.weight.value = weight * self.plastic_mask
            self.pre_trace.value = (
                self.pre_trace.value * self.trace_decay + spikes.astype(jnp.float32)
            )
            self.post_trace.value = (
                self.post_trace.value * self.trace_decay + spikes.astype(jnp.float32)
            )
            return spikes


def route_protocol(n_trials: int = 12):
    """Sequential A-B-C-D teaching cues followed by a boundary-state A seed."""
    cue = jnp.eye(N_PLACES, dtype=jnp.float32)
    sections = []
    for place_index in range(N_PLACES):
        sections.extend(
            [jnp.repeat(cue[place_index][None, :], 9, axis=0), jnp.zeros((3, N_PLACES))]
        )
    trial = jnp.concatenate(sections, axis=0)
    wake = jnp.tile(trial, (n_trials, 1))
    seed = jnp.concatenate(
        [jnp.zeros((30, N_PLACES)), 6.0 * cue[0][None, :]], axis=0
    )
    place_drive = jnp.concatenate([wake, seed], axis=0)
    return jnp.repeat(place_drive, CELLS_PER_PLACE, axis=1) * (0.34 * u.nA)


def recall_protocol():
    """A matched washout, a brief A cue, then an unforced completion window."""
    washout_steps, cue_steps, completion_steps = 60, 1, 90
    place_drive = jnp.zeros(
        (washout_steps + cue_steps + completion_steps, N_PLACES), dtype=jnp.float32
    )
    place_drive = place_drive.at[washout_steps : washout_steps + cue_steps, 0].set(6.0)
    current = jnp.repeat(place_drive, CELLS_PER_PLACE, axis=1) * (0.34 * u.nA)
    gate = jnp.concatenate(
        [jnp.zeros(washout_steps), jnp.ones(cue_steps + completion_steps)]
    )
    return current, gate, washout_steps + cue_steps


def duplicate_current(current):
    return u.math.stack([current, current], axis=1)


def place_activity(spikes):
    return np.asarray(spikes).reshape(spikes.shape[0], len(GROUPS), N_PLACES, -1).sum(axis=3)


def mean_place_weights(weights):
    array = np.asarray(weights).reshape(
        len(GROUPS), N_PLACES, CELLS_PER_PLACE, N_PLACES, CELLS_PER_PLACE
    )
    return array.mean(axis=(2, 4))


def first_onsets(activity, start=0):
    onsets = []
    for place_index in range(N_PLACES):
        candidates = np.flatnonzero(activity[start:, place_index] > 0)
        onsets.append(None if candidates.size == 0 else int(start + candidates[0]))
    return onsets


def replay_label(activity):
    # A fired on the final wake step and is the saved boundary-state seed.
    onsets = [-1, *first_onsets(activity)[1:]]
    observed = [(time, place) for place, time in enumerate(onsets) if time is not None]
    order = [place for _, place in sorted(observed)]
    if all(time is not None for time in onsets) and all(
        onsets[a] < onsets[b] for a, b in ((0, 1), (1, 2), (2, 3))
    ):
        label = "forward"
    elif all(time is not None for time in onsets) and all(
        onsets[a] > onsets[b] for a, b in ((0, 1), (1, 2), (2, 3))
    ):
        label = "backward"
    elif order == [0]:
        label = "none"
    else:
        label = "partial/mixed"
    return label, order, onsets


def recall_score(activity, completion_start):
    onsets = first_onsets(activity, start=completion_start)
    prefix = 0
    previous = completion_start - 1
    for place in (1, 2, 3):
        if onsets[place] is None or onsets[place] <= previous:
            break
        prefix += 1
        previous = onsets[place]
    return prefix / 3.0, onsets


def make_figure(sleep_activity, recall_activity, wake_weights, final_weights, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ("#147d64", "#b3453e")

    for group_index, group in enumerate(GROUPS):
        ax = axes[0, group_index]
        ax.scatter([-1], [0], s=24, color=colors[group_index])
        for place in range(N_PLACES):
            times = np.flatnonzero(sleep_activity[:, group_index, place] > 0)
            ax.scatter(times, np.full(times.shape, place), s=18, color=colors[group_index])
        ax.set(
            title=f"Sleep: {group}", xlabel="Time from sleep onset (ms)",
            yticks=range(N_PLACES), yticklabels=list("ABCD"),
            xlim=(-2, 30), ylim=(-0.5, 3.5),
        )

    width = 0.35
    x = np.arange(3)
    wake_forward = wake_weights[:, range(3), range(1, 4)]
    final_forward = final_weights[:, range(3), range(1, 4)]
    for group_index, group in enumerate(GROUPS):
        axes[1, 0].bar(
            x + (group_index - 0.5) * width,
            final_forward[group_index] - wake_forward[group_index],
            width,
            label=group,
            color=colors[group_index],
        )
    axes[1, 0].set(
        title="Sleep change in forward weights", xlabel="Route connection",
        ylabel="Mean efficacy change", xticks=x, xticklabels=("A→B", "B→C", "C→D"),
    )
    axes[1, 0].legend(frameon=False)

    ax = axes[1, 1]
    for group_index, group in enumerate(GROUPS):
        activity = recall_activity[:, group_index]
        for place in range(N_PLACES):
            times = np.flatnonzero(activity[:, place] > 0)
            ax.scatter(
                times,
                np.full(times.shape, place + group_index * 0.12),
                s=18,
                color=colors[group_index],
                label=group if place == 0 else None,
            )
    ax.set(
        title="Recall after an A cue", xlabel="Recall time (ms)",
        yticks=range(N_PLACES), yticklabels=list("ABCD"), ylim=(-0.5, 3.5),
    )
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_experiment():
    brainstate.random.seed(7)
    with brainstate.environ.context(dt=DT):
        network = RouteNetwork()
        brainstate.nn.vmap_init_all_states(network, axis_size=len(GROUPS))

        mapped_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
            state_filter.OfType(brainstate.LongTermState),
        )
        mapped_step = vmap2(
            network.update,
            in_axes=(None, 0, 0, 0),
            out_axes=0,
            state_in_axes={0: mapped_state},
            state_out_axes={0: mapped_state},
            unexpected_out_state_mapping="raise",
        )

        learning_current = route_protocol()
        learning_steps = learning_current.shape[0]
        learning_times = u.math.arange(learning_steps) * DT
        learning_spikes = brainstate.transform.for_loop(
            mapped_step,
            learning_times,
            duplicate_current(learning_current),
            jnp.zeros((learning_steps, len(GROUPS))),
            jnp.ones((learning_steps, len(GROUPS))),
        )
        wake_weights = network.weight.value

        sleep_steps = 180
        sleep_times = (learning_steps + u.math.arange(sleep_steps)) * DT
        sleep_spikes = brainstate.transform.for_loop(
            mapped_step,
            sleep_times,
            u.math.zeros((sleep_steps, len(GROUPS), N_CELLS), unit=u.nA),
            jnp.repeat(jnp.array([[1.0, 0.0]]), sleep_steps, axis=0),
            jnp.ones((sleep_steps, len(GROUPS))),
        )
        final_weights = network.weight.value

        recall_current, recall_gate, completion_start = recall_protocol()
        recall_steps = recall_current.shape[0]
        recall_times = (learning_steps + sleep_steps + u.math.arange(recall_steps)) * DT
        recall_spikes = brainstate.transform.for_loop(
            mapped_step,
            recall_times,
            duplicate_current(recall_current),
            jnp.repeat(recall_gate[:, None], len(GROUPS), axis=1),
            jnp.zeros((recall_steps, len(GROUPS))),
        )

    sleep_activity = place_activity(sleep_spikes)
    recall_activity = place_activity(recall_spikes)
    wake_place_weights = mean_place_weights(wake_weights)
    final_place_weights = mean_place_weights(final_weights)

    if not np.allclose(np.asarray(wake_weights[0]), np.asarray(wake_weights[1])):
        raise RuntimeError("The matched groups diverged before the sleep intervention.")
    if sleep_activity[:, 1, 1:].sum() != 0:
        raise RuntimeError("The replay-suppression group propagated activity during sleep.")

    results = {"groups": {}}
    for group_index, group in enumerate(GROUPS):
        direction, order, sleep_onsets = replay_label(sleep_activity[:, group_index])
        score, recall_onsets = recall_score(
            recall_activity[:, group_index], completion_start
        )
        results["groups"][group] = {
            "sleep_replay": direction,
            "sleep_boundary_seed": "A at -1 ms",
            "sleep_place_order": ["ABCD"[place] for place in order],
            "sleep_first_spike_ms": sleep_onsets,
            "recall_score": score,
            "recall_first_spike_ms": recall_onsets,
            "recall_completion_latency_ms": (
                None
                if score < 1.0
                else int(recall_onsets[3] - completion_start)
            ),
            "forward_weight_before_sleep": wake_place_weights[
                group_index, range(3), range(1, 4)
            ].tolist(),
            "forward_weight_after_sleep": final_place_weights[
                group_index, range(3), range(1, 4)
            ].tolist(),
        }

    output_dir = Path(__file__).resolve().parent
    (output_dir / "sleep_replay_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(
        sleep_activity,
        recall_activity,
        wake_place_weights,
        final_place_weights,
        output_dir / "sleep_replay.png",
    )
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_experiment()
