"""Learn, replay, suppress, and recall a four-place spiking sequence.

The model is intentionally small enough to inspect. Four six-cell place
populations learn A -> B -> C -> D through pair-based STDP. During sleep the
network receives no place cue; matched intrinsic current fluctuations can
trigger endogenous recurrent sequences. Excitatory recurrent transmission is
disabled only in the replay-suppression lanes.

Run with::

    python sleep_replay.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import brainevent
import brainpy
import brainstate
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


DT = 1.0 * u.ms
N_PLACES = 4
CELLS_PER_PLACE = 6
N_NEURONS = N_PLACES * CELLS_PER_PLACE
PLACE_NAMES = np.asarray(list("ABCD"))

N_PAIRS = 8
N_LANES = 2 * N_PAIRS
REPLAY_ENABLED = 1.0
REPLAY_SUPPRESSED = 0.0

TAU_M = 15.0 * u.ms
TAU_REF = 3.0 * u.ms
TAU_EXC = 5.0 * u.ms
TAU_INH = 8.0 * u.ms
TAU_TRACE = 22.0 * u.ms
V_REST = -65.0 * u.mV
V_RESET = -65.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
E_EXC = 0.0 * u.mV
E_INH = -80.0 * u.mV

WEIGHT_SCALE = 9.0 * u.nS
INHIBITORY_WEIGHT = 0.35 * u.nS
W_MIN = 0.015
W_MAX = 1.0
ETA_PLUS = 0.010
ETA_MINUS = 0.0085

LEARNING_REPEATS = 18
LEARNING_PULSE_STEPS = 9
LEARNING_GAP_STEPS = 6
LEARNING_PAUSE_STEPS = 18
LEARNING_CURRENT = 0.45 * u.nA
SLEEP_STEPS = 1600
SLEEP_NOISE_STD = 0.045 * u.nA
SLEEP_BURST_CURRENT = 0.90 * u.nA
RECALL_STEPS = 170
RECALL_CUE_STEPS = 8
RECALL_CURRENT = 0.85 * u.nA


def _place_ids() -> jax.Array:
    return jnp.repeat(jnp.arange(N_PLACES), CELLS_PER_PLACE)


def initial_weights() -> jax.Array:
    """Dimensionless efficacy matrix; rows are pre, columns are post."""
    place = _place_ids()
    same = place[:, None] == place[None, :]
    adjacent = jnp.abs(place[:, None] - place[None, :]) == 1
    weights = jnp.where(same, 0.18, jnp.where(adjacent, 0.055, W_MIN))
    return weights * (1.0 - jnp.eye(N_NEURONS, dtype=jnp.float32))


def plastic_mask() -> jax.Array:
    """Allow local and neighboring place assemblies to learn."""
    place = _place_ids()
    nearby = jnp.abs(place[:, None] - place[None, :]) <= 1
    return nearby & ~jnp.eye(N_NEURONS, dtype=bool)


class RouteNetwork(brainstate.nn.Module):
    """Unit-aware recurrent LIF place-cell network with online STDP."""

    def __init__(self):
        super().__init__()
        self.neurons = brainpy.state.LIFRef(
            N_NEURONS,
            R=100.0 * u.Mohm,
            tau=TAU_M,
            tau_ref=TAU_REF,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.exc_syn = brainpy.state.Expon(N_NEURONS, tau=TAU_EXC)
        self.inh_syn = brainpy.state.Expon(N_NEURONS, tau=TAU_INH)
        self.exc_out = brainpy.state.COBA(E=E_EXC)
        self.inh_out = brainpy.state.COBA(E=E_INH)
        self.neurons.add_current_input("route_excitation", self.exc_out)
        self.neurons.add_current_input("global_inhibition", self.inh_out)

        self._initial_weight = initial_weights()
        self._plastic_mask = plastic_mask()
        self._inhibitory = (
            jnp.ones((N_NEURONS, N_NEURONS), dtype=jnp.float32)
            * INHIBITORY_WEIGHT
        )

    def init_state(self):
        self.weight = brainstate.LongTermState(self._initial_weight)
        self.pre_trace = brainstate.ShortTermState(
            jnp.zeros(N_NEURONS, dtype=jnp.float32)
        )
        self.post_trace = brainstate.ShortTermState(
            jnp.zeros(N_NEURONS, dtype=jnp.float32)
        )

    def update(
        self,
        t,
        intrinsic_current,
        place_cue,
        recurrent_gate,
        plasticity_gate,
    ):
        with brainstate.environ.context(t=t):
            previous_spikes = self.neurons.get_spike() != 0.0

            exc_drive = brainevent.BinaryArray(previous_spikes) @ (
                self.weight.value * WEIGHT_SCALE
            )
            inh_drive = brainevent.BinaryArray(previous_spikes) @ self._inhibitory
            self.exc_out.bind_cond(self.exc_syn(exc_drive * recurrent_gate))
            self.inh_out.bind_cond(self.inh_syn(inh_drive))

            spikes = self.neurons(intrinsic_current + place_cue) != 0.0

            decay = u.math.exp(-brainstate.environ.get_dt() / TAU_TRACE)
            pre = self.pre_trace.value * decay
            post = self.post_trace.value * decay
            old_weight = self.weight.value

            potentiated = brainevent.update_dense_on_binary_post(
                weight=old_weight,
                post_spike=spikes,
                pre_trace=pre * ETA_PLUS,
                w_min=W_MIN,
                w_max=W_MAX,
            )
            updated = brainevent.update_dense_on_binary_pre(
                weight=potentiated,
                pre_spike=spikes,
                post_trace=-post * ETA_MINUS,
                w_min=W_MIN,
                w_max=W_MAX,
            )
            learned = jnp.where(self._plastic_mask, updated, old_weight)
            self.weight.value = jnp.where(
                plasticity_gate != 0.0, learned, old_weight
            )
            self.pre_trace.value = pre + spikes.astype(jnp.float32)
            self.post_trace.value = post + spikes.astype(jnp.float32)
            return spikes


def learning_protocol() -> u.Quantity:
    """Time-major A-B-C-D place cues repeated along one continuous route."""
    one_repeat = N_PLACES * (LEARNING_PULSE_STEPS + LEARNING_GAP_STEPS)
    one_repeat += LEARNING_PAUSE_STEPS
    protocol = np.zeros(
        (LEARNING_REPEATS * one_repeat, N_NEURONS), dtype=np.float32
    )
    for repeat in range(LEARNING_REPEATS):
        offset = repeat * one_repeat
        for place in range(N_PLACES):
            start = offset + place * (LEARNING_PULSE_STEPS + LEARNING_GAP_STEPS)
            cell_slice = slice(
                place * CELLS_PER_PLACE, (place + 1) * CELLS_PER_PLACE
            )
            protocol[start : start + LEARNING_PULSE_STEPS, cell_slice] = 1.0
    return jnp.asarray(protocol) * LEARNING_CURRENT


def sleep_intrinsic_current(seed: int = 8) -> u.Quantity:
    """Matched intrinsic noise with rare, randomly placed excitability bursts."""
    key_noise, key_time, key_cells = jax.random.split(jax.random.PRNGKey(seed), 3)
    noise = jax.random.normal(key_noise, (SLEEP_STEPS, N_PAIRS, N_NEURONS))
    noise = noise * SLEEP_NOISE_STD

    # Rare compact intrinsic fluctuations let an assembly ignite. Their place
    # and time are random; they never encode the learned route order.
    burst_times = jax.random.randint(key_time, (N_PAIRS, 9), 20, SLEEP_STEPS - 20)
    burst_places = jax.random.randint(key_cells, (N_PAIRS, 9), 0, N_PLACES)
    bursts = jnp.zeros((SLEEP_STEPS, N_PAIRS, N_NEURONS), dtype=jnp.float32)
    for pair in range(N_PAIRS):
        for event in range(9):
            start = int(burst_times[pair, event])
            place = int(burst_places[pair, event])
            cells = slice(place * CELLS_PER_PLACE, (place + 1) * CELLS_PER_PLACE)
            bursts = bursts.at[start : start + 3, pair, cells].add(1.0)
    pair_current = noise + bursts * SLEEP_BURST_CURRENT
    return pair_current.repeat(2, axis=1)


def recall_protocol() -> u.Quantity:
    cue = jnp.zeros((RECALL_STEPS, N_LANES, N_NEURONS), dtype=jnp.float32)
    cue = cue.at[:RECALL_CUE_STEPS, :, :CELLS_PER_PLACE].set(1.0)
    return cue * RECALL_CURRENT


def _lane_gates() -> jax.Array:
    return jnp.tile(jnp.asarray([REPLAY_ENABLED, REPLAY_SUPPRESSED]), N_PAIRS)


def _fast_state_filter():
    return state_filter.Any(
        state_filter.OfType(brainstate.HiddenState),
        state_filter.OfType(brainstate.ShortTermState),
    )


def _all_mutable_state_filter():
    return state_filter.Any(
        _fast_state_filter(),
        state_filter.OfType(brainstate.LongTermState),
    )


def snapshot_fast_state(net: RouteNetwork) -> dict[str, object]:
    return {
        path: state.value
        for path, state in net.states(_fast_state_filter()).items()
    }


def restore_fast_state(net: RouteNetwork, snapshot: dict[str, object]) -> None:
    states = net.states(_fast_state_filter())
    if set(states) != set(snapshot):
        raise ValueError("fast-state checkpoint no longer matches the model graph")
    for path, state in states.items():
        state.value = snapshot[path]


def population_activity(spikes: np.ndarray) -> np.ndarray:
    shape = spikes.shape[:-1] + (N_PLACES, CELLS_PER_PLACE)
    return np.asarray(spikes, dtype=np.float32).reshape(shape).mean(axis=-1)


def detect_replays(
    spikes: np.ndarray,
    *,
    active_fraction: float = 0.50,
    max_gap_steps: int = 22,
) -> list[tuple[int, int, str]]:
    """Return non-overlapping A-B-C-D or D-C-B-A onset sequences.

    Each tuple is ``(lane, start_step, direction)``. A population counts as
    active when at least ``active_fraction`` of its cells fire in one step;
    consecutive places must begin no more than ``max_gap_steps`` apart.
    """
    activity = population_activity(spikes)
    events: list[tuple[int, int, str]] = []
    for lane in range(activity.shape[1]):
        active = activity[:, lane] >= active_fraction
        onset = active & np.vstack(
            [np.ones((1, N_PLACES), dtype=bool), ~active[:-1]]
        )
        time, place = np.nonzero(onset)
        index = 0
        while index <= len(place) - N_PLACES:
            sequence = place[index : index + N_PLACES]
            gaps = np.diff(time[index : index + N_PLACES])
            valid = np.all((gaps > 0) & (gaps <= max_gap_steps))
            if valid and np.array_equal(sequence, np.arange(N_PLACES)):
                events.append((lane, int(time[index]), "forward"))
                index += N_PLACES
            elif valid and np.array_equal(sequence, np.arange(N_PLACES)[::-1]):
                events.append((lane, int(time[index]), "backward"))
                index += N_PLACES
            else:
                index += 1
    return events


def recall_score(spikes: np.ndarray) -> np.ndarray:
    """Fraction of B, C, D activated in order after cueing A."""
    activity = population_activity(spikes)
    scores = np.zeros(activity.shape[1], dtype=np.float32)
    for lane in range(activity.shape[1]):
        cursor = RECALL_CUE_STEPS
        hits = 0
        for place in range(1, N_PLACES):
            rates = activity[cursor:, lane, place]
            candidates = np.flatnonzero(rates >= 0.20)
            if not len(candidates):
                break
            cursor += int(candidates[0]) + 1
            hits += 1
        scores[lane] = hits / (N_PLACES - 1)
    return scores


@dataclass(frozen=True)
class ExperimentResult:
    sleep_spikes: np.ndarray
    recall_spikes: np.ndarray
    learned_weights: np.ndarray
    post_sleep_weights: np.ndarray
    replay_events: list[tuple[int, int, str]]
    recall_scores: np.ndarray


def run_experiment() -> ExperimentResult:
    brainstate.random.seed(12)
    with brainstate.environ.context(dt=DT):
        net = RouteNetwork()
        brainstate.nn.vmap_init_all_states(net, axis_size=N_LANES)
        fast_initial = snapshot_fast_state(net)

        mapped_state = _all_mutable_state_filter()
        mapped_step = vmap2(
            net.update,
            in_axes=(None, 0, 0, 0, 0),
            out_axes=0,
            state_in_axes={0: mapped_state},
            state_out_axes={0: mapped_state},
            unexpected_out_state_mapping="raise",
        )

        learn_cue = learning_protocol()
        learn_times = u.math.arange(0.0 * u.ms, learn_cue.shape[0] * DT, DT)
        zero_intrinsic = jnp.zeros((N_LANES, N_NEURONS)) * u.nA
        ones = jnp.ones(N_LANES, dtype=jnp.float32)

        @brainstate.transform.jit
        def learn():
            def step(t, cue):
                cue_lanes = u.math.broadcast_to(cue, (N_LANES, N_NEURONS))
                return mapped_step(t, zero_intrinsic, cue_lanes, ones, ones)

            return brainstate.transform.for_loop(step, learn_times, learn_cue)

        learn()
        learned_weights = np.asarray(net.weight.value)

        restore_fast_state(net, fast_initial)
        sleep_current = sleep_intrinsic_current()
        sleep_times = u.math.arange(0.0 * u.ms, SLEEP_STEPS * DT, DT)
        zero_cue = jnp.zeros((N_LANES, N_NEURONS)) * u.nA
        replay_gate = _lane_gates()

        @brainstate.transform.jit
        def sleep():
            def step(t, intrinsic):
                return mapped_step(t, intrinsic, zero_cue, replay_gate, ones)

            return brainstate.transform.for_loop(step, sleep_times, sleep_current)

        sleep_spikes = np.asarray(sleep())
        post_sleep_weights = np.asarray(net.weight.value)

        restore_fast_state(net, fast_initial)
        recall_cue = recall_protocol()
        recall_times = u.math.arange(0.0 * u.ms, RECALL_STEPS * DT, DT)
        zero_recall_intrinsic = jnp.zeros((N_LANES, N_NEURONS)) * u.nA
        no_plasticity = jnp.zeros(N_LANES, dtype=jnp.float32)

        @brainstate.transform.jit
        def recall():
            def step(t, cue):
                return mapped_step(
                    t, zero_recall_intrinsic, cue, ones, no_plasticity
                )

            return brainstate.transform.for_loop(step, recall_times, recall_cue)

        recall_spikes = np.asarray(recall())

    events = detect_replays(sleep_spikes)
    scores = recall_score(recall_spikes)
    return ExperimentResult(
        sleep_spikes=sleep_spikes,
        recall_spikes=recall_spikes,
        learned_weights=learned_weights,
        post_sleep_weights=post_sleep_weights,
        replay_events=events,
        recall_scores=scores,
    )


def summarize(result: ExperimentResult) -> str:
    enabled = np.arange(0, N_LANES, 2)
    suppressed = np.arange(1, N_LANES, 2)
    forward = sum(
        lane in enabled and direction == "forward"
        for lane, _, direction in result.replay_events
    )
    backward = sum(
        lane in enabled and direction == "backward"
        for lane, _, direction in result.replay_events
    )
    control_events = sum(
        lane in suppressed for lane, _, _ in result.replay_events
    )
    delta = result.recall_scores[enabled] - result.recall_scores[suppressed]
    return "\n".join(
        [
            f"Replay-enabled sleep: {forward} forward, {backward} backward events",
            f"Replay-suppressed sleep: {control_events} complete route events",
            (
                "Recall score (mean +/- SD): "
                f"replay {result.recall_scores[enabled].mean():.3f} +/- "
                f"{result.recall_scores[enabled].std(ddof=1):.3f}; "
                f"suppressed {result.recall_scores[suppressed].mean():.3f} +/- "
                f"{result.recall_scores[suppressed].std(ddof=1):.3f}"
            ),
            f"Matched-pair recall difference: {delta.mean():+.3f}",
        ]
    )


def save_evidence(result: ExperimentResult, path: Path) -> None:
    event_lane = np.asarray([event[0] for event in result.replay_events], dtype=int)
    event_step = np.asarray([event[1] for event in result.replay_events], dtype=int)
    event_direction = np.asarray(
        [event[2] for event in result.replay_events], dtype="U8"
    )
    np.savez_compressed(
        path,
        sleep_spikes=result.sleep_spikes,
        recall_spikes=result.recall_spikes,
        learned_weights=result.learned_weights,
        post_sleep_weights=result.post_sleep_weights,
        replay_event_lane=event_lane,
        replay_event_step=event_step,
        replay_event_direction=event_direction,
        recall_scores=result.recall_scores,
        replay_gate=np.asarray(_lane_gates()),
        dt_ms=DT.to_decimal(u.ms),
    )


def plot_result(result: ExperimentResult, path: Path) -> None:
    enabled = np.arange(0, N_LANES, 2)
    suppressed = np.arange(1, N_LANES, 2)
    activity = population_activity(result.sleep_spikes)
    replay_lanes = [lane for lane, _, _ in result.replay_events if lane % 2 == 0]
    if replay_lanes:
        pair = int(replay_lanes[0] // 2)
    else:
        pair = int(np.argmax([np.sum(result.sleep_spikes[:, lane]) for lane in enabled]))
    replay_lane = enabled[pair]
    control_lane = suppressed[pair]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    t_sleep = np.arange(SLEEP_STEPS) * DT.to_decimal(u.ms)
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for place, color in enumerate(colors):
        axes[0, 0].plot(
            t_sleep,
            activity[:, replay_lane, place],
            color=color,
            lw=0.8,
            label=PLACE_NAMES[place],
        )
    axes[0, 0].set(title="Replay-enabled sleep", ylabel="Place activity")
    axes[0, 0].legend(ncol=4, frameon=False)

    for place, color in enumerate(colors):
        axes[0, 1].plot(
            t_sleep,
            activity[:, control_lane, place],
            color=color,
            lw=0.8,
        )
    axes[0, 1].set(title="Matched replay suppression")

    place = _place_ids()
    forward_mask = np.asarray(place[:, None] + 1 == place[None, :])
    forward_before = result.learned_weights[:, forward_mask].mean(axis=1)
    forward_after = result.post_sleep_weights[:, forward_mask].mean(axis=1)
    axes[1, 0].plot(
        [0, 1],
        np.c_[forward_before[enabled], forward_after[enabled]].T,
        color="#0072B2",
        alpha=0.45,
    )
    axes[1, 0].plot(
        [0, 1],
        np.c_[forward_before[suppressed], forward_after[suppressed]].T,
        color="#D55E00",
        alpha=0.45,
    )
    axes[1, 0].set(
        xticks=[0, 1],
        xticklabels=["Post-learning", "Post-sleep"],
        ylabel="Mean forward efficacy",
        title="Sleep-dependent weight change",
    )

    x = np.arange(N_PAIRS)
    width = 0.38
    axes[1, 1].bar(
        x - width / 2,
        result.recall_scores[enabled],
        width,
        color="#0072B2",
        label="Replay",
    )
    axes[1, 1].bar(
        x + width / 2,
        result.recall_scores[suppressed],
        width,
        color="#D55E00",
        label="Suppressed",
    )
    axes[1, 1].set(
        xlabel="Matched pair",
        ylabel="Ordered recall score",
        ylim=(0.0, 1.05),
        title="Cue A, then score B-C-D",
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes[0]:
        axis.set(xlabel="Sleep time (ms)", ylim=(-0.03, 1.03))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    result = run_experiment()
    figure = Path("sleep_replay_results.png")
    evidence = Path("sleep_replay_evidence.npz")
    plot_result(result, figure)
    save_evidence(result, evidence)
    print(summarize(result))
    print(f"Figure: {figure.resolve()}")
    print(f"Evidence: {evidence.resolve()}")


if __name__ == "__main__":
    main()
