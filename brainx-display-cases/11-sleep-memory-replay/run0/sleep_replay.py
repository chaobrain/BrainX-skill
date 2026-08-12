"""Learn, replay, suppress, and recall a four-place route with BrainX.

The four place assemblies are represented by four recurrent LIF neurons.  A
small network is intentional here: it makes every plastic synapse and every
decoded replay event directly inspectable.  Independent matched pairs are
executed with BrainState ``vmap2``; time is executed with ``for_loop``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/brainx-matplotlib")

import brainevent
import brainpy
import brainstate
import brainunit as u
import braintools
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.util import filter as state_filter


DT = 1.0 * u.ms
N_PLACES = 4
N_PAIRS = 8
N_LANES = 2 * N_PAIRS

V_REST = -65.0 * u.mV
V_RESET = -65.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
TAU_MEMBRANE = 10.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
MEMBRANE_RESISTANCE = 100.0 * u.Mohm

TAU_SYNAPSE = 4.0 * u.ms
REVERSAL_EXCITATORY = 0.0 * u.mV
MAX_RECURRENT_CONDUCTANCE = 220.0 * u.nS

TAU_STDP = 18.0 * u.ms
POTENTIATION = 0.018
DEPRESSION = 0.014
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0

ROUTE_CURRENT = 0.45 * u.nA
CUE_CURRENT = 0.45 * u.nA
PLACE_BLOCK_STEPS = 9
PLACE_DRIVE_STEPS = 5
N_TRAVERSALS = 7
INTER_TRAVERSAL_STEPS = 45
PRIME_STEPS = 5
SLEEP_STEPS = 75
RECALL_STEPS = 75
RECALL_RECURRENT_GATE = 0.60

PLACE_NAMES = np.asarray(["A", "B", "C", "D"])
ROUTE_TOPOLOGY = jnp.asarray(
    np.equal.outer(np.arange(N_PLACES), np.arange(N_PLACES) + 1)
    | np.equal.outer(np.arange(N_PLACES) + 1, np.arange(N_PLACES)),
    dtype=jnp.float32,
)


class RouteNetwork(brainstate.nn.Module):
    """Four recurrent place cells with conductance transmission and STDP."""

    def __init__(self):
        super().__init__()
        self.cells = brainpy.state.LIFRef(
            N_PLACES,
            R=MEMBRANE_RESISTANCE,
            tau=TAU_MEMBRANE,
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau_ref=TAU_REFRACTORY,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.synapse = brainpy.state.Expon(N_PLACES, tau=TAU_SYNAPSE)
        self.output = brainpy.state.COBA(E=REVERSAL_EXCITATORY)
        self.cells.add_current_input("route_recurrence", self.output)

    def init_state(self):
        self.weight = brainstate.LongTermState(
            jnp.zeros((N_PLACES, N_PLACES), dtype=jnp.float32)
        )
        self.pre_trace = brainstate.ShortTermState(
            jnp.zeros(N_PLACES, dtype=jnp.float32)
        )
        self.post_trace = brainstate.ShortTermState(
            jnp.zeros(N_PLACES, dtype=jnp.float32)
        )

    def reset_state(self):
        # Learned weights deliberately survive an independent recall rollout.
        self.pre_trace.value = jnp.zeros_like(self.pre_trace.value)
        self.post_trace.value = jnp.zeros_like(self.post_trace.value)

    def update(self, t, sensory_drive, recurrent_gate, plasticity_gate):
        with brainstate.environ.context(t=t):
            transmitting_spikes = self.cells.get_spike() != 0.0
            recurrent_efficacy = (
                brainevent.BinaryArray(transmitting_spikes) @ self.weight.value
            )
            recurrent_conductance = (
                recurrent_efficacy
                * recurrent_gate
                * MAX_RECURRENT_CONDUCTANCE
            )
            self.output.bind_cond(self.synapse(recurrent_conductance))
            new_spikes = self.cells(sensory_drive) != 0.0

            trace_decay = u.math.exp(-brainstate.environ.get_dt() / TAU_STDP)
            new_pre_trace = (
                self.pre_trace.value * trace_decay
                + new_spikes.astype(jnp.float32)
            )
            new_post_trace = (
                self.post_trace.value * trace_decay
                + new_spikes.astype(jnp.float32)
            )

            potentiated = brainevent.update_dense_on_binary_post(
                weight=self.weight.value,
                post_spike=new_spikes,
                pre_trace=POTENTIATION * new_pre_trace,
                w_min=WEIGHT_MIN,
                w_max=WEIGHT_MAX,
            )
            updated = brainevent.update_dense_on_binary_pre(
                weight=potentiated,
                pre_spike=new_spikes,
                post_trace=-DEPRESSION * new_post_trace,
                w_min=WEIGHT_MIN,
                w_max=WEIGHT_MAX,
            )
            updated = updated * ROUTE_TOPOLOGY
            self.weight.value = (
                self.weight.value
                + plasticity_gate * (updated - self.weight.value)
            )
            self.pre_trace.value = new_pre_trace
            self.post_trace.value = new_post_trace
            return transmitting_spikes, new_spikes


def route_protocol() -> u.Quantity:
    """Seven A-B-C-D traversals followed by an A reactivation at sleep onset."""
    traversal_steps = N_PLACES * PLACE_BLOCK_STEPS
    trial_steps = traversal_steps + INTER_TRAVERSAL_STEPS
    learned_steps = N_TRAVERSALS * trial_steps
    protocol = np.zeros(
        (learned_steps + PRIME_STEPS, N_PLACES), dtype=np.float32
    )
    for trial in range(N_TRAVERSALS):
        trial_start = trial * trial_steps
        for place in range(N_PLACES):
            start = trial_start + place * PLACE_BLOCK_STEPS
            protocol[start : start + PLACE_DRIVE_STEPS, place] = 1.0
    protocol[-PRIME_STEPS:, 0] = 1.0
    return jnp.asarray(protocol) * ROUTE_CURRENT


def cue_protocol() -> u.Quantity:
    protocol = np.zeros((RECALL_STEPS, N_PLACES), dtype=np.float32)
    protocol[:PLACE_DRIVE_STEPS, 0] = 1.0
    return jnp.asarray(protocol) * CUE_CURRENT


def first_spike_times(events: np.ndarray) -> np.ndarray:
    """Return first event index by lane and place; missing events are infinity."""
    time_index = np.arange(events.shape[0], dtype=np.float32)[:, None, None]
    return np.where(events, time_index, np.inf).min(axis=0)


def direction_scores(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count adjacent first-spike transitions consistent with each direction."""
    first = first_spike_times(events)
    finite = np.isfinite(first)
    adjacent_present = finite[:, 1:] & finite[:, :-1]
    forward = np.sum(
        adjacent_present & (first[:, 1:] > first[:, :-1]), axis=1
    )
    backward = np.sum(
        adjacent_present & (first[:, :-1] > first[:, 1:]), axis=1
    )
    return forward, backward


def recall_scores(events: np.ndarray) -> np.ndarray:
    """Fraction of B-C-D recovered as one ordered prefix after the A cue."""
    first = first_spike_times(events)
    b = np.isfinite(first[:, 1])
    c = b & np.isfinite(first[:, 2]) & (first[:, 2] > first[:, 1])
    d = c & np.isfinite(first[:, 3]) & (first[:, 3] > first[:, 2])
    return (b.astype(float) + c.astype(float) + d.astype(float)) / 3.0


def classify_replay(forward: np.ndarray, backward: np.ndarray) -> str:
    forward_total = int(forward.sum())
    backward_total = int(backward.sum())
    if forward_total > backward_total:
        return "forward"
    if backward_total > forward_total:
        return "backward"
    return "undetermined"


def plot_results(
    sleep_events: np.ndarray,
    before_sleep: np.ndarray,
    after_sleep: np.ndarray,
    recall: np.ndarray,
    output_path: Path,
):
    replay_example = sleep_events[:, 0]
    suppressed_example = sleep_events[:, N_PAIRS]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    for place in range(N_PLACES):
        replay_times = np.flatnonzero(replay_example[:, place]) * DT.to_decimal(u.ms)
        suppressed_times = (
            np.flatnonzero(suppressed_example[:, place]) * DT.to_decimal(u.ms)
        )
        axes[0].scatter(replay_times, np.full_like(replay_times, place), s=28,
                        color="#167d68", marker="|")
        axes[0].scatter(suppressed_times, np.full_like(suppressed_times, place + 4),
                        s=28, color="#b4483d", marker="|")
    axes[0].set(
        title="Sleep activity (matched pair 1)",
        xlabel="Sleep time (ms)",
        ylabel="Place / condition",
        yticks=np.arange(8),
        yticklabels=list(PLACE_NAMES) + [f"{p} blocked" for p in PLACE_NAMES],
    )

    path = np.arange(N_PLACES - 1)
    axes[1].plot(path, before_sleep[0, path, path + 1], "o--", color="#555555",
                 label="Before sleep")
    axes[1].plot(path, after_sleep[:N_PAIRS, path, path + 1].mean(0), "o-",
                 color="#167d68", label="Replay")
    axes[1].plot(path, after_sleep[N_PAIRS:, path, path + 1].mean(0), "s-",
                 color="#b4483d", label="Replay blocked")
    axes[1].set(
        title="Forward route synapses",
        xlabel="Connection",
        ylabel="Dimensionless efficacy",
        xticks=path,
        xticklabels=["A->B", "B->C", "C->D"],
    )
    axes[1].legend(frameon=False)

    jitter = np.linspace(-0.05, 0.05, N_PAIRS)
    axes[2].plot(
        np.vstack((np.zeros(N_PAIRS), np.ones(N_PAIRS))),
        np.vstack((recall[:N_PAIRS], recall[N_PAIRS:])),
        color="#a7a7a7",
        linewidth=1.0,
        zorder=1,
    )
    axes[2].scatter(jitter, recall[:N_PAIRS], color="#167d68", label="Replay", zorder=2)
    axes[2].scatter(1 + jitter, recall[N_PAIRS:], color="#b4483d",
                    label="Replay blocked", zorder=2)
    axes[2].set(
        title="Cue-only route recall",
        ylabel="Ordered recall score",
        xticks=[0, 1],
        xticklabels=["Replay", "Blocked"],
        ylim=(-0.05, 1.05),
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_experiment(output_path: Path = Path("sleep_replay_results.png")) -> dict:
    brainstate.random.seed(7)
    lane_gain = np.linspace(0.94, 1.06, N_PAIRS, dtype=np.float32)
    lane_gain = jnp.asarray(np.concatenate((lane_gain, lane_gain)))
    replay_gate = jnp.asarray(
        np.concatenate((np.ones(N_PAIRS), np.zeros(N_PAIRS))), dtype=jnp.float32
    )
    all_enabled = jnp.ones(N_LANES, dtype=jnp.float32)

    with brainstate.environ.context(dt=DT):
        network = RouteNetwork()
        brainstate.nn.vmap_init_all_states(network, axis_size=N_LANES)
        initial_state = {
            path: state.value for path, state in network.states().items()
        }

        mapped_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
            state_filter.OfType(brainstate.LongTermState),
        )
        mapped_step = brainstate.transform.vmap2(
            network.update,
            in_axes=(None, 0, 0, 0),
            out_axes=0,
            state_in_axes={0: mapped_state},
            state_out_axes={0: mapped_state},
            unexpected_out_state_mapping="raise",
        )

        @brainstate.transform.jit
        def run_phase(times, drives, recurrence, plasticity):
            def step(t, drive):
                return mapped_step(t, drive, recurrence, plasticity)

            return brainstate.transform.for_loop(step, times, drives)

        learning_drive = route_protocol()[:, None, :] * lane_gain[None, :, None]
        learning_times = u.math.arange(
            0.0 * u.ms, learning_drive.shape[0] * DT, DT
        )
        run_phase(
            learning_times,
            learning_drive,
            jnp.zeros_like(all_enabled),
            all_enabled,
        )
        before_sleep = np.asarray(network.weight.value)
        if not np.allclose(before_sleep[:N_PAIRS], before_sleep[N_PAIRS:]):
            raise AssertionError("matched groups diverged before sleep")

        sleep_drive = jnp.zeros((SLEEP_STEPS, N_LANES, N_PLACES)) * u.nA
        sleep_start = learning_drive.shape[0] * DT
        sleep_times = u.math.arange(
            sleep_start, sleep_start + SLEEP_STEPS * DT, DT
        )
        sleep_events, _ = run_phase(
            sleep_times, sleep_drive, replay_gate, all_enabled
        )
        after_sleep = np.asarray(network.weight.value)

        weight_snapshot = np.asarray(network.weight.value).copy()
        recall_state = {
            path: (
                state.value
                if isinstance(state, brainstate.LongTermState)
                else initial_state[path]
            )
            for path, state in network.states().items()
        }
        unexpected, missing = brainstate.nn.assign_state_values(
            network, recall_state
        )
        if unexpected or missing:
            raise AssertionError(
                f"recall state restore mismatch: unexpected={unexpected}, "
                f"missing={missing}"
            )
        if network.cells.V.value.shape[0] != N_LANES:
            raise AssertionError("state restore lost the independent-lane axis")
        if not np.allclose(np.asarray(network.weight.value), weight_snapshot):
            raise AssertionError("recall reset changed learned weights")

        recall_drive = cue_protocol()[:, None, :] * lane_gain[None, :, None]
        recall_times = u.math.arange(0.0 * u.ms, RECALL_STEPS * DT, DT)
        recall_events, _ = run_phase(
            recall_times,
            recall_drive,
            jnp.full_like(all_enabled, RECALL_RECURRENT_GATE),
            jnp.zeros_like(all_enabled),
        )

    sleep_events_np = np.asarray(sleep_events, dtype=bool)
    recall_events_np = np.asarray(recall_events, dtype=bool)
    forward, backward = direction_scores(sleep_events_np[:, :N_PAIRS])
    recall = recall_scores(recall_events_np)
    direction = classify_replay(forward, backward)
    plot_results(sleep_events_np, before_sleep, after_sleep, recall, output_path)

    forward_edges = np.arange(N_PLACES - 1)
    forward_before = before_sleep[:, forward_edges, forward_edges + 1].mean(axis=1)
    forward_after = after_sleep[:, forward_edges, forward_edges + 1].mean(axis=1)
    return {
        "direction": direction,
        "forward_transitions": forward,
        "backward_transitions": backward,
        "matched_pre_sleep_max_difference": float(
            np.max(np.abs(before_sleep[:N_PAIRS] - before_sleep[N_PAIRS:]))
        ),
        "replay_weight_change": float(
            np.mean(forward_after[:N_PAIRS] - forward_before[:N_PAIRS])
        ),
        "blocked_weight_change": float(
            np.mean(forward_after[N_PAIRS:] - forward_before[N_PAIRS:])
        ),
        "replay_recall": recall[:N_PAIRS],
        "blocked_recall": recall[N_PAIRS:],
        "output_path": output_path,
    }


def main():
    result = run_experiment()
    paired_gain = result["replay_recall"] - result["blocked_recall"]
    print(f"Sleep replay direction: {result['direction']}")
    print(
        "Transition evidence (forward / backward): "
        f"{int(result['forward_transitions'].sum())} / "
        f"{int(result['backward_transitions'].sum())}"
    )
    print(
        "Matched pre-sleep max |weight difference|: "
        f"{result['matched_pre_sleep_max_difference']:.3g}"
    )
    print(
        "Mean forward-weight change during sleep (replay / blocked): "
        f"{result['replay_weight_change']:.3f} / "
        f"{result['blocked_weight_change']:.3f}"
    )
    print(
        "Mean ordered recall (replay / blocked): "
        f"{result['replay_recall'].mean():.3f} / "
        f"{result['blocked_recall'].mean():.3f}"
    )
    print(f"Mean paired recall benefit: {paired_gain.mean():.3f}")
    print(f"Saved figure: {result['output_path']}")


if __name__ == "__main__":
    main()
