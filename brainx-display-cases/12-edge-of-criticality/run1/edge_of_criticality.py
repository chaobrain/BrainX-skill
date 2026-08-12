"""Locate the stable edge of criticality in a sparse recurrent E/I network."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-edge-criticality")

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brainstate.transform import vmap2
from brainstate.util import filter as state_filter


DT = 0.5 * u.ms
DURATION = 300.0 * u.ms
SPARK_DURATION = 1.0 * u.ms
SPARK_CURRENT = 240.0 * u.mA
SPARK_NEURONS = 8

N_EXC = 320
N_INH = 80
CONNECTION_PROBABILITY = 0.05

V_REST = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
V_RESET = -60.0 * u.mV
TAU_MEMBRANE = 20.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
TAU_EXCITATORY = 3.0 * u.ms
TAU_INHIBITORY = 6.0 * u.ms

# One excitatory event produces about J_EE * tau_exc / tau_membrane of EPSP.
J_EE = 55.0 * u.mA
J_EE_CV = 0.30
J_EI = 45.0 * u.mA
J_IE = 55.0 * u.mA
J_II = 35.0 * u.mA

DEFAULT_COUPLINGS = np.array(
    [
        0.50,
        0.55,
        0.60,
        0.625,
        0.65,
        0.675,
        0.70,
        0.72,
        0.73,
        0.74,
        0.745,
        0.75,
        0.755,
        0.76,
        0.77,
        0.78,
        0.80,
        0.825,
        0.85,
        0.90,
    ]
)
DEFAULT_REALIZATIONS = 16
BASE_SEED = 90210

AVALANCHE_BIN = 2.0 * u.ms
TAIL_WINDOW = 50.0 * u.ms
TAIL_ACTIVE_BIN_FRACTION = 0.8
MAX_UNSTABLE_FRACTION = 0.10
NEAR_PEAK_FRACTION = 0.90


class RecurrentEINetwork(brainstate.nn.Module):
    """Unit-aware LIF populations with sparse event-driven recurrent input."""

    def __init__(self, n_exc: int = N_EXC, n_inh: int = N_INH):
        super().__init__()
        self.n_exc = n_exc
        self.n_inh = n_inh
        neuron_args = dict(
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_initializer=braintools.init.Constant(V_REST),
        )
        self.exc = brainpy.state.LIFRef(n_exc, **neuron_args)
        self.inh = brainpy.state.LIFRef(n_inh, **neuron_args)
        zero_current = braintools.init.Constant(0.0 * u.mA)
        self.ee = brainpy.state.Expon(
            n_exc, tau=TAU_EXCITATORY, g_initializer=zero_current
        )
        self.ie = brainpy.state.Expon(
            n_exc, tau=TAU_INHIBITORY, g_initializer=zero_current
        )
        self.ei = brainpy.state.Expon(
            n_inh, tau=TAU_EXCITATORY, g_initializer=zero_current
        )
        self.ii = brainpy.state.Expon(
            n_inh, tau=TAU_INHIBITORY, g_initializer=zero_current
        )

    def update(self, t, coupling, graph_seed, spark_target, spark_on):
        with brainstate.environ.context(t=t):
            exc_spikes = brainevent.BinaryArray(self.exc.get_spike() != 0.0)
            inh_spikes = brainevent.BinaryArray(self.inh.get_spike() != 0.0)

            # Seeds define a matched graph: only the E->E weight changes across
            # coupling conditions for a given realization.
            half_width = jnp.sqrt(3.0) * J_EE_CV
            ee_conn = brainevent.JITCUniformR(
                (
                    coupling * J_EE * (1.0 - half_width),
                    coupling * J_EE * (1.0 + half_width),
                    CONNECTION_PROBABILITY,
                    graph_seed,
                ),
                shape=(self.n_exc, self.n_exc),
            )
            ei_conn = brainevent.JITCScalarR(
                (J_EI, CONNECTION_PROBABILITY, graph_seed + 1),
                shape=(self.n_exc, self.n_inh),
            )
            ie_conn = brainevent.JITCScalarR(
                (-J_IE, CONNECTION_PROBABILITY, graph_seed + 2),
                shape=(self.n_inh, self.n_exc),
            )
            ii_conn = brainevent.JITCScalarR(
                (-J_II, CONNECTION_PROBABILITY, graph_seed + 3),
                shape=(self.n_inh, self.n_inh),
            )

            exc_recurrent = self.ee(exc_spikes @ ee_conn) + self.ie(inh_spikes @ ie_conn)
            inh_recurrent = self.ei(exc_spikes @ ei_conn) + self.ii(inh_spikes @ ii_conn)
            spark_indices = (spark_target + jnp.arange(SPARK_NEURONS)) % self.n_exc
            spark = jnn.one_hot(
                spark_indices, self.n_exc, dtype=jnp.float32
            ).sum(axis=0) * spark_on * SPARK_CURRENT

            exc_now = self.exc(exc_recurrent + spark)
            inh_now = self.inh(inh_recurrent)
            return jnp.stack((jnp.sum(exc_now), jnp.sum(inh_now)))


def simulate(couplings, realizations=DEFAULT_REALIZATIONS, base_seed=BASE_SEED):
    """Run all coupling-by-realization lanes in one stateful mapped rollout."""
    couplings = jnp.asarray(couplings, dtype=jnp.float32)
    lane_couplings = jnp.repeat(couplings, realizations)
    realization_ids = jnp.tile(jnp.arange(realizations, dtype=jnp.uint32), couplings.size)
    graph_seeds = jnp.asarray(base_seed, dtype=jnp.uint32) + realization_ids * 8
    spark_targets = realization_ids.astype(jnp.int32) % N_EXC

    with brainstate.environ.context(dt=DT):
        net = RecurrentEINetwork()
        brainstate.nn.vmap_init_all_states(net, axis_size=lane_couplings.size)

        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = vmap2(
            net.update,
            in_axes=(None, 0, 0, 0, None),
            out_axes=0,
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )
        times = u.math.arange(0.0 * u.ms, DURATION, DT)
        spark_envelope = times < SPARK_DURATION

        @brainstate.transform.jit
        def run():
            def step(t, spark_on):
                return mapped_step(
                    t,
                    lane_couplings,
                    graph_seeds,
                    spark_targets,
                    spark_on,
                )

            return brainstate.transform.for_loop(step, times, spark_envelope)

        counts = run()

    lane_first = np.asarray(counts).transpose(1, 0, 2)
    return lane_first.reshape(len(couplings), realizations, -1, 2)


def contiguous_avalanche_sizes(population_counts):
    """Return sizes of nonempty runs in a one-dimensional binned spike train."""
    active = population_counts > 0
    padded = np.pad(active.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return np.asarray([population_counts[start:stop].sum() for start, stop in zip(starts, stops)])


def analyze_activity(counts, couplings):
    """Measure stable avalanche variability and sustained-activity probability."""
    bin_steps = int(round(AVALANCHE_BIN / DT))
    tail_bins = int(round(TAIL_WINDOW / AVALANCHE_BIN))
    n_couplings, n_realizations, n_steps, _ = counts.shape
    usable_steps = n_steps // bin_steps * bin_steps
    binned = counts[:, :, :usable_steps].sum(axis=(3,)).reshape(
        n_couplings, n_realizations, -1, bin_steps
    ).sum(axis=3)

    rows = []
    realization_rows = []
    for coupling_index, coupling in enumerate(couplings):
        condition = binned[coupling_index]
        tail_active_fraction = (condition[:, -tail_bins:] > 0).mean(axis=1)
        unstable = tail_active_fraction >= TAIL_ACTIVE_BIN_FRACTION
        for realization, (tail_fraction, is_unstable, activity) in enumerate(
            zip(tail_active_fraction, unstable, condition)
        ):
            realization_rows.append(
                {
                    "coupling": float(coupling),
                    "realization": realization,
                    "tail_active_fraction": float(tail_fraction),
                    "unstable": bool(is_unstable),
                    "total_spikes": float(activity.sum()),
                }
            )
        stable_sizes = []
        for realization in condition[~unstable]:
            stable_sizes.extend(contiguous_avalanche_sizes(realization))
        stable_sizes = np.asarray(stable_sizes, dtype=float)
        mean_size = float(stable_sizes.mean()) if stable_sizes.size else np.nan
        susceptibility = (
            float(stable_sizes.var(ddof=1) / mean_size)
            if stable_sizes.size > 1 and mean_size > 0.0
            else np.nan
        )
        rows.append(
            {
                "coupling": float(coupling),
                "mean_avalanche_size": mean_size,
                "susceptibility": susceptibility,
                "unstable_fraction": float(unstable.mean()),
                "stable_realizations": int((~unstable).sum()),
                "mean_total_spikes": float(condition.sum(axis=1).mean()),
            }
        )
    return rows, realization_rows, binned


def locate_critical_region(rows):
    """Return a resolved near-peak stable interval, or only the sampled optimum."""
    stable = [
        row
        for row in rows
        if row["unstable_fraction"] <= MAX_UNSTABLE_FRACTION
        and np.isfinite(row["susceptibility"])
        and row["susceptibility"] > 0.0
    ]
    if not stable:
        return [], None
    optimum = max(stable, key=lambda row: row["susceptibility"])
    threshold = NEAR_PEAK_FRACTION * optimum["susceptibility"]
    near_peak = [row for row in stable if row["susceptibility"] >= threshold]

    by_index = {rows.index(row): row for row in near_peak}
    optimum_index = rows.index(optimum)
    left = optimum_index
    right = optimum_index
    while left - 1 in by_index:
        left -= 1
    while right + 1 in by_index:
        right += 1
    region = rows[left : right + 1]
    return (region if len(region) >= 2 else []), optimum


def save_metrics(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_counts(binned, couplings, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        population_spikes=binned,
        couplings=np.asarray(couplings),
        bin_width_ms=AVALANCHE_BIN.to_decimal(u.ms),
    )
    return path


def plot_results(rows, binned, region, optimum, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    couplings = np.asarray([row["coupling"] for row in rows])
    susceptibility = np.asarray([row["susceptibility"] for row in rows])
    unstable = np.asarray([row["unstable_fraction"] for row in rows])
    time_ms = np.arange(binned.shape[2]) * AVALANCHE_BIN.to_decimal(u.ms)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    mean_activity = binned.mean(axis=1)
    image = axes[0].imshow(
        mean_activity,
        origin="lower",
        aspect="auto",
        extent=(time_ms[0], time_ms[-1], couplings[0], couplings[-1]),
        cmap="magma",
    )
    axes[0].set(xlabel="Time after spark (ms)", ylabel="E-E coupling", title="Mean spikes per 2 ms bin")
    fig.colorbar(image, ax=axes[0], label="Spikes")

    axes[1].plot(couplings, susceptibility, "o-", color="black", label="Avalanche susceptibility")
    axes[1].set(xlabel="E-E coupling", ylabel="Variance / mean avalanche size")
    instability_axis = axes[1].twinx()
    instability_axis.plot(couplings, unstable, "s--", color="tab:red", label="Sustained fraction")
    instability_axis.axhline(MAX_UNSTABLE_FRACTION, color="tab:red", linestyle=":", linewidth=1)
    instability_axis.set_ylabel("Sustained-activity fraction", color="tab:red")
    if optimum is not None:
        axes[1].axvline(optimum["coupling"], color="tab:blue", linestyle="--", linewidth=1)
    if region:
        axes[1].axvspan(region[0]["coupling"], region[-1]["coupling"], color="tab:green", alpha=0.16)
    axes[1].legend(loc="upper left")
    instability_axis.legend(loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def print_summary(rows, region, optimum):
    print("coupling  susceptibility  unstable  mean avalanche")
    for row in rows:
        print(
            f"{row['coupling']:8.3f}  {row['susceptibility']:14.3f}  "
            f"{row['unstable_fraction']:8.3f}  {row['mean_avalanche_size']:14.2f}"
        )
    if optimum is None:
        print("No stable coupling produced enough avalanches to score.")
    elif region:
        print(
            f"Resolved critical region: {region[0]['coupling']:.3f} to "
            f"{region[-1]['coupling']:.3f}; sampled optimum {optimum['coupling']:.3f}."
        )
    else:
        print(
            f"Sampled stable optimum: {optimum['coupling']:.3f}. "
            "The near-peak set contains fewer than two adjacent points, so it is not called a region."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS)
    parser.add_argument("--couplings", type=float, nargs="+", default=DEFAULT_COUPLINGS.tolist())
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    couplings = np.asarray(sorted(args.couplings), dtype=float)
    counts = simulate(couplings, realizations=args.realizations, base_seed=args.base_seed)
    rows, realization_rows, binned = analyze_activity(counts, couplings)
    region, optimum = locate_critical_region(rows)
    metrics_path = save_metrics(rows, args.output_dir / "criticality_metrics.csv")
    realization_path = save_metrics(
        realization_rows, args.output_dir / "criticality_realizations.csv"
    )
    counts_path = save_counts(binned, couplings, args.output_dir / "binned_spike_counts.npz")
    figure_path = plot_results(
        rows,
        binned,
        region,
        optimum,
        args.output_dir / "criticality_summary.png",
    )
    print_summary(rows, region, optimum)
    print(f"Metrics: {metrics_path}")
    print(f"Runs:    {realization_path}")
    print(f"Counts:  {counts_path}")
    print(f"Figure:  {figure_path}")


if __name__ == "__main__":
    main()
