"""Locate a phenomenological edge of criticality in a sparse E/I network."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
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
from brainstate.util import filter as state_filter


DT = 0.2 * u.ms
DURATION = 300.0 * u.ms
V_REST = -65.0 * u.mV
V_RESET = -65.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
TAU_MEMBRANE = 20.0 * u.ms
TAU_REFRACTORY = 2.0 * u.ms
TAU_EXCITATORY = 5.0 * u.ms
TAU_INHIBITORY = 10.0 * u.ms
E_EXCITATORY = 0.0 * u.mV
E_INHIBITORY = -80.0 * u.mV
W_EE = 0.55 * u.siemens
W_EI = 0.70 * u.siemens
W_IE = 2.50 * u.siemens
W_II = 2.00 * u.siemens


@dataclass(frozen=True)
class Experiment:
    """Numerical and pre-registered analysis settings for one scan."""

    n_exc: int = 128
    n_inh: int = 32
    exc_fanout: int = 16
    inh_fanout: int = 16
    n_realizations: int = 16
    gains: tuple[float, ...] = tuple(np.linspace(0.8, 2.6, 19))
    first_seed: int = 1701
    bin_ms: float = 1.0
    quiet_ms: float = 20.0
    late_window_ms: float = 50.0
    runaway_rate_hz: float = 15.0
    max_unstable_fraction: float = 0.10
    region_score_fraction: float = 0.90

    @property
    def n_neurons(self) -> int:
        return self.n_exc + self.n_inh


class RecurrentEINetwork(brainstate.nn.Module):
    """Two LIF populations coupled by four event-driven sparse pathways."""

    def __init__(self, config: Experiment):
        super().__init__()
        initializer = braintools.init.Constant(V_REST)
        neuron_args = dict(
            V_rest=V_REST,
            V_th=V_THRESHOLD,
            V_reset=V_RESET,
            tau=TAU_MEMBRANE,
            tau_ref=TAU_REFRACTORY,
            V_initializer=initializer,
        )
        self.exc = brainpy.state.LIFRef(config.n_exc, **neuron_args)
        self.inh = brainpy.state.LIFRef(config.n_inh, **neuron_args)

        self.ee_syn = brainpy.state.Expon(config.n_exc, tau=TAU_EXCITATORY)
        self.ie_syn = brainpy.state.Expon(config.n_exc, tau=TAU_INHIBITORY)
        self.ei_syn = brainpy.state.Expon(config.n_inh, tau=TAU_EXCITATORY)
        self.ii_syn = brainpy.state.Expon(config.n_inh, tau=TAU_INHIBITORY)

        self.ee_out = brainpy.state.COBA(E=E_EXCITATORY)
        self.ie_out = brainpy.state.COBA(E=E_INHIBITORY)
        self.ei_out = brainpy.state.COBA(E=E_EXCITATORY)
        self.ii_out = brainpy.state.COBA(E=E_INHIBITORY)
        self.exc.add_current_input("recurrent_excitation", self.ee_out)
        self.exc.add_current_input("recurrent_inhibition", self.ie_out)
        self.inh.add_current_input("recurrent_excitation", self.ei_out)
        self.inh.add_current_input("recurrent_inhibition", self.ii_out)

        self.n_exc = config.n_exc
        self.n_inh = config.n_inh
        self.ee_prob = config.exc_fanout / config.n_exc
        self.ei_prob = max(1, config.exc_fanout // 4) / config.n_inh
        self.ie_prob = config.inh_fanout / config.n_exc
        self.ii_prob = max(1, config.inh_fanout // 4) / config.n_inh

    def update(self, gain, realization_seed, spark):
        """Advance one lane by one time step and return its spike count."""
        exc_spikes = self.exc.get_spike() != 0.0
        inh_spikes = self.inh.get_spike() != 0.0
        exc_events = exc_spikes.at[0].set(exc_spikes[0] | spark)

        ee = brainevent.JITCScalarR(
            (W_EE * gain, self.ee_prob, realization_seed),
            shape=(self.n_exc, self.n_exc),
        )
        ei = brainevent.JITCScalarR(
            (W_EI * gain, self.ei_prob, realization_seed + 1),
            shape=(self.n_exc, self.n_inh),
        )
        ie = brainevent.JITCScalarR(
            (W_IE, self.ie_prob, realization_seed + 2),
            shape=(self.n_inh, self.n_exc),
        )
        ii = brainevent.JITCScalarR(
            (W_II, self.ii_prob, realization_seed + 3),
            shape=(self.n_inh, self.n_inh),
        )

        self.ee_out.bind_cond(
            self.ee_syn(brainevent.BinaryArray(exc_events) @ ee)
        )
        self.ei_out.bind_cond(
            self.ei_syn(brainevent.BinaryArray(exc_events) @ ei)
        )
        self.ie_out.bind_cond(
            self.ie_syn(brainevent.BinaryArray(inh_spikes) @ ie)
        )
        self.ii_out.bind_cond(
            self.ii_syn(brainevent.BinaryArray(inh_spikes) @ ii)
        )

        new_exc = self.exc(0.0 * u.mA) != 0.0
        new_inh = self.inh(0.0 * u.mA) != 0.0
        return jnp.sum(new_exc) + jnp.sum(new_inh) + spark.astype(jnp.int32)


def sample_realizations(config: Experiment) -> dict[str, np.ndarray]:
    """Generate paired connectivity seeds and initial voltages."""
    arrays: dict[str, list[np.ndarray]] = {
        "seed": [],
        "v_exc_mv": [],
        "v_inh_mv": [],
    }
    for seed in range(config.first_seed, config.first_seed + config.n_realizations):
        rng = np.random.default_rng(seed)
        arrays["seed"].append(np.asarray(seed, dtype=np.int32))
        arrays["v_exc_mv"].append(
            np.clip(rng.normal(-61.5, 2.0, config.n_exc), -65.0, -55.0).astype(np.float32)
        )
        arrays["v_inh_mv"].append(
            np.clip(rng.normal(-61.5, 2.0, config.n_inh), -65.0, -55.0).astype(np.float32)
        )
    return {name: np.stack(values) for name, values in arrays.items()}


def expand_lanes(
    config: Experiment,
    realizations: dict[str, np.ndarray],
    gains: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Cross gains with realizations while retaining common random networks."""
    lane_gains = np.repeat(gains.astype(np.float32), config.n_realizations)
    lanes = {
        name: np.tile(values, (len(gains),) + (1,) * (values.ndim - 1))
        for name, values in realizations.items()
    }
    return lane_gains, lanes


def simulate(
    config: Experiment,
    gains: np.ndarray,
    realizations: dict[str, np.ndarray],
    with_spark: bool,
) -> np.ndarray:
    """Run all gain-realization lanes in one vmapped transition and time loop."""
    lane_gains, lanes = expand_lanes(config, realizations, gains)
    lane_count = len(lane_gains)
    net = RecurrentEINetwork(config)

    with brainstate.environ.context(dt=DT):
        brainstate.nn.vmap_init_all_states(net, axis_size=lane_count)
        net.exc.V.value = jnp.asarray(lanes["v_exc_mv"]) * u.mV
        net.inh.V.value = jnp.asarray(lanes["v_inh_mv"]) * u.mV

        dynamical_state = state_filter.Any(
            state_filter.OfType(brainstate.HiddenState),
            state_filter.OfType(brainstate.ShortTermState),
        )
        mapped_step = brainstate.transform.vmap2(
            net.update,
            in_axes=(0, 0, None),
            out_axes=0,
            state_in_axes={0: dynamical_state},
            state_out_axes={0: dynamical_state},
            unexpected_out_state_mapping="raise",
        )
        times = u.math.arange(0.0 * u.ms, DURATION, DT)
        spark_schedule = jnp.zeros(times.shape[0], dtype=bool)
        if with_spark:
            spark_schedule = spark_schedule.at[0].set(True)

        @brainstate.transform.jit
        def run():
            def step(t, spark):
                with brainstate.environ.context(t=t):
                    return mapped_step(
                        jnp.asarray(lane_gains),
                        jnp.asarray(lanes["seed"]),
                        spark,
                    )

            return brainstate.transform.for_loop(step, times, spark_schedule)

        spike_counts = jax.block_until_ready(run())
    return np.asarray(spike_counts)


def avalanche_observables(
    spike_counts: np.ndarray,
    config: Experiment,
) -> dict[str, np.ndarray]:
    """Measure a seeded avalanche and flag late self-sustained activity."""
    dt_ms = float(DT.to_decimal(u.ms))
    steps_per_bin = int(round(config.bin_ms / dt_ms))
    usable_steps = spike_counts.shape[0] // steps_per_bin * steps_per_bin
    binned = spike_counts[:usable_steps].reshape(
        -1, steps_per_bin, spike_counts.shape[1]
    ).sum(axis=1)
    quiet_bins = max(1, int(round(config.quiet_ms / config.bin_ms)))
    active = binned > 0

    sizes = np.zeros(spike_counts.shape[1], dtype=np.float64)
    durations = np.zeros_like(sizes)
    never_quiet = np.zeros(spike_counts.shape[1], dtype=bool)
    for lane in range(spike_counts.shape[1]):
        quiet = ~active[:, lane]
        run = np.convolve(quiet.astype(np.int16), np.ones(quiet_bins, dtype=np.int16), mode="valid")
        ends = np.flatnonzero(run == quiet_bins)
        if len(ends):
            end_bin = int(ends[0])
        else:
            end_bin = binned.shape[0]
            never_quiet[lane] = True
        sizes[lane] = binned[:end_bin, lane].sum()
        durations[lane] = end_bin * config.bin_ms

    late_steps = max(1, int(round(config.late_window_ms / dt_ms)))
    late_seconds = late_steps * dt_ms / 1000.0
    late_rate_hz = (
        spike_counts[-late_steps:].sum(axis=0)
        / config.n_neurons
        / late_seconds
    )
    unstable = never_quiet | (late_rate_hz >= config.runaway_rate_hz)
    return {
        "size": sizes,
        "duration_ms": durations,
        "late_rate_hz": late_rate_hz,
        "unstable": unstable,
    }


def summarize_scan(
    observables: dict[str, np.ndarray],
    gains: np.ndarray,
    config: Experiment,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Score stable gains by across-realization avalanche-size variability."""
    sizes = observables["size"].reshape(len(gains), config.n_realizations)
    durations = observables["duration_ms"].reshape(len(gains), config.n_realizations)
    late_rates = observables["late_rate_hz"].reshape(len(gains), config.n_realizations)
    unstable = observables["unstable"].reshape(len(gains), config.n_realizations)

    rows: list[dict[str, float]] = []
    for index, gain in enumerate(gains):
        stable_sizes = sizes[index, ~unstable[index]]
        mean_size = float(np.mean(stable_sizes)) if len(stable_sizes) else float("nan")
        variability = (
            float(np.std(stable_sizes, ddof=1) / mean_size)
            if len(stable_sizes) > 1 and mean_size > 0
            else 0.0
        )
        rows.append(
            {
                "gain": float(gain),
                "median_avalanche_size": float(np.median(sizes[index])),
                "q25_avalanche_size": float(np.quantile(sizes[index], 0.25)),
                "q75_avalanche_size": float(np.quantile(sizes[index], 0.75)),
                "median_duration_ms": float(np.median(durations[index])),
                "size_cv_stable": variability,
                "unstable_fraction": float(np.mean(unstable[index])),
                "median_late_rate_hz": float(np.median(late_rates[index])),
            }
        )

    stable_indices = [
        index
        for index, row in enumerate(rows)
        if row["unstable_fraction"] <= config.max_unstable_fraction
    ]
    if not stable_indices:
        raise RuntimeError("No gain met the pre-registered stability criterion.")
    best_index = max(stable_indices, key=lambda index: rows[index]["size_cv_stable"])
    best_score = rows[best_index]["size_cv_stable"]
    if not np.isfinite(best_score) or best_score <= 0.0:
        raise RuntimeError(
            "The stable scan contains no avalanche-size variability; expand or refine the gain range."
        )
    eligible = {
        index
        for index in stable_indices
        if rows[index]["size_cv_stable"] >= config.region_score_fraction * best_score
    }
    region_indices = [best_index]
    lower = best_index - 1
    while lower in eligible:
        region_indices.insert(0, lower)
        lower -= 1
    upper = best_index + 1
    while upper in eligible:
        region_indices.append(upper)
        upper += 1
    summary = {
        "selected_gain": rows[best_index]["gain"],
        "selected_size_cv": best_score,
        "critical_region": [
            rows[min(region_indices)]["gain"],
            rows[max(region_indices)]["gain"],
        ],
        "stability_rule": (
            f"unstable_fraction <= {config.max_unstable_fraction:g}; a realization is "
            f"unstable if it never becomes quiet for {config.quiet_ms:g} ms or its "
            f"last {config.late_window_ms:g} ms rate is >= {config.runaway_rate_hz:g} Hz/neuron"
        ),
        "selection_rule": "maximum avalanche-size CV among stable gains",
        "region_rule": f"stable gains with CV >= {config.region_score_fraction:g} of the maximum",
    }
    return rows, summary


def save_outputs(
    rows: list[dict[str, float]],
    summary: dict[str, object],
    config: Experiment,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "criticality_scan.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    gains = np.asarray([row["gain"] for row in rows])
    medians = np.asarray([row["median_avalanche_size"] for row in rows])
    lower = np.asarray([row["q25_avalanche_size"] for row in rows])
    upper = np.asarray([row["q75_avalanche_size"] for row in rows])
    cv = np.asarray([row["size_cv_stable"] for row in rows])
    unstable = np.asarray([row["unstable_fraction"] for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True, constrained_layout=True)
    axes[0].fill_between(gains, lower, upper, color="#9fc7b5", alpha=0.5, linewidth=0)
    axes[0].plot(gains, medians, color="#175c4c", marker="o", label="Median (IQR)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Avalanche size [spikes]")
    axes[0].legend(frameon=False)
    axes[1].plot(gains, cv, color="#2457a7", marker="o")
    axes[1].set_ylabel("Stable size CV")
    axes[2].plot(gains, unstable, color="#b34332", marker="o")
    axes[2].axhline(
        config.max_unstable_fraction,
        color="#666666",
        linestyle="--",
        linewidth=1,
    )
    axes[2].set_ylabel("Unstable fraction")
    axes[2].set_xlabel("Excitatory coupling gain")
    for axis in axes:
        axis.axvspan(*summary["critical_region"], color="#e0ab35", alpha=0.18)
        axis.grid(alpha=0.2)
    fig.suptitle("Edge of criticality in sparse recurrent E/I networks")
    fig.savefig(output_dir / "criticality_scan.png", dpi=180)
    plt.close(fig)


def run_experiment(config: Experiment, output_dir: Path) -> dict[str, object]:
    gains = np.asarray(config.gains, dtype=np.float32)
    realizations = sample_realizations(config)
    spike_counts = simulate(config, gains, realizations, with_spark=True)
    observables = avalanche_observables(spike_counts, config)
    rows, summary = summarize_scan(observables, gains, config)

    no_spark = simulate(config, gains[-1:], realizations, with_spark=False)
    summary["no_spark_spike_count_at_max_gain"] = int(no_spark.sum())
    summary["n_realizations"] = config.n_realizations
    summary["n_neurons"] = config.n_neurons
    summary["dt_ms"] = float(DT.to_decimal(u.ms))
    summary["duration_ms"] = float(DURATION.to_decimal(u.ms))
    summary["realization_seeds"] = [
        config.first_seed,
        config.first_seed + config.n_realizations - 1,
    ]
    save_outputs(rows, summary, config, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small compilation/smoke-test configuration.",
    )
    args = parser.parse_args()
    config = Experiment()
    if args.quick:
        config = replace(
            config,
            n_exc=32,
            n_inh=8,
            exc_fanout=8,
            inh_fanout=8,
            n_realizations=4,
            gains=tuple(np.arange(2.0, 2.81, 0.1)),
        )
    summary = run_experiment(config, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
