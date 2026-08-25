from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import brainevent
import brainpy
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from scipy import signal


DT = 0.1 * u.ms
TAU_M = 20.0 * u.ms
V_THRESHOLD = 20.0 * u.mV
V_RESET = 10.0 * u.mV
TAU_REF = 2.0 * u.ms
JUMP = 0.1 * u.mV
DELAY = 1.5 * u.ms
NU_THRESHOLD = 10.0 * u.Hz

CONDITIONS = (
    ("synchronous_regular", 3.0, 2.0),
    ("fast_synchronous_irregular", 6.0, 4.0),
    ("asynchronous_irregular", 5.0, 2.0),
    ("slow_synchronous_irregular", 4.5, 0.9),
)
REPEAT_SEEDS = (1729, 2718, 3141, 5772, 8119)


@dataclass(frozen=True)
class NetworkConfig:
    n_exc: int = 10_000
    n_inh: int = 2_500
    exc_indegree: int = 1_000
    inh_indegree: int = 250
    external_indegree: int = 1_000
    burn_ms: float = 500.0
    analysis_ms: float = 2_000.0

    @property
    def num_neurons(self) -> int:
        return self.n_exc + self.n_inh

    @property
    def num_steps(self) -> int:
        return int(round((self.burn_ms + self.analysis_ms) / DT.to_decimal(u.ms)))

    @property
    def burn_steps(self) -> int:
        return int(round(self.burn_ms / DT.to_decimal(u.ms)))


class BrunelLIF(brainpy.state.Neuron):
    """LIF population with exact instantaneous voltage jumps."""

    def __init__(self, in_size: int):
        super().__init__(in_size)

    def init_state(self, *args, **kwargs):
        del args, kwargs
        self.V = brainstate.HiddenState(jnp.full(self.varshape, 10.0) * u.mV)
        self.last_spike = brainstate.ShortTermState(
            jnp.full(self.varshape, -1.0e7) * u.ms
        )
        self.spike = brainstate.ShortTermState(
            jnp.zeros(self.varshape, dtype=bool)
        )

    def update(self, delta_v):
        t = brainstate.environ.get("t")
        dt = brainstate.environ.get_dt()
        leaked = self.V.value * u.math.exp(-dt / TAU_M)
        refractory = (t - self.last_spike.value) <= TAU_REF
        voltage = u.math.where(refractory, self.V.value, leaked + delta_v)
        spike = voltage >= V_THRESHOLD
        self.V.value = u.math.where(spike, V_RESET, voltage)
        self.last_spike.value = u.math.where(spike, t, self.last_spike.value)
        self.spike.value = spike
        return spike

    def get_spike(self):
        return self.spike.value

    def reset_state(self, *args, **kwargs):
        del args, kwargs
        self.V.value = jnp.full(self.varshape, 10.0) * u.mV
        self.last_spike.value = jnp.full(self.varshape, -1.0e7) * u.ms
        self.spike.value = jnp.zeros(self.varshape, dtype=bool)


class SparseEINetwork(brainstate.nn.Module):
    def __init__(self, config: NetworkConfig, external_seed: int, exc_conn, inh_conn):
        super().__init__()
        self.config = config
        self.exc_conn = exc_conn
        self.inh_conn = inh_conn
        self.neurons = BrunelLIF(config.num_neurons)
        self.spike_delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((config.num_neurons,), jnp.bool_), DELAY
        )
        self.external_delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((config.num_neurons,), brainstate.environ.ditype()),
            DELAY,
        )
        self.external_rng = brainstate.random.clone_rng(external_seed)
        self.delay_steps = int(round(DELAY.to_decimal(u.ms) / DT.to_decimal(u.ms)))

    def update(self, t, step_index, g, eta):
        with brainstate.environ.context(t=t, i=step_index):
            external = self.external_rng.poisson(
                lam=eta, size=(self.config.num_neurons,)
            ).astype(brainstate.environ.ditype())
            self.spike_delay.update(self.neurons.get_spike())
            self.external_delay.update(external)
            delayed_spikes = self.spike_delay.retrieve_at_step(
                jnp.asarray(self.delay_steps, dtype=jnp.int32)
            )
            delayed_external = self.external_delay.retrieve_at_step(
                jnp.asarray(self.delay_steps, dtype=jnp.int32)
            )
            exc_count = (
                brainevent.BinaryArray(delayed_spikes[: self.config.n_exc])
                @ self.exc_conn
            )
            inh_count = (
                brainevent.BinaryArray(delayed_spikes[self.config.n_exc :])
                @ self.inh_conn
            )
            delta_v = (exc_count - g * inh_count + delayed_external) * JUMP
            spikes = self.neurons(delta_v)
            return spikes


def sample_fixed_indegree(
    n_pre: int,
    n_post: int,
    indegree: int,
    seed: int,
    target_offset: int,
) -> np.ndarray:
    """Sample unique sources per target, excluding within-population autapses."""
    if indegree >= n_pre:
        raise ValueError("indegree must be smaller than n_pre when autapses are excluded")
    rng = np.random.default_rng(seed)
    indices = np.empty((n_post, indegree), dtype=np.int32)
    for target in range(n_post):
        local_self = target - target_offset
        if 0 <= local_self < n_pre:
            row = rng.choice(n_pre - 1, size=indegree, replace=False)
            row += row >= local_self
        else:
            row = rng.choice(n_pre, size=indegree, replace=False)
        indices[target] = row
    return indices


def make_connectivity(config: NetworkConfig, repeat_seed: int):
    exc_indices = sample_fixed_indegree(
        config.n_exc,
        config.num_neurons,
        config.exc_indegree,
        repeat_seed + 101,
        0,
    )
    inh_indices = sample_fixed_indegree(
        config.n_inh,
        config.num_neurons,
        config.inh_indegree,
        repeat_seed + 202,
        config.n_exc,
    )
    exc_conn = brainevent.FixedNumPerPost(
        jnp.asarray(1.0, dtype=brainstate.environ.dftype()),
        jnp.asarray(exc_indices),
        shape=(config.n_exc, config.num_neurons),
    )
    inh_conn = brainevent.FixedNumPerPost(
        jnp.asarray(1.0, dtype=brainstate.environ.dftype()),
        jnp.asarray(inh_indices),
        shape=(config.n_inh, config.num_neurons),
    )
    return exc_conn, inh_conn, exc_indices, inh_indices


def reset_run(net: SparseEINetwork, initial_seed: int, external_seed: int):
    brainstate.nn.reset_all_states(net)
    rng = np.random.default_rng(initial_seed)
    initial_v = rng.uniform(
        V_RESET.to_decimal(u.mV),
        V_THRESHOLD.to_decimal(u.mV),
        size=net.config.num_neurons,
    ).astype(np.float32)
    net.neurons.V.value = jnp.asarray(initial_v) * u.mV
    net.external_rng.seed(external_seed)


def build_runner(net: SparseEINetwork):
    times = u.math.arange(0.0 * u.ms, net.config.num_steps * DT, DT)
    step_indices = jnp.arange(net.config.num_steps, dtype=jnp.int32)

    @brainstate.transform.jit
    def run(g, eta):
        return brainstate.transform.for_loop(
            lambda t, i: net.update(t, i, g, eta),
            times,
            step_indices,
        )

    return run


def isi_cv_by_population(spikes: np.ndarray, start: int, stop: int) -> dict:
    values = []
    for neuron in range(start, stop):
        spike_steps = np.flatnonzero(spikes[:, neuron])
        if spike_steps.size >= 4:
            intervals = np.diff(spike_steps).astype(np.float64)
            mean_interval = intervals.mean()
            if mean_interval > 0:
                values.append(float(intervals.std(ddof=0) / mean_interval))
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else float("nan"),
        "median": float(np.median(array)) if array.size else float("nan"),
        "eligible_neurons": int(array.size),
        "values": array,
    }


def classify_metrics(metrics: dict) -> str:
    cv = metrics["isi_cv_all_mean"]
    synchronous = (
        metrics["population_rate_cv_1ms"] >= 0.2
        and metrics["spectral_peak_to_background"] >= 5.0
    )
    frequency = metrics["dominant_frequency_hz"]
    neuron_rate = metrics["overall_firing_rate_hz"]
    if cv < 0.5 and synchronous:
        return "synchronous_regular"
    if cv >= 0.7 and synchronous and 100.0 <= frequency <= 300.0:
        if neuron_rate < frequency:
            return "fast_synchronous_irregular"
    if cv >= 0.7 and not synchronous:
        return "asynchronous_irregular"
    if cv >= 0.7 and synchronous and 10.0 <= frequency <= 60.0:
        if neuron_rate < frequency:
            return "slow_synchronous_irregular"
    return "inconclusive"


def analyze_spikes(spikes: np.ndarray, config: NetworkConfig) -> tuple[dict, dict]:
    analyzed = np.asarray(spikes[config.burn_steps :], dtype=bool)
    duration_s = config.analysis_ms / 1_000.0
    dt_s = DT.to_decimal(u.second)
    exc_spikes = analyzed[:, : config.n_exc]
    inh_spikes = analyzed[:, config.n_exc :]
    exc_rate = exc_spikes.sum() / config.n_exc / duration_s
    inh_rate = inh_spikes.sum() / config.n_inh / duration_s
    overall_rate = analyzed.sum() / config.num_neurons / duration_s

    exc_cv = isi_cv_by_population(analyzed, 0, config.n_exc)
    inh_cv = isi_cv_by_population(analyzed, config.n_exc, config.num_neurons)
    eligible = np.concatenate((exc_cv.pop("values"), inh_cv.pop("values")))
    all_cv_mean = float(eligible.mean()) if eligible.size else float("nan")

    exc_counts = exc_spikes.sum(axis=1).astype(np.int32)
    inh_counts = inh_spikes.sum(axis=1).astype(np.int32)
    population_counts = exc_counts + inh_counts
    population_rate = population_counts / config.num_neurons / dt_s

    bin_steps = int(round(1.0 / DT.to_decimal(u.ms)))
    n_bins = population_counts.size // bin_steps
    counts_1ms = population_counts[: n_bins * bin_steps].reshape(n_bins, bin_steps).sum(1)
    rate_1ms = counts_1ms / config.num_neurons / (bin_steps * dt_s)
    rate_cv = float(rate_1ms.std(ddof=0) / rate_1ms.mean())

    frequencies, power = signal.welch(
        population_rate - population_rate.mean(),
        fs=1.0 / dt_s,
        window="hann",
        nperseg=min(8192, population_rate.size),
        noverlap=min(4096, population_rate.size // 2),
        detrend="constant",
        scaling="density",
    )
    search = (frequencies >= 1.0) & (frequencies <= 1_000.0)
    search_indices = np.flatnonzero(search)
    peak_index = int(search_indices[np.argmax(power[search])])
    dominant_frequency = float(frequencies[peak_index])
    background_mask = search & (np.abs(frequencies - dominant_frequency) > 5.0)
    background = float(np.median(power[background_mask]))
    prominence = float(power[peak_index] / background) if background > 0 else float("inf")

    metrics = {
        "exc_firing_rate_hz": float(exc_rate),
        "inh_firing_rate_hz": float(inh_rate),
        "overall_firing_rate_hz": float(overall_rate),
        "exc_isi_cv_mean": exc_cv["mean"],
        "exc_isi_cv_median": exc_cv["median"],
        "exc_isi_cv_eligible_neurons": exc_cv["eligible_neurons"],
        "inh_isi_cv_mean": inh_cv["mean"],
        "inh_isi_cv_median": inh_cv["median"],
        "inh_isi_cv_eligible_neurons": inh_cv["eligible_neurons"],
        "isi_cv_all_mean": all_cv_mean,
        "population_rate_cv_1ms": rate_cv,
        "dominant_frequency_hz": dominant_frequency,
        "spectral_peak_to_background": prominence,
    }
    metrics["classified_state"] = classify_metrics(metrics)
    traces = {
        "exc_counts": exc_counts,
        "inh_counts": inh_counts,
        "frequencies_hz": frequencies.astype(np.float32),
        "power_hz": power.astype(np.float32),
    }
    return metrics, traces


def aggregate_condition(rows: list[dict], expected_state: str) -> dict:
    numeric_keys = (
        "exc_firing_rate_hz",
        "inh_firing_rate_hz",
        "overall_firing_rate_hz",
        "exc_isi_cv_mean",
        "inh_isi_cv_mean",
        "isi_cv_all_mean",
        "population_rate_cv_1ms",
        "dominant_frequency_hz",
        "spectral_peak_to_background",
    )
    aggregate = {key: float(np.median([row[key] for row in rows])) for key in numeric_keys}
    aggregate["classified_state"] = classify_metrics(aggregate)
    matches = sum(row["classified_state"] == expected_state for row in rows)
    aggregate.update(
        expected_state=expected_state,
        matching_repeats=matches,
        total_repeats=len(rows),
        robust=(matches >= 4 and aggregate["classified_state"] == expected_state),
    )
    return aggregate


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _git_identity(root: Path) -> dict:
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short", "--", str(root)],
    }
    result = {}
    for key, command in commands.items():
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result[key] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    return result


def config_payload(config: NetworkConfig, seeds) -> dict:
    return {
        "network": asdict(config),
        "dt_ms": DT.to_decimal(u.ms),
        "tau_m_ms": TAU_M.to_decimal(u.ms),
        "threshold_mv": V_THRESHOLD.to_decimal(u.mV),
        "reset_mv": V_RESET.to_decimal(u.mV),
        "refractory_ms": TAU_REF.to_decimal(u.ms),
        "jump_mv": JUMP.to_decimal(u.mV),
        "delay_ms": DELAY.to_decimal(u.ms),
        "delay_steps": int(round(DELAY / DT)),
        "nu_threshold_hz": NU_THRESHOLD.to_decimal(u.Hz),
        "repeat_seeds": list(seeds),
        "conditions": [
            {"expected_state": name, "g": g, "eta": eta}
            for name, g, eta in CONDITIONS
        ],
        "classification": {
            "regular_cv_lt": 0.5,
            "irregular_cv_gte": 0.7,
            "synchronous_rate_cv_gte": 0.2,
            "synchronous_peak_background_gte": 5.0,
            "fast_band_hz": [100.0, 300.0],
            "slow_band_hz": [10.0, 60.0],
            "robust_repeats_required": 4,
        },
    }


def load_run_contract(path: Path) -> tuple[NetworkConfig, tuple[int, ...]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    config = NetworkConfig(**payload["network"])
    seeds = tuple(int(seed) for seed in payload["repeat_seeds"])
    expected = config_payload(config, seeds)
    if payload != expected:
        raise ValueError("run config does not exactly match the locked BrainX contract")
    if config != NetworkConfig() or seeds != REPEAT_SEEDS:
        raise ValueError("production config must use the approved network and five seeds")
    return config, seeds


def run_experiment(output_dir: Path, config: NetworkConfig, seeds=REPEAT_SEEDS):
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    probe_rng = np.random.default_rng(8675309)
    n_probe_exc = min(50, config.n_exc)
    n_probe_inh = min(50, config.n_inh)
    probe_exc = np.sort(probe_rng.choice(config.n_exc, n_probe_exc, replace=False))
    probe_inh = np.sort(
        probe_rng.choice(config.n_inh, n_probe_inh, replace=False) + config.n_exc
    )
    probe_indices = np.concatenate((probe_exc, probe_inh)).astype(np.int32)

    metric_rows = []
    graph_rows = []
    wall_start = time.perf_counter()

    for repeat_index, repeat_seed in enumerate(seeds):
        exc_conn, inh_conn, exc_indices, inh_indices = make_connectivity(config, repeat_seed)
        graph_rows.append(
            {
                "repeat": repeat_index,
                "seed": repeat_seed,
                "exc_indices_sha256": _hash_array(exc_indices),
                "inh_indices_sha256": _hash_array(inh_indices),
            }
        )
        net = SparseEINetwork(
            config,
            external_seed=repeat_seed + 404,
            exc_conn=exc_conn,
            inh_conn=inh_conn,
        )
        brainstate.nn.init_all_states(net)
        runner = build_runner(net)
        for expected_state, g, eta in CONDITIONS:
            initial_seed = repeat_seed + 303
            external_seed = repeat_seed + 404
            reset_run(net, initial_seed, external_seed)
            start = time.perf_counter()
            spikes = np.asarray(
                jax.block_until_ready(
                    runner(jnp.asarray(g), jnp.asarray(eta))
                ),
                dtype=bool,
            )
            run_seconds = time.perf_counter() - start
            metrics, traces = analyze_spikes(spikes, config)
            row = {
                "repeat": repeat_index,
                "repeat_seed": repeat_seed,
                "expected_state": expected_state,
                "g": g,
                "eta": eta,
                "runtime_seconds": run_seconds,
                **metrics,
            }
            metric_rows.append(row)
            artifact_name = f"repeat-{repeat_index:02d}_{expected_state}.npz"
            np.savez_compressed(
                raw_dir / artifact_name,
                probe_indices=probe_indices,
                raster=spikes[config.burn_steps :, probe_indices],
                **traces,
            )
            print(json.dumps(row, sort_keys=True), flush=True)

    aggregate = {}
    for expected_state, _, _ in CONDITIONS:
        rows = [row for row in metric_rows if row["expected_state"] == expected_state]
        aggregate[expected_state] = aggregate_condition(rows, expected_state)

    run_config = config_payload(config, seeds)
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metric_rows, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (output_dir / "robustness.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (output_dir / "graph-hashes.json").write_text(
        json.dumps(graph_rows, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    provenance = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "git": _git_identity(Path(__file__).resolve().parent),
        "wall_seconds": time.perf_counter() - wall_start,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return metric_rows, aggregate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config and args.smoke:
        raise ValueError("choose either --config production or --smoke")
    if args.config:
        config, seeds = load_run_contract(args.config)
    elif args.smoke:
        config = NetworkConfig(
            n_exc=80,
            n_inh=20,
            exc_indegree=8,
            inh_indegree=2,
            external_indegree=8,
            burn_ms=20.0,
            analysis_ms=80.0,
        )
        seeds = REPEAT_SEEDS[:1]
    else:
        raise ValueError("production requires --config; use --smoke for a reduced run")
    with brainstate.environ.context(dt=DT, precision=32):
        run_experiment(args.output_dir, config, seeds)


if __name__ == "__main__":
    main()
