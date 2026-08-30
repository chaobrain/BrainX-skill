"""Run and analyze four current-based Brunel LIF network conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
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
import scipy.signal


DT = 0.1 * u.ms
TAU_M = 20.0 * u.ms
V_THRESHOLD = 20.0 * u.mV
V_RESET = 10.0 * u.mV
TAU_REF = 2.0 * u.ms
JUMP = 0.1 * u.mV
DELAY = 1.5 * u.ms
NU_THRESHOLD = V_THRESHOLD / (JUMP * 1_000 * TAU_M)

CONDITIONS = (
    ("synchronous_regular", 3.0, 2.0),
    ("fast_synchronous_irregular", 6.0, 4.0),
    ("asynchronous_irregular", 5.0, 2.0),
    ("slow_synchronous_irregular", 4.5, 0.9),
)
REPEAT_SEEDS = (1729, 2718, 3141, 5772, 8119)
PROBE_SEED = 8_675_309

METRIC_ROW_KEYS = {
    "repeat",
    "repeat_seed",
    "expected_state",
    "g",
    "eta",
    "runtime_seconds",
    "exc_firing_rate_hz",
    "inh_firing_rate_hz",
    "overall_firing_rate_hz",
    "exc_isi_cv_mean",
    "exc_isi_cv_median",
    "inh_isi_cv_mean",
    "inh_isi_cv_median",
    "isi_cv_all_mean",
    "exc_isi_cv_eligible_neurons",
    "inh_isi_cv_eligible_neurons",
    "population_rate_cv_1ms",
    "dominant_frequency_hz",
    "spectral_peak_to_background",
    "classified_state",
}
RUNTIME_DISTRIBUTIONS = {
    "BrainX": "BrainX",
    "brainevent": "brainevent",
    "brainpy": "brainpy",
    "brainpy_state": "brainpy-state",
    "brainstate": "brainstate",
    "braintools": "braintools",
    "brainunit": "brainunit",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "numpy": "numpy",
    "saiunit": "saiunit",
    "scipy": "scipy",
}


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

    @property
    def external_lambda_scale(self) -> float:
        return self.external_indegree / self.exc_indegree


class BrunelLIF(brainpy.state.Neuron):
    """LIF population with exact leak and instantaneous voltage jumps."""

    def __init__(self, in_size: int):
        super().__init__(in_size)

    def init_state(self, *args, **kwargs):
        del args, kwargs
        self.V = brainstate.HiddenState(jnp.full(self.varshape, 10.0) * u.mV)
        self.last_spike = brainstate.ShortTermState(
            jnp.full(self.varshape, -1.0e7) * u.ms
        )
        self.spike = brainstate.ShortTermState(jnp.zeros(self.varshape, dtype=bool))

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
            jax.ShapeDtypeStruct(
                (config.num_neurons,), brainstate.environ.ditype()
            ),
            DELAY,
        )
        self.external_rng = brainstate.random.clone_rng(external_seed)
        self.delay_steps = int(round(DELAY.to_decimal(u.ms) / DT.to_decimal(u.ms)))

    def update(self, t, step_index, g, eta):
        with brainstate.environ.context(t=t, i=step_index):
            external = self.external_rng.poisson(
                lam=eta * self.config.external_lambda_scale,
                size=(self.config.num_neurons,),
            ).astype(brainstate.environ.ditype())
            self.spike_delay.update(self.neurons.get_spike())
            self.external_delay.update(external)
            delay_step = jnp.asarray(self.delay_steps, dtype=jnp.int32)
            delayed_spikes = self.spike_delay.retrieve_at_step(delay_step)
            delayed_external = self.external_delay.retrieve_at_step(delay_step)
            exc_count = (
                brainevent.BinaryArray(delayed_spikes[: self.config.n_exc])
                @ self.exc_conn
            )
            inh_count = (
                brainevent.BinaryArray(delayed_spikes[self.config.n_exc :])
                @ self.inh_conn
            )
            delta_v = (exc_count - g * inh_count + delayed_external) * JUMP
            return self.neurons(delta_v)


def sample_fixed_indegree(
    n_pre: int,
    n_post: int,
    indegree: int,
    seed: int,
    target_offset: int,
) -> np.ndarray:
    """Sample unique sources per target and exclude within-population autapses."""
    if indegree >= n_pre:
        raise ValueError("indegree must be smaller than n_pre")
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
    unit_weight = jnp.asarray(1.0, dtype=brainstate.environ.dftype())
    exc_conn = brainevent.FixedNumPerPost(
        unit_weight,
        jnp.asarray(exc_indices),
        shape=(config.n_exc, config.num_neurons),
    )
    inh_conn = brainevent.FixedNumPerPost(
        unit_weight,
        jnp.asarray(inh_indices),
        shape=(config.n_inh, config.num_neurons),
    )
    return exc_conn, inh_conn, exc_indices, inh_indices


def fixed_probe_indices(config: NetworkConfig) -> np.ndarray:
    rng = np.random.default_rng(PROBE_SEED)
    n_exc = min(40, config.n_exc)
    n_inh = min(10, config.n_inh)
    exc = np.sort(rng.choice(config.n_exc, n_exc, replace=False))
    inh = np.sort(rng.choice(config.n_inh, n_inh, replace=False)) + config.n_exc
    return np.concatenate((exc, inh)).astype(np.int32)


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
            if mean_interval > 0.0:
                values.append(float(intervals.std(ddof=0) / mean_interval))
    array = np.asarray(values, dtype=np.float32)
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
    if (
        cv >= 0.7
        and synchronous
        and 100.0 <= frequency <= 300.0
        and neuron_rate < frequency
    ):
        return "fast_synchronous_irregular"
    if cv >= 0.7 and not synchronous:
        return "asynchronous_irregular"
    if (
        cv >= 0.7
        and synchronous
        and 10.0 <= frequency <= 60.0
        and neuron_rate < frequency
    ):
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
    eligible = np.concatenate((exc_cv["values"], inh_cv["values"]))
    all_cv_mean = float(eligible.mean()) if eligible.size else float("nan")

    exc_counts = exc_spikes.sum(axis=1).astype(np.int32)
    inh_counts = inh_spikes.sum(axis=1).astype(np.int32)
    population_counts = exc_counts + inh_counts
    population_rate = population_counts / config.num_neurons / dt_s

    bin_steps = int(round(1.0 / DT.to_decimal(u.ms)))
    n_bins = population_counts.size // bin_steps
    counts_1ms = population_counts[: n_bins * bin_steps]
    counts_1ms = counts_1ms.reshape(n_bins, bin_steps).sum(axis=1)
    rate_1ms = counts_1ms / config.num_neurons / (bin_steps * dt_s)
    rate_cv = float(rate_1ms.std(ddof=0) / rate_1ms.mean())

    frequencies, power = scipy.signal.welch(
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
    prominence = float(power[peak_index] / background) if background > 0.0 else float("inf")

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
        "exc_isi_cv_values": exc_cv["values"],
        "inh_isi_cv_values": inh_cv["values"],
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
    aggregate = {
        key: float(np.median([row[key] for row in rows])) for key in numeric_keys
    }
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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "probe": {"seed": PROBE_SEED, "n_exc": 40, "n_inh": 10},
        "conditions": [
            {"expected_state": name, "g": g, "eta": eta}
            for name, g, eta in CONDITIONS
        ],
        "spectrum": {
            "sampling_hz": 10_000.0,
            "window": "hann",
            "nperseg": 8192,
            "noverlap": 4096,
            "search_band_hz": [1.0, 1_000.0],
        },
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
    if payload != config_payload(config, seeds):
        raise ValueError("run config does not exactly match the locked contract")
    if config != NetworkConfig() or seeds != REPEAT_SEEDS:
        raise ValueError("production requires the approved network and five seeds")
    return config, seeds


def _write_json(path: Path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def _manifest_entries(source_results: Path) -> dict[str, dict]:
    manifest_path = source_results / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("resume artifact manifest has no file list")
    by_path = {}
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in by_path:
            raise ValueError("resume artifact manifest has an invalid path")
        by_path[relative] = entry
    return by_path


def _verify_manifest_file(path: Path, relative: str, entries: dict[str, dict]):
    entry = entries.get(relative)
    if entry is None:
        raise ValueError(f"resume artifact is absent from its manifest: {relative}")
    if path.stat().st_size != int(entry["bytes"]) or _hash_file(path) != entry["sha256"]:
        raise ValueError(f"resume artifact does not match its manifest: {relative}")


def load_resume_rows(
    resume_from: Path,
    raw_dir: Path,
    config: NetworkConfig,
    seeds,
) -> tuple[list[dict], set[tuple[int, str]]]:
    source_results = resume_from / "results"
    manifest_entries = _manifest_entries(source_results)
    metrics_path = source_results / "metrics.partial.json"
    _verify_manifest_file(metrics_path, "metrics.partial.json", manifest_entries)
    rows = json.loads(metrics_path.read_text(encoding="ascii"))
    if not isinstance(rows, list):
        raise ValueError("resume metrics must be a list")
    condition_contract = {name: (g, eta) for name, g, eta in CONDITIONS}
    expected_probe = fixed_probe_indices(config)
    expected_steps = int(round(config.analysis_ms / DT.to_decimal(u.ms)))
    spectrum_bins = 8192 // 2 + 1
    completed = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != METRIC_ROW_KEYS:
            raise ValueError("resume row does not match the locked metric schema")
        repeat = int(row["repeat"])
        name = row["expected_state"]
        if repeat < 0 or repeat >= len(seeds):
            raise ValueError("resume row has an out-of-range repeat")
        if int(row["repeat_seed"]) != seeds[repeat] or name not in condition_contract:
            raise ValueError("resume row does not match the locked seed/condition contract")
        if (float(row["g"]), float(row["eta"])) != condition_contract[name]:
            raise ValueError("resume row has incorrect condition parameters")
        if row["classified_state"] != classify_metrics(row):
            raise ValueError("resume row classification does not match its metrics")
        key = (repeat, name)
        if key in completed:
            raise ValueError("resume data contains a duplicate condition")
        if any(
            not np.isfinite(value)
            for value in row.values()
            if isinstance(value, (int, float))
        ):
            raise ValueError("resume row contains a non-finite metric")
        source_raw = source_results / "raw" / f"repeat-{repeat:02d}_{name}.npz"
        if not source_raw.is_file():
            raise ValueError(f"resume raw artifact is missing: {source_raw}")
        relative_raw = str(source_raw.relative_to(source_results))
        _verify_manifest_file(source_raw, relative_raw, manifest_entries)
        with np.load(source_raw, allow_pickle=False) as artifact:
            expected_arrays = {
                "probe_indices",
                "raster",
                "exc_counts",
                "inh_counts",
                "exc_isi_cv_values",
                "inh_isi_cv_values",
                "frequencies_hz",
                "power_hz",
            }
            if set(artifact.files) != expected_arrays:
                raise ValueError(f"resume raw artifact is incomplete: {source_raw}")
            expected_shapes = {
                "probe_indices": (50,),
                "raster": (expected_steps, 50),
                "exc_counts": (expected_steps,),
                "inh_counts": (expected_steps,),
                "exc_isi_cv_values": (int(row["exc_isi_cv_eligible_neurons"]),),
                "inh_isi_cv_values": (int(row["inh_isi_cv_eligible_neurons"]),),
                "frequencies_hz": (spectrum_bins,),
                "power_hz": (spectrum_bins,),
            }
            if any(artifact[key].shape != shape for key, shape in expected_shapes.items()):
                raise ValueError(f"resume raw artifact has an invalid array shape: {source_raw}")
            if not np.array_equal(artifact["probe_indices"], expected_probe):
                raise ValueError(f"resume raw artifact has the wrong probe identity: {source_raw}")
        shutil.copy2(source_raw, raw_dir / source_raw.name)
        completed.add(key)
    return rows, completed


def runtime_provenance(resume_from: Path | None, wall_seconds: float) -> dict:
    packages = {}
    for label, distribution in RUNTIME_DISTRIBUTIONS.items():
        try:
            packages[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[label] = "unavailable"
    provenance = {
        "sys_executable": sys.executable,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "git": _git_identity(Path(__file__).resolve().parent),
        "resume_from": str(resume_from.resolve()) if resume_from else None,
        "wall_seconds": wall_seconds,
    }
    if resume_from:
        manifest = resume_from / "results" / "artifact-manifest.json"
        provenance["resume_manifest_sha256"] = _hash_file(manifest)
    return provenance


def _write_assessment(output_dir: Path, aggregate: dict):
    lines = [
        "# Result assessment",
        "",
        "The labels below apply the locked prospective predicates without tuning.",
        "",
        "| Requested condition | Aggregate label | Matching repeats | Dominant frequency (median Hz) | Robust |",
        "|---|---|---:|---:|---|",
    ]
    for name, _, _ in CONDITIONS:
        result = aggregate[name]
        lines.append(
            f"| `{name}` | `{result['classified_state']}` | "
            f"{result['matching_repeats']}/{result['total_repeats']} | "
            f"{result['dominant_frequency_hz']:.3f} | {result['robust']} |"
        )
    lines.extend(
        (
            "",
            "## Claim-evidence matrix",
            "",
            "| Claim | Evidence | Boundary |",
            "|---|---|---|",
            "| Each requested state is reproduced robustly or not | `metrics.json` and `robustness.json` | Requires four of five repeat labels and the aggregate label; no category is forced. |",
            "| Dominant frequencies are measured from the global rate | Per-run `frequencies_hz` and `power_hz` arrays in `raw/*.npz` | Welch estimator and search band were frozen before production. |",
            "| E/I rates and irregularity support the labels | `metrics.csv`, `metrics.json`, and per-run ISI-CV arrays | CV excludes neurons with fewer than four analyzed spikes. |",
            "",
            "No continuous phase boundary, exact Brunel RNG parity, or biological realism is claimed.",
            "",
        )
    )
    (output_dir / "result-assessment.md").write_text("\n".join(lines), encoding="ascii")


def _write_manifest(output_dir: Path):
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _hash_file(path),
                }
            )
    _write_json(output_dir / "artifact-manifest.json", {"files": files})


def run_experiment(
    output_dir: Path,
    config: NetworkConfig,
    seeds=REPEAT_SEEDS,
    resume_from: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    probe_indices = fixed_probe_indices(config)
    if resume_from is None:
        metric_rows = []
        completed = set()
    else:
        metric_rows, completed = load_resume_rows(
            resume_from, raw_dir, config, seeds
        )
        _write_json(output_dir / "metrics.partial.json", metric_rows)
    graph_rows = []
    wall_start = time.perf_counter()

    for repeat_index, repeat_seed in enumerate(seeds):
        exc_conn, inh_conn, exc_indices, inh_indices = make_connectivity(
            config, repeat_seed
        )
        graph_rows.append(
            {
                "repeat": repeat_index,
                "seed": repeat_seed,
                "exc_indices_sha256": _hash_array(exc_indices),
                "inh_indices_sha256": _hash_array(inh_indices),
            }
        )
        missing_conditions = [
            condition
            for condition in CONDITIONS
            if (repeat_index, condition[0]) not in completed
        ]
        if not missing_conditions:
            continue
        net = SparseEINetwork(
            config,
            external_seed=repeat_seed + 404,
            exc_conn=exc_conn,
            inh_conn=inh_conn,
        )
        brainstate.nn.init_all_states(net)
        runner = build_runner(net)
        for expected_state, g, eta in missing_conditions:
            reset_run(net, repeat_seed + 303, repeat_seed + 404)
            start = time.perf_counter()
            spikes = np.asarray(
                jax.block_until_ready(runner(jnp.asarray(g), jnp.asarray(eta))),
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
            np.savez_compressed(
                raw_dir / f"repeat-{repeat_index:02d}_{expected_state}.npz",
                probe_indices=probe_indices,
                raster=spikes[config.burn_steps :, probe_indices],
                **traces,
            )
            _write_json(output_dir / "metrics.partial.json", metric_rows)
            print(json.dumps(row, sort_keys=True), flush=True)
            del spikes

    aggregate = {}
    condition_order = {name: index for index, (name, _, _) in enumerate(CONDITIONS)}
    metric_rows.sort(key=lambda row: (row["repeat"], condition_order[row["expected_state"]]))
    for expected_state, _, _ in CONDITIONS:
        rows = [row for row in metric_rows if row["expected_state"] == expected_state]
        aggregate[expected_state] = aggregate_condition(rows, expected_state)

    _write_json(output_dir / "config.json", config_payload(config, seeds))
    _write_json(output_dir / "metrics.json", metric_rows)
    _write_json(output_dir / "robustness.json", aggregate)
    _write_json(output_dir / "graph-hashes.json", graph_rows)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    _write_json(
        output_dir / "provenance.json",
        runtime_provenance(resume_from, time.perf_counter() - wall_start),
    )
    _write_assessment(output_dir, aggregate)
    _write_manifest(output_dir)
    return metric_rows, aggregate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config and args.smoke:
        raise ValueError("choose either --config production or --smoke")
    if args.resume_from and not args.config:
        raise ValueError("--resume-from requires the locked production --config")
    if args.config:
        config, seeds = load_run_contract(args.config)
    elif args.smoke:
        config = NetworkConfig(
            n_exc=80,
            n_inh=20,
            exc_indegree=8,
            inh_indegree=2,
            external_indegree=8,
            burn_ms=100.0,
            analysis_ms=2_000.0,
        )
        seeds = REPEAT_SEEDS[:1]
    else:
        raise ValueError("production requires --config; use --smoke for a reduced run")
    with brainstate.environ.context(dt=DT, precision=32):
        run_experiment(args.output_dir, config, seeds, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
