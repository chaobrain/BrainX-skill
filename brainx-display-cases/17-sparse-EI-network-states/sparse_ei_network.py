"""Fresh BrainX reproduction of Brunel (2000), Figure 8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import brainevent
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


DT = 0.1 * u.ms
TAU_M = 20.0 * u.ms
TAU_REF = 2.0 * u.ms
V_THRESHOLD = 20.0 * u.mV
V_RESET = 10.0 * u.mV
JUMP = 0.1 * u.mV
DELAY = 1.5 * u.ms
NU_THRESHOLD = V_THRESHOLD / (JUMP * 1000 * TAU_M)

CONDITIONS = (
    ("A", "synchronous_regular", 3.0, 2.0, 100.0),
    ("B", "fast_synchronous_irregular", 6.0, 4.0, 200.0),
    ("C", "asynchronous_irregular", 5.0, 2.0, 200.0),
    ("D", "slow_synchronous_irregular", 4.5, 0.9, 200.0),
)

PAPER_TARGETS = {
    "fast_synchronous_irregular": {"rate_hz": 60.7, "frequency_hz": 180.0},
    "asynchronous_irregular": {"rate_hz": 37.7, "frequency_hz": None},
    "slow_synchronous_irregular": {"rate_hz": 5.5, "frequency_hz": 22.0},
}

DEPENDENCIES = (
    "brainpy",
    "brainevent",
    "brainstate",
    "brainunit",
    "jax",
    "jaxlib",
    "matplotlib",
    "numpy",
    "scipy",
)


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
        duration_ms = self.burn_ms + self.analysis_ms
        return int(round(duration_ms / DT.to_decimal(u.ms)))

    @property
    def burn_steps(self) -> int:
        return int(round(self.burn_ms / DT.to_decimal(u.ms)))

    @property
    def analysis_steps(self) -> int:
        return int(round(self.analysis_ms / DT.to_decimal(u.ms)))


class BrunelLIF(brainstate.nn.Module):
    """Model-A LIF neurons with delta-voltage input and absolute refractory time."""

    def __init__(self, num_neurons: int):
        super().__init__()
        self.num_neurons = num_neurons

    def init_state(self, *args, **kwargs):
        del args, kwargs
        self.V = brainstate.HiddenState(
            jnp.full((self.num_neurons,), V_RESET.to_decimal(u.mV)) * u.mV
        )
        self.last_spike = brainstate.ShortTermState(
            jnp.full((self.num_neurons,), -1.0e7) * u.ms
        )
        self.spike = brainstate.ShortTermState(
            jnp.zeros((self.num_neurons,), dtype=bool)
        )

    def reset_state(self, *args, **kwargs):
        del args, kwargs
        self.V.value = (
            jnp.full((self.num_neurons,), V_RESET.to_decimal(u.mV)) * u.mV
        )
        self.last_spike.value = jnp.full((self.num_neurons,), -1.0e7) * u.ms
        self.spike.value = jnp.zeros((self.num_neurons,), dtype=bool)

    def update(self, delta_v):
        t = brainstate.environ.get("t")
        dt = brainstate.environ.get_dt()
        leaked = self.V.value * u.math.exp(-dt / TAU_M)
        in_refractory = (t - self.last_spike.value) <= TAU_REF
        voltage = u.math.where(
            in_refractory,
            self.V.value,
            leaked + delta_v,
        )
        spike = voltage >= V_THRESHOLD
        self.V.value = u.math.where(spike, V_RESET, voltage)
        self.last_spike.value = u.math.where(spike, t, self.last_spike.value)
        self.spike.value = spike
        return spike

    def get_spike(self):
        return self.spike.value


class SparseEINetwork(brainstate.nn.Module):
    def __init__(self, config: NetworkConfig, exc_conn, inh_conn, external_seed: int):
        super().__init__()
        self.config = config
        self.exc_conn = exc_conn
        self.inh_conn = inh_conn
        self.neurons = BrunelLIF(config.num_neurons)
        self.delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((config.num_neurons,), jnp.bool_),
            DELAY,
        )
        self.external_rng = brainstate.random.clone_rng(external_seed)
        self.delay_steps = int(round(DELAY.to_decimal(u.ms) / DT.to_decimal(u.ms)))

    def update(self, t, step_index, g, eta):
        with brainstate.environ.context(t=t, i=step_index):
            delayed_spikes = self.delay.retrieve_at_step(
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
            external_count = self.external_rng.poisson(
                lam=eta,
                size=(self.config.num_neurons,),
            ).astype(brainstate.environ.ditype())
            delta_v = (exc_count - g * inh_count + external_count) * JUMP
            spike = self.neurons(delta_v)
            self.delay.update(spike)
            return spike


def sample_fixed_indegree(
    n_pre: int,
    n_post: int,
    indegree: int,
    seed: int,
    target_offset: int,
) -> np.ndarray:
    """Sample unique sources for every target and exclude population autapses."""
    if indegree >= n_pre:
        raise ValueError("indegree must be smaller than n_pre")
    rng = np.random.default_rng(seed)
    sources = np.empty((n_post, indegree), dtype=np.int32)
    for target in range(n_post):
        local_self = target - target_offset
        if 0 <= local_self < n_pre:
            row = rng.choice(n_pre - 1, size=indegree, replace=False)
            row += row >= local_self
        else:
            row = rng.choice(n_pre, size=indegree, replace=False)
        sources[target] = row
    return sources


def make_connectivity(config: NetworkConfig, graph_seed: int):
    exc_indices = sample_fixed_indegree(
        config.n_exc,
        config.num_neurons,
        config.exc_indegree,
        graph_seed + 101,
        0,
    )
    inh_indices = sample_fixed_indegree(
        config.n_inh,
        config.num_neurons,
        config.inh_indegree,
        graph_seed + 202,
        config.n_exc,
    )
    weight = jnp.asarray(1.0, dtype=brainstate.environ.dftype())
    exc_conn = brainevent.FixedNumPerPost(
        weight,
        jnp.asarray(exc_indices),
        shape=(config.n_exc, config.num_neurons),
    )
    inh_conn = brainevent.FixedNumPerPost(
        weight,
        jnp.asarray(inh_indices),
        shape=(config.n_inh, config.num_neurons),
    )
    return exc_conn, inh_conn, exc_indices, inh_indices


def choose_probe_indices(config: NetworkConfig, probe_seed: int) -> np.ndarray:
    rng = np.random.default_rng(probe_seed)
    n_exc = min(40, config.n_exc)
    n_inh = min(10, config.n_inh)
    exc = np.sort(rng.choice(config.n_exc, n_exc, replace=False))
    inh = np.sort(rng.choice(config.n_inh, n_inh, replace=False)) + config.n_exc
    return np.concatenate((exc, inh)).astype(np.int32)


def reset_run(net: SparseEINetwork, initial_seed: int, external_seed: int):
    brainstate.nn.reset_all_states(net)
    rng = np.random.default_rng(initial_seed)
    initial_mv = rng.uniform(
        V_RESET.to_decimal(u.mV),
        V_THRESHOLD.to_decimal(u.mV),
        size=net.config.num_neurons,
    ).astype(np.float32)
    net.neurons.V.value = jnp.asarray(initial_mv) * u.mV
    net.external_rng.seed(external_seed)


def build_runner(net: SparseEINetwork):
    times = u.math.arange(0.0 * u.ms, net.config.num_steps * DT, DT)
    indices = jnp.arange(net.config.num_steps, dtype=jnp.int32)

    @brainstate.transform.jit
    def run(g, eta):
        return brainstate.transform.for_loop(
            lambda t, i: net.update(t, i, g, eta),
            times,
            indices,
        )

    return run


def isi_cv_values(spikes: np.ndarray, start: int, stop: int) -> np.ndarray:
    values = []
    for neuron in range(start, stop):
        steps = np.flatnonzero(spikes[:, neuron])
        if steps.size >= 4:
            intervals = np.diff(steps).astype(np.float64)
            mean_interval = intervals.mean()
            if mean_interval > 0:
                values.append(intervals.std(ddof=0) / mean_interval)
    return np.asarray(values, dtype=np.float32)


def summarize_cv(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "eligible_neurons": int(values.size),
    }


def analyze_spikes(spikes: np.ndarray, config: NetworkConfig) -> tuple[dict, dict]:
    analysis = spikes[config.burn_steps :]
    duration_s = config.analysis_ms / 1000.0
    exc_counts = analysis[:, : config.n_exc].sum(axis=1, dtype=np.int32)
    inh_counts = analysis[:, config.n_exc :].sum(axis=1, dtype=np.int32)
    global_counts = exc_counts + inh_counts
    exc_cv = isi_cv_values(analysis, 0, config.n_exc)
    inh_cv = isi_cv_values(analysis, config.n_exc, config.num_neurons)
    all_cv = np.concatenate((exc_cv, inh_cv))

    global_rate_hz = global_counts / (
        config.num_neurons * DT.to_decimal(u.second)
    )
    bin_steps = int(round(1.0 / DT.to_decimal(u.ms)))
    n_bins = global_counts.size // bin_steps
    rate_1ms = global_counts[: n_bins * bin_steps].reshape(n_bins, bin_steps).sum(1)
    rate_1ms = rate_1ms / (config.num_neurons * 0.001)
    rate_cv_1ms = float(rate_1ms.std(ddof=0) / rate_1ms.mean())

    centered = global_rate_hz - global_rate_hz.mean()
    frequencies_hz, power_hz = scipy.signal.welch(
        centered,
        fs=1.0 / DT.to_decimal(u.second),
        window="hann",
        nperseg=min(8192, centered.size),
        noverlap=min(4096, centered.size // 2),
        detrend="constant",
        scaling="density",
    )
    search = (frequencies_hz >= 1.0) & (frequencies_hz <= 1000.0)
    search_power = power_hz[search]
    search_freq = frequencies_hz[search]
    dominant_index = int(np.argmax(search_power))
    dominant_frequency = float(search_freq[dominant_index])
    peak_to_background = float(
        search_power[dominant_index] / max(float(np.median(search_power)), 1e-30)
    )

    exc_summary = summarize_cv(exc_cv)
    inh_summary = summarize_cv(inh_cv)
    all_summary = summarize_cv(all_cv)
    metrics = {
        "exc_firing_rate_hz": float(exc_counts.sum() / config.n_exc / duration_s),
        "inh_firing_rate_hz": float(inh_counts.sum() / config.n_inh / duration_s),
        "overall_firing_rate_hz": float(global_counts.sum() / config.num_neurons / duration_s),
        "exc_isi_cv_mean": exc_summary["mean"],
        "exc_isi_cv_median": exc_summary["median"],
        "exc_isi_cv_eligible_neurons": exc_summary["eligible_neurons"],
        "inh_isi_cv_mean": inh_summary["mean"],
        "inh_isi_cv_median": inh_summary["median"],
        "inh_isi_cv_eligible_neurons": inh_summary["eligible_neurons"],
        "isi_cv_all_mean": all_summary["mean"],
        "population_rate_cv_1ms": rate_cv_1ms,
        "dominant_frequency_hz": dominant_frequency,
        "spectral_peak_to_background": peak_to_background,
    }
    raw = {
        "exc_counts": exc_counts,
        "inh_counts": inh_counts,
        "global_rate_hz": global_rate_hz.astype(np.float32),
        "exc_isi_cv_values": exc_cv,
        "inh_isi_cv_values": inh_cv,
        "frequencies_hz": frequencies_hz.astype(np.float32),
        "power_hz": power_hz.astype(np.float32),
    }
    return metrics, raw


def _within(value: float, target: float, relative_tolerance: float) -> bool:
    return abs(value - target) <= relative_tolerance * target


def assess_condition(name: str, metrics: dict) -> tuple[str, list[str]]:
    cv = metrics["isi_cv_all_mean"]
    rate = metrics["overall_firing_rate_hz"]
    frequency = metrics["dominant_frequency_hz"]
    failures = []
    if name == "synchronous_regular":
        if not cv < 0.5:
            failures.append("mean ISI CV is not below 0.5")
        if not metrics["population_rate_cv_1ms"] >= 0.2:
            failures.append("1 ms population-rate CV does not show synchrony")
    elif name == "fast_synchronous_irregular":
        if not cv >= 0.7:
            failures.append("mean ISI CV is below 0.7")
        if not _within(rate, 60.7, 0.2):
            failures.append("firing rate differs from 60.7 Hz by more than 20%")
        if not _within(frequency, 180.0, 0.2):
            failures.append("global frequency differs from 180 Hz by more than 20%")
        if not rate < frequency:
            failures.append("neuron rate is not below global frequency")
    elif name == "asynchronous_irregular":
        if not cv >= 0.7:
            failures.append("mean ISI CV is below 0.7")
        if not _within(rate, 37.7, 0.2):
            failures.append("firing rate differs from 37.7 Hz by more than 20%")
        if not metrics["population_rate_cv_1ms"] < 0.2:
            failures.append("1 ms population-rate CV is not below 0.2")
    elif name == "slow_synchronous_irregular":
        if not cv >= 0.7:
            failures.append("mean ISI CV is below 0.7")
        if not _within(rate, 5.5, 0.3):
            failures.append("firing rate differs from 5.5 Hz by more than 30%")
        if not _within(frequency, 22.0, 0.3):
            failures.append("global frequency differs from 22 Hz by more than 30%")
        if not rate < frequency:
            failures.append("neuron rate is not below global frequency")
    else:
        raise ValueError(f"unknown condition {name!r}")
    return ("reproduced" if not failures else "not_reproduced"), failures


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_config(config: NetworkConfig, seed: int) -> dict:
    return {
        "network": asdict(config),
        "dt_ms": DT.to_decimal(u.ms),
        "tau_m_ms": TAU_M.to_decimal(u.ms),
        "tau_ref_ms": TAU_REF.to_decimal(u.ms),
        "threshold_mv": V_THRESHOLD.to_decimal(u.mV),
        "reset_mv": V_RESET.to_decimal(u.mV),
        "jump_mv": JUMP.to_decimal(u.mV),
        "delay_ms": DELAY.to_decimal(u.ms),
        "nu_threshold_hz": NU_THRESHOLD.to_decimal(u.Hz),
        "conditions": [
            {"panel": panel, "name": name, "g": g, "eta": eta, "display_ms": display_ms}
            for panel, name, g, eta, display_ms in CONDITIONS
        ],
        "seed": seed,
        "graph_seed": seed,
        "probe_seed": seed + 303,
        "initial_seed_base": seed + 1_000,
        "external_seed_base": seed + 2_000,
    }


def runtime_provenance() -> dict:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "dependencies": versions,
    }


def write_manifest(output_dir: Path) -> Path:
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
    manifest_path = output_dir / "artifact-manifest.json"
    manifest_path.write_text(json.dumps({"files": files}, indent=2), encoding="ascii")
    return manifest_path


def write_assessment(output_dir: Path, rows: list[dict]) -> Path:
    lines = [
        "# Result assessment",
        "",
        "The verdicts apply the locked prospective criteria without tuning.",
        "",
        "| Panel | Requested state | Verdict | Rate (Hz) | ISI CV | Dominant frequency (Hz) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['panel']} | `{row['condition']}` | `{row['verdict']}` | "
            f"{row['overall_firing_rate_hz']:.3f} | {row['isi_cv_all_mean']:.3f} | "
            f"{row['dominant_frequency_hz']:.3f} |"
        )
    lines.extend(["", "## Deterministic findings", ""])
    for row in rows:
        if row["failures"]:
            lines.append(
                f"- Panel {row['panel']}: " + "; ".join(row["failures"]) + "."
            )
        else:
            lines.append(f"- Panel {row['panel']}: all locked predicates passed.")
    lines.extend(
        [
            "",
            "## Claim-evidence matrix",
            "",
            "| Claim | Evidence | Boundary |",
            "|---|---|---|",
            "| Each panel is or is not reproduced | `metrics.json` and `raw/*.npz` | Applies only to this finite seeded realization and the locked predicates. |",
            "| Frequencies come from global activity | `frequencies_hz` and `power_hz` in each raw file | Welch settings and search band are fixed in source code. |",
            "| The final image derives from accepted evidence | Figure hashes and raw-file hashes | The renderer does not alter simulation data. |",
            "",
            "No phase region, pixel identity, or exact author-RNG parity is claimed.",
        ]
    )
    path = output_dir / "result-assessment.md"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def run_experiment(config: NetworkConfig, seed: int, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    config_payload = scientific_config(config, seed)
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="ascii"
    )

    rows = []
    probe_indices = choose_probe_indices(config, seed + 303)
    with brainstate.environ.context(dt=DT, precision=32):
        exc_conn, inh_conn, exc_indices, inh_indices = make_connectivity(config, seed)
        graph_hashes = {
            "exc_indices_sha256": _hash_array(exc_indices),
            "inh_indices_sha256": _hash_array(inh_indices),
            "probe_indices_sha256": _hash_array(probe_indices),
        }
        (output_dir / "graph-hashes.json").write_text(
            json.dumps(graph_hashes, indent=2), encoding="ascii"
        )
        net = SparseEINetwork(
            config,
            exc_conn=exc_conn,
            inh_conn=inh_conn,
            external_seed=seed + 2_000,
        )
        brainstate.nn.init_all_states(net)
        runner = build_runner(net)

        for condition_index, (panel, name, g, eta, display_ms) in enumerate(CONDITIONS):
            reset_run(
                net,
                initial_seed=seed + 1_000 + condition_index,
                external_seed=seed + 2_000 + condition_index,
            )
            started = time.perf_counter()
            spikes = np.asarray(
                jax.block_until_ready(
                    runner(
                        jnp.asarray(g, dtype=brainstate.environ.dftype()),
                        jnp.asarray(eta, dtype=brainstate.environ.dftype()),
                    )
                ),
                dtype=bool,
            )
            runtime_seconds = time.perf_counter() - started
            metrics, raw = analyze_spikes(spikes, config)
            verdict, failures = assess_condition(name, metrics)
            analysis = spikes[config.burn_steps :]
            display_steps = int(round(display_ms / DT.to_decimal(u.ms)))
            display_raster = analysis[-display_steps:, probe_indices]
            display_rate = raw["global_rate_hz"][-display_steps:]
            raw_path = raw_dir / f"panel-{panel}_{name}.npz"
            np.savez_compressed(
                raw_path,
                panel=np.asarray(panel),
                condition=np.asarray(name),
                g=np.asarray(g, dtype=np.float32),
                eta=np.asarray(eta, dtype=np.float32),
                dt_ms=np.asarray(DT.to_decimal(u.ms), dtype=np.float32),
                display_time_ms=(
                    np.arange(display_steps, dtype=np.float32) * DT.to_decimal(u.ms)
                ),
                probe_indices=probe_indices,
                display_raster=display_raster,
                display_global_rate_hz=display_rate,
                **raw,
            )
            row = {
                "panel": panel,
                "condition": name,
                "g": g,
                "eta": eta,
                "runtime_seconds": runtime_seconds,
                "verdict": verdict,
                "failures": failures,
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del spikes, analysis

    (output_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="ascii"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="ascii") as stream:
        csv_rows = [{**row, "failures": "; ".join(row["failures"])} for row in rows]
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_dir / "provenance.json").write_text(
        json.dumps(runtime_provenance(), indent=2), encoding="ascii"
    )
    write_assessment(output_dir, rows)
    write_manifest(output_dir)
    return rows


def render_figure8(results_dir: Path, output_path: Path) -> Path:
    """Render the paper layout from review-accepted raw arrays."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 8), constrained_layout=True)
    positions = {
        "A": (0, 0),
        "B": (0, 1),
        "C": (2, 0),
        "D": (2, 1),
    }
    for panel, name, g, eta, _ in CONDITIONS:
        data = np.load(results_dir / "raw" / f"panel-{panel}_{name}.npz")
        raster_row, column = positions[panel]
        raster_ax = axes[raster_row, column]
        rate_ax = axes[raster_row + 1, column]
        time_ms = data["display_time_ms"]
        step, neuron = np.nonzero(data["display_raster"])
        raster_ax.scatter(time_ms[step], neuron, s=2.0, color="black", marker=".")
        raster_ax.set_title(f"{panel}  g={g:g},  nu_ext/nu_thr={eta:g}")
        raster_ax.set_ylabel("Neuron")
        raster_ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
        raster_ax.set_ylim(-1, 50)
        raster_ax.set_xticklabels([])

        rate_hz = data["display_global_rate_hz"]
        rate_ax.plot(time_ms, rate_hz, color="black", linewidth=0.7)
        rate_ax.axhline(float(rate_hz.mean()), color="black", linestyle="--", linewidth=0.8)
        rate_ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
        rate_ax.set_ylim(bottom=0)
        rate_ax.set_ylabel("Rate (Hz)")
        rate_ax.set_xlabel("Time (ms)")
    fig.suptitle("Brunel (2000), Figure 8 reproduction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def smoke_config() -> NetworkConfig:
    return NetworkConfig(
        n_exc=800,
        n_inh=200,
        exc_indegree=80,
        inh_indegree=20,
        external_indegree=80,
        burn_ms=100.0,
        analysis_ms=1_000.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=17_729)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figure-from", type=Path)
    parser.add_argument("--figure-path", type=Path)
    args = parser.parse_args()

    if args.figure_from is not None:
        if args.figure_path is None:
            parser.error("--figure-path is required with --figure-from")
        print(render_figure8(args.figure_from, args.figure_path))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for simulation")
    run_experiment(smoke_config() if args.smoke else NetworkConfig(), args.seed, args.output_dir)


if __name__ == "__main__":
    main()
