from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-brainx-lif")

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy import signal


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_or_verify_json(path: Path, value: Any) -> None:
    serialized = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"frozen artifact differs from active run: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def derive_seed_streams(root_seed: int) -> dict[str, Any]:
    connectivity_seed, initialization_seed, external_seed = np.random.SeedSequence(
        root_seed
    ).spawn(3)
    return {
        "connectivity": connectivity_seed,
        "initialization": initialization_seed,
        "external": int(external_seed.generate_state(1, dtype=np.uint32)[0]),
    }


def _fixed_sources(
    rng: np.random.Generator,
    n_pre: int,
    n_post: int,
    degree: int,
    source_offset: int,
) -> np.ndarray:
    if degree >= n_pre:
        raise ValueError("degree must leave room to exclude an autapse")
    indices = np.empty((n_post, degree), dtype=np.int32)
    for target in range(n_post):
        local_target = target - source_offset
        if 0 <= local_target < n_pre:
            row = rng.choice(n_pre - 1, size=degree, replace=False, shuffle=False)
            row += row >= local_target
        else:
            row = rng.choice(n_pre, size=degree, replace=False, shuffle=False)
        indices[target] = row
    return indices


def build_random_structures(
    root_seed: int,
    *,
    ne: int,
    ni: int,
    ce: int,
    ci: int,
    sample_size: int,
    reset_mV: float,
    threshold_mV: float,
) -> dict[str, Any]:
    streams = derive_seed_streams(root_seed)
    conn_rng = np.random.default_rng(streams["connectivity"])
    init_rng = np.random.default_rng(streams["initialization"])
    n_total = ne + ni
    exc_indices = _fixed_sources(conn_rng, ne, n_total, ce, 0)
    inh_indices = _fixed_sources(conn_rng, ni, n_total, ci, ne)
    initial_voltage_mV = init_rng.uniform(
        reset_mV, threshold_mV, size=n_total
    ).astype(np.float32)
    sample_ids = np.sort(
        init_rng.choice(n_total, size=sample_size, replace=False).astype(np.int32)
    )
    return {
        "exc_indices": exc_indices,
        "inh_indices": inh_indices,
        "initial_voltage_mV": initial_voltage_mV,
        "sample_ids": sample_ids,
        "external_seed": streams["external"],
    }


def validate_connectivity(
    exc_indices: np.ndarray,
    inh_indices: np.ndarray,
    *,
    ne: int,
    ni: int,
    ce: int,
    ci: int,
) -> None:
    n_total = ne + ni
    if exc_indices.shape != (n_total, ce):
        raise ValueError(f"wrong E index shape: {exc_indices.shape}")
    if inh_indices.shape != (n_total, ci):
        raise ValueError(f"wrong I index shape: {inh_indices.shape}")
    if exc_indices.min() < 0 or exc_indices.max() >= ne:
        raise ValueError("E source index out of bounds")
    if inh_indices.min() < 0 or inh_indices.max() >= ni:
        raise ValueError("I source index out of bounds")
    if np.any(np.diff(np.sort(exc_indices, axis=1), axis=1) == 0):
        raise ValueError("duplicate E source within a target")
    if np.any(np.diff(np.sort(inh_indices, axis=1), axis=1) == 0):
        raise ValueError("duplicate I source within a target")
    if np.any(exc_indices[np.arange(ne)] == np.arange(ne)[:, None]):
        raise ValueError("E autapse detected")
    inhibitory_targets = np.arange(ni)
    if np.any(inh_indices[ne + inhibitory_targets] == inhibitory_targets[:, None]):
        raise ValueError("I autapse detected")


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


class FixedFanInComm(brainstate.nn.Module):
    def __init__(
        self,
        indices: np.ndarray,
        *,
        n_pre: int,
        n_post: int,
        weight: u.Quantity,
    ):
        super().__init__()
        self.connectivity = brainevent.FixedNumPerPost(
            (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(indices)),
            shape=(n_pre, n_post),
        )
        self.weight = brainstate.State(weight)

    def update(self, spikes: jax.Array) -> u.Quantity:
        counts = brainevent.BinaryArray(spikes) @ self.connectivity
        return counts * self.weight.value


class ExternalJumpComm(brainstate.nn.Module):
    def __init__(self, weight: u.Quantity):
        super().__init__()
        self.weight = weight

    def update(self, counts: jax.Array) -> u.Quantity:
        return counts * self.weight


class CurrentLIFNetwork(brainstate.nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        random_structures: dict[str, Any],
    ):
        super().__init__()
        model = config["model"]
        protocol = config["protocol"]
        self.ne = int(model["NE"])
        self.ni = int(model["NI"])
        self.n_total = self.ne + self.ni
        self.je = float(model["JE_mV"]) * u.mV
        self.dt = float(model["dt_ms"]) * u.ms
        self.delay_steps = int(round(model["delay_ms"] / model["dt_ms"]))
        self.transient_step = int(round(protocol["transient_ms"] / model["dt_ms"]))
        self.initial_voltage = (
            jnp.asarray(random_structures["initial_voltage_mV"]) * u.mV
        )
        self.sample_ids = jnp.asarray(random_structures["sample_ids"])
        self.external_seed = int(random_structures["external_seed"])

        self.neurons = brainpy.state.LIFRef(
            self.n_total,
            R=1.0 * u.ohm,
            tau=float(model["tau_m_ms"]) * u.ms,
            tau_ref=float(model["t_ref_ms"]) * u.ms,
            V_rest=float(model["rest_mV"]) * u.mV,
            V_th=float(model["threshold_mV"]) * u.mV,
            V_reset=float(model["reset_mV"]) * u.mV,
            spk_reset="hard",
            V_initializer=braintools.init.Constant(
                float(model["reset_mV"]) * u.mV
            ),
        )
        self.exc_comm = FixedFanInComm(
            random_structures["exc_indices"],
            n_pre=self.ne,
            n_post=self.n_total,
            weight=self.je,
        )
        self.inh_comm = FixedFanInComm(
            random_structures["inh_indices"],
            n_pre=self.ni,
            n_post=self.n_total,
            weight=-self.je,
        )
        self.exc_projection = brainpy.state.DeltaProj(
            comm=self.exc_comm, post=self.neurons, label="recurrent_exc"
        )
        self.inh_projection = brainpy.state.DeltaProj(
            comm=self.inh_comm, post=self.neurons, label="recurrent_inh"
        )
        self.external_projection = brainpy.state.DeltaProj(
            comm=ExternalJumpComm(self.je), post=self.neurons, label="external"
        )
        self.delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((self.n_total,), jnp.bool_),
            float(model["delay_ms"]) * u.ms,
        )
        self.external_lambda = brainstate.State(jnp.asarray(0.0, dtype=jnp.float32))
        self.external_rng = brainstate.random.RandomState(self.external_seed)
        self.last_spike_step = brainstate.HiddenState(
            jnp.full(self.n_total, -1, dtype=jnp.int32)
        )
        self.isi_count = brainstate.HiddenState(
            jnp.zeros(self.n_total, dtype=jnp.int32)
        )
        self.isi_sum = brainstate.HiddenState(
            jnp.zeros(self.n_total, dtype=jnp.float32)
        )
        self.isi_sq_sum = brainstate.HiddenState(
            jnp.zeros(self.n_total, dtype=jnp.float32)
        )
        self.zero_current = jnp.zeros(self.n_total, dtype=jnp.float32) * u.mA

    def set_condition(self, g: float, eta: float, config: dict[str, Any]) -> None:
        model = config["model"]
        nu_ext = (
            eta
            * float(model["threshold_mV"])
            * u.mV
            / (
                self.je
                * int(model["CE"])
                * (float(model["tau_m_ms"]) * u.ms)
            )
        )
        external_lambda = float(int(model["Cext"]) * nu_ext * self.dt)
        self.inh_comm.weight.value = -float(g) * self.je
        self.external_lambda.value = jnp.asarray(external_lambda, dtype=jnp.float32)

    def reset(self) -> None:
        brainstate.nn.reset_all_states(self)
        self.neurons.V.value = self.initial_voltage
        self.external_rng.seed(self.external_seed)
        self.last_spike_step.value = jnp.full(
            self.n_total, -1, dtype=jnp.int32
        )
        self.isi_count.value = jnp.zeros(self.n_total, dtype=jnp.int32)
        self.isi_sum.value = jnp.zeros(self.n_total, dtype=jnp.float32)
        self.isi_sq_sum.value = jnp.zeros(self.n_total, dtype=jnp.float32)

    def update(self, t: u.Quantity, step: jax.Array):
        with brainstate.environ.context(t=t, i=step):
            # The source spike is available one loop after emission. Buffer step 14
            # is therefore 15 simulation steps behind this target update.
            self.delay.update(self.neurons.get_spike() != 0.0)
            delayed = self.delay.retrieve_at_step(
                jnp.asarray(self.delay_steps - 1, dtype=jnp.int32)
            )
            self.exc_projection(delayed[: self.ne])
            self.inh_projection(delayed[self.ne :])
            external_counts = self.external_rng.poisson(
                lam=self.external_lambda.value,
                size=(self.n_total,),
            )
            self.external_projection(external_counts)
            spikes = self.neurons(self.zero_current) != 0.0

            in_analysis = step >= self.transient_step
            previous = self.last_spike_step.value
            interval = (step - previous).astype(jnp.float32)
            valid_interval = spikes & in_analysis & (previous >= self.transient_step)
            self.isi_count.value = self.isi_count.value + valid_interval.astype(jnp.int32)
            self.isi_sum.value = self.isi_sum.value + jnp.where(
                valid_interval, interval, 0.0
            )
            self.isi_sq_sum.value = self.isi_sq_sum.value + jnp.where(
                valid_interval, interval * interval, 0.0
            )
            self.last_spike_step.value = jnp.where(
                spikes & in_analysis, step, previous
            )

            return (
                jnp.sum(spikes[: self.ne], dtype=jnp.int32),
                jnp.sum(spikes[self.ne :], dtype=jnp.int32),
                spikes[self.sample_ids],
            )

    def isi_cv(self, minimum_intervals: int) -> tuple[np.ndarray, np.ndarray]:
        count = np.asarray(self.isi_count.value)
        total = np.asarray(self.isi_sum.value)
        total_sq = np.asarray(self.isi_sq_sum.value)
        valid = count >= minimum_intervals
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        second = np.divide(
            total_sq, count, out=np.zeros_like(total_sq), where=count > 0
        )
        variance = np.maximum(second - mean * mean, 0.0)
        cv = np.full(self.n_total, np.nan, dtype=np.float32)
        cv[valid] = np.sqrt(variance[valid]) / mean[valid]
        return cv, valid


def create_rollout(
    network: CurrentLIFNetwork,
    config: dict[str, Any],
):
    model = config["model"]
    protocol = config["protocol"]
    n_steps = int(round(protocol["duration_ms"] / model["dt_ms"]))
    steps = jnp.arange(n_steps, dtype=jnp.int32)
    times = steps * (float(model["dt_ms"]) * u.ms)

    def rollout():
        return brainstate.transform.for_loop(network.update, times, steps)

    return brainstate.transform.jit(rollout)


def spectral_metrics(
    global_rate_hz: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    model = config["model"]
    spectrum = config["spectrum"]
    fs_hz = 1000.0 / float(model["dt_ms"])
    frequencies, power = signal.welch(
        global_rate_hz - np.mean(global_rate_hz),
        fs=fs_hz,
        window=spectrum["window"],
        nperseg=int(spectrum["nperseg"]),
        noverlap=int(spectrum["noverlap"]),
        detrend=spectrum["detrend"],
        scaling=spectrum["scaling"],
    )
    low_hz, high_hz = spectrum["search_hz"]
    search = (frequencies >= low_hz) & (frequencies <= high_hz)
    search_indices = np.flatnonzero(search)
    peak_index = search_indices[np.argmax(power[search])]
    dominant_hz = float(frequencies[peak_index])
    exclusion = float(spectrum["background_exclusion_hz"])
    background_mask = search & (np.abs(frequencies - dominant_hz) > exclusion)
    background = float(np.median(power[background_mask]))
    prominence = float(power[peak_index] / max(background, np.finfo(float).tiny))
    half_width = float(spectrum["narrowband_half_width_hz"])
    narrowband = search & (np.abs(frequencies - dominant_hz) <= half_width)
    fraction = float(np.sum(power[narrowband]) / np.sum(power[search]))
    significant = bool(
        prominence >= float(spectrum["min_prominence_ratio"])
        and fraction >= float(spectrum["min_narrowband_fraction"])
    )
    return {
        "frequencies_hz": frequencies,
        "power_hz": power,
        "dominant_frequency_hz": dominant_hz,
        "peak_prominence_ratio": prominence,
        "narrowband_power_fraction": fraction,
        "significant_narrowband_peak": significant,
    }


def analyze_rollout(
    exc_counts: np.ndarray,
    inh_counts: np.ndarray,
    sample_spikes: np.ndarray,
    network: CurrentLIFNetwork,
    config: dict[str, Any],
) -> dict[str, Any]:
    model = config["model"]
    protocol = config["protocol"]
    regularity = config["regularity"]
    dt_seconds = float(model["dt_ms"]) * 1e-3
    transient_step = int(round(protocol["transient_ms"] / model["dt_ms"]))
    exc_rate = exc_counts.astype(np.float64) / (int(model["NE"]) * dt_seconds)
    inh_rate = inh_counts.astype(np.float64) / (int(model["NI"]) * dt_seconds)
    global_rate = (exc_counts + inh_counts).astype(np.float64) / (
        (int(model["NE"]) + int(model["NI"])) * dt_seconds
    )
    cv, cv_valid = network.isi_cv(
        int(regularity["minimum_intervals_per_neuron"])
    )
    spectral = spectral_metrics(global_rate[transient_step:], config)
    post_exc = exc_rate[transient_step:]
    post_inh = inh_rate[transient_step:]
    post_global = global_rate[transient_step:]
    final_voltage_mV = np.asarray(network.neurons.V.value.to_decimal(u.mV))
    final_voltage_all_finite = bool(np.all(np.isfinite(final_voltage_mV)))
    if not final_voltage_all_finite:
        raise FloatingPointError("non-finite final membrane voltage")
    return {
        "exc_rate_hz": exc_rate,
        "inh_rate_hz": inh_rate,
        "global_rate_hz": global_rate,
        "sample_spikes": sample_spikes.astype(bool),
        "isi_cv": cv,
        "isi_cv_valid": cv_valid,
        "final_voltage_mV": final_voltage_mV,
        "final_voltage_all_finite": final_voltage_all_finite,
        "final_voltage_sha256": array_sha256(final_voltage_mV),
        "final_voltage_min_mV": float(np.min(final_voltage_mV)),
        "final_voltage_max_mV": float(np.max(final_voltage_mV)),
        "mean_exc_rate_hz": float(np.mean(post_exc)),
        "mean_inh_rate_hz": float(np.mean(post_inh)),
        "mean_global_rate_hz": float(np.mean(post_global)),
        "global_rate_cv": float(np.std(post_global) / np.mean(post_global)),
        "valid_isi_cv_fraction": float(np.mean(cv_valid)),
        "median_isi_cv": float(np.nanmedian(cv)),
        **spectral,
    }


def save_run_result(
    output_dir: Path,
    *,
    seed: int,
    condition: dict[str, Any],
    sample_ids: np.ndarray,
    analysis: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    run_id = f"seed-{seed}_{condition['id']}"
    npz_path = output_dir / "raw" / f"{run_id}.npz"
    json_path = output_dir / "metrics" / f"{run_id}.json"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        exc_rate_hz=analysis["exc_rate_hz"],
        inh_rate_hz=analysis["inh_rate_hz"],
        global_rate_hz=analysis["global_rate_hz"],
        sample_spikes=analysis["sample_spikes"],
        sample_ids=sample_ids,
        isi_cv=analysis["isi_cv"],
        isi_cv_valid=analysis["isi_cv_valid"],
        final_voltage_mV=analysis["final_voltage_mV"],
        frequencies_hz=analysis["frequencies_hz"],
        power_hz=analysis["power_hz"],
    )
    metric = {
        "run_id": run_id,
        "seed": seed,
        "condition_id": condition["id"],
        "g": condition["g"],
        "eta": condition["eta"],
        "requested_regime": condition["requested_regime"],
        "elapsed_seconds": elapsed_seconds,
        "mean_exc_rate_hz": analysis["mean_exc_rate_hz"],
        "mean_inh_rate_hz": analysis["mean_inh_rate_hz"],
        "mean_global_rate_hz": analysis["mean_global_rate_hz"],
        "global_rate_cv": analysis["global_rate_cv"],
        "valid_isi_cv_fraction": analysis["valid_isi_cv_fraction"],
        "median_isi_cv": analysis["median_isi_cv"],
        "final_voltage_all_finite": analysis["final_voltage_all_finite"],
        "final_voltage_sha256": analysis["final_voltage_sha256"],
        "final_voltage_min_mV": analysis["final_voltage_min_mV"],
        "final_voltage_max_mV": analysis["final_voltage_max_mV"],
        "dominant_frequency_hz": analysis["dominant_frequency_hz"],
        "peak_prominence_ratio": analysis["peak_prominence_ratio"],
        "narrowband_power_fraction": analysis["narrowband_power_fraction"],
        "significant_narrowband_peak": analysis["significant_narrowband_peak"],
        "raw_path": str(npz_path.relative_to(output_dir)),
    }
    json_path.write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
    return metric


def classify_conditions(
    run_metrics: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    spectrum = config["spectrum"]
    regularity = config["regularity"]
    classifications = []
    for condition in config["protocol"]["conditions"]:
        rows = [row for row in run_metrics if row["condition_id"] == condition["id"]]
        peaks = np.asarray([row["dominant_frequency_hz"] for row in rows])
        significant = np.asarray(
            [row["significant_narrowband_peak"] for row in rows], dtype=bool
        )
        consistency_limit = max(
            float(spectrum["peak_consistency_abs_hz"]),
            float(spectrum["peak_consistency_relative"]) * float(np.median(peaks)),
        )
        if np.all(significant) and float(np.ptp(peaks)) <= consistency_limit:
            synchrony = "synchronous"
        elif not np.any(significant):
            synchrony = "asynchronous"
        else:
            synchrony = "synchrony-indeterminate"

        valid_fractions = np.asarray([row["valid_isi_cv_fraction"] for row in rows])
        cv_available = bool(
            np.all(valid_fractions >= float(regularity["minimum_valid_fraction"]))
        )
        cv_values = []
        for row in rows:
            with np.load(output_dir / row["raw_path"]) as data:
                cv_values.append(data["isi_cv"][data["isi_cv_valid"]])
        pooled_cv = np.concatenate(cv_values) if cv_values else np.asarray([])
        pooled_median_cv = float(np.median(pooled_cv)) if pooled_cv.size else float("nan")
        if not cv_available:
            regularity_label = "regularity-indeterminate"
        elif pooled_median_cv <= float(regularity["regular_max_median_cv"]):
            regularity_label = "regular"
        elif pooled_median_cv >= float(regularity["irregular_min_median_cv"]):
            regularity_label = "irregular"
        else:
            regularity_label = "regularity-indeterminate"

        median_peak = float(np.median(peaks))
        speed = (
            "slow"
            if median_peak < float(spectrum["slow_fast_boundary_hz"])
            else "fast"
        )
        if synchrony == "synchronous" and regularity_label == "regular":
            measured = "synchronous regular"
        elif synchrony == "synchronous" and regularity_label == "irregular":
            measured = f"{speed} synchronous irregular"
        elif synchrony == "asynchronous" and regularity_label == "irregular":
            measured = "asynchronous irregular"
        else:
            measured = f"{synchrony}, {regularity_label}"
        classifications.append(
            {
                "condition_id": condition["id"],
                "g": condition["g"],
                "eta": condition["eta"],
                "requested_regime": condition["requested_regime"],
                "measured_regime": measured,
                "verified": measured == condition["requested_regime"],
                "synchrony": synchrony,
                "regularity": regularity_label,
                "median_dominant_frequency_hz": median_peak,
                "peak_frequency_range_hz": float(np.ptp(peaks)),
                "peak_consistency_limit_hz": consistency_limit,
                "pooled_median_isi_cv": pooled_median_cv,
                "mean_exc_rate_hz": float(
                    np.mean([row["mean_exc_rate_hz"] for row in rows])
                ),
                "mean_inh_rate_hz": float(
                    np.mean([row["mean_inh_rate_hz"] for row in rows])
                ),
                "seed_metrics": rows,
            }
        )
    path = output_dir / "condition_assessment.json"
    path.write_text(json.dumps(classifications, indent=2) + "\n", encoding="utf-8")
    return classifications


def environment_record() -> dict[str, Any]:
    packages = {}
    for package_name in (
        "BrainX",
        "brainpy",
        "brainstate",
        "brainevent",
        "braintools",
        "brainunit",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
    ):
        try:
            from importlib.metadata import version

            packages[package_name] = version(package_name)
        except Exception:
            packages[package_name] = "unknown"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "packages": packages,
    }


def write_metrics_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    fields = [key for key in metrics[0] if key != "raw_path"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in metrics)


def resolve_run_config(
    config: dict[str, Any],
    *,
    smoke: bool = False,
    selected_seed: int | None = None,
) -> dict[str, Any]:
    model = dict(config["model"])
    protocol = dict(config["protocol"])
    spectrum = dict(config["spectrum"])
    if smoke:
        model.update({"NE": 80, "NI": 20, "CE": 8, "CI": 2})
        protocol.update(
            {
                "duration_ms": 40.0,
                "transient_ms": 10.0,
                "seeds": [11],
                "sample_size": 10,
                "conditions": [protocol["conditions"][0]],
            }
        )
        spectrum.update({"nperseg": 256, "noverlap": 128})
    elif selected_seed is not None:
        if selected_seed not in protocol["seeds"]:
            raise ValueError(f"seed {selected_seed} is not in the locked seed set")
        protocol["seeds"] = [selected_seed]
    return {
        **config,
        "model": model,
        "protocol": protocol,
        "spectrum": spectrum,
    }


def run_experiment(
    config: dict[str, Any],
    output_dir: Path,
    *,
    smoke: bool = False,
    selected_seed: int | None = None,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    active_config = resolve_run_config(
        config, smoke=smoke, selected_seed=selected_seed
    )
    model = active_config["model"]
    protocol = active_config["protocol"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_or_verify_json(output_dir / "config.json", active_config)
    write_or_verify_json(output_dir / "environment.json", environment_record())

    run_metrics: list[dict[str, Any]] = []
    connectivity_records = []
    with brainstate.environ.context(dt=float(model["dt_ms"]) * u.ms, fit=False):
        for seed in protocol["seeds"]:
            random_structures = build_random_structures(
                seed,
                ne=int(model["NE"]),
                ni=int(model["NI"]),
                ce=int(model["CE"]),
                ci=int(model["CI"]),
                sample_size=int(protocol["sample_size"]),
                reset_mV=float(model["reset_mV"]),
                threshold_mV=float(model["threshold_mV"]),
            )
            validate_connectivity(
                random_structures["exc_indices"],
                random_structures["inh_indices"],
                ne=int(model["NE"]),
                ni=int(model["NI"]),
                ce=int(model["CE"]),
                ci=int(model["CI"]),
            )
            conn_path = output_dir / "connectivity" / f"seed-{seed}.npz"
            conn_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                conn_path,
                exc_indices=random_structures["exc_indices"],
                inh_indices=random_structures["inh_indices"],
                initial_voltage_mV=random_structures["initial_voltage_mV"],
                sample_ids=random_structures["sample_ids"],
                external_seed=np.asarray(random_structures["external_seed"]),
            )
            connectivity_records.append(
                {
                    "seed": seed,
                    "path": str(conn_path.relative_to(output_dir)),
                    "exc_sha256": array_sha256(random_structures["exc_indices"]),
                    "inh_sha256": array_sha256(random_structures["inh_indices"]),
                }
            )
            network = CurrentLIFNetwork(active_config, random_structures)
            brainstate.nn.init_all_states(network)
            rollout = create_rollout(network, active_config)
            for condition in protocol["conditions"]:
                network.set_condition(condition["g"], condition["eta"], active_config)
                network.reset()
                started = time.perf_counter()
                exc_counts, inh_counts, sample_spikes = rollout()
                jax.block_until_ready(sample_spikes)
                elapsed = time.perf_counter() - started
                analysis = analyze_rollout(
                    np.asarray(exc_counts),
                    np.asarray(inh_counts),
                    np.asarray(sample_spikes),
                    network,
                    active_config,
                )
                metric = save_run_result(
                    output_dir,
                    seed=seed,
                    condition=condition,
                    sample_ids=random_structures["sample_ids"],
                    analysis=analysis,
                    elapsed_seconds=elapsed,
                )
                run_metrics.append(metric)
                message = json.dumps(metric, sort_keys=True)
                print(message, flush=True)
                if log_path is not None:
                    with log_path.open("a", encoding="utf-8") as log_file:
                        log_file.write(message + "\n")

    (output_dir / "connectivity_manifest.json").write_text(
        json.dumps(connectivity_records, indent=2) + "\n", encoding="utf-8"
    )
    write_metrics_csv(run_metrics, output_dir / "run_metrics.csv")
    classify_conditions(run_metrics, output_dir, active_config)
    return run_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    status_path = args.output_dir / "status.json"
    started_at = utc_now()
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "pid": os.getpid(),
                "started_at": started_at,
                "finished_at": None,
                "exit_code": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            json.dumps(
                {
                    "event": "start",
                    "time": started_at,
                    "pid": os.getpid(),
                    "mode": args.mode,
                    "seed": args.seed,
                },
                sort_keys=True,
            )
            + "\n"
        )
    try:
        run_experiment(
            config,
            args.output_dir,
            smoke=args.mode == "smoke",
            selected_seed=args.seed,
            log_path=log_path,
        )
    except Exception:
        finished_at = utc_now()
        with log_path.open("a", encoding="utf-8") as log_file:
            traceback.print_exc(file=log_file)
        status_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "exit_code": 1,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    finished_at = utc_now()
    status_path.write_text(
        json.dumps(
            {
                "status": "done",
                "pid": os.getpid(),
                "started_at": started_at,
                "finished_at": finished_at,
                "exit_code": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "exit_code").write_text("0\n", encoding="utf-8")


if __name__ == "__main__":
    main()
