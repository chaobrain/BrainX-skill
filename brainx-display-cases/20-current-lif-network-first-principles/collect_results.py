from __future__ import annotations

import hashlib
import json
from pathlib import Path

import braintools
import brainunit as u
import numpy as np

import lif_network


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
COMBINED = RUNS / "production-v2-combined-20260903"
SEED_DIRS = {
    11: RUNS / "production-v2-seed-11-cpu-20260903",
    29: RUNS / "production-v2-seed-29-cpu-20260903",
    47: RUNS / "production-v2-seed-47-cpu-20260903",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectral_summary(
    frequencies: np.ndarray,
    power: np.ndarray,
    config: dict,
) -> dict:
    spectrum = config["spectrum"]
    low_hz, high_hz = spectrum["search_hz"]
    search = (frequencies >= low_hz) & (frequencies <= high_hz)
    search_indices = np.flatnonzero(search)
    peak_index = int(search_indices[np.argmax(power[search])])
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
        "dominant_frequency_hz": dominant_hz,
        "peak_prominence_ratio": prominence,
        "narrowband_power_fraction": fraction,
        "significant_narrowband_peak": significant,
    }


def classify_spectral_variant(
    run_metrics: list[dict],
    base_assessment: list[dict],
    config: dict,
) -> list[dict]:
    spectrum = config["spectrum"]
    by_condition = {row["condition_id"]: row for row in base_assessment}
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
        regularity = by_condition[condition["id"]]["regularity"]
        speed = (
            "slow"
            if float(np.median(peaks)) < float(spectrum["slow_fast_boundary_hz"])
            else "fast"
        )
        if synchrony == "synchronous" and regularity == "regular":
            measured = "synchronous regular"
        elif synchrony == "synchronous" and regularity == "irregular":
            measured = f"{speed} synchronous irregular"
        elif synchrony == "asynchronous" and regularity == "irregular":
            measured = "asynchronous irregular"
        else:
            measured = f"{synchrony}, {regularity}"
        classifications.append(
            {
                "condition_id": condition["id"],
                "synchrony": synchrony,
                "measured_regime": measured,
                "median_dominant_frequency_hz": float(np.median(peaks)),
            }
        )
    return classifications


def braintools_psd_parity(
    metrics: list[dict],
    assessment: list[dict],
    config: dict,
) -> dict:
    model = config["model"]
    protocol = config["protocol"]
    spectrum = config["spectrum"]
    transient_step = int(round(protocol["transient_ms"] / model["dt_ms"]))
    alternate_metrics = []
    comparisons = []
    for metric in metrics:
        raw_path = COMBINED / metric["raw_path"]
        with np.load(raw_path) as data:
            trace = np.asarray(data["global_rate_hz"][transient_step:])
            scipy_frequencies = np.asarray(data["frequencies_hz"])
            scipy_power = np.asarray(data["power_hz"])
        frequencies, power = braintools.metric.power_spectral_density(
            trace - np.mean(trace),
            dt=float(model["dt_ms"]) * u.ms,
            nperseg=int(spectrum["nperseg"]),
            noverlap=int(spectrum["noverlap"]),
        )
        frequencies = np.asarray(frequencies)
        power = np.asarray(power)
        if frequencies.shape != scipy_frequencies.shape or not np.array_equal(
            frequencies, scipy_frequencies
        ):
            raise RuntimeError(f"BrainTools frequency-grid mismatch: {metric['run_id']}")
        summary = spectral_summary(frequencies, power, config)
        alternate_metrics.append({**metric, **summary})
        comparisons.append(
            {
                "run_id": metric["run_id"],
                "frequency_grid_exact": True,
                "scipy_dominant_frequency_hz": metric["dominant_frequency_hz"],
                "braintools_dominant_frequency_hz": summary["dominant_frequency_hz"],
                "dominant_frequency_equal": bool(
                    metric["dominant_frequency_hz"]
                    == summary["dominant_frequency_hz"]
                ),
                "scipy_significant_narrowband_peak": metric[
                    "significant_narrowband_peak"
                ],
                "braintools_significant_narrowband_peak": summary[
                    "significant_narrowband_peak"
                ],
                "significance_equal": bool(
                    metric["significant_narrowband_peak"]
                    == summary["significant_narrowband_peak"]
                ),
                "maximum_absolute_psd_difference": float(
                    np.max(np.abs(scipy_power - power))
                ),
                "psd_arrays_close_rtol_1e-4_atol_1e-5": bool(
                    np.allclose(scipy_power, power, rtol=1e-4, atol=1e-5)
                ),
            }
        )
    alternate_assessment = classify_spectral_variant(
        alternate_metrics, assessment, config
    )
    base_by_id = {row["condition_id"]: row for row in assessment}
    condition_comparisons = []
    for alternate in alternate_assessment:
        base = base_by_id[alternate["condition_id"]]
        condition_comparisons.append(
            {
                "condition_id": alternate["condition_id"],
                "scipy_measured_regime": base["measured_regime"],
                "braintools_measured_regime": alternate["measured_regime"],
                "classification_equal": bool(
                    base["measured_regime"] == alternate["measured_regime"]
                ),
                "scipy_median_dominant_frequency_hz": base[
                    "median_dominant_frequency_hz"
                ],
                "braintools_median_dominant_frequency_hz": alternate[
                    "median_dominant_frequency_hz"
                ],
            }
        )
    parity = {
        "purpose": "BrainTools parity at the documented SciPy Welch boundary",
        "all_frequency_grids_exact": all(
            row["frequency_grid_exact"] for row in comparisons
        ),
        "all_dominant_frequencies_equal": all(
            row["dominant_frequency_equal"] for row in comparisons
        ),
        "all_significance_predicates_equal": all(
            row["significance_equal"] for row in comparisons
        ),
        "all_condition_classifications_equal": all(
            row["classification_equal"] for row in condition_comparisons
        ),
        "run_comparisons": comparisons,
        "condition_comparisons": condition_comparisons,
    }
    if not parity["all_condition_classifications_equal"]:
        raise RuntimeError("BrainTools PSD changes a final condition classification")
    return parity


def main() -> None:
    COMBINED.mkdir(parents=True, exist_ok=True)
    config = lif_network.load_config()
    metrics = []
    checks = []
    manifest = []
    for seed, run_dir in SEED_DIRS.items():
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        exit_code = int((run_dir / "exit_code").read_text(encoding="utf-8"))
        if status["status"] != "done" or exit_code != 0:
            raise RuntimeError(f"seed {seed} did not complete mechanically")
        metric_paths = sorted((run_dir / "metrics").glob("*.json"))
        raw_paths = sorted((run_dir / "raw").glob("*.npz"))
        if len(metric_paths) != 4 or len(raw_paths) != 4:
            raise RuntimeError(f"seed {seed} has incomplete condition artifacts")
        for metric_path in metric_paths:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            raw_path = run_dir / metric["raw_path"]
            with np.load(raw_path) as data:
                expected_shapes = {
                    "exc_rate_hz": (50_000,),
                    "inh_rate_hz": (50_000,),
                    "global_rate_hz": (50_000,),
                    "sample_spikes": (50_000, 50),
                    "sample_ids": (50,),
                    "isi_cv": (12_500,),
                    "isi_cv_valid": (12_500,),
                    "final_voltage_mV": (12_500,),
                }
                for key, shape in expected_shapes.items():
                    if data[key].shape != shape:
                        raise RuntimeError(
                            f"{raw_path}: {key} shape {data[key].shape} != {shape}"
                        )
                finite_keys = (
                    "exc_rate_hz",
                    "inh_rate_hz",
                    "global_rate_hz",
                    "frequencies_hz",
                    "power_hz",
                    "final_voltage_mV",
                )
                if not all(np.all(np.isfinite(data[key])) for key in finite_keys):
                    raise RuntimeError(f"non-finite raw values in {raw_path}")
                valid_cv = data["isi_cv"][data["isi_cv_valid"]]
                if not np.all(np.isfinite(valid_cv)):
                    raise RuntimeError(f"non-finite valid CV values in {raw_path}")
                final_voltage_hash = lif_network.array_sha256(
                    data["final_voltage_mV"]
                )
                if not metric["final_voltage_all_finite"]:
                    raise RuntimeError(f"invalid final-voltage metric in {metric_path}")
                if final_voltage_hash != metric["final_voltage_sha256"]:
                    raise RuntimeError(f"final-voltage hash mismatch in {raw_path}")
            relative_raw = Path("..") / run_dir.name / metric["raw_path"]
            metric["raw_path"] = str(relative_raw)
            metrics.append(metric)
            checks.append(
                {
                    "run_id": metric["run_id"],
                    "status": "done",
                    "exit_code": exit_code,
                    "required_shapes": "pass",
                    "finite_values": "pass",
                    "final_voltage_shape": "pass",
                    "final_voltage_sha256": final_voltage_hash,
                    "raw_sha256": sha256(raw_path),
                }
            )
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                manifest.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

    metrics.sort(key=lambda row: (row["condition_id"], row["seed"]))
    lif_network.write_metrics_csv(metrics, COMBINED / "run_metrics.csv")
    assessment = lif_network.classify_conditions(metrics, COMBINED, config)
    psd_parity = braintools_psd_parity(metrics, assessment, config)
    (COMBINED / "braintools_psd_parity.json").write_text(
        json.dumps(psd_parity, indent=2) + "\n", encoding="utf-8"
    )
    (COMBINED / "mechanical_validation.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    (COMBINED / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (COMBINED / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Result assessment",
        "",
        "All twelve immutable CPU runs completed with exit code 0. Each raw file has the frozen shapes, finite rate/CV/frequency/power values, and a finite 12,500-neuron final membrane vector whose hash matches its metric record. BrainTools PSD parity preserves all final classifications.",
        "",
        "| Condition | Requested | Measured | Verified | Dominant frequency | Pooled median ISI CV | Mean E/I rate |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in assessment:
        lines.append(
            "| `(g, eta) = ({g:g}, {eta:g})` | {requested} | {measured} | {verified} | {frequency:.1f} Hz | {cv:.3f} | {e:.2f} / {i:.2f} Hz |".format(
                g=row["g"],
                eta=row["eta"],
                requested=row["requested_regime"],
                measured=row["measured_regime"],
                verified="yes" if row["verified"] else "no",
                frequency=row["median_dominant_frequency_hz"],
                cv=row["pooled_median_isi_cv"],
                e=row["mean_exc_rate_hz"],
                i=row["mean_inh_rate_hz"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim-evidence matrix",
            "",
            "| Claim | Evidence | Outcome |",
            "|---|---|---|",
        ]
    )
    for row in assessment:
        seed_summary = ", ".join(
            f"seed {seed['seed']}: {seed['dominant_frequency_hz']:.0f} Hz, CV {seed['median_isi_cv']:.3f}, peak={seed['significant_narrowband_peak']}"
            for seed in row["seed_metrics"]
        )
        lines.append(
            f"| `{row['requested_regime']}` at `(g, eta)=({row['g']:g}, {row['eta']:g})` | {seed_summary}; pooled CV `{row['pooled_median_isi_cv']:.3f}`; measured `{row['measured_regime']}` | {'supported' if row['verified'] else 'not supported'} |"
        )
    lines.extend(
        [
            "",
            "## Allowed conclusion",
            "",
            "Report only the frozen finite-simulation classifications above. Do not relabel indeterminate or contradictory conditions to match the requested names, and do not generalize beyond this implementation, initialization, timestep, duration, or seed set.",
            "",
        ]
    )
    (COMBINED / "RESULT_ASSESSMENT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
