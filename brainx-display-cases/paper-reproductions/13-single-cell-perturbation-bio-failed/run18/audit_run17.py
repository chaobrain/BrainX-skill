#!/usr/bin/env python3
"""Read-only integrity and statistical-support audit of completed run17."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "run17"
results = json.loads((SOURCE / "results.json").read_text(encoding="ascii"))
arrays = np.load(SOURCE / "trial_summaries.npz")
with (SOURCE / "pair_influences.csv").open(newline="", encoding="ascii") as handle:
    pair_rows = list(csv.DictReader(handle))

signal = np.asarray([float(row["signal_correlation"]) for row in pair_rows])
quantile_edges = np.quantile(signal, np.linspace(0.0, 1.0, 7))
quantile_bin = np.clip(np.digitize(signal, quantile_edges[1:-1]), 0, 5)
estimated = arrays["estimated_e_preference_deg"]
unique_pref, pref_counts = np.unique(estimated, return_counts=True)
audit = {
    "source_run": str(SOURCE),
    "status": results["status"],
    "all_pairing_checks_pass": bool(
        results["pairing_checks"]["noise_and_stimulus_max_abs_difference_nA"] == 0.0
        and results["pairing_checks"]["pre_photostim_exc_spike_mismatches"] == 0
        and results["pairing_checks"]["pre_photostim_inh_spike_mismatches"] == 0
    ),
    "csv_pair_rows": len(pair_rows),
    "reported_eligible_pairs": results["sample_counts"]["eligible_target_neighbor_pairs"],
    "all_numeric_results_finite": bool(np.isfinite(signal).all()),
    "signal_correlation": {
        "min": float(signal.min()),
        "max": float(signal.max()),
        "unique_values": int(np.unique(signal).size),
        "six_quantile_bin_counts": np.bincount(quantile_bin, minlength=6).tolist(),
        "descriptive_plot_has_empty_bins": bool(np.any(np.bincount(quantile_bin, minlength=6) == 0)),
    },
    "tuning": {
        "nonzero_direction_mean_fraction": float(np.mean(arrays["tuning_counts"] > 0)),
        "estimated_preference_counts": {
            str(float(direction)): int(count)
            for direction, count in zip(unique_pref, pref_counts)
        },
        "estimated_zero_degree_fraction": float(np.mean(estimated == 0.0)),
    },
    "failed_tests": [key for key, passed in results["tests"].items() if not passed],
}
(HERE / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="ascii")
print(json.dumps(audit, indent=2))
